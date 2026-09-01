import numpy as np
import pytest

from reservoir_backend.exceptions import AssimilationError
from reservoir_backend.io.udp_api import TwinUDPProtocol
from reservoir_backend.twin.online import OnlineAssimilationWorkflow


def test_checkpoint_rollback_restores_members() -> None:
    rng = np.random.default_rng(2)

    class _Twin:
        grid = type("G", (), {"n_cells": 2})()
        physics = type("P", (), {"model": "compositional_dpdp", "p_init": 1.0e7})()
        parameterization = type("M", (), {"n_params": 1})()

    members = np.array([[0.1, -0.2, 0.0]])
    wf = OnlineAssimilationWorkflow(twin=_Twin(), members=members.copy(), q_std=0.01)  # type: ignore[arg-type]
    ckpt = wf.snapshot(rng)
    wf.members = members + 1.0
    wf.restore(ckpt, rng)
    assert wf.members == pytest.approx(members)


def test_assimilate_rolls_back_when_qc_rejects_all() -> None:
    rng = np.random.default_rng(3)

    class _Twin:
        grid = type("G", (), {"n_cells": 2})()
        physics = type("P", (), {"model": "compositional_dpdp", "p_init": 1.0e7})()
        parameterization = type("M", (), {"n_params": 1})()

    members = np.array([[0.0, 0.1, -0.1]])
    wf = OnlineAssimilationWorkflow(twin=_Twin(), members=members.copy(), q_std=0.01)  # type: ignore[arg-type]
    y = np.full((1, 3), np.nan)
    with pytest.raises(AssimilationError):
        wf.assimilate(y, np.array([1.0]), np.array([0.2]), rng)
    assert wf.members == pytest.approx(members)


def test_udp_status_and_checkpoint() -> None:
    class _Twin:
        grid = type("G", (), {"n_cells": 2})()
        physics = type("P", (), {"model": "compositional_dpdp", "p_init": 1.0e7})()
        parameterization = type("M", (), {"n_params": 1})()

    members = np.array([[0.0, 0.2]])
    wf = OnlineAssimilationWorkflow(twin=_Twin(), members=members.copy())  # type: ignore[arg-type]
    proto = TwinUDPProtocol(workflow=wf)
    status = proto.handle_bytes(b'{"cmd":"status"}')
    assert b'"ok": true' in status
    ck = proto.handle_bytes(b'{"cmd":"checkpoint","request_id":"r1"}')
    assert b"checkpoint" in ck
    assert b'"request_id": "r1"' in ck or b'"request_id":"r1"' in ck
    proto.workflow.members = members + 5.0
    rb = proto.handle_bytes(b'{"cmd":"rollback"}')
    assert b'"ok": true' in rb
    assert proto.workflow.members == pytest.approx(members)
    start = proto.handle_bytes(b'{"cmd":"start"}')
    assert b'"ok": true' in start
    field = proto.handle_bytes(b'{"cmd":"request_field","path":"results/p.npz"}')
    assert b"npz" in field
    assert b"27k" not in field


def test_cycle_forecast_then_rollback_keeps_time() -> None:
    rng = np.random.default_rng(1)

    class _Twin:
        grid = type("G", (), {"n_cells": 2})()
        physics = type("P", (), {"model": "compositional_dpdp", "p_init": 1.0e7})()
        parameterization = type("M", (), {"n_params": 1})()

    members = np.array([[0.0, 0.2, -0.1]])
    wf = OnlineAssimilationWorkflow(twin=_Twin(), members=members.copy(), q_std=0.0)  # type: ignore[arg-type]
    wf.time_s = 3.0
    wf.controls_cursor = 2

    def pred(m):
        return np.ones((1, m.shape[1])) * (m[0] + 1.0)

    xa = wf.cycle(pred, np.array([1.0]), np.array([0.2]), rng)
    assert wf.time_s == pytest.approx(3.0)
    assert wf.controls_cursor == 2
    assert xa.shape == members.shape
    assert wf.predicted_posterior is not None
    assert wf.predicted_posterior.shape[1] == members.shape[1]

