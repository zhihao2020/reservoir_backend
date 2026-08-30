"""Gate 1: per-member DualState checkpoints and window-only online forward."""

import numpy as np

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.twin.loops import OnlineMemberState, TwinLoops
from reservoir_backend.twin.offline import Posterior


def test_online_member_state_holds_dual() -> None:
    from reservoir_backend.comp.dual_state import CompositionalContinuumState

    dual = DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([1.0e7]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([1.2e7]), np.ones((1, 2)) * 4.0e-4),
        time_s=30.0,
    )
    st = OnlineMemberState(theta=np.array([0.1]), dual_state=dual)
    assert st.dual_state is not None
    assert st.dual_state.time_s == 30.0


def test_from_posterior_copies_last_dual_to_members() -> None:
    from reservoir_backend.comp.dual_state import CompositionalContinuumState
    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.solver.impes import Trajectory

    members = np.array([[1.0, 2.0, 3.0]])
    ens = PosteriorEnsemble(
        theta_members=members.T,
        k_members=np.ones((3, 1)),
        k_mean=np.ones(1),
        k_std=np.ones(1),
        theta_mean=np.array([2.0]),
        theta_std=np.array([1.0]),
    )
    post = Posterior(
        theta=np.array([2.0]),
        k=np.ones(1),
        theta_std=np.array([1.0]),
        assimilate_rmse=0.0,
        holdout_rmse=0.0,
        forecast_rmse=None,
        identifiability=np.array([1.0]),
        history=Trajectory(times_s=np.array([0.0, 30.0]), states=[], reports=[], port_rates=[]),
        notes=[],
        ensemble=ens,
    )
    dual = DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([1.0e7]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([1.2e7]), np.ones((1, 2)) * 4.0e-4),
        time_s=30.0,
    )

    class _Twin:
        _last_dual = dual

    loops = TwinLoops.from_posterior(_Twin(), post, slow_interval_s=30.0)  # type: ignore[arg-type]
    assert loops.dual_states is not None
    assert len(loops.dual_states) == 3
    assert loops.dual_states[0].time_s == 30.0
    loops.dual_states[0].time_s = 0.0
    assert dual.time_s == 30.0


def test_forward_ensemble_passes_state0() -> None:
    from reservoir_backend.domain.types import ObservationSeries
    from reservoir_backend.twin.history_match import _forward_ensemble

    seen: list[float | None] = []

    class _Twin:
        grid = type("G", (), {"n_cells": 2})()

        def _forward_vector(self, theta, series, *, t_end=None, state0=None, **kwargs):
            seen.append(None if state0 is None else float(getattr(state0, "time_s", -1.0)))
            self._last_dual = state0
            return np.array([1.0])

    duals = [type("D", (), {"time_s": 30.0, "copy": lambda self=None: type("D2", (), {"time_s": 30.0})()})() for _ in range(2)]
    for d in duals:
        d.copy = lambda d=d: d
    members = np.array([[0.0, 0.1]])
    series = [
        ObservationSeries("P1", "pressure", np.array([60.0]), np.array([1.0]), np.array([0.1]), False)
    ]
    y, failed, n, out = _forward_ensemble(_Twin(), members, series, 60.0, dual_states=duals)  # type: ignore[arg-type]
    assert failed == []
    assert n == 2
    assert seen == [30.0, 30.0]
