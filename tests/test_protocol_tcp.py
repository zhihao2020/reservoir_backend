"""TCP protocol v2: STEP encode/decode, framing, and record validation."""

import struct
from pathlib import Path

import numpy as np
import pytest

from src.programs.input_tcp import StepDecodeError, records_to_arrays
from src.programs.protocol import (
    PROBE_Q_P,
    PROBE_Q_SG,
    PROBE_Q_SO,
    PROBE_Q_SW,
    PROTOCOL_VERSION,
    NAK_BAD_INDEX,
    NAK_BAD_KIND,
    NAK_BAD_QUANTITY,
    TCP_STEP,
    WELL_K_INJ,
    WELL_K_PROD_G,
    WELL_K_PROD_O,
    WELL_K_PW,
    frame_tcp,
    pack_step,
    unpack_step,
)


def test_step_roundtrip():
    probes = [(0, PROBE_Q_P, 19.1e6), (3, PROBE_Q_SG, 0.2)]
    wells = [(1, WELL_K_INJ, 8.3e-8), (0, WELL_K_PROD_O, 1.3e-8)]
    raw = pack_step(7, 2592000.0, probes, wells)
    step = unpack_step(raw)
    assert step.seq == 7
    assert step.time_s == 2592000.0
    assert list(step.probe_records) == [(0, PROBE_Q_P, 19.1e6), (3, PROBE_Q_SG, 0.2)]
    assert list(step.well_records) == [(1, WELL_K_INJ, 8.3e-8), (0, WELL_K_PROD_O, 1.3e-8)]


def test_step_truncated_raises():
    raw = pack_step(0, 1.0, [(0, PROBE_Q_P, 1.0)], [])
    with pytest.raises(ValueError):
        unpack_step(raw[:-1])


def test_frame_layout():
    frame = frame_tcp(TCP_STEP, b"\x01\x02")
    (n,) = struct.unpack("<I", frame[:4])
    assert n == len(frame) - 4  # length counts version + type + payload
    assert frame[4] == PROTOCOL_VERSION
    assert frame[5] == TCP_STEP
    assert frame[6:] == b"\x01\x02"


def test_records_to_arrays_sign_convention():
    raw = pack_step(
        0,
        1.0,
        [(0, PROBE_Q_P, 19.1e6), (0, PROBE_Q_SW, 0.0), (0, PROBE_Q_SO, 0.8), (0, PROBE_Q_SG, 0.2)],
        [(0, WELL_K_PW, 19.3e6), (0, WELL_K_INJ, 8.3e-8), (1, WELL_K_PROD_O, 1.3e-8), (1, WELL_K_PROD_G, 0.4e-8)],
    )
    step = unpack_step(raw)
    arr = records_to_arrays(step, 4, 2)
    assert arr.pressure[0] == 19.1e6
    assert arr.sw[0] == 0.0 and arr.so[0] == 0.8 and arr.sg[0] == 0.2
    # injector: q and qg both positive
    assert arr.well_q[0] == 8.3e-8
    assert arr.well_qg[0] == 8.3e-8
    # producer: qo/qg negative, total q = -(qo+qg)
    assert arr.well_qo[1] == pytest.approx(-1.3e-8)
    assert arr.well_qg[1] == pytest.approx(-0.4e-8)
    assert arr.well_q[1] == pytest.approx(-1.7e-8)
    # unobserved probe stays NaN
    assert np.isnan(arr.pressure[1])


def test_records_to_arrays_bad_index():
    raw = pack_step(0, 1.0, [(4, PROBE_Q_P, 1.0)], [])
    step = unpack_step(raw)
    with pytest.raises(StepDecodeError) as exc:
        records_to_arrays(step, 4, 1)
    assert exc.value.code == NAK_BAD_INDEX


def test_records_to_arrays_bad_quantity():
    raw = pack_step(0, 1.0, [(0, 99, 1.0)], [])
    step = unpack_step(raw)
    with pytest.raises(StepDecodeError) as exc:
        records_to_arrays(step, 4, 1)
    assert exc.value.code == NAK_BAD_QUANTITY


def test_records_to_arrays_bad_kind():
    raw = pack_step(0, 1.0, [], [(0, 99, 1.0)])
    step = unpack_step(raw)
    with pytest.raises(StepDecodeError) as exc:
        records_to_arrays(step, 4, 1)
    assert exc.value.code == NAK_BAD_KIND


def test_step_before_ready_is_nak():
    # A STEP sent before the READY handshake must be rejected with not_ready,
    # not silently processed.
    from src.cli.main import _Last, _StepHandler
    from src.core.lab_case import load_lab_case
    from src.programs.protocol import NAK_NOT_READY, TCP_NAK, loads
    from src.programs.session import InversionSession

    case = load_lab_case(Path(__file__).resolve().parents[1] / "example" / "case.yaml")
    session = InversionSession.open(case, window=24)
    handler = _StepHandler(None, session, _Last())
    payload = pack_step(0, 2592000.0, [], [])
    replies = handler.on_message(PROTOCOL_VERSION, TCP_STEP, payload)
    assert len(replies) == 1
    (n,) = struct.unpack("<I", replies[0][:4])
    assert n == len(replies[0]) - 4
    assert replies[0][5] == TCP_NAK
    nak = loads(replies[0][6:])
    assert nak["code"] == NAK_NOT_READY
