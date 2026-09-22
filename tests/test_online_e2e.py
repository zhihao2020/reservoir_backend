"""End-to-end online joint-debugging: collector (TCP) -> backend -> UDP -> receiver.

Wires the real online path end to end on loopback ports: a TCP collector sends
HELLO/READY/STEP, the backend session reconstructs and publishes the field over
UDP, and ``receive_fields`` reassembles the full 6-field stream. Verifies the
whole chain produces a finite ``(n_times, n_cells)`` field set.
"""

from __future__ import annotations

import socket
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

from src.cli.main import _Last, _StepHandler
from src.core.lab_case import load_lab_case
from src.programs.input_tcp import serve
from src.programs.protocol import (
    PROBE_Q_P,
    PROTOCOL_VERSION,
    TCP_ACK,
    TCP_BYE,
    TCP_HELLO,
    TCP_READY,
    TCP_STEP,
    WELL_K_PW,
    frame_tcp,
    json_frame,
    loads,
    pack_step,
    read_frame,
)
from src.programs.session import InversionSession
from src.programs.udp import receive_fields

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "small" / "case.yaml"
SHALE_OIL = ROOT / "examples" / "shale_oil" / "case.yaml"


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _empty_case(path):
    """Load a case but zero out its observation history (fresh online session)."""
    case = load_lab_case(path)
    n_p, n_w = len(case.probes), len(case.wells)
    case.times = np.zeros(0, dtype=float)
    case.pressure = np.zeros((0, n_p))
    case.sw = np.zeros((0, n_p))
    case.so = np.zeros((0, n_p))
    case.sg = np.zeros((0, n_p))
    case.well_pw = np.zeros((0, n_w))
    case.well_q = np.zeros((0, n_w))
    case.well_qw = np.zeros((0, n_w))
    case.well_qo = np.zeros((0, n_w))
    case.well_qg = np.zeros((0, n_w))
    return case


def _run_online_e2e(path: Path):
    """Send one STEP through the real online chain and reassemble the UDP
    stream, returning ``(manifest, fields, results)``."""
    full = load_lab_case(path)          # has the observations to send
    session = InversionSession.open(_empty_case(path))

    tcp_port = _free_tcp_port()
    lab_port = _free_udp_port()

    # bind the UDP receiver FIRST (as in the real 4-terminal startup order),
    # so the backend's published datagrams are not dropped.
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind(("127.0.0.1", lab_port))

    args = Namespace(output=None, ip="127.0.0.1", lab_port=lab_port, field_port=None)
    handler = _StepHandler(args, session, _Last())

    # backend TCP server (listens + serves one client)
    threading.Thread(target=serve, args=(tcp_port, handler), daemon=True).start()

    # receiver (dashboard) reassembles the published UDP stream concurrently so
    # the socket buffer never overflows (one shale_oil stream is ~160 KB).
    recv_result: dict = {}

    def _receive():
        recv_result["manifest"], recv_result["fields"], recv_result["results"] = receive_fields(
            lab_port, timeout=10.0, sock=recv_sock
        )

    recv_thread = threading.Thread(target=_receive, daemon=True)
    recv_thread.start()

    # collector: connect (retry until the server has bound)
    sock: socket.socket | None = None
    for _ in range(50):
        try:
            sock = socket.create_connection(("127.0.0.1", tcp_port), timeout=1.0)
            break
        except OSError:
            time.sleep(0.1)
    assert sock is not None, "backend did not start listening"

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # block until the ACK returns; the first STEP pays any one-time warmup
    sock.settimeout(None)
    _version, msg_type, payload = read_frame(sock)
    assert msg_type == TCP_HELLO
    manifest = loads(payload)
    assert manifest["n_probes"] == len(full.probe_ids)

    sock.sendall(json_frame(TCP_READY, {"version": PROTOCOL_VERSION, "ok": True}))

    # one STEP: first time slice's probe pressures + well BHP
    t = 0
    probes = [(i, PROBE_Q_P, float(full.pressure[t, i])) for i in range(len(full.probe_ids))]
    wells = [(0, WELL_K_PW, float(full.well_pw[t, 0]))]
    sock.sendall(frame_tcp(TCP_STEP, pack_step(0, float(full.times[t]), probes, wells)))
    _v, _t, reply = read_frame(sock)
    assert _t == TCP_ACK, f"expected ACK, got {loads(reply)}"
    sock.sendall(frame_tcp(TCP_BYE))
    sock.close()

    recv_thread.join(timeout=12.0)
    assert "manifest" in recv_result, "receiver did not reassemble a UDP stream"
    return full, (recv_result["manifest"], recv_result["fields"], recv_result["results"]), recv_sock


def test_online_end_to_end():
    full, (manifest, fields, results), recv_sock = _run_online_e2e(SMALL)
    try:
        n_cells = full.nx * full.ny * full.nz
        assert manifest["n_cells"] == n_cells
        # all six fields, finite, one time slice
        for name in ("p", "sw", "so", "sg", "phi", "k"):
            assert name in fields
            assert fields[name].shape == (1, n_cells)
            assert np.isfinite(fields[name]).all(), f"{name} has non-finite values"
        # well/probe results present
        assert results["n_times"] == 1
    finally:
        recv_sock.close()


def test_online_end_to_end_shale_oil():
    """The full online chain at shale_oil scale (27 probes / 5 wells / 15³)."""
    full, (manifest, fields, results), recv_sock = _run_online_e2e(SHALE_OIL)
    try:
        n_cells = full.nx * full.ny * full.nz
        assert manifest["n_cells"] == n_cells
        assert len(manifest["probes"]) == 27
        assert len(manifest["wells"]) == 5
        for name in ("p", "sw", "so", "sg", "phi", "k"):
            assert name in fields
            assert fields[name].shape == (1, n_cells)
            assert np.isfinite(fields[name]).all(), f"{name} has non-finite values"
        assert results["n_times"] == 1
        assert len(results["probes"]) == 27
        assert len(results["wells"]) == 5
    finally:
        recv_sock.close()
