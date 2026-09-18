"""Persistent TCP input for the v2 protocol (HELLO/READY/STEP/ACK/NAK/BYE).

The wire framing, record encoding, enums, and NAK codes live in
:mod:`src.programs.protocol`; this module provides the server loop and the
record → dense-array decoding (including the injection/production sign
convention). Message payloads are JSON except STEP, which is packed binary.

Probe ``quantity``: 1 pressure Pa, 2 Sw, 3 So, 4 Sg.
Well ``kind``: 1 BHP Pa, 2 injection m3/s, 3 produced water, 4 produced oil,
5 produced gas. Flow values are positive; signs are applied here.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidObservation
from .protocol import (
    PROBE_Q_P,
    PROBE_Q_SG,
    PROBE_Q_SO,
    PROBE_Q_SW,
    WELL_K_INJ,
    WELL_K_PROD_G,
    WELL_K_PROD_O,
    WELL_K_PROD_W,
    WELL_K_PW,
    NAK_BAD_INDEX,
    NAK_BAD_KIND,
    NAK_BAD_QUANTITY,
    NAK_TRUNCATED,
    StepFrame,
    read_frame,
)

__all__ = [
    "PROBE_Q_P",
    "PROBE_Q_SG",
    "PROBE_Q_SO",
    "PROBE_Q_SW",
    "WELL_K_INJ",
    "WELL_K_PROD_G",
    "WELL_K_PROD_O",
    "WELL_K_PROD_W",
    "WELL_K_PW",
    "StepArrays",
    "StepDecodeError",
    "records_to_arrays",
    "serve",
]


class StepDecodeError(ValueError):
    """A STEP payload failed decode/validation, with a NAK reason ``code``."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class StepArrays:
    time_s: float
    pressure: NDArray[np.float64]
    sw: NDArray[np.float64]
    so: NDArray[np.float64]
    sg: NDArray[np.float64]
    well_pw: NDArray[np.float64]
    well_q: NDArray[np.float64]
    well_qw: NDArray[np.float64]
    well_qo: NDArray[np.float64]
    well_qg: NDArray[np.float64]


def records_to_arrays(step: StepFrame, n_probes: int, n_wells: int) -> StepArrays:
    """Decode raw records into dense SI arrays, applying range/enum checks and
    the injection/production sign convention."""
    n_p, n_w = int(n_probes), int(n_wells)
    pressure = np.full(n_p, np.nan)
    sw = np.full(n_p, np.nan)
    so = np.full(n_p, np.nan)
    sg = np.full(n_p, np.nan)
    well_pw = np.full(n_w, np.nan)
    well_q = np.zeros(n_w)
    well_qw = np.zeros(n_w)
    well_qo = np.zeros(n_w)
    well_qg = np.zeros(n_w)
    for index, quantity, value in step.probe_records:
        if not 0 <= index < n_p:
            raise StepDecodeError(NAK_BAD_INDEX, f"probe index {index} out of range")
        if quantity == PROBE_Q_P:
            pressure[index] = value
        elif quantity == PROBE_Q_SW:
            sw[index] = value
        elif quantity == PROBE_Q_SO:
            so[index] = value
        elif quantity == PROBE_Q_SG:
            sg[index] = value
        else:
            raise StepDecodeError(NAK_BAD_QUANTITY, f"unknown probe quantity {quantity}")
    for index, kind, value in step.well_records:
        if not 0 <= index < n_w:
            raise StepDecodeError(NAK_BAD_INDEX, f"well index {index} out of range")
        mag = abs(float(value))
        if kind == WELL_K_PW:
            well_pw[index] = float(value)
        elif kind == WELL_K_INJ:
            well_q[index] += mag
            well_qg[index] += mag
        elif kind == WELL_K_PROD_W:
            well_qw[index] -= mag
            well_q[index] -= mag
        elif kind == WELL_K_PROD_O:
            well_qo[index] -= mag
            well_q[index] -= mag
        elif kind == WELL_K_PROD_G:
            well_qg[index] -= mag
            well_q[index] -= mag
        else:
            raise StepDecodeError(NAK_BAD_KIND, f"unknown well kind {kind}")
    return StepArrays(
        time_s=step.time_s,
        pressure=pressure,
        sw=sw,
        so=so,
        sg=sg,
        well_pw=well_pw,
        well_q=well_q,
        well_qw=well_qw,
        well_qo=well_qo,
        well_qg=well_qg,
    )


class Handler(Protocol):
    """Server-side message handler. All returned bytes are *already-framed*."""

    def on_connect(self, host: str, port: int) -> list[bytes]: ...

    def on_message(self, version: int, msg_type: int, payload: bytes) -> list[bytes]: ...

    def on_disconnect(self) -> None: ...


def _nodelay(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def serve(port: int, handler: Handler, *, host: str = "0.0.0.0") -> None:
    """Listen forever on ``port``; one client at a time, reconnect keeps history.

    On accept, the frames returned by ``handler.on_connect`` are sent first (the
    HELLO), then every inbound frame is dispatched to ``handler.on_message`` and
    its reply frames are written back. EOF or BYE closes the connection and calls
    ``handler.on_disconnect``; a new client is then accepted.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, int(port)))
    server.listen(1)
    try:
        while True:
            conn, addr = server.accept()
            _nodelay(conn)
            try:
                for frame in handler.on_connect(str(addr[0]), int(addr[1])) or ():
                    conn.sendall(frame)
                while True:
                    frame = read_frame(conn)
                    if frame is None:
                        break
                    version, msg_type, payload = frame
                    if msg_type == 0x06:  # BYE
                        break
                    for reply in handler.on_message(version, msg_type, payload) or ():
                        conn.sendall(reply)
            finally:
                conn.close()
                handler.on_disconnect()
    finally:
        server.close()
