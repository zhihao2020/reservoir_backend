"""Lightweight UDP interface for small JSON commands."""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

import numpy as np

from reservoir_backend.inversion.resistivity_archie import ArchieInverter


def handle_udp_command(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle a small JSON command without transferring large field payloads."""
    command = payload.get("command")
    if command == "ping":
        return {"status": "ok", "message": "pong"}
    if command == "archie_compute":
        inverter = ArchieInverter(
            a=float(payload.get("a", 1.0)),
            m=float(payload.get("m", 2.0)),
            n=float(payload.get("n", 2.0)),
            swi=float(payload.get("swi", 0.0)),
            sor=float(payload.get("sor", 0.0)),
            invalid_policy=payload.get("invalid_policy", "raise"),
        )
        sw, confidence = inverter.invert_with_confidence(
            rt=payload["rt"],
            rw=payload["rw"],
            phi=payload["phi"],
        )
        return {
            "status": "ok",
            "sw": _json_value(sw),
            "confidence": _json_value(confidence),
            "unit": "fraction",
        }
    return {"status": "error", "message": f"unknown command: {command}"}


class UDPArchieServer:
    """Tiny local UDP JSON server used by regression tests and demos."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, max_packet_size: int = 8192) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(0.1)
        self._max_packet_size = max_packet_size
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Return bound server address."""
        host, port = self._socket.getsockname()
        return str(host), int(port)

    def start(self) -> None:
        """Start serving UDP requests in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background server."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._socket.close()

    def _serve(self) -> None:
        while self._running:
            try:
                message, client = self._socket.recvfrom(self._max_packet_size)
            except TimeoutError:
                continue
            except OSError:
                break

            try:
                payload = json.loads(message.decode("utf-8"))
                response = handle_udp_command(payload)
            except Exception as exc:  # UDP boundary converts exceptions into JSON errors.
                response = {"status": "error", "message": str(exc)}
            self._socket.sendto(json.dumps(response).encode("utf-8"), client)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
