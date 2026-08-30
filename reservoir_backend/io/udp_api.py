"""Peripheral UDP JSON API. Not imported by solvers.

Packets are JSON objects with a ``cmd`` field. Binding a socket is the
caller's job; this module only encodes the request/response contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reservoir_backend.twin.online import OnlineAssimilationWorkflow, TwinCheckpoint


@dataclass
class TwinUDPProtocol:
    """Translate UDP payloads to checkpoint / observe / status commands."""

    workflow: OnlineAssimilationWorkflow | None = None
    last_checkpoint: TwinCheckpoint | None = None
    rng_seed: int = 0
    notes: list[str] = field(default_factory=list)

    def handle_bytes(self, payload: bytes) -> bytes:
        req_id = None
        try:
            msg = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return json.dumps(
                {"ok": False, "error_code": 1, "error": f"invalid json: {exc}", "request_id": None}
            ).encode("utf-8")
        if not isinstance(msg, dict):
            return json.dumps(
                {"ok": False, "error_code": 1, "error": "payload must be an object", "request_id": None}
            ).encode("utf-8")
        req_id = msg.get("request_id")
        cmd = str(msg.get("cmd", "")).strip().lower()
        try:
            out = self._dispatch(cmd, msg)
            out.setdefault("ok", True)
            out.setdefault("error_code", 0)
        except Exception as exc:
            out = {"ok": False, "error_code": 2, "error": f"{type(exc).__name__}: {exc}"}
        out["request_id"] = req_id
        return json.dumps(out).encode("utf-8")

    def _dispatch(self, cmd: str, msg: dict[str, Any]) -> dict[str, Any]:
        if cmd in {"status", "ping"}:
            n = 0 if self.workflow is None else int(np.asarray(self.workflow.members).size)
            return {"ok": True, "cmd": "status", "n_theta_ensemble": n, "time_s": None if self.workflow is None else self.workflow.time_s}
        if cmd == "checkpoint":
            if self.workflow is None:
                raise ValueError("no workflow bound")
            rng = np.random.default_rng(int(msg.get("seed", self.rng_seed)))
            self.last_checkpoint = self.workflow.snapshot(rng)
            return {"ok": True, "cmd": "checkpoint", "time_s": self.last_checkpoint.time_s, "hash": self.last_checkpoint.config_hash}
        if cmd == "rollback":
            if self.workflow is None or self.last_checkpoint is None:
                raise ValueError("no checkpoint to roll back")
            rng = np.random.default_rng(int(msg.get("seed", self.rng_seed)))
            self.workflow.restore(self.last_checkpoint, rng)
            return {"ok": True, "cmd": "rollback", "time_s": self.workflow.time_s}
        if cmd in {"observe", "observation"}:
            if self.workflow is None:
                raise ValueError("no workflow bound")
            d = np.asarray(msg.get("values"), dtype=float).ravel()
            sig = np.asarray(msg.get("sigma", np.ones(d.size)), dtype=float).ravel()
            y = np.asarray(msg.get("predicted"), dtype=float)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            rng = np.random.default_rng(int(msg.get("seed", self.rng_seed)))
            xa = self.workflow.assimilate(y, d, sig, rng)
            return {"ok": True, "cmd": "observe", "theta_mean": np.mean(xa, axis=1).tolist()}
        raise ValueError(f"unknown cmd {cmd!r}")
