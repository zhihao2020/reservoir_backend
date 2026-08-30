"""Online parameter filter: forecast → F(m) → QC → analysis → rerun F.

Does not put pressure or saturation in the EnKF state. Checkpoint is the
rollback unit when assimilation fails.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.exceptions import AssimilationError
from reservoir_backend.inverse.parameter_enkf import analysis_parameters, forecast_parameters
from reservoir_backend.observation.qc import ObservationStatus, classify_observations
from reservoir_backend.twin.offline import DigitalTwin


def _config_hash(twin: DigitalTwin) -> str:
    payload = {
        "n_cells": int(twin.grid.n_cells),
        "model": str(twin.physics.model),
        "p_init": float(twin.physics.p_init),
        "n_theta": int(twin.parameterization.n_params),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TwinCheckpoint:
    """Snapshot that can roll the online filter back to t_k."""

    time_s: float
    parameter_ensemble: NDArray[np.float64]
    dual_states: list[DualCompositionalState] | None
    rng_state: dict
    config_hash: str
    controls_cursor: int = 0


@dataclass
class OnlineAssimilationWorkflow:
    """Parameter-only EnKF around an existing twin."""

    twin: DigitalTwin
    members: NDArray[np.float64]
    q_std: float = 0.02
    time_s: float = 0.0
    dual_states: list[DualCompositionalState] | None = None
    controls_cursor: int = 0
    predicted_posterior: NDArray[np.float64] | None = None
    notes: list[str] = field(default_factory=list)

    def snapshot(self, rng: np.random.Generator) -> TwinCheckpoint:
        duals = None if self.dual_states is None else [s.copy() for s in self.dual_states]
        return TwinCheckpoint(
            time_s=float(self.time_s),
            parameter_ensemble=np.asarray(self.members, dtype=float).copy(),
            dual_states=duals,
            rng_state=rng.bit_generator.state,
            config_hash=_config_hash(self.twin),
            controls_cursor=int(self.controls_cursor),
        )

    def restore(self, checkpoint: TwinCheckpoint, rng: np.random.Generator) -> None:
        if checkpoint.config_hash != _config_hash(self.twin):
            raise AssimilationError("checkpoint config hash does not match the twin")
        self.time_s = float(checkpoint.time_s)
        self.members = np.asarray(checkpoint.parameter_ensemble, dtype=float).copy()
        self.dual_states = None if checkpoint.dual_states is None else [s.copy() for s in checkpoint.dual_states]
        self.controls_cursor = int(checkpoint.controls_cursor)
        rng.bit_generator.state = checkpoint.rng_state

    def forecast(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Parameter random walk. Caller then recomputes y = H(F(m^f))."""
        self.members = forecast_parameters(self.members, self.q_std, rng)
        return self.members

    def assimilate(
        self,
        predicted: NDArray[np.float64],
        observations: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Analysis only. ``predicted`` must be H(F(self.members)) after forecast."""
        checkpoint = self.snapshot(rng)
        try:
            status = classify_observations(predicted, observations, sigma)
            active = status == ObservationStatus.ACTIVE.value
            if not np.any(active):
                raise AssimilationError("no ACTIVE observations after QC")
            xa = analysis_parameters(
                self.members,
                predicted[active],
                np.asarray(observations, dtype=float).ravel()[active],
                np.asarray(sigma, dtype=float).ravel()[active],
                rng,
            )
            self.members = xa
            return xa
        except Exception:
            self.restore(checkpoint, rng)
            raise

    def cycle(
        self,
        predicted_fn,
        observations: NDArray[np.float64],
        sigma: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """checkpoint → forecast → H(F) → QC/EnKF → rollback → posterior members.

        ``predicted_fn(members)`` must return H(F(members)) from the checkpoint
        physical time. Physical rerun of the posterior is the caller's job with
        the returned ensemble; this method never writes p/S into the filter.
        """
        checkpoint = self.snapshot(rng)
        try:
            self.forecast(rng)
            y = predicted_fn(self.members)
            xa = self.assimilate(y, observations, sigma, rng)
            physical = checkpoint.dual_states
            cursor = checkpoint.controls_cursor
            t = checkpoint.time_s
            self.restore(checkpoint, rng)
            self.members = xa
            self.dual_states = physical
            self.controls_cursor = cursor
            self.time_s = t
            self.predicted_posterior = predicted_fn(xa)
            return xa
        except Exception:
            self.restore(checkpoint, rng)
            raise
