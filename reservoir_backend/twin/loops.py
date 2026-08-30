"""High-frequency frozen-λ pressure + low-frequency ES-MDA.

Industrial twins do not rerun ES-MDA every sensor sample. The fast loop
propagates p_f, p_m with λ frozen from the last full DPDP flash. The slow
loop assimilates a window of observations and reruns compositional F.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.domain.types import ObservationSeries, State
from reservoir_backend.solver.fi_comp_dual import dual_to_state
from reservoir_backend.solver.frozen_pressure import step_frozen_pressure
from reservoir_backend.solver.impes import Trajectory
from reservoir_backend.twin.offline import DigitalTwin, Posterior


@dataclass
class TwinLoops:
    """1 s frozen-λ pressure; ES-MDA on ``slow_interval_s``, not every second."""

    twin: DigitalTwin
    slow_interval_s: float = 30.0
    last_slow_s: float = 0.0
    last_traj: Trajectory | None = None
    last_theta: NDArray[np.float64] | None = None
    notes: list[str] = field(default_factory=list)

    def fast_state(self, t: float) -> State:
        """State at time t from the last slow forward (nearest sample)."""
        if self.last_traj is None:
            raise RuntimeError("slow loop has not produced a trajectory yet")
        return self.last_traj.state_at(float(t))

    def fast_step(self, dt: float) -> State:
        """Advance pressures with frozen λ. No flash, no ES-MDA."""
        dual: DualCompositionalState | None = getattr(self.twin, "_last_dual", None)
        if dual is None:
            raise RuntimeError("no DPDP state; run a slow forward first")
        ctx = self.twin.dpdp_context()
        rock = getattr(self.twin, "_last_dual_rock", None)
        if ctx is None or rock is None:
            raise RuntimeError("DPDP context missing")
        pf, pm = step_frozen_pressure(
            self.twin.grid,
            ctx,
            rock,
            self.twin.transfer_operator(),
            dual.fracture.pressure,
            dual.matrix.pressure,
            np.asarray(self.twin._lam_f, dtype=float),
            np.asarray(self.twin._lam_m, dtype=float),
            float(dt),
        )
        dual.fracture.pressure = pf
        dual.matrix.pressure = pm
        dual.time_s = float(dual.time_s) + float(dt)
        if self.twin.physics.fluid is None:
            raise RuntimeError("fluid spec required")
        return dual_to_state(self.twin.physics.fluid, dual, rock)

    def maybe_slow(
        self,
        t: float,
        *,
        observations: list[ObservationSeries] | None = None,
        force: bool = False,
    ) -> Posterior | None:
        """Run ES-MDA when the interval elapsed or ``force`` is set."""
        t = float(t)
        if not force and (t - self.last_slow_s) < float(self.slow_interval_s) - 1.0e-12:
            return None
        if observations is not None:
            self.twin.experiment.observations = list(observations)
        post = self.twin.calibrate()
        self.last_traj = post.history
        self.last_theta = np.asarray(post.theta, dtype=float)
        self.last_slow_s = t
        self.notes.append(f"slow assimilate at t={t}")
        return post
