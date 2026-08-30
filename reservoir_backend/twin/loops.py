"""High-frequency frozen-λ pressure + low-frequency Parameter EnKF.

Industrial twins do not rerun ES-MDA every sensor sample. The fast loop
propagates p_f, p_m with λ frozen from the last full DPDP flash. The slow
loop updates the previous posterior ensemble with a parameter EnKF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.domain.types import ObservationSeries, State
from reservoir_backend.inverse.ensemble import sample_log_prior
from reservoir_backend.inverse.parameter_enkf import analysis_parameters, forecast_parameters
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN
from reservoir_backend.solver.frozen_pressure import step_frozen_pressure
from reservoir_backend.solver.impes import Trajectory
from reservoir_backend.twin.offline import (
    DigitalTwin,
    Posterior,
    split_history_observations,
    stack_observations,
)


@dataclass
class TwinLoops:
    """1 s frozen-λ pressure; Parameter EnKF on ``slow_interval_s``."""

    twin: DigitalTwin
    slow_interval_s: float = 30.0
    last_slow_s: float = 0.0
    last_traj: Trajectory | None = None
    last_theta: NDArray[np.float64] | None = None
    members: NDArray[np.float64] | None = None
    notes: list[str] = field(default_factory=list)
    q_std: float = 0.02
    rng_seed: int | None = None

    def fast_state(self, t: float) -> State:
        """State at time t from the last slow forward (nearest sample)."""
        if self.last_traj is None:
            raise RuntimeError("slow loop has not produced a trajectory yet")
        return self.last_traj.state_at(float(t))

    def fast_step(self, dt: float) -> State:
        """Advance pressures with frozen λ. No flash, no assimilation."""
        dual: DualCompositionalState | None = getattr(self.twin, "_last_dual", None)
        if dual is None:
            raise RuntimeError("no DPDP state; run a slow forward first")
        ctx = self.twin.dpdp_context()
        rock = getattr(self.twin, "_last_dual_rock", None)
        if ctx is None or rock is None:
            raise RuntimeError("DPDP context missing")
        t_eval = float(dual.time_s) + float(dt)
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
            ct_fracture=None if self.twin._ct_f is None else np.asarray(self.twin._ct_f, dtype=float),
            ct_matrix=None if self.twin._ct_m is None else np.asarray(self.twin._ct_m, dtype=float),
            ports=list(self.twin.ports),
            controls=list(self.twin.experiment.controls),
            t_eval=t_eval,
            v_mix_fracture=None if self.twin._v_mix_f is None else np.asarray(self.twin._v_mix_f, dtype=float),
            v_mix_matrix=None if self.twin._v_mix_m is None else np.asarray(self.twin._v_mix_m, dtype=float),
        )
        dual.fracture.pressure = pf
        dual.matrix.pressure = pm
        dual.time_s = t_eval
        phi_f = np.asarray(rock.fracture.porosity, dtype=float)
        phi_m = np.asarray(rock.matrix.porosity, dtype=float)
        sw_f = np.asarray(self.twin._sw_f if self.twin._sw_f is not None else np.zeros(pf.size), dtype=float)
        sg_f = np.asarray(self.twin._sg_f if self.twin._sg_f is not None else np.zeros(pf.size), dtype=float)
        sw_m = np.asarray(self.twin._sw_m if self.twin._sw_m is not None else np.zeros(pm.size), dtype=float)
        sg_m = np.asarray(self.twin._sg_m if self.twin._sg_m is not None else np.zeros(pm.size), dtype=float)
        return State(
            pressure=pf.copy(),
            sw=sw_f.copy(),
            sg=sg_f.copy(),
            moles=dual.fracture.moles.copy(),
            moles_matrix=dual.matrix.moles.copy(),
            time_s=t_eval,
            pressure_matrix=pm.copy(),
            sw_matrix=sw_m.copy(),
            sg_matrix=sg_m.copy(),
            phi_fracture=phi_f.copy(),
            phi_matrix=phi_m.copy(),
            saturations_held=True,
        )

    def maybe_slow(
        self,
        t: float,
        *,
        observations: list[ObservationSeries] | None = None,
        force: bool = False,
    ) -> Posterior | None:
        """Parameter EnKF from the previous posterior ensemble. Not a full ES-MDA rerun."""
        t = float(t)
        if not force and (t - self.last_slow_s) < float(self.slow_interval_s) - 1.0e-12:
            return None
        if observations is not None:
            self.twin.experiment.observations = list(observations)
        from reservoir_backend.twin.history_match import _clip_members, _forward_ensemble

        series, _hold = split_history_observations(self.twin.experiment.observations, self.twin.experiment.history_end_s)
        if not series:
            series = list(self.twin.experiment.observations)
        if not series:
            raise ValueError("no observations for the slow loop")
        d_obs = stack_observations(series)
        n_ens = int(self.twin.inverse.ensemble_size)
        n_theta = int(self.twin.parameterization.n_params)
        lo = float(getattr(self.twin.parameterization, "log_min", LOGK_MIN))
        hi = float(getattr(self.twin.parameterization, "log_max", LOGK_MAX))
        seed = int(self.rng_seed if self.rng_seed is not None else self.twin.inverse.seed)
        rng = np.random.default_rng(seed + int(t))
        if self.members is None:
            pmean = getattr(self.twin.parameterization, "prior_mean", self.twin.inverse.prior_mean)
            pstd = getattr(self.twin.parameterization, "prior_std", self.twin.inverse.prior_std)
            self.members = sample_log_prior(pmean, pstd, n_theta, n_ens, rng, log_min=lo, log_max=hi)
            self.members = _clip_members(self.twin, self.members)
        members = forecast_parameters(self.members, self.q_std, rng)
        members = _clip_members(self.twin, members)
        predicted, _failed, n_fwd = _forward_ensemble(self.twin, members, series, t)
        xa = analysis_parameters(members, predicted, d_obs.values, d_obs.sigma, rng)
        xa = _clip_members(self.twin, xa)
        self.members = xa
        theta_mean = np.mean(xa, axis=1)
        theta_std = np.std(xa, axis=1, ddof=1) if xa.shape[1] > 1 else np.zeros_like(theta_mean)
        hist = self.twin.simulate(parameters=theta_mean, t_end=t, report_times=d_obs.times)
        k_mean = np.asarray(self.twin.parameterization.expand(theta_mean), dtype=float).ravel()
        post = Posterior(
            theta=theta_mean,
            k=k_mean,
            theta_std=theta_std,
            assimilate_rmse=float("nan"),
            holdout_rmse=float("nan"),
            forecast_rmse=None,
            identifiability=np.zeros_like(theta_mean),
            history=hist,
            notes=[f"parameter EnKF at t={t} Ne={xa.shape[1]}"],
            n_forward=int(n_fwd) + 1,
        )
        self.last_traj = hist
        self.last_theta = np.asarray(theta_mean, dtype=float)
        self.last_slow_s = t
        self.notes.append(f"slow parameter EnKF at t={t}")
        return post
