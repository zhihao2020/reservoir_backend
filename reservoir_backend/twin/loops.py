"""High-frequency frozen-λ pressure + low-frequency Parameter EnKF.

Industrial twins do not rerun ES-MDA every sensor sample. The fast loop
propagates p_f, p_m with λ frozen from the last full DPDP flash. The slow
loop updates the previous posterior ensemble with a parameter EnKF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.domain.types import ObservationSeries, State
from reservoir_backend.inverse.ensemble import sample_log_prior
from reservoir_backend.inverse.parameter_enkf import analysis_parameters, forecast_parameters
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN
from reservoir_backend.solver.frozen_pressure import FrozenPressureContext, step_frozen_pressure
from reservoir_backend.solver.impes import Trajectory
from reservoir_backend.exceptions import AssimilationError
from reservoir_backend.inverse.ensemble import replace_failed_members
from reservoir_backend.observation.qc import ObservationStatus, classify_observations
from reservoir_backend.twin.offline import (
    DigitalTwin,
    Posterior,
    stack_observations,
    window_observations,
)


@dataclass
class OnlineMemberState:
    """Per-member sequential-filter checkpoint. Flash cache is a guess only."""

    theta: NDArray[np.float64]
    dual_state: DualCompositionalState | None = None
    flash_cache: object | None = None  # FlashCache; guess only


@dataclass
class TwinLoops:
    """1 s frozen-λ pressure; Parameter EnKF on ``slow_interval_s``."""

    twin: DigitalTwin
    slow_interval_s: float = 30.0
    last_slow_s: float = 0.0
    last_traj: Trajectory | None = None
    last_theta: NDArray[np.float64] | None = None
    members: NDArray[np.float64] | None = None
    dual_states: list[DualCompositionalState] | None = None
    notes: list[str] = field(default_factory=list)
    q_std: float = 0.02
    rng_seed: int | None = None
    last_full_s: float = 0.0
    last_cycle_s: float = 0.0
    cycle_safety: float = 1.2
    eta_threshold: float = 2.0
    eta_streak_need: int = 3
    eta_streak: int = 0
    last_fast_pressure: NDArray[np.float64] | None = None
    last_fast_error: float = 0.0
    last_fast_error_inf: float = 0.0
    flash_caches: list | None = None
    _frozen: FrozenPressureContext | None = field(default=None, repr=False)

    @classmethod
    def from_posterior(cls, twin: DigitalTwin, posterior: Posterior, **kwargs) -> TwinLoops:
        """Continue from an offline posterior ensemble. Does not resample the prior."""
        if posterior.ensemble is None:
            raise ValueError("posterior has no ensemble; run ES-MDA before the online loop")
        loops = cls(twin, **kwargs)
        loops.members = np.asarray(posterior.ensemble.theta_members, dtype=float).T
        loops.last_theta = np.asarray(posterior.theta, dtype=float)
        loops.last_traj = posterior.history
        if posterior.history is not None and posterior.history.times_s.size:
            loops.last_slow_s = float(posterior.history.times_s[-1])
        ens_duals = getattr(posterior.ensemble, "dual_states", None)
        if loops.last_slow_s > 1.0e-15:
            if not ens_duals or any(s is None for s in ens_duals) or len(ens_duals) != loops.members.shape[1]:
                raise ValueError("online start at t>0 requires a DualState for every ensemble member")
        if ens_duals:
            loops.dual_states = [None if s is None else s.copy() for s in ens_duals]
        ens_cache = getattr(posterior.ensemble, "flash_caches", None)
        if ens_cache:
            loops.flash_caches = [None if c is None else c.copy() for c in ens_cache]
        return loops

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
        if self._frozen is None:
            self._frozen = FrozenPressureContext()
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
            factor=self._frozen,
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
        st = State(
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
        self.last_fast_pressure = st.pressure.copy()
        return st

    def maybe_slow(
        self,
        t: float,
        *,
        observations: list[ObservationSeries] | None = None,
        force: bool = False,
    ) -> Posterior | None:
        """Parameter EnKF from the previous posterior ensemble. Not a full ES-MDA rerun."""
        t = float(t)
        if observations is not None:
            if getattr(self.twin, "experiment", None) is None:
                raise RuntimeError("twin has no experiment to attach observations")
            self.twin.experiment.observations = list(observations)
        from reservoir_backend.twin.history_match import _clip_members, _forward_ensemble
        from reservoir_backend.twin.offline import predict_from_trajectory

        exp = getattr(self.twin, "experiment", None)
        if exp is None:
            interval = max(
                float(self.slow_interval_s),
                float(self.last_full_s),
                float(self.last_cycle_s) * float(self.cycle_safety),
            )
            if not force and (t - self.last_slow_s) < interval - 1.0e-12:
                return None
            return None
        series = window_observations(list(exp.observations), self.last_slow_s, t)
        if series and self.last_traj is not None:
            d_tmp = stack_observations(series)
            try:
                pred = predict_from_trajectory(self.twin.operator, self.twin.experiment, self.last_traj, series)
                eta = float(np.sqrt(np.mean(((pred - d_tmp.values) / np.maximum(d_tmp.sigma, 1.0e-12)) ** 2)))
            except Exception:
                eta = 0.0
            if eta > float(self.eta_threshold):
                self.eta_streak += 1
            else:
                self.eta_streak = 0
            if self.eta_streak >= int(self.eta_streak_need):
                force = True
                self.notes.append(f"innovation trigger eta={eta:.3g} streak={self.eta_streak}")
        interval = max(
            float(self.slow_interval_s),
            float(self.last_full_s),
            float(self.last_cycle_s) * float(self.cycle_safety),
        )
        if not force and (t - self.last_slow_s) < interval - 1.0e-12:
            return None
        if not series:
            self.notes.append(f"slow skip, no new observations in ({self.last_slow_s},{t}]")
            self.last_slow_s = t
            return None
        d_obs = stack_observations(series)
        n_ens = int(self.twin.inverse.ensemble_size)
        n_theta = int(self.twin.parameterization.n_params)
        lo = float(getattr(self.twin.parameterization, "log_min", LOGK_MIN))
        hi = float(getattr(self.twin.parameterization, "log_max", LOGK_MAX))
        pstd = getattr(self.twin.parameterization, "prior_std", self.twin.inverse.prior_std)
        seed = int(self.rng_seed if self.rng_seed is not None else self.twin.inverse.seed)
        rng = np.random.default_rng(seed + int(t))
        if self.members is None:
            pmean = getattr(self.twin.parameterization, "prior_mean", self.twin.inverse.prior_mean)
            self.members = sample_log_prior(pmean, pstd, n_theta, n_ens, rng, log_min=lo, log_max=hi)
            self.members = _clip_members(self.twin, self.members)
        t_cycle0 = time.perf_counter()
        members = forecast_parameters(self.members, self.q_std, rng)
        members = _clip_members(self.twin, members)
        checkpoints = self.dual_states
        t_fc0 = time.perf_counter()
        predicted, failed, n_fwd, _ = _forward_ensemble(
            self.twin, members, series, t, dual_states=checkpoints
        )
        t_forecast = time.perf_counter() - t_fc0
        failed_mask = np.array([not np.all(np.isfinite(predicted[:, j])) for j in range(members.shape[1])])
        if np.any(failed_mask):
            members = replace_failed_members(members, failed_mask, rng, pstd)
            members = _clip_members(self.twin, members)
            predicted, failed, n_fwd2, _ = _forward_ensemble(
                self.twin, members, series, t, dual_states=checkpoints
            )
            n_fwd += n_fwd2
            failed_mask = np.array([not np.all(np.isfinite(predicted[:, j])) for j in range(members.shape[1])])
            if np.any(failed_mask):
                raise AssimilationError("online ensemble member still failed after replacement")
        status = classify_observations(predicted, d_obs.values, d_obs.sigma)
        active = status == ObservationStatus.ACTIVE.value
        if not np.any(active):
            raise AssimilationError("no ACTIVE observations after QC")
        t_an0 = time.perf_counter()
        xa = analysis_parameters(members, predicted[active], d_obs.values[active], d_obs.sigma[active], rng)
        xa = _clip_members(self.twin, xa)
        t_analysis = time.perf_counter() - t_an0
        self.members = xa
        theta_mean = np.mean(xa, axis=1)
        theta_std = np.std(xa, axis=1, ddof=1) if xa.shape[1] > 1 else np.zeros_like(theta_mean)
        t_po0 = time.perf_counter()
        _, _, n_post, duals_post = _forward_ensemble(self.twin, xa, series, t, dual_states=checkpoints)
        t_posterior = time.perf_counter() - t_po0
        n_fwd += n_post
        self.dual_states = duals_post
        j_mean = int(np.argmin(np.linalg.norm(xa - theta_mean[:, None], axis=0)))
        state0 = None if checkpoints is None else checkpoints[j_mean]
        t0 = time.perf_counter()
        hist = self.twin.simulate(
            parameters=theta_mean, t_end=t, report_times=d_obs.times, state0=state0
        )
        t_mean = time.perf_counter() - t0
        self.last_full_s = t_mean
        self.last_cycle_s = time.perf_counter() - t_cycle0
        if self.last_fast_pressure is not None and hist.states:
            pfull = np.asarray(hist.states[-1].pressure, dtype=float).ravel()
            pfast = np.asarray(self.last_fast_pressure, dtype=float).ravel()
            n = min(pfull.size, pfast.size)
            denom = max(float(np.linalg.norm(pfull[:n])), 1.0)
            self.last_fast_error = float(np.linalg.norm(pfast[:n] - pfull[:n]) / denom)
            self.last_fast_error_inf = float(np.max(np.abs(pfast[:n] - pfull[:n])))
            if self.last_fast_error > 0.10:
                self.slow_interval_s = max(5.0, 0.5 * float(self.slow_interval_s))
            elif self.last_fast_error < 0.01:
                self.slow_interval_s = min(300.0, 1.25 * float(self.slow_interval_s))
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
            notes=[
                f"parameter EnKF at t={t} Ne={xa.shape[1]}",
                f"T_forecast={t_forecast:.4f}",
                f"T_analysis={t_analysis:.4f}",
                f"T_posterior={t_posterior:.4f}",
                f"T_mean={t_mean:.4f}",
                f"T_cycle={self.last_cycle_s:.4f}",
            ],
            n_forward=int(n_fwd) + 1,
        )
        self.last_traj = hist
        self.last_theta = np.asarray(theta_mean, dtype=float)
        self.last_slow_s = t
        self._frozen = None
        self.notes.append(f"slow parameter EnKF at t={t}")
        if self.last_full_s > float(self.slow_interval_s):
            self.notes.append(
                f"slow interval stretched to {self.last_full_s:.3g}s (full solve)"
            )
        return post
