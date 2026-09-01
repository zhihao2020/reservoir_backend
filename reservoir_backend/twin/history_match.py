"""Offline history-match workflow: ensemble forward + ES-MDA on log parameters.

The smoother updates ``m`` only. Pressure / saturation / composition come from
a fresh forward run after the parameter update.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import os

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ObservationSeries
from reservoir_backend.exceptions import AssimilationError, PhysicsConvergenceError, TimeStepUnderflow
from reservoir_backend.inverse.ensemble import (
    replace_failed_member_bundle,
    sample_log_prior,
)
from reservoir_backend.inverse.esmda import esmda_update, inflation_schedule
from reservoir_backend.observation.qc import ObservationStatus, classify_observations
from reservoir_backend.inverse.lm import identifiability
from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN
from reservoir_backend.twin.offline import (
    DigitalTwin,
    Posterior,
    predict_from_trajectory,
    split_history_observations,
    stack_observations,
)


def _clip_members(twin: DigitalTwin, members: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.asarray(members, dtype=float).copy()
    project = getattr(twin.parameterization, "project", None)
    lo = float(getattr(twin.parameterization, "log_min", LOGK_MIN))
    hi = float(getattr(twin.parameterization, "log_max", LOGK_MAX))
    for j in range(x.shape[1]):
        col = x[:, j]
        if callable(project):
            col = np.asarray(project(col), dtype=float)
        x[:, j] = np.clip(col, lo, hi)
    return x


_WORKER: tuple[DigitalTwin, list[ObservationSeries], float] | None = None


def _init_worker(twin: DigitalTwin, series: list[ObservationSeries], t_end: float) -> None:
    """Load immutable twin context once per process. Cap inner flash threads."""
    from reservoir_backend.eos.threads import cap_flash_threads, ensemble_flash_threads

    cap_flash_threads(ensemble_flash_threads())
    global _WORKER
    _WORKER = (twin, series, float(t_end))


def _forward_theta(job: tuple) -> tuple[int, NDArray[np.float64] | None, str, object]:
    """Picklable worker: (index, theta[, dual_state]) against the process-local twin."""
    j, theta, *rest = job
    dual0 = rest[0] if rest else None
    if _WORKER is None:
        return j, None, "worker twin is not initialized", None
    twin, series, t_end = _WORKER
    try:
        yj = np.asarray(twin._forward_vector(theta, series, t_end=t_end, state0=dual0), dtype=float)
        dual1 = getattr(twin, "_last_dual", None)
        if dual1 is not None:
            dual1 = dual1.copy()
        if not np.all(np.isfinite(yj)):
            return j, None, "NaN predicted observation", dual1
        return j, yj, "", dual1
    except (PhysicsConvergenceError, TimeStepUnderflow, ValueError, ArithmeticError) as exc:
        return j, None, f"{type(exc).__name__}: {exc}", None


def _forward_one(payload: tuple) -> tuple[int, NDArray[np.float64] | None, str]:
    """Serial path: (index, theta, series, t_end, twin[, dual]) → predicted column."""
    j, theta, series, t_end, twin, *rest = payload
    dual0 = rest[0] if rest else None
    global _WORKER
    prev = _WORKER
    _WORKER = (twin, series, float(t_end))
    try:
        return _forward_theta((j, theta, dual0))
    finally:
        _WORKER = prev


def _worker_count(n_ens: int, n_cells: int, requested: int | None) -> int:
    cpu = max(int(os.cpu_count() or 1), 1)
    if n_cells < 125 and requested is None:
        return 1
    mem_cap = max(1, 8 if n_cells >= 8000 else 16 if n_cells >= 1000 else cpu)
    want = cpu if requested is None else max(int(requested), 1)
    return max(1, min(n_ens, cpu, mem_cap, want))


def _forward_ensemble(
    twin: DigitalTwin,
    members: NDArray[np.float64],
    series: list[ObservationSeries],
    t_end: float,
    *,
    n_workers: int | None = None,
    dual_states: list | None = None,
) -> tuple[NDArray[np.float64], list[dict[str, str]], int, list]:
    n_obs = stack_observations(series).values.size
    n_ens = members.shape[1]
    y = np.full((n_obs, n_ens), np.nan, dtype=float)
    duals: list = [None] * n_ens
    failed: list[dict[str, str]] = []
    workers = _worker_count(n_ens, int(twin.grid.n_cells), n_workers)
    from reservoir_backend.eos.threads import cap_flash_threads, ensemble_flash_threads, production_flash_threads

    cap_flash_threads(ensemble_flash_threads() if workers > 1 else production_flash_threads())
    jobs = []
    for j in range(n_ens):
        dual0 = None if dual_states is None or j >= len(dual_states) else dual_states[j]
        jobs.append((j, members[:, j], dual0))
    if workers <= 1:
        rows = [
            _forward_one((j, members[:, j], series, t_end, twin, None if dual_states is None else dual_states[j]))
            for j in range(n_ens)
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(twin, series, float(t_end)),
        ) as pool:
            rows = list(pool.map(_forward_theta, jobs))
    for j, yj, reason, dual1 in rows:
        duals[int(j)] = dual1
        if yj is None:
            failed.append({"member": str(j), "reason": reason})
        else:
            y[:, int(j)] = yj
    return y, failed, n_ens, duals


def joint_phase_schedule(n_a: int, n_theta: int) -> list[str]:
    """Hierarchical ES-MDA: freeze C_f while T_mf fits matrix P/S, then freeze T_mf
    while C_f fits fracture pressure.

    A single Kalman update on both parameters from the prior lets T_mf absorb
    the matrix signal and drives C_f to the bound. Freeze the other coordinate
    until the last steps.
    """
    n_a = int(n_a)
    if int(n_theta) < 2 or n_a < 2:
        return ["joint"] * max(n_a, 1)
    # Two T_mf steps so the first C_f update sees a usable T_mf (one step leaves
    # C_f ~16% high). Then T_mf / C_f so C_f is not frozen at a T_mf-compensated
    # value, and the schedule ends on C_f.
    if n_a == 2:
        return ["tmf", "cf"]
    if n_a == 3:
        return ["tmf", "cf", "tmf"]
    if n_a == 4:
        return ["tmf", "tmf", "cf", "cf"]
    if n_a == 5:
        return ["tmf", "tmf", "cf", "tmf", "cf"]
    n_tmf = max(1, (n_a + 1) // 2)
    n_cf = n_a - n_tmf
    return ["tmf"] * n_tmf + ["cf"] * n_cf


def observation_mask_for_phase(twin: DigitalTwin, d_obs, phase: str) -> NDArray[np.bool_]:
    n = int(d_obs.values.size)
    if phase == "joint" or int(twin.parameterization.n_params) < 2:
        return np.ones(n, dtype=bool)
    smap = twin.experiment.sensor_map()
    mask = np.zeros(n, dtype=bool)
    sat_kinds = {"saturation", "gas_saturation", "oil_saturation"}
    for i, (name, kind) in enumerate(zip(d_obs.names, d_obs.kinds)):
        sen = smap.get(name)
        med = str(getattr(sen, "medium", "fracture") if sen is not None else "fracture")
        if phase == "cf":
            # T_mf is frozen; fracture ΔP identifies C_f. Matrix/sat stay out so
            # residual T_mf error cannot compensate through C_f.
            mask[i] = med == "fracture" and kind == "pressure"
        else:
            mask[i] = med == "matrix" or kind in sat_kinds
    if not np.any(mask):
        return np.ones(n, dtype=bool)
    return mask


def _whitened_misfit(predicted: NDArray[np.float64], d: NDArray[np.float64], sigma: NDArray[np.float64]) -> float:
    finite = np.all(np.isfinite(predicted), axis=0)
    if not np.any(finite):
        return float("inf")
    y_mean = np.mean(predicted[:, finite], axis=1)
    return float(np.sqrt(np.mean(((y_mean - d) / sigma) ** 2)))


@dataclass
class HistoryMatchWorkflow:
    """Build ensemble → forward → ES-MDA → posterior. Serial V1."""

    notes: list[str] = field(default_factory=list)

    def run(
        self,
        twin: DigitalTwin,
        observations: list[ObservationSeries] | None = None,
        parameter_prior: tuple[NDArray[np.float64] | float, NDArray[np.float64] | float] | None = None,
        config: dict | None = None,
    ) -> Posterior:
        cfg = config or {}
        history_end = twin.experiment.history_end_s
        series = observations if observations is not None else twin.experiment.observations
        assim, hold = split_history_observations(series, history_end)
        if not assim:
            raise ValueError("no assimilating observations in the history window")
        d_obs = stack_observations(assim)
        t_hist = float(history_end) if history_end is not None else float(np.max(d_obs.times))

        n_ens = int(cfg.get("ensemble_size", twin.inverse.ensemble_size))
        n_a = int(cfg.get("assimilation_steps", twin.inverse.assimilation_steps))
        seed = int(cfg.get("seed", twin.inverse.seed))
        alpha_cfg = cfg.get("alpha", twin.inverse.alpha)
        n_workers = cfg.get("n_workers", getattr(twin.inverse, "n_workers", None))
        rng = np.random.default_rng(seed)
        alphas = inflation_schedule(n_a, None if alpha_cfg is None else np.asarray(alpha_cfg, dtype=float))

        if parameter_prior is None:
            pmean = getattr(twin.parameterization, "prior_mean", twin.inverse.prior_mean)
            pstd = getattr(twin.parameterization, "prior_std", twin.inverse.prior_std)
        else:
            pmean, pstd = parameter_prior
        n_theta = int(twin.parameterization.n_params)
        lo = float(getattr(twin.parameterization, "log_min", LOGK_MIN))
        hi = float(getattr(twin.parameterization, "log_max", LOGK_MAX))
        members = sample_log_prior(pmean, pstd, n_theta, n_ens, rng, log_min=lo, log_max=hi)
        members = _clip_members(twin, members)

        misfit: list[float] = []
        n_forward = 0
        failed_all: list[dict[str, str]] = []
        predicted = np.zeros((d_obs.values.size, n_ens), dtype=float)
        phases = joint_phase_schedule(n_a, n_theta)
        notes_phases: list[str] = []
        for step, alpha in enumerate(alphas):
            phase = phases[step] if step < len(phases) else "joint"
            predicted, failed, n_fwd, _ = _forward_ensemble(
                twin, members, assim, t_hist, n_workers=n_workers
            )
            n_forward += n_fwd
            failed_mask = np.array([not np.all(np.isfinite(predicted[:, j])) for j in range(n_ens)])
            if np.any(failed_mask):
                failed_all.extend({"step": str(step), **row} for row in failed)
                members, _, _ = replace_failed_member_bundle(members, failed_mask, rng, pstd)
                members = _clip_members(twin, members)
                predicted, failed2, n_fwd2, _ = _forward_ensemble(
                    twin, members, assim, t_hist, n_workers=n_workers
                )
                n_forward += n_fwd2
                failed_all.extend({"step": str(step), "retry": "1", **row} for row in failed2)
                failed_mask = np.array([not np.all(np.isfinite(predicted[:, j])) for j in range(n_ens)])
                if np.any(failed_mask):
                    raise AssimilationError(
                        "ensemble member still failed after replacement: "
                        + ", ".join(r["reason"] for r in failed2)
                    )
            misfit.append(_whitened_misfit(predicted, d_obs.values, d_obs.sigma))
            print(f"ES-MDA {step + 1}/{n_a} phase={phase} misfit={misfit[-1]:.4g}", flush=True)
            status = classify_observations(
                predicted,
                d_obs.values,
                d_obs.sigma,
                outlier_nsigma=float(cfg.get("outlier_nsigma", getattr(twin.inverse, "outlier_nsigma", 8.0))),
            )
            phase_mask = observation_mask_for_phase(twin, d_obs, phase)
            active = (status == ObservationStatus.ACTIVE.value) & phase_mask
            if not np.any(active):
                active = status == ObservationStatus.ACTIVE.value
            if not np.any(active):
                raise AssimilationError("no ACTIVE observations after QC")
            frozen = members.copy()
            members = esmda_update(
                members,
                predicted[active],
                d_obs.values[active],
                d_obs.sigma[active],
                float(alpha),
                rng,
                clip_innovation=bool(cfg.get("clip_innovation", twin.inverse.clip_innovation)),
            )
            if phase == "cf" and n_theta >= 2:
                members[1:, :] = frozen[1:, :]
            elif phase == "tmf" and n_theta >= 2:
                members[0, :] = frozen[0, :]
            members = _clip_members(twin, members)
            notes_phases.append(phase)

        predicted, failed, n_fwd, duals_post = _forward_ensemble(
            twin, members, assim, t_hist, n_workers=n_workers
        )
        n_forward += n_fwd
        failed_mask = np.array([d is None or not np.all(np.isfinite(predicted[:, j])) for j, d in enumerate(duals_post)])
        if np.any(failed_mask):
            failed_all.extend({"step": "posterior", **row} for row in failed)
            members, duals_post, _ = replace_failed_member_bundle(
                members, failed_mask, rng, pstd, dual_states=duals_post
            )
            members = _clip_members(twin, members)
            predicted, failed2, n_fwd2, duals_post = _forward_ensemble(
                twin, members, assim, t_hist, n_workers=n_workers
            )
            n_forward += n_fwd2
            failed_all.extend({"step": "posterior", "retry": "1", **row} for row in failed2)
            failed_mask = np.array([d is None or not np.all(np.isfinite(predicted[:, j])) for j, d in enumerate(duals_post)])
            if np.any(failed_mask):
                raise AssimilationError("posterior ensemble member still failed after replacement")
        misfit.append(_whitened_misfit(predicted, d_obs.values, d_obs.sigma))

        theta_mean = np.mean(members, axis=1)
        theta_std = np.std(members, axis=1, ddof=1)
        k_list = [np.asarray(twin.parameterization.expand(members[:, j]), dtype=float).ravel() for j in range(n_ens)]
        k_members = np.stack(k_list, axis=0)
        k_mean = np.mean(k_members, axis=0)
        k_std = np.std(k_members, axis=0, ddof=1)
        if twin.uses_dpdp():
            hist = twin.simulate(parameters=theta_mean, t_end=t_hist, report_times=d_obs.times)
        else:
            rock = twin.rock_from_k(k_mean)
            hist = twin.simulate(rock, t_end=t_hist, report_times=d_obs.times)
        n_forward += 1
        d_post = predict_from_trajectory(twin.operator, twin.experiment, hist, assim)
        assim_rmse = float(np.sqrt(np.mean(((d_post - d_obs.values) / d_obs.sigma) ** 2)))
        hold_rmse = float("nan")
        if hold:
            d_h = stack_observations(hold)
            pred_h = predict_from_trajectory(twin.operator, twin.experiment, hist, hold)
            hold_rmse = float(np.sqrt(np.mean(((pred_h - d_h.values) / d_h.sigma) ** 2)))
        prior_spread = np.broadcast_to(np.asarray(pstd, dtype=float).ravel(), theta_mean.shape)
        ident = identifiability(prior_spread, theta_std)
        notes = list(self.notes)
        notes.append(f"ES-MDA Ne={n_ens} Na={n_a} seed={seed}")
        notes.append(f"phases={notes_phases}")
        notes.append(f"alpha={alphas.tolist()}")
        notes.append(f"assimilation whitened RMSE={assim_rmse:.4g}")
        notes.append(f"hold-out whitened RMSE={hold_rmse:.4g}")
        n_failed_forward = len(failed_all)
        fail_rate = float(n_failed_forward) / float(max(n_forward, 1))
        member_steps: dict[str, set[str]] = {}
        for row in failed_all:
            member_steps.setdefault(str(row.get("member")), set()).add(str(row.get("step")))
        repeated_fail = any(len(steps) > 1 for steps in member_steps.values())
        notes.append(f"fail_rate={fail_rate:.4g} n_failed_forward={n_failed_forward} repeated_fail={repeated_fail}")
        if failed_all:
            notes.append(f"failed members: {failed_all}")
        q = np.quantile(members, [0.05, 0.50, 0.95], axis=1)
        notes.append(f"theta P05={q[0].tolist()} P50={q[1].tolist()} P95={q[2].tolist()}")
        from reservoir_backend.twin.offline import physical_from_theta

        phys_members = [physical_from_theta(twin.parameterization, members[:, j]) for j in range(n_ens)]
        cfs = np.array([p["cf_m2"] for p in phys_members], dtype=float)
        betas = np.array([p["tmf_multiplier"] for p in phys_members], dtype=float)
        notes.append(f"Cf P50={float(np.quantile(cfs, 0.50)):.4g} Tmf P50={float(np.quantile(betas, 0.50)):.4g}")
        ensemble = PosteriorEnsemble(
            theta_members=members.T,
            k_members=k_members,
            k_mean=k_mean,
            k_std=k_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
            dual_states=duals_post,
        )
        return Posterior(
            theta=theta_mean,
            k=k_mean,
            theta_std=theta_std,
            assimilate_rmse=assim_rmse,
            holdout_rmse=hold_rmse,
            forecast_rmse=None,
            identifiability=ident,
            history=hist,
            notes=notes,
            n_forward=n_forward,
            n_failed_forward=n_failed_forward,
            fail_rate=fail_rate,
            repeated_fail=repeated_fail,
            misfit=misfit,
            ensemble=ensemble,
        )
