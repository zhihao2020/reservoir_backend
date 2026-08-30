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
from reservoir_backend.inverse.ensemble import replace_failed_members, sample_log_prior
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


def _forward_one(payload: tuple) -> tuple[int, NDArray[np.float64] | None, str]:
    """Picklable worker: (index, theta, series, t_end, twin) → predicted column."""
    j, theta, series, t_end, twin = payload
    try:
        yj = np.asarray(twin._forward_vector(theta, series, t_end=t_end), dtype=float)
        if not np.all(np.isfinite(yj)):
            return j, None, "NaN predicted observation"
        return j, yj, ""
    except (PhysicsConvergenceError, TimeStepUnderflow, ValueError, ArithmeticError) as exc:
        return j, None, f"{type(exc).__name__}: {exc}"


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
) -> tuple[NDArray[np.float64], list[dict[str, str]], int]:
    n_obs = stack_observations(series).values.size
    n_ens = members.shape[1]
    y = np.full((n_obs, n_ens), np.nan, dtype=float)
    failed: list[dict[str, str]] = []
    workers = _worker_count(n_ens, int(twin.grid.n_cells), n_workers)
    jobs = [(j, members[:, j], series, t_end, twin) for j in range(n_ens)]
    if workers <= 1:
        rows = [_forward_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_forward_one, jobs))
    for j, yj, reason in rows:
        if yj is None:
            failed.append({"member": str(j), "reason": reason})
        else:
            y[:, int(j)] = yj
    return y, failed, n_ens


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
        for step, alpha in enumerate(alphas):
            predicted, failed, n_fwd = _forward_ensemble(
                twin, members, assim, t_hist, n_workers=n_workers
            )
            n_forward += n_fwd
            failed_mask = np.array([not np.all(np.isfinite(predicted[:, j])) for j in range(n_ens)])
            if np.any(failed_mask):
                failed_all.extend({"step": str(step), **row} for row in failed)
                members = replace_failed_members(members, failed_mask, rng, pstd)
                members = _clip_members(twin, members)
                predicted, failed2, n_fwd2 = _forward_ensemble(
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
            status = classify_observations(predicted, d_obs.values, d_obs.sigma)
            active = status == ObservationStatus.ACTIVE.value
            if not np.any(active):
                raise AssimilationError("no ACTIVE observations after QC")
            members = esmda_update(
                members,
                predicted[active],
                d_obs.values[active],
                d_obs.sigma[active],
                float(alpha),
                rng,
                clip_innovation=bool(cfg.get("clip_innovation", twin.inverse.clip_innovation)),
            )
            members = _clip_members(twin, members)

        predicted, failed, n_fwd = _forward_ensemble(
            twin, members, assim, t_hist, n_workers=n_workers
        )
        n_forward += n_fwd
        if failed:
            failed_all.extend({"step": "posterior", **row} for row in failed)
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
        notes.append(f"alpha={alphas.tolist()}")
        notes.append(f"assimilation whitened RMSE={assim_rmse:.4g}")
        notes.append(f"hold-out whitened RMSE={hold_rmse:.4g}")
        if failed_all:
            notes.append(f"failed members: {failed_all}")
        q = np.quantile(members, [0.05, 0.50, 0.95], axis=1)
        notes.append(f"Cf latent P05={q[0].tolist()} P50={q[1].tolist()} P95={q[2].tolist()}")
        ensemble = PosteriorEnsemble(
            theta_members=members.T,
            k_members=k_members,
            k_mean=k_mean,
            k_std=k_std,
            theta_mean=theta_mean,
            theta_std=theta_std,
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
            misfit=misfit,
            ensemble=ensemble,
        )
