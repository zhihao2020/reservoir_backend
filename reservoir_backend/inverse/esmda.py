"""Offline ES-MDA on a fixed parameterization. No posterior/nowcast blending."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import EnsembleMemberFailure
from reservoir_backend.inverse.algorithms import next_rs_alpha, plan_alphas
from reservoir_backend.inverse.ensemble import esmda_update_step
from reservoir_backend.inverse.parallel import map_members, resolve_n_workers


class Parameterization(Protocol):
    n_params: int

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def sample_prior(
        self,
        n_ensemble: int,
        mean: NDArray[np.float64] | float,
        std: NDArray[np.float64] | float,
        seed: int,
    ) -> NDArray[np.float64]: ...


ForwardFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class ESMdaDiagnostics:
    alpha_schedule: list[float]
    data_mismatch: list[float]
    quadratic_mismatch: list[float]
    ensemble_spread: list[float]
    update_norm: list[float]
    failed_members: list[int]
    notes: list[str] = field(default_factory=list)


@dataclass
class ESMdaResult:
    theta_ensemble: NDArray[np.float64]
    theta_mean: NDArray[np.float64]
    theta_std: NDArray[np.float64]
    k_mean: NDArray[np.float64]
    k_std: NDArray[np.float64]
    k_q10: NDArray[np.float64]
    k_q50: NDArray[np.float64]
    k_q90: NDArray[np.float64]
    diagnostics: ESMdaDiagnostics
    prior_theta: NDArray[np.float64]


def _safe_forward(
    forward: ForwardFn,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, str | None]:
    try:
        d = np.asarray(forward(theta), dtype=float).ravel()
    except Exception as exc:
        return None, str(exc)
    if not np.all(np.isfinite(d)):
        return None, "non-finite predicted data"
    return d, None


def _recover_member(
    forward: ForwardFn,
    theta_e: NDArray[np.float64],
    theta_mean: NDArray[np.float64],
    theta_std: NDArray[np.float64],
    rng: np.random.Generator,
    n_obs: int,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, str | None]:
    """Run F(θ). If it fails, pull θ toward the mean / resample. Never reuse another member's d."""
    d, err = _safe_forward(forward, theta_e)
    if d is not None and d.size == n_obs:
        return np.asarray(theta_e, dtype=float).copy(), d, None
    pulled = 0.5 * (np.asarray(theta_e, dtype=float) + theta_mean)
    d, err2 = _safe_forward(forward, pulled)
    if d is not None and d.size == n_obs:
        return pulled, d, f"pulled toward mean ({err})"
    std = np.maximum(np.asarray(theta_std, dtype=float), 1.0e-6)
    trial = theta_mean + 0.35 * std * rng.normal(size=theta_mean.size)
    d, err3 = _safe_forward(forward, trial)
    if d is not None and d.size == n_obs:
        return trial, d, f"resampled ({err})"
    return None, None, err or err2 or err3


def _repair_member(
    forward: ForwardFn,
    theta_e: NDArray[np.float64],
    theta_mean: NDArray[np.float64],
    theta_std: NDArray[np.float64],
    rng: np.random.Generator,
    n_obs: int,
    first_err: str | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, str | None]:
    """Pull / resample after the first (already failed) forecast."""
    pulled = 0.5 * (np.asarray(theta_e, dtype=float) + theta_mean)
    d, err2 = _safe_forward(forward, pulled)
    if d is not None and d.size == n_obs:
        return pulled, d, f"pulled toward mean ({first_err})"
    std = np.maximum(np.asarray(theta_std, dtype=float), 1.0e-6)
    trial = theta_mean + 0.35 * std * rng.normal(size=theta_mean.size)
    d, err3 = _safe_forward(forward, trial)
    if d is not None and d.size == n_obs:
        return trial, d, f"resampled ({first_err})"
    return None, None, first_err or err2 or err3


def run_esmda(
    parameterization: Parameterization,
    forward: ForwardFn,
    obs: NDArray[np.float64],
    r_diag: NDArray[np.float64],
    *,
    n_ensemble: int = 40,
    n_assimilations: int = 4,
    prior_mean: NDArray[np.float64] | float = np.log(1.0e-12),
    prior_std: NDArray[np.float64] | float = 1.0,
    seed: int = 7,
    inflation: float = 1.02,
    fail_fraction: float = 0.30,
    theta0: NDArray[np.float64] | None = None,
    time_limit_s: float | None = None,
    early_stop: bool = True,
    algorithm: str = "esmda",
    n_workers: int | None = None,
) -> ESMdaResult:
    """Ensemble smoother family in θ-space (ES / ES-MDA / ES-MDA-RS).

    ``forward(theta) -> d_sim`` must be a complete observation vector aligned
    with ``obs`` / ``r_diag``. Failed members are pulled toward the mean or
    resampled — their predicted data is never replaced by another member's.
    If the failure fraction exceeds ``fail_fraction`` the run aborts.
    """
    obs = np.asarray(obs, dtype=float).ravel()
    r_diag = np.asarray(r_diag, dtype=float).ravel()
    if obs.size != r_diag.size:
        raise ValueError("obs and r_diag must have the same length")
    algo = str(algorithm).strip().lower()
    alphas = plan_alphas(algo, int(n_assimilations))
    adaptive = alphas is None
    max_steps = int(n_assimilations) if adaptive else int(alphas.size)
    if theta0 is not None:
        rng = np.random.default_rng(seed)
        base = np.asarray(theta0, dtype=float).ravel()
        noise = rng.normal(0.0, float(np.mean(np.asarray(prior_std, dtype=float))), size=(n_ensemble, base.size))
        theta = base[None, :] + noise
    else:
        theta = parameterization.sample_prior(n_ensemble, prior_mean, prior_std, seed)
    prior_theta = theta.copy()
    rng = np.random.default_rng(int(seed) + 11)
    diag = ESMdaDiagnostics(
        alpha_schedule=[],
        data_mismatch=[],
        quadratic_mismatch=[],
        ensemble_spread=[],
        update_norm=[],
        failed_members=[],
        notes=[
            f"{algo} ne={n_ensemble} max_steps={max_steps} n_theta={parameterization.n_params}",
        ],
    )

    n_obs = obs.size
    workers = resolve_n_workers(n_workers, n_ensemble)
    diag.notes.append(f"ensemble workers={workers}")
    t0 = time.perf_counter()
    remaining_inv = 1.0
    for ia in range(max_steps):
        if time_limit_s is not None and ia > 0 and (time.perf_counter() - t0) >= float(time_limit_s):
            diag.notes.append(f"time_limit_s={time_limit_s:g} stop after {ia} assimilations")
            break
        mean0 = np.mean(theta, axis=0)
        std0 = np.std(theta, axis=0)
        first = map_members(lambda th: _safe_forward(forward, th), [theta[e] for e in range(n_ensemble)], workers)
        kept_th: list[NDArray[np.float64]] = []
        kept_d: list[NDArray[np.float64]] = []
        failed = 0
        for e, (d0, err0) in enumerate(first):
            if d0 is not None and d0.size == n_obs:
                kept_th.append(np.asarray(theta[e], dtype=float).copy())
                kept_d.append(d0)
                continue
            th, d, note = _repair_member(forward, theta[e], mean0, std0, rng, n_obs, err0)
            if th is None or d is None:
                failed += 1
                diag.notes.append(f"step {ia} member {e} dropped: {note}")
                continue
            if note:
                diag.notes.append(f"step {ia} member {e}: {note}")
            kept_th.append(th)
            kept_d.append(d)
        n_keep = len(kept_th)
        frac = failed / max(n_ensemble, 1)
        diag.failed_members.append(failed)
        if n_keep < max(4, int(np.ceil((1.0 - fail_fraction) * n_ensemble))):
            raise EnsembleMemberFailure(
                f"failed member fraction {frac:.2f} exceeds {fail_fraction:.2f} at step {ia}"
            )
        theta = np.stack(kept_th, axis=0)
        d_sim = np.stack(kept_d, axis=0)
        sig = np.sqrt(np.maximum(r_diag, 1.0e-30))
        resid = np.mean(d_sim, axis=0) - obs
        nrmse = float(np.sqrt(np.mean((resid / sig) ** 2)))
        if early_stop and ia >= 2 and diag.data_mismatch and nrmse > diag.data_mismatch[-1] * 1.08:
            diag.notes.append(f"early stop at step {ia}: nRMSE {nrmse:.4g} > {diag.data_mismatch[-1]:.4g}")
            break
        if adaptive:
            alpha = next_rs_alpha(nrmse, remaining_inv)
        else:
            alpha = float(alphas[ia])
        remaining_inv = max(remaining_inv - 1.0 / float(alpha), 0.0)
        diag.alpha_schedule.append(float(alpha))
        quad = float(resid @ (resid / np.maximum(r_diag, 1.0e-30)))
        spread = float(np.mean(np.std(theta, axis=0)))
        theta_new = esmda_update_step(
            theta,
            d_sim,
            obs,
            r_diag,
            float(alpha),
            rng,
            inflation=float(inflation),
        )
        update = float(np.linalg.norm(np.mean(theta_new - theta, axis=0)))
        if theta_new.shape[0] < n_ensemble:
            extra = n_ensemble - theta_new.shape[0]
            pick = rng.integers(0, theta_new.shape[0], size=extra)
            jitter = 0.05 * np.std(theta_new, axis=0) * rng.normal(size=(extra, theta_new.shape[1]))
            theta = np.vstack([theta_new, theta_new[pick] + jitter])
        else:
            theta = theta_new
        project = getattr(parameterization, "project", None)
        if callable(project):
            theta = np.stack([project(th) for th in theta], axis=0)
        diag.data_mismatch.append(nrmse)
        diag.quadratic_mismatch.append(quad)
        diag.ensemble_spread.append(spread)
        diag.update_norm.append(update)
        diag.notes.append(
            f"step {ia} {algo} α={float(alpha):.3g} nRMSE={nrmse:.4g} J={quad:.4g} spread={spread:.4g} kept={n_keep}"
        )
        if adaptive and (remaining_inv <= 1.0e-9 or nrmse <= 1.0):
            diag.notes.append(f"esmda_rs stop remaining_inv={remaining_inv:.3g} nRMSE={nrmse:.4g}")
            break

    n_final = int(theta.shape[0])
    theta_mean = np.mean(theta, axis=0)
    k_members = np.stack([parameterization.expand(theta[e]) for e in range(n_final)], axis=0)
    return ESMdaResult(
        theta_ensemble=theta,
        theta_mean=theta_mean,
        theta_std=np.std(theta, axis=0),
        k_mean=parameterization.expand(theta_mean),
        k_std=np.std(k_members, axis=0),
        k_q10=np.quantile(k_members, 0.10, axis=0),
        k_q50=np.quantile(k_members, 0.50, axis=0),
        k_q90=np.quantile(k_members, 0.90, axis=0),
        diagnostics=diag,
        prior_theta=prior_theta,
    )


def identifiability(prior_std: NDArray[np.float64], post_std: NDArray[np.float64]) -> NDArray[np.float64]:
    """posterior/prior standard-deviation ratio. ~1 means unconstrained."""
    return np.asarray(post_std, dtype=float) / np.maximum(np.asarray(prior_std, dtype=float), 1.0e-30)
