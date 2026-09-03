"""Levenberg–Marquardt on a fixed low-dimensional θ. Whitened well-history misfit."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN


class Parameterization(Protocol):
    n_params: int

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]: ...


ForwardFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass
class LMResult:
    theta: NDArray[np.float64]
    k: NDArray[np.float64]
    theta_std: NDArray[np.float64]
    theta_cov: NDArray[np.float64]
    misfit: list[float]
    n_forward: int
    notes: list[str] = field(default_factory=list)


def identifiability(prior_std: NDArray[np.float64], post_std: NDArray[np.float64]) -> NDArray[np.float64]:
    """posterior/prior standard-deviation ratio. ~1 means unconstrained."""
    return np.asarray(post_std, dtype=float) / np.maximum(np.asarray(prior_std, dtype=float), 1.0e-30)


# post/prior σ. Above this, LM did not pin the coordinate — escalate to ES-MDA.
IDENT_RATIO_ESCALATE = 0.70
HOLDOUT_WHITENED_ESCALATE = 2.0
HOLDOUT_VS_ASSIM = 1.25


def should_run_ensemble(
    ident: NDArray[np.float64] | list[float],
    *,
    assimilate_rmse: float,
    holdout_rmse: float,
    uq: bool = False,
) -> tuple[bool, str]:
    """Point estimate first. Ensemble only if θ is weakly pinned or UQ is required.

    Spatial localization is not used: V1 θ is a global scalar or pair, not a field.
    ``uq`` with a well-pinned LM fit should use the LM covariance (post_ensemble),
    not a full ES-MDA rerun — this function then returns False with reason
    ``interval_from_lm``.
    """
    ratio = np.asarray(ident, dtype=float).ravel()
    if ratio.size == 0 or not np.all(np.isfinite(ratio)):
        return True, "identifiability_undefined"
    weak = bool(np.any(ratio > IDENT_RATIO_ESCALATE))
    hold = float(holdout_rmse)
    assim = float(assimilate_rmse)
    hold_bad = np.isfinite(hold) and hold > HOLDOUT_WHITENED_ESCALATE
    hold_worse = (
        np.isfinite(hold)
        and np.isfinite(assim)
        and hold > HOLDOUT_VS_ASSIM * max(assim, 1.0e-12)
    )
    if weak:
        return True, "weak_identifiability"
    if hold_bad or hold_worse:
        return True, "holdout"
    if uq:
        return False, "interval_from_lm"
    return False, "lm_sufficient"


def prior_theta(parameterization: Parameterization, mean: NDArray[np.float64] | float) -> NDArray[np.float64]:
    n = int(parameterization.n_params)
    custom = getattr(parameterization, "prior_mean", None)
    raw = np.asarray(mean, dtype=float)
    if custom is not None and raw.size == 1:
        mu = np.asarray(custom, dtype=float).ravel().copy()
        if mu.size != n:
            mu = np.broadcast_to(mu, (n,)).astype(float).copy()
    else:
        mu = np.broadcast_to(raw, (n,)).astype(float).copy()
    log_c = getattr(parameterization, "log_contrast_mean", None)
    if log_c is not None and n >= 2 and raw.size == 1 and custom is None:
        mu[1] = float(log_c)
    return _project(parameterization, mu)


def _project(parameterization: Parameterization, theta: NDArray[np.float64]) -> NDArray[np.float64]:
    th = np.asarray(theta, dtype=float).ravel()
    project = getattr(parameterization, "project", None)
    if callable(project):
        return np.asarray(project(th), dtype=float).ravel()
    if hasattr(parameterization, "region_id") or hasattr(parameterization, "nx"):
        return np.clip(th, LOGK_MIN, LOGK_MAX)
    return th


def _nrmse(resid: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(np.square(resid))))


def _cost(resid: NDArray[np.float64], theta: NDArray[np.float64], theta0: NDArray[np.float64], inv_var: NDArray[np.float64]) -> float:
    prior = theta - theta0
    return 0.5 * float(resid @ resid + prior @ (inv_var * prior))


def run_lm(
    parameterization: Parameterization,
    forward: ForwardFn,
    obs: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    theta0: NDArray[np.float64] | float | None = None,
    prior_mean: NDArray[np.float64] | float = np.log(1.0e-12),
    prior_std: NDArray[np.float64] | float = 0.8,
    max_iter: int = 8,
    fd_rel: float = 0.05,
    time_limit_s: float | None = None,
) -> LMResult:
    """Minimize whitened ||F(θ)−d|| plus Tikhonov to the prior mean."""
    obs = np.asarray(obs, dtype=float).ravel()
    sig = np.maximum(np.asarray(sigma, dtype=float).ravel(), 1.0e-30)
    if obs.size != sig.size:
        raise ValueError("obs and sigma must have the same length")
    n_theta = int(parameterization.n_params)
    th0 = prior_theta(parameterization, prior_mean if theta0 is None else theta0)
    if th0.size != n_theta:
        raise ValueError(f"theta0 size {th0.size} != {n_theta}")
    pstd = np.broadcast_to(np.asarray(prior_std, dtype=float), (n_theta,)).astype(float).copy()
    pstd = np.maximum(pstd, 1.0e-8)
    inv_var = 1.0 / np.square(pstd)
    notes = [f"lm n_theta={n_theta} max_iter={int(max_iter)}"]
    t0 = time.perf_counter()
    n_forward = 0

    def _fwd(th: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal n_forward
        if time_limit_s is not None and n_forward > 0 and (time.perf_counter() - t0) >= float(time_limit_s):
            raise TimeoutError(f"time_limit_s={time_limit_s:g}")
        n_forward += 1
        d = np.asarray(forward(th), dtype=float).ravel()
        if d.size != obs.size:
            raise ValueError(f"forward length {d.size} != obs {obs.size}")
        if not np.all(np.isfinite(d)):
            raise RuntimeError("non-finite predicted data")
        return d

    theta = th0.copy()
    d_sim = _fwd(theta)
    resid = (d_sim - obs) / sig
    misfit = [_nrmse(resid)]
    cost = _cost(resid, theta, th0, inv_var)
    lam = 1.0e-3
    jac: NDArray[np.float64] | None = None

    for it in range(max(int(max_iter), 0)):
        try:
            jac = np.zeros((obs.size, n_theta), dtype=float)
            for j in range(n_theta):
                h = float(fd_rel) * max(float(pstd[j]), 1.0e-3)
                trial = theta.copy()
                trial[j] = trial[j] + h
                trial = _project(parameterization, trial)
                step = float(trial[j] - theta[j])
                if abs(step) < 1.0e-12:
                    continue
                d_pert = _fwd(trial)
                jac[:, j] = ((d_pert - d_sim) / sig) / step
        except TimeoutError:
            notes.append(f"time_limit stop before jacobian at iter {it}")
            break

        jtj = jac.T @ jac
        grad = jac.T @ resid + inv_var * (theta - th0)
        diag = np.maximum(np.diag(jtj), 1.0e-12)
        accepted = False
        d_trial: NDArray[np.float64] | None = None
        for _ in range(8):
            hess = jtj + np.diag(inv_var) + lam * np.diag(diag)
            try:
                delta = -np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(hess) @ grad
            trial = _project(parameterization, theta + delta)
            try:
                d_trial = _fwd(trial)
            except TimeoutError:
                notes.append(f"time_limit stop during line search at iter {it}")
                accepted = False
                d_trial = None
                break
            except Exception as exc:
                notes.append(f"iter {it} trial rejected: {exc}")
                lam = min(lam * 10.0, 1.0e8)
                continue
            r_trial = (d_trial - obs) / sig
            cost_trial = _cost(r_trial, trial, th0, inv_var)
            if cost_trial <= cost * (1.0 + 1.0e-12):
                theta = trial
                d_sim = d_trial
                resid = r_trial
                cost = cost_trial
                misfit.append(_nrmse(resid))
                lam = max(lam / 10.0, 1.0e-12)
                accepted = True
                notes.append(
                    f"iter {it} nRMSE={misfit[-1]:.4g} cost={cost:.4g} lam={lam:.3g} n_forward={n_forward}"
                )
                break
            lam = min(lam * 10.0, 1.0e8)
        if d_trial is None:
            break
        if not accepted:
            notes.append(f"iter {it} no descent, stop nRMSE={misfit[-1]:.4g}")
            break
        if misfit[-1] <= 1.0e-4:
            notes.append("nRMSE floor")
            break
        if len(misfit) >= 2 and abs(misfit[-2] - misfit[-1]) <= 1.0e-4 * max(misfit[-2], 1.0e-12):
            notes.append("misfit stalled")
            break

    if jac is None:
        jac = np.zeros((obs.size, n_theta), dtype=float)
    hess = jac.T @ jac + np.diag(inv_var)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    notes.append(f"n_forward={n_forward}")
    return LMResult(
        theta=theta,
        k=parameterization.expand(theta),
        theta_std=std,
        theta_cov=np.asarray(cov, dtype=float),
        misfit=misfit,
        n_forward=int(n_forward),
        notes=notes,
    )
