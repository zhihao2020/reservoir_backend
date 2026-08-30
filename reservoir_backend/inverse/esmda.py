"""ES-MDA analysis step. Emerick & Reynolds 2013; numerical ideas from ERT / IES.

Does not import ``references/``. Does not use ``np.linalg.inv``.
Updates parameters only; the physical state is recomputed by the forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from reservoir_backend.exceptions import AssimilationError

_RIDGE = 1.0e-10


def inflation_schedule(
    n_assimilations: int,
    alpha: NDArray[np.float64] | list[float] | None = None,
) -> NDArray[np.float64]:
    """Return α such that ``sum_k 1/α_k = 1``. Default ``α_k = N_a``."""
    n = int(n_assimilations)
    if n < 1:
        raise AssimilationError("assimilation_steps must be >= 1")
    if alpha is None:
        raw = np.full(n, float(n), dtype=float)
    else:
        raw = np.asarray(alpha, dtype=float).ravel()
        if raw.size != n:
            raise AssimilationError(f"alpha length {raw.size} != {n}")
        if np.any(raw <= 0.0) or not np.all(np.isfinite(raw)):
            raise AssimilationError("alpha must be positive and finite")
    inv = 1.0 / raw
    return raw * float(np.sum(inv))


def _spd_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve A X = B for SPD A. Cholesky, then truncated SVD. Never invert A."""
    a = 0.5 * (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T)
    n = a.shape[0]
    scale = float(np.mean(np.abs(np.diag(a)))) if n else 1.0
    a = a + _RIDGE * max(scale, 1.0) * np.eye(n)
    rhs = np.asarray(b, dtype=float)
    try:
        cho, lower = linalg.cho_factor(a, check_finite=False)
        return np.asarray(linalg.cho_solve((cho, lower), rhs, check_finite=False), dtype=float)
    except linalg.LinAlgError:
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        cutoff = 1.0e-12 * float(s[0]) if s.size else 0.0
        s_inv = np.where(s > cutoff, 1.0 / s, 0.0)
        return (vt.T * s_inv) @ (u.T @ rhs)


def esmda_update(
    parameters: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    sigma: NDArray[np.float64],
    alpha: float,
    rng: np.random.Generator,
    *,
    clip_innovation: bool = False,
) -> NDArray[np.float64]:
    """One MDA step: ``m^a = m^f + C_my (C_yy + α R)^{-1} (d_j - y_j)``.

    ``parameters`` is (n_theta, n_ensemble) in log-parameter space.
    ``predicted`` / ``observations`` / ``sigma`` are observation-space, SI.
    Observation covariance R is diagonal. Perturbations use ``√α σ``.
    """
    x = np.asarray(parameters, dtype=float)
    y = np.asarray(predicted, dtype=float)
    d = np.asarray(observations, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float).ravel()
    if x.ndim != 2 or y.ndim != 2:
        raise AssimilationError("ensemble arrays must be 2-D")
    n_theta, n_ens = x.shape
    n_obs = y.shape[0]
    if y.shape[1] != n_ens:
        raise AssimilationError("predicted ensemble width != parameter ensemble")
    if d.size != n_obs or sig.size != n_obs:
        raise AssimilationError("observation / sigma length != predicted rows")
    if n_ens < 2:
        raise AssimilationError("ensemble size must be >= 2")
    if float(alpha) <= 0.0 or not np.isfinite(alpha):
        raise AssimilationError("alpha must be positive")
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise AssimilationError("observation sigma must be positive and finite")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise AssimilationError("NaN in ensemble before update")

    xa = x - np.mean(x, axis=1, keepdims=True)
    ya = y - np.mean(y, axis=1, keepdims=True)
    scale = 1.0 / float(n_ens - 1)
    a = float(alpha)
    s = 1.0 / np.sqrt(np.maximum(a * sig * sig, 1.0e-30))
    ys = ya * s[:, None]
    c_scaled = scale * (ys @ ys.T) + np.eye(n_obs)
    pert = rng.standard_normal((n_obs, n_ens)) * (np.sqrt(a) * sig)[:, None]
    innov = d[:, None] + pert - y
    if clip_innovation:
        cap = 5.0 * np.sqrt(a) * sig
        innov = np.clip(innov, -cap[:, None], cap[:, None])
    rhs = s[:, None] * innov
    w = _spd_solve(c_scaled, rhs)
    w = s[:, None] * w
    c_my = scale * (xa @ ya.T)
    return x + c_my @ w


@dataclass
class ESMDAResult:
    members: NDArray[np.float64]
    predicted: NDArray[np.float64]
    misfit: list[float]
    n_forward: int
    failed: list[dict[str, str]]
    alphas: NDArray[np.float64]
    seed: int
