"""Observation covariance R for M1c. Never treat dense sampling as independent.

R is assembled from sensor white noise, optional common pressure bias, and
optional temporal correlation on the same channel:

    R_ij = σ_i σ_j * (δ_{name} exp(-|Δt|/τ) + ρ_bias 1_{pressure,pressure})

Diagonal (τ=None, ρ=0) recovers the old Σ(Δy/σ)² detectability.
Does not use np.linalg.inv.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

_RIDGE = 1.0e-10


def observation_covariance(
    names: list[str],
    times: NDArray[np.float64],
    sigma: NDArray[np.float64],
    kinds: list[str],
    *,
    rho_bias: float = 0.0,
    tau_s: float | None = None,
) -> NDArray[np.float64]:
    names = [str(n) for n in names]
    kinds = [str(k) for k in kinds]
    t = np.asarray(times, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float).ravel()
    n = sig.size
    if t.size != n or len(names) != n or len(kinds) != n:
        raise ValueError("names, times, sigma, kinds must have the same length")
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("observation sigma must be positive and finite")
    r = np.diag(sig * sig)
    rho = float(np.clip(rho_bias, 0.0, 0.95))
    tau = None if tau_s is None else float(tau_s)
    for i in range(n):
        for j in range(i + 1, n):
            corr = 0.0
            if tau is not None and tau > 0.0 and names[i] == names[j]:
                corr += float(np.exp(-abs(t[i] - t[j]) / tau))
            if rho > 0.0 and kinds[i] == "pressure" and kinds[j] == "pressure":
                corr += rho
            corr = float(np.clip(corr, 0.0, 0.99))
            if corr > 0.0:
                val = corr * sig[i] * sig[j]
                r[i, j] = val
                r[j, i] = val
    return r


def _spd_solve(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
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


def mahalanobis_d(dy: NDArray[np.float64], r: NDArray[np.float64]) -> float:
    """sqrt(dy^T R^{-1} dy) via SPD solve. Never inverts R."""
    dy = np.asarray(dy, dtype=float).ravel()
    x = _spd_solve(r, dy)
    val = float(dy @ x)
    return float(np.sqrt(max(val, 0.0)))


def fisher_from_sensitivity(s: NDArray[np.float64], r: NDArray[np.float64]) -> NDArray[np.float64]:
    """F = S^T R^{-1} S. ``s`` is (n_obs, n_theta), unwhitened."""
    s = np.asarray(s, dtype=float)
    wr = _spd_solve(r, s)
    return s.T @ wr
