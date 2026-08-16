"""ES-MDA linear algebra. Independent of the forward model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_alpha_weights(alpha: int | NDArray[np.float64]) -> NDArray[np.float64]:
    """Positive weights with ``sum(1/alpha_i) == 1``."""
    if isinstance(alpha, (int, np.integer)):
        n = int(alpha)
        if n < 1:
            raise ValueError("alpha integer must be >= 1")
        arr = np.full(n, float(n), dtype=float)
    else:
        arr = np.asarray(alpha, dtype=float).ravel()
        if arr.size < 1 or np.any(arr <= 0.0) or not np.all(np.isfinite(arr)):
            raise ValueError("alpha array must be positive finite")
    inv = 1.0 / arr
    return arr * float(np.sum(inv))


def inflate_ensemble(members: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
    f = float(factor)
    if f <= 0.0:
        raise ValueError("inflation factor must be positive")
    if abs(f - 1.0) < 1.0e-15:
        return members
    mean = np.mean(members, axis=0, keepdims=True)
    return mean + f * (members - mean)


def gaspari_cohn(distance: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    r = float(radius)
    if r <= 0.0:
        raise ValueError("radius must be positive")
    z = np.abs(np.asarray(distance, dtype=float) / r)
    out = np.zeros_like(z, dtype=float)
    a = z < 1.0
    b = (z >= 1.0) & (z < 2.0)
    za = z[a]
    zb = z[b]
    out[a] = ((((-0.25 * za + 0.5) * za + 0.625) * za - 5.0 / 3.0) * za) * za + 1.0
    out[b] = (
        ((((zb / 12.0 - 0.5) * zb + 0.625) * zb + 5.0 / 3.0) * zb - 5.0) * zb
        + 4.0
        - 2.0 / (3.0 * zb)
    )
    out = np.clip(out, 0.0, 1.0)
    out[z >= 2.0] = 0.0
    return out


def solve_obs_system(
    cov_dd: NDArray[np.float64],
    r_diag: NDArray[np.float64],
    alpha: float,
    rhs: NDArray[np.float64],
) -> NDArray[np.float64]:
    r_diag = np.asarray(r_diag, dtype=float).ravel()
    cov_dd = np.asarray(cov_dd, dtype=float)
    n = r_diag.size
    if cov_dd.shape != (n, n):
        raise ValueError("cov_dd shape must be (n_obs, n_obs)")
    scale = 1.0 / np.sqrt(np.maximum(r_diag, 1.0e-30))
    s = np.diag(scale)
    a = s @ (cov_dd + float(alpha) * np.diag(r_diag)) @ s
    a = 0.5 * (a + a.T)
    b = s @ rhs
    try:
        y = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(a, b, rcond=None)[0]
    return scale[:, None] * y if y.ndim == 2 else scale * y


def esmda_update_step(
    m_ens: NDArray[np.float64],
    d_sim: NDArray[np.float64],
    obs: NDArray[np.float64],
    r_diag: NDArray[np.float64],
    alpha: float,
    rng: np.random.Generator,
    *,
    md_localization: NDArray[np.float64] | None = None,
    inflation: float = 1.0,
) -> NDArray[np.float64]:
    ne, _n_m = m_ens.shape
    n_obs = obs.size
    if d_sim.shape != (ne, n_obs):
        raise ValueError("d_sim shape must be (ne, n_obs)")
    m_mean = np.mean(m_ens, axis=0)
    d_mean = np.mean(d_sim, axis=0)
    am = m_ens - m_mean
    ad = d_sim - d_mean
    denom = max(ne - 1, 1)
    cov_md = (am.T @ ad) / denom
    cov_dd = (ad.T @ ad) / denom
    if md_localization is not None:
        cov_md = cov_md * md_localization
    inv_cols = solve_obs_system(cov_dd, r_diag, alpha, np.eye(n_obs))
    gain = cov_md @ inv_cols
    sigma = np.sqrt(np.maximum(r_diag, 0.0))
    out = np.empty_like(m_ens)
    for e in range(ne):
        eps = rng.normal(0.0, 1.0, size=n_obs) * sigma
        innov = obs + np.sqrt(float(alpha)) * eps - d_sim[e]
        out[e] = m_ens[e] + gain @ innov
    if inflation != 1.0:
        out = inflate_ensemble(out, inflation)
    return out
