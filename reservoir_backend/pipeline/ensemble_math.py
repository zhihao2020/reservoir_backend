"""Ensemble analysis math for ES-MDA (self-contained; paper-inspired).

Ideas aligned with Emerick & Reynolds (2013) and common open-source practice
(normalized alpha, diagonal R preconditioning, optional Gaspari–Cohn taper,
post-update inflation). Implemented independently — do not import references/.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_alpha_weights(alpha: int | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return positive alpha weights with ``sum(1/alpha_i) == 1``.

    - ``int Na`` → equal weights ``[Na, ..., Na]`` (Na times), then normalized.
    - 1-D array → scaled so harmonic-type constraint holds.
    """
    if isinstance(alpha, (int, np.integer)):
        n = int(alpha)
        if n < 1:
            raise ValueError("alpha integer must be >= 1")
        arr = np.ones(n, dtype=float) * float(n)
    else:
        arr = np.asarray(alpha, dtype=float).ravel()
        if arr.size < 1 or np.any(arr <= 0) or not np.all(np.isfinite(arr)):
            raise ValueError("alpha array must be positive finite")
    inv = 1.0 / arr
    s = float(np.sum(inv))
    return arr * s  # scale so sum(1/alpha)=1


def inflate_ensemble(
    members: NDArray[np.float64],
    factor: float,
) -> NDArray[np.float64]:
    """Linear inflation of rows around the ensemble mean: ``m + f*(m-mean)``."""
    f = float(factor)
    if f <= 0.0:
        raise ValueError("inflation factor must be positive")
    if abs(f - 1.0) < 1.0e-15:
        return members
    mean = np.mean(members, axis=0, keepdims=True)
    return mean + f * (members - mean)


def gaspari_cohn(distance: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    """Gaspari–Cohn compact correlation; zero for ``distance >= 2*radius``.

    ``radius`` is the length scale ``L`` in the standard formulation
    (support radius ``2L``).
    """
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
    """Solve ``(C_dd + alpha * R) x = rhs`` with diagonal R preconditioning.

    ``r_diag`` is the diagonal of R (length n_obs). ``rhs`` is (n_obs,) or (n_obs, k).
    """
    r_diag = np.asarray(r_diag, dtype=float).ravel()
    cov_dd = np.asarray(cov_dd, dtype=float)
    n = r_diag.size
    if cov_dd.shape != (n, n):
        raise ValueError("cov_dd shape must be (n_obs, n_obs)")
    scale = 1.0 / np.sqrt(np.maximum(r_diag, 1.0e-30))
    s = np.diag(scale)
    a = s @ (cov_dd + float(alpha) * np.diag(r_diag)) @ s
    # symmetrize
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
    """One ES-MDA analysis step.

    Parameters
    ----------
    m_ens :
        Ensemble parameters, shape ``(ne, n_m)``.
    d_sim :
        Predicted data, shape ``(ne, n_obs)``.
    obs :
        Observations, shape ``(n_obs,)``.
    r_diag :
        Diagonal observation error variances.
    alpha :
        Inflation factor for R at this assimilation (from normalize_alpha_weights).
    md_localization :
        Optional Schur product mask on ``C_md``, shape ``(n_m, n_obs)``.
    inflation :
        Post-update ensemble inflation factor (>=1 damps collapse).
    """
    ne, n_m = m_ens.shape
    n_obs = obs.size
    if d_sim.shape != (ne, n_obs):
        raise ValueError("d_sim shape must be (ne, n_obs)")

    m_mean = np.mean(m_ens, axis=0)
    d_mean = np.mean(d_sim, axis=0)
    am = m_ens - m_mean
    ad = d_sim - d_mean
    denom = max(ne - 1, 1)
    cov_md = (am.T @ ad) / denom  # (n_m, n_obs)
    cov_dd = (ad.T @ ad) / denom  # (n_obs, n_obs)
    if md_localization is not None:
        cov_md = cov_md * md_localization

    # Kalman gain K = C_md @ inv(C_dd + alpha R)
    # For each ensemble member innovation: obs + sqrt(alpha) eps - d_sim
    # Solve (C_dd+alpha R) X = I  then K = C_md @ X, or per-innovation.
    eye = np.eye(n_obs, dtype=float)
    inv_cols = solve_obs_system(cov_dd, r_diag, alpha, eye)
    gain = cov_md @ inv_cols  # (n_m, n_obs)

    sigma = np.sqrt(np.maximum(r_diag, 0.0))
    out = np.empty_like(m_ens)
    for e in range(ne):
        eps = rng.normal(0.0, 1.0, size=n_obs) * sigma
        innov = obs + np.sqrt(float(alpha)) * eps - d_sim[e]
        out[e] = m_ens[e] + gain @ innov
    if inflation != 1.0:
        out = inflate_ensemble(out, inflation)
    return out


def well_parameter_localization(
    mesh_xyz: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    radius_m: float,
) -> NDArray[np.float64]:
    """Build Gaspari–Cohn MD localization ``(n_cells, n_wells)``.

    ``mesh_xyz`` is ``(n_cells, 3)``, ``well_xyz`` is ``(n_wells, 3)``.
    """
    mesh_xyz = np.asarray(mesh_xyz, dtype=float)
    well_xyz = np.asarray(well_xyz, dtype=float)
    n_m = mesh_xyz.shape[0]
    n_w = well_xyz.shape[0]
    loc = np.zeros((n_m, n_w), dtype=float)
    for j in range(n_w):
        d = np.linalg.norm(mesh_xyz - well_xyz[j], axis=1)
        loc[:, j] = gaspari_cohn(d, radius_m)
    return loc
