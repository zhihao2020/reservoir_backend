"""Low-dimensional geological permeability parameterization.

Greenfield model (wells only — no CMG):

    log k(x) = log_k_bg * (1 - w(x)) + log_k_ch * w(x)

``w`` is a soft tube between injector and producer with width, vertical bias,
and a single sine meander (dogleg / undulating channel). Parameter count is
small (6-D) so ES-MDA stays well-posed with sparse probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle

# θ = [log_k_bg, log_k_ch, log_width, z_bias, meander_amp, meander_phase]
N_K_PARAMS = 6
LOG_K_MIN = np.log(1.0e-18)
LOG_K_MAX = np.log(1.0e-10)


@dataclass(frozen=True)
class KParamPrior:
    """Prior mean/std for θ (natural log k in m²)."""

    mean: NDArray[np.float64]  # (6,)
    std: NDArray[np.float64]  # (6,)


def default_k_param_prior(k_mean_m2: float = 1.0e-13) -> KParamPrior:
    log_bg = float(np.log(max(float(k_mean_m2), 1.0e-20)))
    mean = np.array([log_bg, log_bg + 3.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    std = np.array([0.75, 1.1, 0.55, 0.4, 0.45, 0.8], dtype=float)
    return KParamPrior(mean=mean, std=std)


def sample_k_param_ensemble(
    prior: KParamPrior,
    *,
    ne: int,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Draw θ ensemble, shape ``(ne, N_K_PARAMS)``."""
    rng = np.random.default_rng(seed)
    ens = rng.normal(prior.mean[None, :], prior.std[None, :], size=(int(ne), N_K_PARAMS))
    return _clip_theta(ens)


def expand_k_from_params(
    mesh: MeshBundle,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map θ → full-grid permeability [m²]."""
    th = np.asarray(theta, dtype=float).ravel()
    if th.size != N_K_PARAMS:
        raise ValueError(f"theta must have length {N_K_PARAMS}")
    log_bg, log_ch, log_w, z_bias, m_amp, m_ph = (float(x) for x in th)
    log_bg = float(np.clip(log_bg, LOG_K_MIN, LOG_K_MAX))
    log_ch = float(np.clip(log_ch, LOG_K_MIN, LOG_K_MAX))
    if log_ch < log_bg:
        log_ch, log_bg = log_bg, log_ch

    w = _path_weight(
        mesh,
        width_scale=float(np.exp(log_w)),
        z_bias=z_bias,
        meander_amp=m_amp,
        meander_phase=m_ph,
    )
    log_k = log_bg * (1.0 - w) + log_ch * w
    return np.clip(np.exp(log_k), 1.0e-18, 1.0e-10)


def expand_k_ensemble(
    mesh: MeshBundle,
    theta_ens: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expand θ ensemble → k ensemble ``(ne, nz, ny, nx)``."""
    th = np.asarray(theta_ens, dtype=float)
    if th.ndim != 2 or th.shape[1] != N_K_PARAMS:
        raise ValueError(f"theta_ens must be (ne, {N_K_PARAMS})")
    return np.stack([expand_k_from_params(mesh, th[e]) for e in range(th.shape[0])], axis=0)


def project_k_to_params(
    mesh: MeshBundle,
    k_field: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Best-effort θ from a full k field (warm start)."""
    k = np.clip(np.asarray(k_field, dtype=float), 1.0e-30, None)
    w = _path_weight(mesh, width_scale=1.0, z_bias=0.0, meander_amp=0.0, meander_phase=0.0)
    w_flat = w.ravel()
    log_k = np.log(k).ravel()
    bg_m = float(np.average(log_k, weights=np.maximum(1.0 - w_flat, 1.0e-6)))
    ch_m = float(np.average(log_k, weights=np.maximum(w_flat, 1.0e-6)))
    if ch_m < bg_m:
        ch_m = bg_m + 0.5
    return _clip_theta(np.array([bg_m, ch_m, 0.0, 0.0, 0.0, 0.0], dtype=float))


def enforce_theta_contrast(
    theta: NDArray[np.float64],
    *,
    min_log_ratio: float = 1.2,
) -> NDArray[np.float64]:
    """Ensure log_k_ch - log_k_bg >= min_log_ratio (~3.3× linear if 1.2)."""
    th = _clip_theta(np.asarray(theta, dtype=float).ravel().copy())
    if th[1] < th[0] + float(min_log_ratio):
        mid = 0.5 * (th[0] + th[1])
        half = 0.5 * float(min_log_ratio)
        th[0] = mid - half
        th[1] = mid + half
    return _clip_theta(th)


def enforce_k_channel_contrast(
    mesh: MeshBundle,
    k_field: NDArray[np.float64],
    theta: NDArray[np.float64],
    *,
    min_ratio: float = 2.5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Protect channel/matrix contrast after grid operations.

    If mean(k|corridor) / mean(k|matrix) < min_ratio, rebuild k from θ with
    a forced log-contrast (does not invent geometry outside the path model).
    Returns (k_fixed, theta_fixed, ratio_after).
    """
    th = enforce_theta_contrast(theta, min_log_ratio=float(np.log(max(min_ratio, 1.01))))
    k = np.asarray(k_field, dtype=float)
    w = _path_weight(
        mesh,
        width_scale=float(np.exp(th[2])),
        z_bias=float(th[3]),
        meander_amp=float(th[4]),
        meander_phase=float(th[5]),
    )
    high = w >= 0.45
    low = w <= 0.20
    if not np.any(high) or not np.any(low):
        k2 = expand_k_from_params(mesh, th)
        return k2, th, float("nan")
    ratio = float(np.mean(k[high]) / max(float(np.mean(k[low])), 1.0e-30))
    if ratio >= float(min_ratio):
        return np.clip(k, 1.0e-18, 1.0e-10), th, ratio
    # rebuild purely from parametric model with enforced contrast
    k2 = expand_k_from_params(mesh, th)
    ratio2 = float(np.mean(k2[high]) / max(float(np.mean(k2[low])), 1.0e-30))
    return k2, th, ratio2


def boost_theta_from_indicator(
    mesh: MeshBundle,
    theta: NDArray[np.float64],
    indicator: NDArray[np.float64],
    *,
    strength: float = 0.55,
) -> NDArray[np.float64]:
    """Increase channel contrast if ΔSw indicator aligns with the corridor."""
    th = np.asarray(theta, dtype=float).ravel().copy()
    w = _path_weight(
        mesh,
        width_scale=float(np.exp(th[2])),
        z_bias=float(th[3]),
        meander_amp=float(th[4]),
        meander_phase=float(th[5]),
    )
    ind = np.asarray(indicator, dtype=float)
    corr = _corr(w, ind)
    # positive alignment → raise channel / slightly lower background
    s = float(strength) * max(corr, 0.0)
    gap = max(float(th[1] - th[0]), 0.5)
    th[1] = th[1] + s * 0.85 * gap
    th[0] = th[0] - s * 0.18 * gap
    # aligned ΔSw corridor should support large channel/matrix contrast
    min_lr = 2.7 if corr > 0.15 else 1.5  # ~15× vs ~4.5×
    return enforce_theta_contrast(th, min_log_ratio=min_lr)


def fit_corridor_to_indicator(
    mesh: MeshBundle,
    theta: NDArray[np.float64],
    indicator: NDArray[np.float64],
    *,
    n_amp: int = 9,
    n_phase: int = 12,
    n_width: int = 5,
) -> tuple[NDArray[np.float64], float]:
    """Grid-search meander/width so corridor weight aligns with ΔSw indicator.

    Maximizes a blend of Pearson correlation and high-w mean indicator.
    Does not use CMG truth — only multi-time shape indicator from sensors.
    """
    th0 = _clip_theta(np.asarray(theta, dtype=float).ravel().copy())
    ind = np.asarray(indicator, dtype=float)
    if float(np.std(ind)) < 1e-12:
        return th0, 0.0

    best_score = -1.0e30
    best = th0.copy()
    amp_grid = np.linspace(-1.35, 1.35, max(3, int(n_amp)))
    phase_grid = np.linspace(-np.pi, np.pi, max(4, int(n_phase)), endpoint=False)
    # width: explore around current log_width
    w0 = float(th0[2])
    width_grid = np.linspace(w0 - 0.7, w0 + 0.7, max(3, int(n_width)))

    for amp in amp_grid:
        for ph in phase_grid:
            for lw in width_grid:
                th = th0.copy()
                th[2] = float(lw)
                th[4] = float(amp)
                th[5] = float(ph)
                w = _path_weight(
                    mesh,
                    width_scale=float(np.exp(th[2])),
                    z_bias=float(th[3]),
                    meander_amp=float(th[4]),
                    meander_phase=float(th[5]),
                )
                score = _alignment_score(w, ind)
                if score > best_score:
                    best_score = score
                    best = th

    return _clip_theta(best), float(best_score)


def _alignment_score(weight: NDArray[np.float64], indicator: NDArray[np.float64]) -> float:
    """Higher when high-indicator mass sits on the corridor."""
    w = np.asarray(weight, dtype=float)
    ind = np.asarray(indicator, dtype=float)
    corr = _corr(w, ind)
    wsum = float(np.sum(w)) + 1.0e-30
    high = float(np.sum(ind * w) / wsum)
    low_m = w <= 0.2
    low = float(np.mean(ind[low_m])) if np.any(low_m) else float(np.mean(ind))
    contrast = high - low
    return 0.55 * corr + 0.45 * contrast


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    af, bf = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if float(np.std(af)) < 1e-12 or float(np.std(bf)) < 1e-12:
        return 0.0
    c = float(np.corrcoef(af, bf)[0, 1])
    return c if np.isfinite(c) else 0.0


def _clip_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    th = np.asarray(theta, dtype=float).copy()
    if th.ndim == 1:
        th[0] = np.clip(th[0], LOG_K_MIN, LOG_K_MAX)
        th[1] = np.clip(th[1], LOG_K_MIN, LOG_K_MAX)
        th[2] = np.clip(th[2], -1.2, 1.2)
        th[3] = np.clip(th[3], -1.0, 1.0)
        th[4] = np.clip(th[4], -1.5, 1.5)
        th[5] = np.clip(th[5], -np.pi, np.pi)
        return th
    th[:, 0] = np.clip(th[:, 0], LOG_K_MIN, LOG_K_MAX)
    th[:, 1] = np.clip(th[:, 1], LOG_K_MIN, LOG_K_MAX)
    th[:, 2] = np.clip(th[:, 2], -1.2, 1.2)
    th[:, 3] = np.clip(th[:, 3], -1.0, 1.0)
    th[:, 4] = np.clip(th[:, 4], -1.5, 1.5)
    th[:, 5] = np.clip(th[:, 5], -np.pi, np.pi)
    return th


def _path_weight(
    mesh: MeshBundle,
    *,
    width_scale: float,
    z_bias: float,
    meander_amp: float = 0.0,
    meander_phase: float = 0.0,
) -> NDArray[np.float64]:
    """Soft [0,1] weight along inj–prod path with optional planar meander."""
    inj = _first_role(mesh, "injector")
    prod = _first_role(mesh, "producer")
    if inj is None or prod is None:
        names = list(mesh.well_cell_id.keys())
        if len(names) < 2:
            return np.zeros(mesh.grid.shape, dtype=float)
        c0, c1 = mesh.well_cell_id[names[0]], mesh.well_cell_id[names[1]]
    else:
        c0, c1 = inj, prod

    x0, y0, z0 = float(mesh.x[c0]), float(mesh.y[c0]), float(mesh.z[c0])
    x1, y1, z1 = float(mesh.x[c1]), float(mesh.y[c1]), float(mesh.z[c1])
    dxi = float(np.mean(np.asarray(mesh.grid.dx, dtype=float)))
    dyj = float(np.mean(np.asarray(mesh.grid.dy, dtype=float)))
    base_r = 1.6 * np.sqrt(dxi * dxi + dyj * dyj)
    radius = max(base_r * max(float(width_scale), 0.25), 0.5 * base_r)

    vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
    # planar length for meander amplitude in metres
    Lxy = float(np.sqrt(vx * vx + vy * vy) + 1.0e-30)
    # unit along-path and left-normal in plan view
    ux, uy = vx / Lxy, vy / Lxy
    nx_, ny_ = -uy, ux
    amp = float(meander_amp) * base_r * 2.5  # scale-free amp → metres

    out = np.zeros(mesh.grid.shape, dtype=float)
    vv = vx * vx + vy * vy + vz * vz + 1.0e-30
    zmin, zmax = float(np.min(mesh.z)), float(np.max(mesh.z))
    zspan = max(zmax - zmin, 1.0e-30)

    for n in range(mesh.n_cells):
        px, py, pz = float(mesh.x[n]), float(mesh.y[n]), float(mesh.z[n])
        wx, wy, wz = px - x0, py - y0, pz - z0
        t = float(np.clip((wx * vx + wy * vy + wz * vz) / vv, 0.0, 1.0))
        # meander: offset path center in plan-normal direction
        shift = amp * float(np.sin(2.0 * np.pi * t + float(meander_phase)))
        cx = x0 + t * vx + shift * nx_
        cy = y0 + t * vy + shift * ny_
        cz = z0 + t * vz
        dist = float(np.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2))
        if dist > radius:
            continue
        w = 1.0 - dist / radius
        zn = (pz - zmin) / zspan
        w *= float(np.clip(1.0 + float(z_bias) * (zn - 0.5), 0.25, 1.75))
        w = float(np.clip(w, 0.0, 1.0))
        i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
        out[k, j, i] = max(out[k, j, i], w)
    return out


def _first_role(mesh: MeshBundle, role: str) -> int | None:
    for name, r in mesh.well_role.items():
        if r == role and name in mesh.well_cell_id:
            return int(mesh.well_cell_id[name])
    return None
