"""Low-dimensional geological permeability parameterization.

Greenfield model (wells only — no CMG):

    log k(x) = log_k_bg * (1 - w(x)) + log_k_ch * w(x)

where weight ``w`` is a soft tube between injector/producer (main path),
with optional width and vertical bias. Parameter vector is small (4-D),
so ES-MDA is well-posed with sparse well/probe data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle

# θ = [log_k_bg, log_k_ch, log_width, z_bias]
N_K_PARAMS = 4
# physical bounds
LOG_K_MIN = np.log(1.0e-18)
LOG_K_MAX = np.log(1.0e-10)


@dataclass(frozen=True)
class KParamPrior:
    """Prior mean/std for θ (natural log k in m²)."""

    mean: NDArray[np.float64]  # (4,)
    std: NDArray[np.float64]  # (4,)


def default_k_param_prior(k_mean_m2: float = 1.0e-13) -> KParamPrior:
    log_bg = float(np.log(max(float(k_mean_m2), 1.0e-20)))
    # channel typically 1–2 orders higher in log10 ≈ 2.3–4.6 in ln
    mean = np.array([log_bg, log_bg + 2.8, 0.0, 0.0], dtype=float)
    std = np.array([0.8, 1.0, 0.5, 0.4], dtype=float)
    return KParamPrior(mean=mean, std=std)


def sample_k_param_ensemble(
    prior: KParamPrior,
    *,
    ne: int,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Draw θ ensemble, shape ``(ne, N_K_PARAMS)``."""
    rng = np.random.default_rng(seed)
    ne = int(ne)
    ens = rng.normal(prior.mean[None, :], prior.std[None, :], size=(ne, N_K_PARAMS))
    return _clip_theta(ens)


def expand_k_from_params(
    mesh: MeshBundle,
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map θ → full-grid permeability [m²]."""
    th = np.asarray(theta, dtype=float).ravel()
    if th.size != N_K_PARAMS:
        raise ValueError(f"theta must have length {N_K_PARAMS}")
    log_bg, log_ch, log_w, z_bias = (float(th[0]), float(th[1]), float(th[2]), float(th[3]))
    log_bg = float(np.clip(log_bg, LOG_K_MIN, LOG_K_MAX))
    log_ch = float(np.clip(log_ch, LOG_K_MIN, LOG_K_MAX))
    # ensure channel not below background in mean sense
    if log_ch < log_bg:
        log_ch, log_bg = log_bg, log_ch

    w = _path_weight(mesh, width_scale=float(np.exp(log_w)), z_bias=z_bias)
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
    members = [expand_k_from_params(mesh, th[e]) for e in range(th.shape[0])]
    return np.stack(members, axis=0)


def project_k_to_params(
    mesh: MeshBundle,
    k_field: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Best-effort θ from a full k field (for warm start)."""
    k = np.clip(np.asarray(k_field, dtype=float), 1.0e-30, None)
    w = _path_weight(mesh, width_scale=1.0, z_bias=0.0)
    w_flat = w.ravel()
    log_k = np.log(k).ravel()
    # weighted means
    bg_m = float(np.average(log_k, weights=np.maximum(1.0 - w_flat, 1.0e-6)))
    ch_m = float(np.average(log_k, weights=np.maximum(w_flat, 1.0e-6)))
    if ch_m < bg_m:
        ch_m = bg_m + 0.5
    return _clip_theta(np.array([bg_m, ch_m, 0.0, 0.0], dtype=float))


def _clip_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    th = np.asarray(theta, dtype=float).copy()
    if th.ndim == 1:
        th[0] = np.clip(th[0], LOG_K_MIN, LOG_K_MAX)
        th[1] = np.clip(th[1], LOG_K_MIN, LOG_K_MAX)
        th[2] = np.clip(th[2], -1.2, 1.2)
        th[3] = np.clip(th[3], -1.0, 1.0)
        return th
    th[:, 0] = np.clip(th[:, 0], LOG_K_MIN, LOG_K_MAX)
    th[:, 1] = np.clip(th[:, 1], LOG_K_MIN, LOG_K_MAX)
    th[:, 2] = np.clip(th[:, 2], -1.2, 1.2)
    th[:, 3] = np.clip(th[:, 3], -1.0, 1.0)
    return th


def _path_weight(
    mesh: MeshBundle,
    *,
    width_scale: float,
    z_bias: float,
) -> NDArray[np.float64]:
    """Soft [0,1] weight along injector–producer corridor (+ vertical bias)."""
    inj = _first_role(mesh, "injector")
    prod = _first_role(mesh, "producer")
    if inj is None or prod is None:
        # fall back: first two wells
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

    out = np.zeros(mesh.grid.shape, dtype=float)
    vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
    vv = vx * vx + vy * vy + vz * vz + 1.0e-30
    z_all = mesh.z
    zmin, zmax = float(np.min(z_all)), float(np.max(z_all))
    zspan = max(zmax - zmin, 1.0e-30)

    for n in range(mesh.n_cells):
        px, py, pz = float(mesh.x[n]), float(mesh.y[n]), float(mesh.z[n])
        wx, wy, wz = px - x0, py - y0, pz - z0
        t = float(np.clip((wx * vx + wy * vy + wz * vz) / vv, 0.0, 1.0))
        dx_, dy_, dz_ = wx - t * vx, wy - t * vy, wz - t * vz
        dist = float(np.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_))
        if dist > radius:
            continue
        w = 1.0 - dist / radius
        # vertical bias: positive z_bias favors upper layers (smaller z if z down? use normalized)
        zn = (pz - zmin) / zspan  # 0 bottom-ish → 1 top-ish depending on coord
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
