"""Infer reservoir / channel shape indicators from multi-time fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import FieldBundle, MeshBundle


def infer_shape_indicator(
    mesh: MeshBundle,
    history: list[FieldBundle],
    *,
    permeability: NDArray[np.float64] | None = None,
    sw_weight: float = 1.0,
    k_weight: float = 1.0,
    pressure_weight: float = 0.5,
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Build a [0, 1] shape indicator on the mesh (mountain / channel proxy).

    Combines:
    - cumulative absolute water-saturation change across time samples;
    - high permeability relative to the field median;
    - pressure drawdown / buildup contrast relative to domain mean.
    """
    if not history:
        raise ValueError("history must contain at least one FieldBundle")

    shape = mesh.grid.shape
    cum_dsw = np.zeros(shape, dtype=float)
    for a, b in zip(history[:-1], history[1:]):
        cum_dsw += np.abs(np.asarray(b.sw, dtype=float) - np.asarray(a.sw, dtype=float))

    if history:
        # also include deviation of last sw from first (mobilized oil/water footprint)
        cum_dsw += 0.5 * np.abs(
            np.asarray(history[-1].sw, dtype=float) - np.asarray(history[0].sw, dtype=float)
        )

    k = (
        np.asarray(permeability, dtype=float)
        if permeability is not None
        else np.asarray(history[-1].permeability, dtype=float)
    )
    k_med = float(np.median(k)) + 1.0e-30
    k_score = np.clip((k / k_med - 1.0) / 3.0, 0.0, 1.0)

    p = np.asarray(history[-1].pressure, dtype=float)
    p_mean = float(np.mean(p))
    p_score = np.clip(np.abs(p - p_mean) / (float(np.std(p)) + 1.0e-30) / 3.0, 0.0, 1.0)

    dsw_score = cum_dsw / (float(np.max(cum_dsw)) + 1.0e-30) if np.max(cum_dsw) > 0 else cum_dsw

    indicator = (
        float(sw_weight) * dsw_score
        + float(k_weight) * k_score
        + float(pressure_weight) * p_score
    )
    indicator = indicator / (float(sw_weight + k_weight + pressure_weight) + 1.0e-30)
    indicator = np.clip(indicator, 0.0, 1.0)

    # Boost cells on the well-to-well corridor if two+ wells exist (channel prior).
    corridor = None
    if len(mesh.well_cell_id) >= 2:
        corridor = _well_corridor_mask(mesh)
        indicator = np.clip(indicator + 0.28 * corridor.astype(float), 0.0, 1.0)

    stats = {
        "indicator_mean": float(np.mean(indicator)),
        "indicator_p90": float(np.quantile(indicator, 0.9)),
        "active_fraction_at_0.4": float(np.mean(indicator >= 0.4)),
        "max_cum_dsw": float(np.max(cum_dsw)),
        "corridor_fraction": float(np.mean(corridor)) if corridor is not None else 0.0,
    }
    return indicator, stats


def enhance_permeability_from_indicator(
    permeability: NDArray[np.float64],
    indicator: NDArray[np.float64],
    *,
    strength: float = 0.55,
    clip: tuple[float, float] = (1.0e-18, 1.0e-10),
    asymmetric: bool = True,
) -> NDArray[np.float64]:
    """Log-space k boost on high-indicator cells (preferential flow paths).

    Does not use external truth masks — only the multi-time shape indicator
    built from ΔSw / pressure contrast (and optionally prior k).

    With ``asymmetric=True`` (default), high-activity cells are boosted more
    than quiet cells are reduced, improving channel/matrix contrast.
    """
    k = np.asarray(permeability, dtype=float)
    ind = np.asarray(indicator, dtype=float)
    if k.shape != ind.shape:
        raise ValueError("permeability and indicator shapes must match")
    mu = float(np.mean(ind))
    sd = float(np.std(ind)) + 1.0e-12
    z = np.clip((ind - mu) / sd, -2.5, 2.5)
    s = float(strength)
    if asymmetric:
        # boost path (z>0) more; mild damp of matrix (z<0)
        scale = np.where(z >= 0.0, s * 0.55 * z, s * 0.25 * z)
    else:
        scale = s * 0.40 * z
    k_new = k * np.exp(scale)
    return np.clip(k_new, float(clip[0]), float(clip[1]))


def indicator_to_active_mask(
    indicator: NDArray[np.float64],
    *,
    threshold: float = 0.35,
    dilate: int = 1,
) -> NDArray[np.bool_]:
    """Threshold indicator and optionally dilate in i-j for connectivity."""
    mask = np.asarray(indicator, dtype=float) >= float(threshold)
    for _ in range(max(0, int(dilate))):
        mask = _dilate3(mask)
    # keep at least something active
    if not np.any(mask):
        mask = np.asarray(indicator, dtype=float) >= float(np.quantile(indicator, 0.7))
    return mask


def _dilate3(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    out = mask.copy()
    nz, ny, nx = mask.shape
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if mask[k, j, i]:
                    for dk in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            for di in (-1, 0, 1):
                                kk, jj, ii = k + dk, j + dj, i + di
                                if 0 <= kk < nz and 0 <= jj < ny and 0 <= ii < nx:
                                    out[kk, jj, ii] = True
    return out


def _well_corridor_mask(mesh: MeshBundle) -> NDArray[np.float64]:
    """Soft tube between first two wells (mountain/channel seed along main path)."""
    names = list(mesh.well_cell_id.keys())
    c0 = mesh.well_cell_id[names[0]]
    c1 = mesh.well_cell_id[names[1]]
    x0, y0, z0 = mesh.x[c0], mesh.y[c0], mesh.z[c0]
    x1, y1, z1 = mesh.x[c1], mesh.y[c1], mesh.z[c1]
    grid = mesh.grid
    out = np.zeros(grid.shape, dtype=float)
    # radius ~ 1.5 cell diagonals of mean spacing
    dxi = float(np.mean(np.asarray(grid.dx, dtype=float)))
    dyj = float(np.mean(np.asarray(grid.dy, dtype=float)))
    radius = 1.5 * np.sqrt(dxi * dxi + dyj * dyj)
    for n in range(mesh.n_cells):
        px, py, pz = mesh.x[n], mesh.y[n], mesh.z[n]
        # distance from point to segment
        vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
        wx, wy, wz = px - x0, py - y0, pz - z0
        vv = vx * vx + vy * vy + vz * vz + 1.0e-30
        t = np.clip((wx * vx + wy * vy + wz * vz) / vv, 0.0, 1.0)
        dx_, dy_, dz_ = wx - t * vx, wy - t * vy, wz - t * vz
        dist = np.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_)
        if dist <= radius:
            i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
            out[k, j, i] = 1.0 - dist / radius
    return out
