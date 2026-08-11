"""Estimate permeability and porosity from pressure, saturation, and flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.pipeline.state import MeshBundle
from reservoir_backend.solver.velocity import compute_face_fluxes


def invert_rock_properties(
    mesh: MeshBundle,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    *,
    viscosity_pa_s: float = 1.0e-3,
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    pressure_prev: NDArray[np.float64] | None = None,
    sw_prev: NDArray[np.float64] | None = None,
    dt: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Return ``permeability [m^2]`` and ``porosity`` arrays on the mesh.

    Upgraded assumptions:
    - Isotropic k from Darcy scaling using face fluxes formed with the
      **array (or scalar) permeability prior**, then blended with that prior.
    - Porosity uses prior; with two saturation snapshots and ``dt``, applies a
      continuity-style update ``φ ≈ -div(u) / (∂Sw/∂t)`` where |∂Sw/∂t| is
      significant, otherwise a weak |ΔSw| regularizer around the prior.
    """
    notes: list[str] = [
        "k inversion uses Darcy scaling k ~ mu * |u| / |grad p| with array prior blend",
    ]
    grid = mesh.grid
    k_prior = _as_field(grid.shape, permeability_prior_m2, name="permeability_prior")
    phi_prior = _as_field(grid.shape, porosity_prior, name="porosity_prior")
    k = k_prior.copy()
    phi = phi_prior.copy()

    p_field = Field3D(grid=grid, values=np.asarray(pressure, dtype=float), name="pressure", unit="Pa")
    try:
        fluxes = compute_face_fluxes(
            grid,
            p_field,
            k_prior,
            k_prior,
            k_prior,
            viscosity_pa_s,
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"flux computation failed ({exc}); returning priors only")
        return k, phi, notes

    fx = fluxes.flux_x
    fy = fluxes.flux_y
    fz = fluxes.flux_z
    dxi = np.asarray(grid.dx, dtype=float)
    dyj = np.asarray(grid.dy, dtype=float)
    dzk = np.asarray(grid.dz, dtype=float)
    if dxi.ndim == 0:
        dxi = np.full(grid.nx, float(dxi))
    if dyj.ndim == 0:
        dyj = np.full(grid.ny, float(dyj))
    if dzk.ndim == 0:
        dzk = np.full(grid.nz, float(dzk))

    p = np.asarray(pressure, dtype=float)
    div_u = np.zeros(grid.shape, dtype=float)

    for k_idx in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                if 0 < i < grid.nx - 1:
                    dpx = (p[k_idx, j, i + 1] - p[k_idx, j, i - 1]) / (
                        0.5 * (dxi[i - 1] + 2 * dxi[i] + dxi[i + 1])
                    )
                elif i == 0 and grid.nx > 1:
                    dpx = (p[k_idx, j, 1] - p[k_idx, j, 0]) / (0.5 * (dxi[0] + dxi[1]))
                elif grid.nx > 1:
                    dpx = (p[k_idx, j, -1] - p[k_idx, j, -2]) / (0.5 * (dxi[-2] + dxi[-1]))
                else:
                    dpx = 0.0

                if 0 < j < grid.ny - 1:
                    dpy = (p[k_idx, j + 1, i] - p[k_idx, j - 1, i]) / (
                        0.5 * (dyj[j - 1] + 2 * dyj[j] + dyj[j + 1])
                    )
                elif j == 0 and grid.ny > 1:
                    dpy = (p[k_idx, 1, i] - p[k_idx, 0, i]) / (0.5 * (dyj[0] + dyj[1]))
                elif grid.ny > 1:
                    dpy = (p[k_idx, -1, i] - p[k_idx, -2, i]) / (0.5 * (dyj[-2] + dyj[-1]))
                else:
                    dpy = 0.0

                if 0 < k_idx < grid.nz - 1:
                    dpz = (p[k_idx + 1, j, i] - p[k_idx - 1, j, i]) / (
                        0.5 * (dzk[k_idx - 1] + 2 * dzk[k_idx] + dzk[k_idx + 1])
                    )
                elif k_idx == 0 and grid.nz > 1:
                    dpz = (p[1, j, i] - p[0, j, i]) / (0.5 * (dzk[0] + dzk[1]))
                elif grid.nz > 1:
                    dpz = (p[-1, j, i] - p[-2, j, i]) / (0.5 * (dzk[-2] + dzk[-1]))
                else:
                    dpz = 0.0

                grad = float(np.sqrt(dpx * dpx + dpy * dpy + dpz * dpz))
                ax = float(dyj[j] * dzk[k_idx])
                ay = float(dxi[i] * dzk[k_idx])
                az = float(dxi[i] * dyj[j])
                vol = float(dxi[i] * dyj[j] * dzk[k_idx])
                ux = 0.5 * (abs(fx[k_idx, j, i]) + abs(fx[k_idx, j, i + 1])) / max(ax, 1e-30)
                uy = 0.5 * (abs(fy[k_idx, j, i]) + abs(fy[k_idx, j + 1, i])) / max(ay, 1e-30)
                uz = 0.5 * (abs(fz[k_idx, j, i]) + abs(fz[k_idx + 1, j, i])) / max(az, 1e-30)
                speed = float(np.sqrt(ux * ux + uy * uy + uz * uz))

                # divergence of volumetric flux [1/s]
                div = (
                    (fx[k_idx, j, i + 1] - fx[k_idx, j, i])
                    + (fy[k_idx, j + 1, i] - fy[k_idx, j, i])
                    + (fz[k_idx + 1, j, i] - fz[k_idx, j, i])
                ) / max(vol, 1e-30)
                div_u[k_idx, j, i] = div

                if grad > 1.0e-12 and speed > 0.0:
                    k_est = float(viscosity_pa_s) * speed / grad
                    prior = float(k_prior[k_idx, j, i])
                    # stronger weight on data when gradient is well-resolved
                    w = 0.65 if grad > 1.0e-6 else 0.40
                    k[k_idx, j, i] = w * k_est + (1.0 - w) * prior
                k[k_idx, j, i] = float(np.clip(k[k_idx, j, i], 1.0e-18, 1.0e-10))

    if sw_prev is not None and dt is not None and float(dt) > 0.0:
        dsw = np.asarray(sw, dtype=float) - np.asarray(sw_prev, dtype=float)
        dsw_dt = dsw / float(dt)
        # Continuity proxy: φ ∂Sw/∂t + ∇·u ≈ 0  →  φ ≈ -div(u) / (∂Sw/∂t)
        phi_mb = phi_prior.copy()
        significant = np.abs(dsw_dt) > 1.0e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.where(
                significant,
                -div_u / np.where(significant, dsw_dt, 1.0),
                phi_prior,
            )
        est = np.clip(est, 1.0e-3, 0.5)
        # blend material-balance estimate with prior; damp noisy cells
        w_mb = np.where(significant, 0.55, 0.0)
        # additional weak |ΔSw| regularizer toward slightly higher φ where fluid moves
        scale = np.clip(np.abs(dsw) * 4.0, 0.0, 0.15)
        phi_reg = np.clip(phi_prior + scale - 0.03, 1.0e-3, 0.5)
        phi = (1.0 - w_mb) * phi_reg + w_mb * est
        phi = np.clip(phi, 1.0e-3, 0.5)
        notes.append(
            f"phi updated from continuity/mass-balance proxy over dt={dt} "
            f"(active_frac={float(np.mean(significant)):.3f})"
        )
    else:
        notes.append("phi held at prior (need sw_prev and dt>0 for mass-balance update)")

    _ = (so, sg)
    return k, phi, notes


def _as_field(
    shape: tuple[int, ...],
    value: float | NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match grid {shape}")
    return arr.astype(float, copy=True)
