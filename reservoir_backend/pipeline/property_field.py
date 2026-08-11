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
    permeability_prior_m2: float = 1.0e-13,
    porosity_prior: float = 0.2,
    pressure_prev: NDArray[np.float64] | None = None,
    sw_prev: NDArray[np.float64] | None = None,
    dt: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Return ``permeability [m^2]`` and ``porosity`` arrays on the mesh.

    MVP assumptions (documented in notes):
    - Isotropic permeability estimated from local pressure gradients and
      face fluxes under a single-phase Darcy scaling with effective viscosity.
    - Porosity defaults to ``porosity_prior`` unless two saturation snapshots
      and ``dt`` are provided, in which case a bulk water storage estimate is
      used as a weak update with clipping to (1e-3, 0.5).
    """
    notes: list[str] = [
        "k inversion uses Darcy scaling k ~ mu * |u| / |grad p| with prior floor",
        "phi uses prior unless two-time saturation change is supplied",
    ]
    grid = mesh.grid
    k = np.full(grid.shape, float(permeability_prior_m2), dtype=float)
    phi = np.full(grid.shape, float(porosity_prior), dtype=float)

    # Fluxes under prior k (consistent with pressure reconstruction prior).
    p_field = Field3D(grid=grid, values=np.asarray(pressure, dtype=float), name="pressure", unit="Pa")
    try:
        fluxes = compute_face_fluxes(
            grid,
            p_field,
            permeability_prior_m2,
            permeability_prior_m2,
            permeability_prior_m2,
            viscosity_pa_s,
        )
    except Exception as exc:  # pragma: no cover
        notes.append(f"flux computation failed ({exc}); returning priors only")
        return k, phi, notes

    # Approximate cell-centered velocity magnitude and pressure gradient.
    fx = fluxes.flux_x
    fy = fluxes.flux_y
    fz = fluxes.flux_z
    # areas ~ dy*dz etc using mean spacing
    dxi = np.asarray(grid.spacing_i if hasattr(grid, "spacing_i") else grid.dx, dtype=float)
    dyj = np.asarray(grid.spacing_j if hasattr(grid, "spacing_j") else grid.dy, dtype=float)
    dzk = np.asarray(grid.spacing_k if hasattr(grid, "spacing_k") else grid.dz, dtype=float)
    if dxi.ndim == 0:
        dxi = np.full(grid.nx, float(dxi))
    if dyj.ndim == 0:
        dyj = np.full(grid.ny, float(dyj))
    if dzk.ndim == 0:
        dzk = np.full(grid.nz, float(dzk))

    p = np.asarray(pressure, dtype=float)
    for k_idx in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                # pressure gradient components (central / one-sided)
                if 0 < i < grid.nx - 1:
                    dpx = (p[k_idx, j, i + 1] - p[k_idx, j, i - 1]) / (0.5 * (dxi[i - 1] + 2 * dxi[i] + dxi[i + 1]))
                elif i == 0 and grid.nx > 1:
                    dpx = (p[k_idx, j, 1] - p[k_idx, j, 0]) / (0.5 * (dxi[0] + dxi[1]))
                elif grid.nx > 1:
                    dpx = (p[k_idx, j, -1] - p[k_idx, j, -2]) / (0.5 * (dxi[-2] + dxi[-1]))
                else:
                    dpx = 0.0

                if 0 < j < grid.ny - 1:
                    dpy = (p[k_idx, j + 1, i] - p[k_idx, j - 1, i]) / (0.5 * (dyj[j - 1] + 2 * dyj[j] + dyj[j + 1]))
                elif j == 0 and grid.ny > 1:
                    dpy = (p[k_idx, 1, i] - p[k_idx, 0, i]) / (0.5 * (dyj[0] + dyj[1]))
                elif grid.ny > 1:
                    dpy = (p[k_idx, -1, i] - p[k_idx, -2, i]) / (0.5 * (dyj[-2] + dyj[-1]))
                else:
                    dpy = 0.0

                if 0 < k_idx < grid.nz - 1:
                    dpz = (p[k_idx + 1, j, i] - p[k_idx - 1, j, i]) / (0.5 * (dzk[k_idx - 1] + 2 * dzk[k_idx] + dzk[k_idx + 1]))
                elif k_idx == 0 and grid.nz > 1:
                    dpz = (p[1, j, i] - p[0, j, i]) / (0.5 * (dzk[0] + dzk[1]))
                elif grid.nz > 1:
                    dpz = (p[-1, j, i] - p[-2, j, i]) / (0.5 * (dzk[-2] + dzk[-1]))
                else:
                    dpz = 0.0

                grad = np.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
                # average absolute face flux into velocity-like magnitude
                ax = float(dyj[j] * dzk[k_idx])
                ay = float(dxi[i] * dzk[k_idx])
                az = float(dxi[i] * dyj[j])
                ux = 0.5 * (abs(fx[k_idx, j, i]) + abs(fx[k_idx, j, i + 1])) / max(ax, 1e-30)
                uy = 0.5 * (abs(fy[k_idx, j, i]) + abs(fy[k_idx, j + 1, i])) / max(ay, 1e-30)
                uz = 0.5 * (abs(fz[k_idx, j, i]) + abs(fz[k_idx + 1, j, i])) / max(az, 1e-30)
                speed = np.sqrt(ux * ux + uy * uy + uz * uz)

                if grad > 1.0e-12 and speed > 0.0:
                    k_est = float(viscosity_pa_s) * speed / grad
                    # blend with prior for stability
                    k[k_idx, j, i] = 0.5 * k_est + 0.5 * float(permeability_prior_m2)
                k[k_idx, j, i] = float(np.clip(k[k_idx, j, i], 1.0e-18, 1.0e-10))

    if (
        pressure_prev is not None
        and sw_prev is not None
        and dt is not None
        and float(dt) > 0.0
    ):
        dsw = np.asarray(sw, dtype=float) - np.asarray(sw_prev, dtype=float)
        # weak storage estimate: phi ~ |dsw| / (|dp| * c + eps) not physical compressibility
        # use saturation change magnitude mapped into (0.05, 0.4) around prior
        scale = np.clip(np.abs(dsw) * 5.0, 0.0, 0.2)
        phi = np.clip(float(porosity_prior) + scale - 0.05, 1.0e-3, 0.5)
        notes.append(f"phi weakly updated from saturation change over dt={dt}")

    # saturations currently unused in k path but accepted for API completeness
    _ = (sw, so, sg)
    return k, phi, notes
