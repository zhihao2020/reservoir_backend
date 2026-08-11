"""Explicit upwind water saturation transport driven by Darcy face fluxes.

Used after sparse-sensor IDW to pull Sw toward a flow-consistent footprint.
This is a simplified single-phase water proxy (not full black-oil IMPES).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.pipeline.state import MeshBundle, SensorSample
from reservoir_backend.solver.velocity import compute_face_fluxes


def transport_water_saturation(
    mesh: MeshBundle,
    sw0: NDArray[np.float64],
    pressure: NDArray[np.float64],
    permeability_m2: float | NDArray[np.float64],
    sample: SensorSample,
    *,
    porosity: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    dt: float,
    n_substeps: int = 8,
) -> tuple[NDArray[np.float64], list[str]]:
    """Advance Sw with explicit upwind finite volume using TPFA face fluxes.

    Parameters
    ----------
    sw0 :
        Initial water saturation ``(nz,ny,nx)``.
    dt :
        Physical time step (same units as sample.time, treated as seconds for
        CFL scaling after non-dimensional clamp).
    """
    notes: list[str] = [
        "saturation transport: explicit upwind with Darcy fluxes (water proxy)",
    ]
    grid = mesh.grid
    sw = np.clip(np.asarray(sw0, dtype=float), 0.0, 1.0).copy()
    phi = _as_field(grid.shape, porosity)
    p_field = Field3D(grid=grid, values=np.asarray(pressure, dtype=float), name="p", unit="Pa")
    fluxes = compute_face_fluxes(
        grid, p_field, permeability_m2, permeability_m2, permeability_m2, viscosity_pa_s
    )
    fx, fy, fz = fluxes.flux_x, fluxes.flux_y, fluxes.flux_z

    dxi = np.asarray(grid.dx, dtype=float).ravel()
    dyj = np.asarray(grid.dy, dtype=float).ravel()
    dzk = np.asarray(grid.dz, dtype=float).ravel()
    if dxi.size == 1:
        dxi = np.full(grid.nx, float(dxi[0]))
    if dyj.size == 1:
        dyj = np.full(grid.ny, float(dyj[0]))
    if dzk.size == 1:
        dzk = np.full(grid.nz, float(dzk[0]))

    vol = np.zeros(grid.shape, dtype=float)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                vol[k, j, i] = dxi[i] * dyj[j] * dzk[k]

    # CFL in the same time unit as ``dt`` (software sample times: often "days").
    # Face fluxes from SI TPFA are m^3/s; if dt is in days, scale Q → m^3/day.
    dt_phys = float(dt)
    q_scale = 86400.0 if dt_phys >= 0.5 else 1.0  # day labels → convert Q to /day
    if q_scale != 1.0:
        fx = fx * q_scale
        fy = fy * q_scale
        fz = fz * q_scale
        notes.append("scaled Darcy fluxes to m^3/day for day-based dt")
    qmax = max(
        float(np.max(np.abs(fx))),
        float(np.max(np.abs(fy))),
        float(np.max(np.abs(fz))),
        1.0e-30,
    )
    vmin = float(np.min(vol * np.maximum(phi, 1.0e-3))) + 1.0e-30
    dt_cfl = 0.4 * vmin / qmax
    n_sub = max(1, int(n_substeps))
    dt_sub = min(dt_phys / n_sub, dt_cfl)
    n_eff = max(1, int(np.ceil(dt_phys / max(dt_sub, 1.0e-30))))
    n_eff = min(n_eff, 40)  # hard cap for interactive use
    dt_sub = dt_phys / n_eff
    notes.append(f"transport substeps={n_eff} dt_sub={dt_sub:.3e} (same unit as sample.time)")

    for _ in range(n_eff):
        sw_new = sw.copy()
        for k in range(grid.nz):
            for j in range(grid.ny):
                for i in range(grid.nx):
                    # net water rate into cell via upwind
                    q_net = 0.0
                    # x-faces
                    qx_w = fx[k, j, i]
                    if qx_w > 0.0:
                        sw_up = sw[k, j, i - 1] if i > 0 else sw[k, j, i]
                        q_net += qx_w * sw_up
                    else:
                        sw_up = sw[k, j, i]
                        q_net += qx_w * sw_up
                    qx_e = fx[k, j, i + 1]
                    if qx_e > 0.0:
                        sw_up = sw[k, j, i]
                        q_net -= qx_e * sw_up
                    else:
                        sw_up = sw[k, j, i + 1] if i + 1 < grid.nx else sw[k, j, i]
                        q_net -= qx_e * sw_up
                    # y-faces
                    qy_s = fy[k, j, i]
                    if qy_s > 0.0:
                        sw_up = sw[k, j - 1, i] if j > 0 else sw[k, j, i]
                        q_net += qy_s * sw_up
                    else:
                        sw_up = sw[k, j, i]
                        q_net += qy_s * sw_up
                    qy_n = fy[k, j + 1, i]
                    if qy_n > 0.0:
                        sw_up = sw[k, j, i]
                        q_net -= qy_n * sw_up
                    else:
                        sw_up = sw[k, j + 1, i] if j + 1 < grid.ny else sw[k, j, i]
                        q_net -= qy_n * sw_up
                    # z-faces
                    qz_b = fz[k, j, i]
                    if qz_b > 0.0:
                        sw_up = sw[k - 1, j, i] if k > 0 else sw[k, j, i]
                        q_net += qz_b * sw_up
                    else:
                        sw_up = sw[k, j, i]
                        q_net += qz_b * sw_up
                    qz_t = fz[k + 1, j, i]
                    if qz_t > 0.0:
                        sw_up = sw[k, j, i]
                        q_net -= qz_t * sw_up
                    else:
                        sw_up = sw[k + 1, j, i] if k + 1 < grid.nz else sw[k, j, i]
                        q_net -= qz_t * sw_up

                    pv = max(float(phi[k, j, i]) * float(vol[k, j, i]), 1.0e-30)
                    sw_new[k, j, i] = sw[k, j, i] + dt_sub * q_net / pv

        sw = np.clip(sw_new, 0.0, 1.0)
        # re-pin well sensors each substep
        for name, phases in sample.well_saturation.items():
            if name not in mesh.well_cell_id:
                continue
            c = mesh.well_cell_id[name]
            i, j, k = mesh.grid.ijk(c)
            sw[k, j, i] = float(np.clip(phases[0], 0.0, 1.0))

    return sw, notes


def phases_from_sw(
    sw: NDArray[np.float64],
    *,
    sample: SensorSample,
    mesh: MeshBundle,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build so, sg with so=1-sw, sg=0; re-apply well phase triples if given."""
    sw = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
    so = 1.0 - sw
    sg = np.zeros_like(sw)
    for name, phases in sample.well_saturation.items():
        if name not in mesh.well_cell_id:
            continue
        c = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(c)
        swv, sov, sgv = float(phases[0]), float(phases[1]), float(phases[2])
        tot = swv + sov + sgv
        if tot > 0:
            swv, sov, sgv = swv / tot, sov / tot, sgv / tot
        sw[k, j, i], so[k, j, i], sg[k, j, i] = swv, sov, sgv
    return sw, so, sg


def _as_field(shape: tuple[int, ...], value: float | NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"field shape {arr.shape} != {shape}")
    return arr.astype(float, copy=True)
