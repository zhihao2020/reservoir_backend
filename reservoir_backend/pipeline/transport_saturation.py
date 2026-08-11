"""Explicit upwind water transport with fractional flow and well rates.

Advances Sw using total Darcy face fluxes times ``f_w(S)`` (Corey proxy),
plus optional well volumetric sources (injection/production).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.pipeline.fractional_flow import water_fractional_flow
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
    oil_viscosity_pa_s: float = 5.0e-3,
    dt: float,
    n_substeps: int = 8,
) -> tuple[NDArray[np.float64], list[str]]:
    """Advance Sw with explicit upwind FV: water flux = Q_total * f_w(S_up)."""
    notes: list[str] = [
        "saturation transport: upwind total flux * f_w(S) (Corey fractional flow)",
    ]
    grid = mesh.grid
    sw = np.clip(np.asarray(sw0, dtype=float), 0.0, 1.0).copy()
    phi = _as_field(grid.shape, porosity)
    p_field = Field3D(grid=grid, values=np.asarray(pressure, dtype=float), name="p", unit="Pa")
    fluxes = compute_face_fluxes(
        grid, p_field, permeability_m2, permeability_m2, permeability_m2, viscosity_pa_s
    )
    fx = fluxes.flux_x.copy()
    fy = fluxes.flux_y.copy()
    fz = fluxes.flux_z.copy()

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

    dt_phys = float(dt)
    q_scale = 86400.0 if dt_phys >= 0.5 else 1.0
    if q_scale != 1.0:
        fx *= q_scale
        fy *= q_scale
        fz *= q_scale
        notes.append("scaled Darcy fluxes to m^3/day for day-based dt")

    # well rates: sample in m^3/s → convert if day-based
    well_q: dict[str, float] = {}
    for name, q in (sample.well_rate or {}).items():
        qq = float(q) * q_scale
        well_q[name] = qq
    if well_q:
        notes.append(f"well rate sources active: {list(well_q.keys())}")

    qmax = max(
        float(np.max(np.abs(fx))),
        float(np.max(np.abs(fy))),
        float(np.max(np.abs(fz))),
        max((abs(v) for v in well_q.values()), default=0.0),
        1.0e-30,
    )
    vmin = float(np.min(vol * np.maximum(phi, 1.0e-3))) + 1.0e-30
    dt_cfl = 0.35 * vmin / qmax
    n_sub = max(1, int(n_substeps))
    dt_sub = min(dt_phys / n_sub, dt_cfl)
    n_eff = max(1, int(np.ceil(dt_phys / max(dt_sub, 1.0e-30))))
    n_eff = min(n_eff, 40)
    dt_sub = dt_phys / n_eff
    notes.append(f"transport substeps={n_eff} dt_sub={dt_sub:.3e}")

    mu_w = float(viscosity_pa_s)
    mu_o = float(oil_viscosity_pa_s)

    def fw_at(sval: float) -> float:
        return float(water_fractional_flow(sval, mu_w=mu_w, mu_o=mu_o))

    def upwind_water(q_face: float, sw_left: float, sw_right: float) -> float:
        """Water volumetric rate across face; positive left→right."""
        if q_face >= 0.0:
            return q_face * fw_at(sw_left)
        return q_face * fw_at(sw_right)

    for _ in range(n_eff):
        sw_new = sw.copy()
        for k in range(grid.nz):
            for j in range(grid.ny):
                for i in range(grid.nx):
                    # water flux into cell
                    qw_in = 0.0
                    # west face of cell i
                    sw_w = sw[k, j, i - 1] if i > 0 else sw[k, j, i]
                    qw_in += upwind_water(fx[k, j, i], sw_w, sw[k, j, i])
                    # east face
                    sw_e = sw[k, j, i + 1] if i + 1 < grid.nx else sw[k, j, i]
                    qw_in -= upwind_water(fx[k, j, i + 1], sw[k, j, i], sw_e)
                    # south
                    sw_s = sw[k, j - 1, i] if j > 0 else sw[k, j, i]
                    qw_in += upwind_water(fy[k, j, i], sw_s, sw[k, j, i])
                    # north
                    sw_n = sw[k, j + 1, i] if j + 1 < grid.ny else sw[k, j, i]
                    qw_in -= upwind_water(fy[k, j + 1, i], sw[k, j, i], sw_n)
                    # bottom
                    sw_b = sw[k - 1, j, i] if k > 0 else sw[k, j, i]
                    qw_in += upwind_water(fz[k, j, i], sw_b, sw[k, j, i])
                    # top
                    sw_t = sw[k + 1, j, i] if k + 1 < grid.nz else sw[k, j, i]
                    qw_in -= upwind_water(fz[k + 1, j, i], sw[k, j, i], sw_t)

                    # well source/sink water
                    # (matched after loop by name→cell)
                    pv = max(float(phi[k, j, i]) * float(vol[k, j, i]), 1.0e-30)
                    sw_new[k, j, i] = sw[k, j, i] + dt_sub * qw_in / pv

        # well volumetric sources (after flux update, same substep)
        for name, q in well_q.items():
            if name not in mesh.well_cell_id:
                continue
            c = mesh.well_cell_id[name]
            i, j, k = mesh.grid.ijk(c)
            pv = max(float(phi[k, j, i]) * float(vol[k, j, i]), 1.0e-30)
            if q >= 0.0:
                # injection: add water at well Sw sensor if available else 1
                sw_inj = 1.0
                if name in sample.well_saturation:
                    sw_inj = float(sample.well_saturation[name][0])
                sw_new[k, j, i] += dt_sub * q * sw_inj / pv
            else:
                # production: remove mixture at local fw
                sw_new[k, j, i] += dt_sub * q * fw_at(float(sw[k, j, i])) / pv

        sw = np.clip(sw_new, 0.0, 1.0)
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
