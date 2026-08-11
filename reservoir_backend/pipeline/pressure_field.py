"""Reconstruct full-grid pressure from sparse well and boundary sensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.wells import Well
from reservoir_backend.pipeline.state import MeshBundle, SensorSample
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.transmissibility import validate_viscosity


def reconstruct_pressure(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64] = 1.0e-13,
    viscosity_pa_s: float = 1.0e-3,
) -> tuple[NDArray[np.float64], list[str]]:
    """Step 2: well/observer P + boundary P/flux → full-grid pressure.

    Injector/producer **and observer (测点)** pressures are assembled as true
    cell Dirichlet constraints. Observers do not inject/produce fluid.

    Face Dirichlet pressures and optional Neumann fluxes (m^3/s into domain)
    use the TPFA face boundary treatment. Permeability forms transmissibility.
    """
    notes: list[str] = [
        "pressure reconstruction uses permeability prior for TPFA transmissibility",
        "observer probes: hard p Dirichlet, no fluid rate (interior sensors)",
    ]
    grid = mesh.grid
    validate_viscosity(viscosity_pa_s)

    boundaries = {
        key: float(value)
        for key, value in sample.boundary.pressure.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    neumann = {
        key: float(value)
        for key, value in sample.boundary.flux.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    cell_dirichlet = _well_cell_dirichlet(mesh, sample)
    rate_wells = _rate_wells_from_sample(mesh, sample)
    obs_names = sample.observation_names(mesh)
    if cell_dirichlet:
        notes.append(f"hard pressure sensors (wells+observers) count={len(cell_dirichlet)}")
    if obs_names:
        notes.append(f"observer probes: {obs_names}")
    if rate_wells:
        notes.append(f"well rate sources count={len(rate_wells)}")
    if neumann:
        notes.append(f"neumann flux faces={sorted(neumann.keys())}")

    if grid.nx > 1 and grid.ny > 1 and grid.nz > 1:
        try:
            result = solve_steady_state_pressure_3d(
                grid=grid,
                kx=permeability_m2,
                ky=permeability_m2,
                kz=permeability_m2,
                mu=viscosity_pa_s,
                dirichlet_boundaries=boundaries or None,
                wells=rate_wells or None,
                reference_pressure=float(next(iter(sample.well_pressure.values()), 0.0)),
                cell_dirichlet=cell_dirichlet or None,
                neumann_fluxes=neumann or None,
            )
            pressure = result.pressure.values.copy()
            # Dirichlet wells already fixed in matrix; pin again for numerics
            pressure = _pin_well_pressures(mesh, sample, pressure)
            n_bc = int(result.report.get("cell_dirichlet_count", 0))
            notes.append(
                "used finite-volume TPFA with matrix well-cell Dirichlet"
                f" (n={n_bc})"
            )
            return pressure, notes
        except Exception as exc:  # pragma: no cover
            notes.append(f"TPFA path failed ({exc}); falling back to sparse blending")

    pressure = _blend_from_sensors(mesh, sample)
    notes.append("used inverse-distance blending of well and boundary sensors")
    return pressure, notes


def _well_cell_dirichlet(mesh: MeshBundle, sample: SensorSample) -> dict[int, float]:
    out: dict[int, float] = {}
    for name, value in sample.well_pressure.items():
        if name not in mesh.well_cell_id:
            raise KeyError(f"sensor well {name} is not on the mesh")
        out[int(mesh.well_cell_id[name])] = float(value)
    return out


def _rate_wells_from_sample(mesh: MeshBundle, sample: SensorSample) -> list[Well]:
    """Build rate wells only for flowing injectors/producers (never observers).

    If the same cell also has Dirichlet BHP, the pressure solver skips the
    rate source on that cell (Dirichlet wins) — rates still feed transport.
    """
    wells: list[Well] = []
    for name, q in (sample.well_rate or {}).items():
        if name not in mesh.well_cell_id:
            continue
        # never treat observers as rate sources even if mis-specified
        if mesh.well_role.get(name) == "observer":
            continue
        q = float(q)
        if not np.isfinite(q) or abs(q) < 1.0e-30:
            continue
        cell = int(mesh.well_cell_id[name])
        i, j, k = mesh.grid.ijk(cell)
        wtype = "injection" if q > 0.0 else "production"
        wells.append(
            Well(
                name=f"{name}_rate",
                well_type=wtype,
                grid=mesh.grid,
                i=i,
                j=j,
                k=k,
                control="rate",
                rate=abs(q),
            )
        )
    return wells


def _pin_well_pressures(
    mesh: MeshBundle,
    sample: SensorSample,
    pressure: NDArray[np.float64],
) -> NDArray[np.float64]:
    out = pressure.copy()
    for name, value in sample.well_pressure.items():
        cell = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(cell)
        out[k, j, i] = float(value)
    return out


def _blend_from_sensors(mesh: MeshBundle, sample: SensorSample) -> NDArray[np.float64]:
    grid = mesh.grid
    anchors_xyz: list[tuple[float, float, float]] = []
    anchors_p: list[float] = []

    for name, value in sample.well_pressure.items():
        cell = mesh.well_cell_id[name]
        anchors_xyz.append((float(mesh.x[cell]), float(mesh.y[cell]), float(mesh.z[cell])))
        anchors_p.append(float(value))

    bounds = mesh.bounds
    if bounds is not None:
        for side, value in sample.boundary.pressure.items():
            if side == "left":
                anchors_xyz.append((bounds.xmin, 0.5 * (bounds.ymin + bounds.ymax), 0.5 * (bounds.zmin + bounds.zmax)))
            elif side == "right":
                anchors_xyz.append((bounds.xmax, 0.5 * (bounds.ymin + bounds.ymax), 0.5 * (bounds.zmin + bounds.zmax)))
            elif side == "front":
                anchors_xyz.append((0.5 * (bounds.xmin + bounds.xmax), bounds.ymin, 0.5 * (bounds.zmin + bounds.zmax)))
            elif side == "back":
                anchors_xyz.append((0.5 * (bounds.xmin + bounds.xmax), bounds.ymax, 0.5 * (bounds.zmin + bounds.zmax)))
            elif side == "bottom":
                anchors_xyz.append((0.5 * (bounds.xmin + bounds.xmax), 0.5 * (bounds.ymin + bounds.ymax), bounds.zmin))
            elif side == "top":
                anchors_xyz.append((0.5 * (bounds.xmin + bounds.xmax), 0.5 * (bounds.ymin + bounds.ymax), bounds.zmax))
            else:
                continue
            anchors_p.append(float(value))

    if not anchors_p:
        raise ValueError("at least one well or boundary pressure sensor is required")

    field = np.zeros(grid.shape, dtype=float)
    power = 2.0
    eps = 1.0e-12
    for idx in range(mesh.n_cells):
        px, py, pz = mesh.x[idx], mesh.y[idx], mesh.z[idx]
        weights = []
        values = []
        for (ax, ay, az), ap in zip(anchors_xyz, anchors_p):
            dist = np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
            if dist < eps:
                weights = [1.0]
                values = [ap]
                break
            w = 1.0 / (dist**power)
            weights.append(w)
            values.append(ap)
        wsum = float(np.sum(weights))
        field.flat[idx] = float(np.dot(weights, values) / wsum)
    return field


def pressure_as_field(mesh: MeshBundle, pressure: NDArray[np.float64]) -> Field3D:
    return Field3D(grid=mesh.grid, values=pressure, name="pressure", unit="Pa")
