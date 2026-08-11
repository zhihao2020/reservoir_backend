"""Point-first four-field workflow matching the sensor design intent.

Workflow
--------
1. Build mesh.
2. Pressure-only hard points (injectors/producers with p, observer_p) → interpolate
   full-grid pressure (TPFA or IDW). Every other hard location (e.g. observer_s)
   receives p from this field.
3. Saturation-only hard points (injectors/producers with S, observer_s) → interpolate
   full-grid saturation. Pressure-only probes receive S from this field.
4. At each hard location that now has both p and S, estimate **point** k and φ
   from local Darcy / continuity.
5. Spatially IDW those point k, φ values onto the full mesh.

A single observer probe measures **either** pressure **or** saturation, never both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.pipeline.pressure_field import reconstruct_pressure
from reservoir_backend.pipeline.property_field import invert_rock_properties
from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
from reservoir_backend.pipeline.spatial_interp import idw_points_to_grid
from reservoir_backend.pipeline.state import FieldBundle, MeshBundle, SensorSample
from reservoir_backend.pipeline.transport_saturation import (
    phases_from_sw,
    transport_water_saturation,
)
from reservoir_backend.solver.velocity import compute_face_fluxes


# Roles that may carry pressure / saturation hard data
_PRESSURE_ROLES = frozenset({"injector", "producer", "observer_p", "observer"})
_SAT_ROLES = frozenset({"injector", "producer", "observer_s", "observer"})
_OBS_P = frozenset({"observer_p"})
_OBS_S = frozenset({"observer_s"})


def validate_exclusive_observers(mesh: MeshBundle, sample: SensorSample) -> list[str]:
    """Enforce: observer_p has only p; observer_s has only S; no rates on observers."""
    notes: list[str] = []
    p_names = set(sample.well_pressure or {})
    s_names = set(sample.well_saturation or {})
    r_names = set(sample.well_rate or {})

    for name, role in mesh.well_role.items():
        if role in _OBS_P:
            if name in s_names:
                raise ValueError(
                    f"probe {name} is observer_p (pressure-only) but has saturation data"
                )
            if name in r_names:
                raise ValueError(f"probe {name} is observer_p and must not have well_rate")
            if name not in p_names:
                notes.append(f"warning: observer_p {name} missing pressure reading")
        elif role in _OBS_S:
            if name in p_names:
                raise ValueError(
                    f"probe {name} is observer_s (saturation-only) but has pressure data"
                )
            if name in r_names:
                raise ValueError(f"probe {name} is observer_s and must not have well_rate")
            if name not in s_names:
                notes.append(f"warning: observer_s {name} missing saturation reading")
        elif role == "observer":
            # legacy: if both provided, reject — one probe cannot measure both
            if name in p_names and name in s_names:
                raise ValueError(
                    f"probe {name}: a single measurement point cannot provide both "
                    f"pressure and saturation; use observer_p or observer_s"
                )
            if name in r_names:
                raise ValueError(f"probe {name} is observer and must not have well_rate")
    return notes


def filter_sample_for_pressure(sample: SensorSample, mesh: MeshBundle) -> SensorSample:
    """Keep only names allowed to contribute pressure hard data."""
    allowed = {
        n
        for n, r in mesh.well_role.items()
        if r in _PRESSURE_ROLES and r not in _OBS_S
    }
    # also include pressure names not on mesh roles (legacy)
    for n in sample.well_pressure:
        if n not in mesh.well_role:
            allowed.add(n)
    # exclude pure saturation observers
    for n, r in mesh.well_role.items():
        if r in _OBS_S:
            allowed.discard(n)
    well_p = {n: v for n, v in sample.well_pressure.items() if n in allowed}
    # rates only for injectors/producers
    rates = {
        n: v
        for n, v in (sample.well_rate or {}).items()
        if mesh.well_role.get(n) in ("injector", "producer")
    }
    return SensorSample(
        time=sample.time,
        well_pressure=well_p,
        well_saturation={},  # unused for pressure path
        boundary=sample.boundary,
        well_rate=rates,
    )


def filter_sample_for_saturation(sample: SensorSample, mesh: MeshBundle) -> SensorSample:
    """Keep only names allowed to contribute saturation hard data."""
    allowed = {
        n
        for n, r in mesh.well_role.items()
        if r in _SAT_ROLES and r not in _OBS_P
    }
    for n in sample.well_saturation:
        if n not in mesh.well_role:
            allowed.add(n)
    for n, r in mesh.well_role.items():
        if r in _OBS_P:
            allowed.discard(n)
    well_s = {n: v for n, v in sample.well_saturation.items() if n in allowed}
    return SensorSample(
        time=sample.time,
        well_pressure={},
        well_saturation=well_s,
        boundary=sample.boundary,
        well_rate={},
    )


@dataclass
class PointPropertyTable:
    """Point estimates of rock properties at hard sensor locations."""

    names: list[str]
    xyz: NDArray[np.float64]  # (n, 3)
    pressure: NDArray[np.float64]
    sw: NDArray[np.float64]
    permeability: NDArray[np.float64]
    porosity: NDArray[np.float64]


def hard_sensor_names(mesh: MeshBundle, sample: SensorSample) -> list[str]:
    """All named locations that participate as hard sensors this sample."""
    names = set(sample.well_pressure) | set(sample.well_saturation) | set(sample.well_rate or {})
    names &= set(mesh.well_cell_id)
    return sorted(names)


def build_point_properties(
    mesh: MeshBundle,
    sample: SensorSample,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    *,
    viscosity_pa_s: float,
    permeability_prior_m2: float | NDArray[np.float64],
    porosity_prior: float | NDArray[np.float64],
    pressure_prev: NDArray[np.float64] | None = None,
    sw_prev: NDArray[np.float64] | None = None,
    dt: float | None = None,
) -> tuple[PointPropertyTable, list[str], dict[str, NDArray[np.float64]]]:
    """Estimate k, φ only at hard points using full-field p/S for local Darcy."""
    # Full-grid invert for fluxes and cell-wise estimates, then sample at points
    k_grid, phi_grid, notes, fluxes = invert_rock_properties(
        mesh,
        pressure,
        sw,
        so,
        sg,
        viscosity_pa_s=viscosity_pa_s,
        permeability_prior_m2=permeability_prior_m2,
        porosity_prior=porosity_prior,
        pressure_prev=pressure_prev,
        sw_prev=sw_prev,
        dt=dt,
    )
    names = hard_sensor_names(mesh, sample)
    if not names:
        # fall back: use all well locations on mesh
        names = sorted(mesh.well_cell_id.keys())
    xyz = []
    p_list, sw_list, k_list, phi_list = [], [], [], []
    for name in names:
        c = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(c)
        xyz.append([mesh.x[c], mesh.y[c], mesh.z[c]])
        # complementary fill already in fields
        p_list.append(float(pressure[k, j, i]))
        sw_list.append(float(sw[k, j, i]))
        k_list.append(float(k_grid[k, j, i]))
        phi_list.append(float(phi_grid[k, j, i]))
    table = PointPropertyTable(
        names=names,
        xyz=np.asarray(xyz, dtype=float),
        pressure=np.asarray(p_list, dtype=float),
        sw=np.asarray(sw_list, dtype=float),
        permeability=np.asarray(k_list, dtype=float),
        porosity=np.asarray(phi_list, dtype=float),
    )
    notes = [
        "point-first rock: k,φ estimated at hard sensor locations only",
        f"hard points n={len(names)}: {names}",
    ] + notes
    return table, notes, fluxes


def interpolate_rock_from_points(
    mesh: MeshBundle,
    table: PointPropertyTable,
    *,
    k_fill: float,
    phi_fill: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Spatial IDW of point k and φ onto the full grid."""
    if table.xyz.size == 0:
        return (
            np.full(mesh.grid.shape, k_fill, dtype=float),
            np.full(mesh.grid.shape, phi_fill, dtype=float),
            ["no hard points; used prior fill for k and phi"],
        )
    k = idw_points_to_grid(mesh, table.xyz, table.permeability, fill=k_fill)
    phi = idw_points_to_grid(mesh, table.xyz, table.porosity, fill=phi_fill)
    k = np.clip(k, 1.0e-18, 1.0e-10)
    phi = np.clip(phi, 1.0e-3, 0.5)
    notes = [
        f"spatial IDW of point k,φ onto grid from {len(table.names)} points",
    ]
    return k, phi, notes


def run_point_first_slice(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    previous: FieldBundle | None = None,
    dt: float | None = None,
    n_k_iterations: int = 2,
    use_transport: bool = True,
) -> FieldBundle:
    """User-described workflow: complementary p/S fields → point k,φ → IDW grid."""
    vnotes = validate_exclusive_observers(mesh, sample)

    if previous is not None:
        k_work: float | NDArray[np.float64] = previous.permeability
        phi_work: float | NDArray[np.float64] = previous.porosity
    else:
        k_work = permeability_prior_m2
        phi_work = porosity_prior

    k_fill = float(np.mean(np.asarray(k_work, dtype=float)))
    phi_fill = float(np.mean(np.asarray(phi_work, dtype=float)))

    sample_p = filter_sample_for_pressure(sample, mesh)
    sample_s = filter_sample_for_saturation(sample, mesh)

    iters = max(1, int(n_k_iterations))
    pressure = np.zeros(mesh.grid.shape, dtype=float)
    sw = so = sg = np.zeros(mesh.grid.shape, dtype=float)
    k = np.full(mesh.grid.shape, k_fill, dtype=float)
    phi = np.full(mesh.grid.shape, phi_fill, dtype=float)
    flux_dict: dict[str, NDArray[np.float64]] = {}
    p_notes: list[str] = []
    s_notes: list[str] = []
    t_notes: list[str] = []
    r_notes: list[str] = []
    i_notes: list[str] = []

    for it in range(iters):
        # --- full-field pressure from pressure sensors only ---
        pressure, p_notes = reconstruct_pressure(
            mesh,
            sample_p,
            permeability_m2=k_work,
            viscosity_pa_s=viscosity_pa_s,
        )
        p_notes = [
            "step2: pressure field from pressure-hard points only "
            f"(n={len(sample_p.well_pressure)})"
        ] + p_notes

        # --- full-field saturation from saturation sensors only ---
        sw, so, sg, s_notes = reconstruct_saturation(
            mesh, sample_s, pressure=pressure
        )
        s_notes = [
            "step3: saturation field from saturation-hard points only "
            f"(n={len(sample_s.well_saturation)})"
        ] + s_notes

        if (
            use_transport
            and previous is not None
            and dt is not None
            and float(dt) > 0.0
        ):
            sw_init = 0.5 * previous.sw + 0.5 * sw
            sw_t, t_notes = transport_water_saturation(
                mesh,
                sw_init,
                pressure,
                k_work,
                sample,  # full sample for rates + sat pins
                porosity=phi_work,
                viscosity_pa_s=viscosity_pa_s,
                dt=float(dt),
                n_substeps=8,
            )
            sw, so, sg = phases_from_sw(sw_t, sample=sample_s, mesh=mesh)

        # complementary fill is automatic: observer_s cells have p from pressure field;
        # observer_p cells have S from saturation field.

        # --- point k,φ then spatial IDW ---
        table, r_notes, flux_dict = build_point_properties(
            mesh,
            sample,
            pressure,
            sw,
            so,
            sg,
            viscosity_pa_s=viscosity_pa_s,
            permeability_prior_m2=k_work,
            porosity_prior=phi_work,
            pressure_prev=None if previous is None else previous.pressure,
            sw_prev=None if previous is None else previous.sw,
            dt=dt,
        )
        k, phi, i_notes = interpolate_rock_from_points(
            mesh, table, k_fill=k_fill, phi_fill=phi_fill
        )
        k_work = k
        if it == 0:
            phi_work = phi

    notes = (
        [
            "point-first workflow: p-interp → S-interp → point k,φ → IDW rock grid",
            "observers measure only p OR only S; complementary values from fields",
        ]
        + vnotes
        + p_notes
        + s_notes
        + t_notes
        + r_notes
        + i_notes
        + [f"k-pressure fixed-point iterations={iters}"]
    )
    return FieldBundle(
        time=sample.time,
        pressure=pressure,
        sw=sw,
        so=so,
        sg=sg,
        permeability=k,
        porosity=phi,
        notes=notes,
        flux_x=flux_dict.get("flux_x"),
        flux_y=flux_dict.get("flux_y"),
        flux_z=flux_dict.get("flux_z"),
    )
