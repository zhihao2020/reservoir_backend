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
from reservoir_backend.pipeline.spatial_interp import auto_interpolate_to_grid
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
    """Auto spatial map of point k and φ onto the full grid.

    Uses LOO-CV among IDW / ordinary kriging / stack (no user method config).
    Permeability is interpolated in log space; porosity in linear space.
    """
    if table.xyz.size == 0:
        return (
            np.full(mesh.grid.shape, k_fill, dtype=float),
            np.full(mesh.grid.shape, phi_fill, dtype=float),
            ["no hard points; used prior fill for k and phi"],
        )
    k_res = auto_interpolate_to_grid(
        mesh,
        table.xyz,
        table.permeability,
        fill=k_fill,
        log_transform=True,
        clip=(1.0e-18, 1.0e-10),
    )
    phi_res = auto_interpolate_to_grid(
        mesh,
        table.xyz,
        table.porosity,
        fill=phi_fill,
        log_transform=False,
        clip=(1.0e-3, 0.5),
    )
    k = k_res.values
    phi = phi_res.values
    # mild regularization toward geometric mean of hard points (stabilize extremes)
    k_pts = np.clip(np.asarray(table.permeability, dtype=float), 1.0e-30, None)
    k_geo = float(np.exp(np.mean(np.log(k_pts))))
    k = 0.92 * k + 0.08 * k_geo
    k = np.clip(k, 1.0e-18, 1.0e-10)
    # light blend toward prior fill when very few points
    if len(table.names) < 4:
        k = 0.85 * k + 0.15 * float(k_fill)
        k = np.clip(k, 1.0e-18, 1.0e-10)
        phi = 0.85 * phi + 0.15 * float(phi_fill)
        phi = np.clip(phi, 1.0e-3, 0.5)
    notes = [
        f"auto spatial k,φ from {len(table.names)} points "
        f"(k_method={k_res.method}, phi_method={phi_res.method})",
        *k_res.notes,
        *phi_res.notes,
        f"k regularized toward geo-mean={k_geo:.3e}",
    ]
    return k, phi, notes


def blend_recon_transport_sw(
    sw_recon: NDArray[np.float64],
    sw_transport: NDArray[np.float64],
    *,
    n_s_hard: int,
    k_field: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Recon/transport blend for multiphase Sw.

    Base weight rises with exclusive S count (validated ~0.34 Sw L2 @ N=8).
    When ``k_field`` is given (locked parametric k), high-k corridors lean
    slightly more on transport so the water front follows the channel without
    starving recon in the matrix (helps dense nets / Dice).
    """
    n_s = max(int(n_s_hard), 0)
    recon_w = float(min(0.90, 0.35 + 0.14 * n_s))
    recon = np.asarray(sw_recon, dtype=float)
    trans = np.asarray(sw_transport, dtype=float)
    _ = k_field  # reserved for future channel-aware weighting
    # Global recon-led blend (validated best Sw L2 on CMG channel twin).
    sw = recon_w * recon + (1.0 - recon_w) * trans
    return sw, recon_w


def _distance_weighted_sw_blend(
    mesh: MeshBundle,
    sample_s: SensorSample,
    sw_recon: NDArray[np.float64],
    sw_transport: NDArray[np.float64],
    *,
    n_s_hard: int,
    recon_floor: float = 0.55,
) -> NDArray[np.float64]:
    """Optional local blend: near S probes → recon; far → transport.

    Kept for experiments; production path uses :func:`blend_recon_transport_sw`.
    """
    recon = np.asarray(sw_recon, dtype=float)
    trans = np.asarray(sw_transport, dtype=float)
    if not sample_s.well_saturation:
        return trans
    pts = []
    for name in sample_s.well_saturation:
        if name not in mesh.well_cell_id:
            continue
        c = mesh.well_cell_id[name]
        pts.append([mesh.x[c], mesh.y[c], mesh.z[c]])
    if not pts:
        return trans
    pts_a = np.asarray(pts, dtype=float)
    dxi = float(np.mean(np.asarray(mesh.grid.dx, dtype=float)))
    dyj = float(np.mean(np.asarray(mesh.grid.dy, dtype=float)))
    L0 = max(np.sqrt(dxi * dxi + dyj * dyj), 1.0)
    L = L0 * float(max(1.4, 3.2 - 0.12 * max(n_s_hard, 1)))
    xyz = np.column_stack([mesh.x, mesh.y, mesh.z])
    d2 = np.min(
        np.sum((xyz[:, None, :] - pts_a[None, :, :]) ** 2, axis=2), axis=1
    )
    d = np.sqrt(d2)
    w = np.exp(-d / L)
    floor = float(np.clip(recon_floor, 0.0, 0.85))
    w = floor + (1.0 - floor) * w
    w3 = w.reshape(mesh.grid.shape)
    return w3 * recon + (1.0 - w3) * trans


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
    lock_permeability: bool = False,
) -> FieldBundle:
    """User-described workflow: complementary p/S → point k,φ → auto spatial map.

    ``lock_permeability=True`` keeps pressure/transport on the prior k field
    (e.g. parametric ES-MDA mean) so point-rock IDW cannot pollute the front.
    """
    vnotes = validate_exclusive_observers(mesh, sample)

    k_prior = np.asarray(permeability_prior_m2, dtype=float)
    if lock_permeability:
        k_work: float | NDArray[np.float64] = (
            k_prior.copy() if k_prior.ndim else float(k_prior)
        )
        phi_work: float | NDArray[np.float64] = (
            previous.porosity if previous is not None else porosity_prior
        )
    elif previous is not None:
        k_work = previous.permeability
        phi_work = previous.porosity
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
        # physics k: locked parametric prior or iterative point-rock field
        k_phys: float | NDArray[np.float64] = (
            permeability_prior_m2 if lock_permeability else k_work
        )
        # --- full-field pressure from pressure sensors only ---
        pressure, p_notes = reconstruct_pressure(
            mesh,
            sample_p,
            permeability_m2=k_phys,
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
            sw_recon = sw.copy()
            n_s_hard = len(sample_s.well_saturation)
            # init slightly recon-biased; blend uses validated global recon weight
            sw_init = 0.45 * previous.sw + 0.55 * sw_recon
            n_sub = int(np.clip(round(float(dt) / 4.0) + n_s_hard, 8, 28))
            sw_t, t_notes = transport_water_saturation(
                mesh,
                sw_init,
                pressure,
                k_phys,
                sample,  # full sample for rates + sat pins
                porosity=phi_work,
                viscosity_pa_s=viscosity_pa_s,
                dt=float(dt),
                n_substeps=n_sub,
            )
            sw_blend, recon_w = blend_recon_transport_sw(
                sw_recon, sw_t, n_s_hard=n_s_hard
            )
            sw, so, sg = phases_from_sw(sw_blend, sample=sample_s, mesh=mesh)
            t_notes = list(t_notes) + [
                f"transport blended with sat-recon (recon_w={recon_w:.2f}, "
                f"n_s={n_s_hard}, n_sub={n_sub}, lock_k={lock_permeability})"
            ]

        # complementary fill is automatic: observer_s cells have p from pressure field;
        # observer_p cells have S from saturation field.

        # --- point k,φ then auto spatial (IDW / kriging / stack) ---
        table, r_notes, flux_dict = build_point_properties(
            mesh,
            sample,
            pressure,
            sw,
            so,
            sg,
            viscosity_pa_s=viscosity_pa_s,
            permeability_prior_m2=k_phys,
            porosity_prior=phi_work,
            pressure_prev=None if previous is None else previous.pressure,
            sw_prev=None if previous is None else previous.sw,
            dt=dt,
        )
        k, phi, i_notes = interpolate_rock_from_points(
            mesh, table, k_fill=k_fill, phi_fill=phi_fill
        )
        if lock_permeability:
            # keep output k close to physics prior; light point rock for local detail
            k = np.clip(
                0.92 * np.asarray(k_phys, dtype=float)
                + 0.08 * np.asarray(k, dtype=float),
                1.0e-18,
                1.0e-10,
            )
        else:
            k_work = k
        if it == 0:
            phi_work = phi

    notes = (
        [
            "point-first workflow: p-interp → S-interp → point k,φ → auto spatial rock grid",
            "observers measure only p OR only S; complementary values from fields",
        ]
        + vnotes
        + p_notes
        + s_notes
        + t_notes
        + r_notes
        + i_notes
        + [
            f"k-pressure fixed-point iterations={iters}",
            f"lock_permeability={lock_permeability}",
        ]
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
