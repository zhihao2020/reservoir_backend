"""Run the four reconstruction programs and write CMG-aligned fields."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from ..core.cartesian import CartesianGrid
from ..core.lab_case import LabCase, summarize_lab_case
from ..core.units import MD_TO_M2
from ..version import __version__
from .mesh import MeshResult, build_mesh
from .forward import forward_saturations
from .pressure import interpolate_pressure, interpolate_pressure_wells
from .results import summarize_results
from .rock import invert_rock, invert_rock_three_phase, solution_gas_ratio, transient_weights, RockDiagnostics
from .saturation import (
    interpolate_saturation,
    project_saturations3,
    smooth_fields,
)
from .similarity import (
    MODEL_HEIGHT_M,
    MODEL_LENGTH_M,
    MODEL_VOLUME_M3,
    MODEL_WIDTH_M,
    field_to_lab_rate,
    field_to_lab_time,
    field_to_lab_volume,
    lab_to_field_rate,
    lab_to_field_time,
    lab_to_field_volume,
    similarity_ratios,
)


@dataclass(frozen=True)
class ProgramFields:
    mesh: MeshResult
    times: NDArray[np.float64]
    p: NDArray[np.float64]
    sw: NDArray[np.float64]
    so: NDArray[np.float64]
    sg: NDArray[np.float64]
    phi: NDArray[np.float64]
    k: NDArray[np.float64]
    # Solution-gas (CO2-in-oil) extension. ``rs`` is the dissolved gas-oil ratio
    # (Rs = rs_slope * p), ``co2`` the total CO2 component (sg + Rs*so). Offline
    # outputs only; the frozen wire protocol (p/sw/so/sg/phi/k) is unchanged.
    rs: NDArray[np.float64] | None = None
    co2: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_mesh(case: LabCase) -> MeshResult:
    return build_mesh(
        case.origin,
        case.extent,
        case.nx,
        case.ny,
        case.nz,
        case.probe_ids,
        case.probe_xyz,
        case.well_ids,
        case.well_xyz,
        case.well_trajectories,
    )


def _invert_static(case: LabCase, mesh: MeshResult, p, sw, so, sg):
    """Invert static k/phi from pressure + saturation, or return the known
    constant values for a homogeneous rock (where the inversion is ill-posed)."""
    n_c = mesh.grid.n_cells
    if case.k_homogeneous:
        diag = RockDiagnostics()
        diag.k_mean_error = 0.0
        diag.phi_mean_error = 0.0
        diag.converged = True
        return np.full(n_c, float(case.phi0)), np.full(n_c, float(case.k0)), diag
    if case.rock_model == "black_oil_3phase":
        tw = transient_weights(case.times) if case.transient else None
        return invert_rock_three_phase(
            mesh.grid,
            p,
            sw,
            so,
            sg,
            case.times,
            mesh.probes,
            mesh.wells,
            case.well_qw,
            case.well_qo,
            case.well_qg,
            phi0=case.phi0,
            k0=case.k0,
            params=case.black_oil,
            relperm=case.relperm,
            time_weights=tw,
            fractional_weight=case.fractional_weight,
            well_bhp=case.well_pw,
            well_params=case.well,
            smoothness=case.k_smoothness,
        )
    return invert_rock(
        mesh.grid,
        p,
        sw,
        so,
        sg,
        case.times,
        mesh.probes,
        mesh.wells,
        case.well_q,
        phi0=case.phi0,
        k0=case.k0,
        params=case.black_oil,
        well_bhp=case.well_pw,
        well_params=case.well,
    )


def run_pipeline(case: LabCase, mesh: MeshResult | None = None) -> ProgramFields:
    mesh = mesh or run_mesh(case)
    n_t = int(case.times.size)
    n_c = mesh.grid.n_cells
    p = np.zeros((n_t, n_c))
    sw = np.zeros((n_t, n_c))
    so = np.zeros((n_t, n_c))
    sg = np.zeros((n_t, n_c))
    pressure_method = str(case.method).strip().lower()
    sat_method = pressure_method
    well_cells = tuple(np.asarray(c, dtype=np.int64) for c in mesh.wells.cells)
    for t in range(n_t):
        p[t] = interpolate_pressure_wells(
            mesh.grid,
            case.probe_xyz,
            case.pressure[t],
            well_cells,
            case.well_pw[t],
            method=pressure_method,
        )
        sw_obs, so_obs, sg_obs = _saturation_slice(case, t)
        sw[t], so[t], sg[t] = interpolate_saturation(
            mesh.grid,
            case.probe_xyz,
            sw_obs,
            so_obs,
            sg_obs,
            method=sat_method,
            swc=case.black_oil.swc,
            sgc=case.black_oil.sgc,
        )
    p = smooth_fields(p)
    sw = smooth_fields(sw)
    so = smooth_fields(so)
    for t in range(n_t):
        sw[t], so[t], sg[t] = project_saturations3(sw[t], so[t], sg[t])

    phi_static, k_static, rock_diag = _invert_static(case, mesh, p, sw, so, sg)
    phi = np.repeat(phi_static[None, :], n_t, axis=0)
    k = np.repeat(k_static[None, :], n_t, axis=0)
    # P3: forward-simulate the saturations (mass-conserving) from the initial
    # reservoir state at t=0, prepending a synthetic t=0 with the first observed
    # rates so the first report time is also forward-simulated (not the
    # over-smoothed kriged field). The forward-simulated saturation replaces the
    # kriged one as the final output.
    oil = case.black_oil
    if case.forward_model in ("none", "off", "skip", "kriging"):
        # Skip the forward model: keep the kriged saturation (fast path).
        pass
    else:
        sw0 = np.full(n_c, oil.swc)
        sg0 = np.full(n_c, oil.sgc)
        so0 = np.full(n_c, max(0.0, 1.0 - oil.swc - oil.sgc))
        times_fwd = np.concatenate([[0.0], case.times])
        p_fwd = np.vstack([p[:1], p])
        qw_fwd = np.vstack([case.well_qw[:1], case.well_qw])
        qo_fwd = np.vstack([case.well_qo[:1], case.well_qo])
        qg_fwd = np.vstack([case.well_qg[:1], case.well_qg])
        sw_f, so_f, sg_f = forward_saturations(
            case.forward_model,
            mesh.grid,
            p_fwd,
            k_static,
            phi_static,
            oil,
            mesh.wells,
            qw_fwd,
            qo_fwd,
            qg_fwd,
            times_fwd,
            sw0,
            so0,
            sg0,
        )
        sw, so, sg = sw_f[1:], so_f[1:], sg_f[1:]
    # Re-invert k/phi against the forward-simulated (mass-conserving) saturation,
    # so the permeability field is consistent with the corrected plume rather than
    # the over-smoothed kriged field. This is the fixed-point coupling between the
    # inversion and the forward model (one extra pass).
    phi_static, k_static, rock_diag = _invert_static(case, mesh, p, sw, so, sg)
    phi = np.repeat(phi_static[None, :], n_t, axis=0)
    k = np.repeat(k_static[None, :], n_t, axis=0)
    # Solution-gas fields: dissolved gas-oil ratio (Rs) and total CO2 component
    # (free gas + dissolved gas). Derived from the reconstructed pressure, which
    # is the accurate part of the reconstruction.
    rs = solution_gas_ratio(p, case.black_oil)
    co2 = sg + rs * so
    diagnostics = rock_diag.as_dict()
    diagnostics["pressure_method"] = pressure_method
    diagnostics["saturation_method"] = sat_method
    diagnostics["holdout"] = holdout_probe_errors(
        case, mesh, p, sw, so,
        pressure_method=pressure_method,
        saturation_method=sat_method,
    )
    return ProgramFields(mesh=mesh, times=case.times, p=p, sw=sw, so=so, sg=sg, phi=phi, k=k, rs=rs, co2=co2, diagnostics=diagnostics)


def holdout_probe_errors(
    case: LabCase,
    mesh: MeshResult,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    n_hold: int = 3,
    *,
    pressure_method: str = "auto",
    saturation_method: str = "auto",
) -> dict[str, Any]:
    n_probe = int(case.probe_xyz.shape[0])
    if n_probe < 6:
        return {"skipped": True, "reason": "need at least six probes"}
    hold = np.arange(n_probe)[-min(n_hold, n_probe // 3) :]
    keep = np.ones(n_probe, dtype=bool)
    keep[hold] = False
    t = 0
    p_wo = interpolate_pressure(
        mesh.grid,
        case.probe_xyz[keep],
        case.pressure[t, keep],
        case.well_xyz,
        case.well_pw[t],
        method=pressure_method,
    )
    sw_obs, so_obs, sg_obs = _saturation_slice(case, t)
    sw_wo, so_wo, _ = interpolate_saturation(
        mesh.grid,
        case.probe_xyz[keep],
        sw_obs[keep],
        so_obs[keep],
        sg_obs[keep],
        method=saturation_method,
        swc=case.black_oil.swc,
        sgc=case.black_oil.sgc,
    )
    cells = mesh.probes.cell[hold]
    return {
        "probes": [case.probe_ids[int(i)] for i in hold],
        "p_rmse": float(np.sqrt(np.mean((p_wo[cells] - case.pressure[t, hold]) ** 2))),
        "sw_rmse": float(np.sqrt(np.mean((sw_wo[cells] - sw_obs[hold]) ** 2))),
        "so_rmse": float(np.sqrt(np.mean((so_wo[cells] - so_obs[hold]) ** 2))),
    }


def field_scale_view(
    case: LabCase, fields: ProgramFields
) -> tuple[CartesianGrid, NDArray[np.float64], dict[str, float]]:
    """Lab reconstruction mapped to field geometry and time (similarity ratios)."""
    model_flow_path = case.model_flow_path_m if case.model_flow_path_m is not None else MODEL_LENGTH_M
    ratios = similarity_ratios(
        case.field_length_m,
        case.field_width_m,
        case.field_height_m,
        case.field_flow_path_m,
        model_flow_path_m=model_flow_path,
    )
    grid = fields.mesh.grid.scaled(
        (ratios["lambda_x"], ratios["lambda_y"], ratios["lambda_z"])
    )
    times = np.asarray(fields.times, dtype=float) * float(ratios["c_t"])
    return grid, times, ratios


def write_output(folder: Path, case: LabCase, fields: ProgramFields) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    grid = fields.mesh.grid
    np.savez_compressed(
        folder / "fields.npz",
        time=fields.times,
        p=fields.p,
        sw=fields.sw,
        so=fields.so,
        sg=fields.sg,
        phi=fields.phi,
        k=fields.k,
        rs=fields.rs,
        co2=fields.co2,
        k_md=fields.k / MD_TO_M2,
        nx=grid.nx,
        ny=grid.ny,
        nz=grid.nz,
        origin=np.array(grid.origin),
        extent=np.array(grid.size_m()),
        scale="lab",
    )
    field_grid, field_times, ratios = field_scale_view(case, fields)
    model_flow_path = case.model_flow_path_m if case.model_flow_path_m is not None else MODEL_LENGTH_M
    np.savez_compressed(
        folder / "fields_field.npz",
        time=field_times,
        p=fields.p,
        sw=fields.sw,
        so=fields.so,
        sg=fields.sg,
        phi=fields.phi,
        k=fields.k,
        rs=fields.rs,
        co2=fields.co2,
        k_md=fields.k / MD_TO_M2,
        nx=field_grid.nx,
        ny=field_grid.ny,
        nz=field_grid.nz,
        origin=np.array(field_grid.origin),
        extent=np.array(field_grid.size_m()),
        scale="field",
    )
    lam = np.array([ratios["lambda_x"], ratios["lambda_y"], ratios["lambda_z"]])
    scaled = not np.allclose(lam, 1.0)
    wells_field_rows: list[dict[str, object]] = []
    if scaled:
        # 井位换算：四口采收井（及注入井）在矿场尺度的等效井位
        wells_field_rows = _well_field_rows(case, lam)
        _write_wells_field(folder / "wells_field.csv", wells_field_rows)
    # 无量纲时间（注入孔隙体积倍数）——相似准则之一
    times_a = np.asarray(fields.times, dtype=float)
    if times_a.size > 1:
        dt = np.diff(times_a)
        q_inj = np.maximum(case.well_q[:-1], 0.0)
        injected = float(np.sum(q_inj * dt[:, None]))
    else:
        injected = 0.0
    pore_volume = float(np.average(fields.phi[0], weights=grid.cell_volumes())) * float(grid.total_volume())
    pore_volumes = injected / pore_volume if pore_volume > 0.0 else 0.0
    for name, arr in (
        ("p", fields.p),
        ("sw", fields.sw),
        ("so", fields.so),
        ("sg", fields.sg),
        ("phi", fields.phi),
        ("k", fields.k),
        ("rs", fields.rs),
        ("co2", fields.co2),
        ("k_md", fields.k / MD_TO_M2),
    ):
        np.save(folder / f"{name}.npy", arr)
    _write_mapping(folder / "mesh.csv", fields.mesh.mapping_rows("probe") + fields.mesh.mapping_rows("well"))
    _write_well_series(folder / "wells.csv", case)
    vol = grid.cell_volumes()
    summary = {
        "library": "reservoir-backend",
        "version": __version__,
        "n_times": int(fields.times.size),
        "n_cells": grid.n_cells,
        "shape_ijk": list(grid.shape_ijk),
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "phi_mean": float(np.average(fields.phi[0], weights=vol)),
        "k_md_mean": float(np.average(fields.k[0], weights=vol) / MD_TO_M2),
        "phi0": case.phi0,
        "k0_md": case.k0 / MD_TO_M2,
        "probes": case.probe_ids,
        "wells": case.well_ids,
        "times_s": fields.times.tolist(),
    }
    summary.update(summarize_lab_case(case))
    summary["similarity"] = {
        **ratios,
        "model": {
            "length_m": MODEL_LENGTH_M,
            "width_m": MODEL_WIDTH_M,
            "height_m": MODEL_HEIGHT_M,
            "volume_m3": MODEL_VOLUME_M3,
            "flow_path_m": model_flow_path,
        },
        "volume_basis": case.volume_basis,
        "pore_volumes": pore_volumes,
        "conversion": {
            "field_to_lab": {
                "time_min_per_day": field_to_lab_time(1.0, ratios["c_lc"]),
                "rate_ml_min_per_m3_d": field_to_lab_rate(1.0, ratios["c_lc"], ratios["c_v"]),
                "volume_ml_per_m3": field_to_lab_volume(1.0, ratios["c_v"]),
            },
            "lab_to_field": {
                "time_d_per_min": lab_to_field_time(1.0, ratios["c_lc"]),
                "rate_m3_d_per_ml_min": lab_to_field_rate(1.0, ratios["c_lc"], ratios["c_v"]),
                "volume_m3_per_ml": lab_to_field_volume(1.0, ratios["c_v"]),
            },
        },
    }
    if wells_field_rows:
        summary["similarity"]["wells_field"] = wells_field_rows
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (folder / "validation.json").write_text(
        json.dumps(fields.diagnostics, indent=2, default=float),
        encoding="utf-8",
    )
    (folder / "results.json").write_text(
        json.dumps(summarize_results(case, fields.mesh, fields), indent=2, default=float),
        encoding="utf-8",
    )
    return summary


def _saturation_slice(case: LabCase, t: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """One time slice of the phase saturations, carrying forward the previous
    time when a phase is unobserved at this time. Missing phases stay NaN so
    ``interpolate_saturation`` can recover them by closure or residual fill."""
    sw = np.array(case.sw[t], dtype=float, copy=True)
    so = np.array(case.so[t], dtype=float, copy=True)
    sg = np.array(case.sg[t], dtype=float, copy=True)
    if not np.isfinite(sw).any() and t > 0 and np.isfinite(case.sw[t - 1]).any():
        sw = np.array(case.sw[t - 1], dtype=float, copy=True)
    if not np.isfinite(so).any() and t > 0 and np.isfinite(case.so[t - 1]).any():
        so = np.array(case.so[t - 1], dtype=float, copy=True)
    if not np.isfinite(sg).any() and t > 0 and np.isfinite(case.sg[t - 1]).any():
        sg = np.array(case.sg[t - 1], dtype=float, copy=True)
    return sw, so, sg


def _write_mapping(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["kind", "id", "x", "y", "z", "i", "j", "k", "cell"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _well_field_rows(case: LabCase, lam: NDArray[np.float64]) -> list[dict[str, object]]:
    """井位换算：把模型井位（跟/趾端）逐轴放大到矿场尺度。"""
    rows: list[dict[str, object]] = []
    for w in case.wells:
        row: dict[str, object] = {
            "id": w.id,
            "kind": w.kind,
            "x_m": float(w.x * lam[0]),
            "y_m": float(w.y * lam[1]),
            "z_m": float(w.z * lam[2]),
        }
        if w.x2 is not None and w.y2 is not None and w.z2 is not None:
            row.update(
                {
                    "x2_m": float(w.x2 * lam[0]),
                    "y2_m": float(w.y2 * lam[1]),
                    "z2_m": float(w.z2 * lam[2]),
                }
            )
        rows.append(row)
    return rows


def _write_wells_field(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "kind", "x_m", "y_m", "z_m", "x2_m", "y2_m", "z2_m"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_well_series(path: Path, case: LabCase) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_s", "well", "kind", "pw_pa", "q_m3s", "qw", "qo", "qg"],
        )
        writer.writeheader()
        for t, time in enumerate(case.times):
            for j, well in enumerate(case.wells):
                writer.writerow(
                    {
                        "time_s": float(time),
                        "well": well.id,
                        "kind": well.kind,
                        "pw_pa": float(case.well_pw[t, j]),
                        "q_m3s": float(case.well_q[t, j]),
                        "qw": float(case.well_qw[t, j]),
                        "qo": float(case.well_qo[t, j]),
                        "qg": float(case.well_qg[t, j]),
                    }
                )
