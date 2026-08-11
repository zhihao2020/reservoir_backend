"""Saturation transport benchmark hardening suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.reference_case_loader import get_reference_case, load_reference_arrays
from reservoir_backend.core.exceptions import CFLViolationError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.cfl import compute_cfl_number
from reservoir_backend.solver.saturation_diagnostics import (
    build_saturation_diagnostics_report,
    check_saturation_bounds,
    compute_material_balance_error,
)
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    advance_saturation_3d,
)


DEFAULT_RELPERM = {
    "swi": 0.2,
    "sor": 0.2,
    "krw0": 1.0,
    "kro0": 1.0,
    "nw": 2.0,
    "no": 2.0,
    "mu_w": 1.0e-3,
    "mu_o": 5.0e-3,
}


def run_saturation_transport_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run saturation transport benchmark cases and write summary reports."""
    cases = [
        _buckley_leverett_qualitative_case(),
        _mrst_buckley_leverett_reference_case(),
        _boundedness_case(),
        _cfl_stability_case(),
        _material_balance_case(),
        _areal_waterflood_case(),
        _opm_spe1_saturation_sanity_case(),
    ]
    success = all(case["success"] for case in cases)
    summary = {
        "benchmark_name": "saturation_transport_benchmark",
        "success": bool(success),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cases": cases,
        "overall_material_balance_error": float(
            max(abs(_metric(case, "material_balance_error", "relative_material_balance_error", default=0.0)) for case in cases)
        ),
        "overall_max_cfl": float(max(_metric(case, "max_cfl", "stable_max_cfl", default=0.0) for case in cases)),
        "open_source_references_used": [
            "MRST buckleyLeverett1D",
            "MRST simpleIncompTPFA",
            "OPM SPE1CASE1 metadata",
        ],
        "has_nan": bool(any(case["has_nan"] for case in cases)),
        "has_inf": bool(any(case["has_inf"] for case in cases)),
        "warnings": [],
        "recommendations": [
            "Keep BL/front, CFL, boundedness, and material-balance checks as regression gates.",
            "Use MRST/OPM references as adapted metadata or benchmark ideas only.",
            "Do not claim full MRST reproduction, OPM Flow equivalence, or black-oil behavior.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _buckley_leverett_qualitative_case() -> dict:
    grid = Grid3D(nx=100, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    phi = 0.2
    dt = 500.0
    steps = 40
    flux = _x_flux_1d(grid, 1.0e-5)
    sw0 = np.full(grid.shape, DEFAULT_RELPERM["swi"], dtype=float)
    sw_field = Field3D(grid=grid, values=sw0, name="sw", unit="fraction")
    initial_front = _front_position(sw0, threshold=0.25, dx=float(grid.dx[0]))
    injected = 0.0
    produced = 0.0
    max_cfl = 0.0
    last_report: dict = {}
    for _ in range(steps):
        result = advance_saturation_1d(grid, sw_field, phi, flux, dt, DEFAULT_RELPERM, max_cfl=1.0)
        sw_field = result.sw
        last_report = result.report
        injected += float(result.report["injected_water_volume"])
        produced += float(result.report["produced_water_volume"])
        max_cfl = max(max_cfl, float(result.report["max_cfl"]))
    sw = sw_field.values
    final_front = _front_position(sw, threshold=0.25, dx=float(grid.dx[0]))
    pore_volume = phi * grid.cell_volume
    balance = compute_material_balance_error(sw0, sw, injected, produced, pore_volume)
    metrics = {
        "initial_front_position": initial_front,
        "final_front_position": final_front,
        "front_moved_downstream": bool(final_front is not None and (initial_front is None or final_front > initial_front)),
        "inlet_sw_increase": float(sw[0, 0, 0] - sw0[0, 0, 0]),
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "material_balance_error": float(max(balance["relative_material_balance_error"], float(last_report["material_balance_error"]))),
        "max_cfl": max_cfl,
        "has_nan": bool(np.isnan(sw).any()),
        "has_inf": bool(np.isinf(sw).any()),
    }
    success = (
        metrics["front_moved_downstream"]
        and metrics["inlet_sw_increase"] > 0.0
        and metrics["sw_min"] >= DEFAULT_RELPERM["swi"] - 1.0e-12
        and metrics["sw_max"] <= 1.0 - DEFAULT_RELPERM["sor"] + 1.0e-12
        and metrics["material_balance_error"] < 1.0e-10
        and metrics["max_cfl"] <= 1.0
    )
    return _case(
        "buckley_leverett_1d_qualitative",
        grid.shape,
        metrics,
        success,
        source="MRST buckleyLeverett1D.m",
        limitations=[
            "qualitative benchmark idea only; not exact MRST reproduction",
            "no MATLAB runtime dependency",
            "uses current oil-water Corey upwind transport solver",
        ],
    )


def _mrst_buckley_leverett_reference_case() -> dict:
    reference = get_reference_case("mrst_buckley_leverett_1d_reference")
    arrays = load_reference_arrays()
    grid = arrays["mrst_bl_grid"]
    metrics = {
        "metadata_loaded": True,
        "grid_shape": [int(grid[0]), int(grid[1])],
        "porosity": float(arrays["mrst_bl_porosity"][0]),
        "permeability_md": float(arrays["mrst_bl_perm_md"][0]),
        "mentions_explicit_transport": bool(reference["mentions_explicit_transport"]),
        "mentions_implicit_transport": bool(reference["mentions_implicit_transport"]),
        "has_nan": False,
        "has_inf": False,
    }
    return {
        "case_name": "mrst_buckley_leverett_1d_reference",
        "source": "MRST buckleyLeverett1D.m",
        "is_exact_reproduction": False,
        "success": True,
        "grid_shape": metrics["grid_shape"],
        "key_metrics": metrics,
        "warnings": [],
        "has_nan": False,
        "has_inf": False,
        "limitations": [
            "metadata/reference note only",
            "benchmark does not parse or execute buckleyLeverett1D.m",
            "no MRST runtime dependency",
        ],
    }


def _boundedness_case() -> dict:
    grid = Grid3D(nx=20, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    phi = 0.2
    flux = _x_flux_1d(grid, 5.0e-6)
    rng = np.random.default_rng(1234)
    cases = [
        np.full(grid.shape, 0.2),
        _step_x(grid, 0.7, 0.2),
        rng.uniform(0.2, 0.8, size=grid.shape),
        np.full(grid.shape, 0.200001),
        np.full(grid.shape, 0.799999),
    ]
    bounded = 0
    mins = []
    maxs = []
    violations = 0
    clipped_cells = 0
    has_nan = False
    has_inf = False
    for sw0 in cases:
        result = advance_saturation_1d(grid, sw0, phi, flux, 100.0, DEFAULT_RELPERM)
        bounds = check_saturation_bounds(result.sw.values, DEFAULT_RELPERM["swi"], 1.0 - DEFAULT_RELPERM["sor"])
        bounded += int(bounds["bounded"])
        violations += int(bounds["num_below_lower"]) + int(bounds["num_above_upper"])
        clipped_cells += int(result.report["clipped_cells"])
        mins.append(float(np.min(result.sw.values)))
        maxs.append(float(np.max(result.sw.values)))
        has_nan = has_nan or bool(result.report["has_nan"])
        has_inf = has_inf or bool(result.report["has_inf"])
    metrics = {
        "num_cases": len(cases),
        "num_bounded": bounded,
        "sw_min_global": float(min(mins)),
        "sw_max_global": float(max(maxs)),
        "num_bound_violations": violations,
        "clipped_cells": clipped_cells,
        "bound_handling": "solver clips to [Swi, 1-Sor] when needed",
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    return _case(
        "saturation_boundedness",
        grid.shape,
        metrics,
        bounded == len(cases) and violations == 0 and not has_nan and not has_inf,
        source="internal boundedness regression set",
        limitations=["small deterministic oil-water cases only"],
    )


def _cfl_stability_case() -> dict:
    grid = Grid3D(nx=10, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    phi = 0.2
    flux = _x_flux_1d(grid, 1.0e-5)
    flags: dict[str, str] = {}
    warnings: list[str] = []

    stable = advance_saturation_1d(grid, 0.2, phi, flux, 1000.0, DEFAULT_RELPERM)
    near = advance_saturation_1d(grid, 0.2, phi, flux, 9000.0, DEFAULT_RELPERM)
    try:
        advance_saturation_1d(grid, 0.2, phi, flux, 20000.0, DEFAULT_RELPERM)
        too_large_max_cfl = None
        flags["too_large"] = "unexpected_pass"
        warnings.append("too-large dt silently passed")
    except CFLViolationError as exc:
        _, cfl_report = compute_cfl_number(
            grid,
            phi,
            flux,
            np.zeros((1, 2, grid.nx)),
            np.zeros((2, 1, grid.nx)),
            20000.0,
        )
        too_large_max_cfl = float(cfl_report["max_cfl"])
        flags["too_large"] = "cfl_violation"
        warnings.append(str(exc))
    metrics = {
        "stable_max_cfl": float(stable.report["max_cfl"]),
        "near_limit_max_cfl": float(near.report["max_cfl"]),
        "too_large_max_cfl": too_large_max_cfl,
        "num_cfl_warnings": len(warnings),
        "stability_flags": {
            "stable": "success",
            "near_limit": "success",
            **flags,
        },
        "max_cfl": float(max(stable.report["max_cfl"], near.report["max_cfl"], too_large_max_cfl or 0.0)),
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "cfl_stability",
        grid.shape,
        metrics,
        flags.get("too_large") == "cfl_violation",
        source="internal CFL stability regression",
        warnings=warnings,
        limitations=["diagnoses existing CFL behavior without changing CFL algorithm"],
    )


def _material_balance_case() -> dict:
    grid = Grid3D(nx=30, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    phi = 0.25
    sw0 = np.full(grid.shape, 0.2)
    result = advance_saturation_1d(grid, sw0, phi, _x_flux_1d(grid, 1.0e-5), 1000.0, DEFAULT_RELPERM)
    pore_volume = phi * grid.cell_volume
    balance = compute_material_balance_error(
        sw0,
        result.sw.values,
        result.report["injected_water_volume"],
        result.report["produced_water_volume"],
        pore_volume,
    )
    metrics = {
        "initial_water_volume": float(np.sum(sw0 * pore_volume)),
        "final_water_volume": float(np.sum(result.sw.values * pore_volume)),
        "injected_water_volume": float(result.report["injected_water_volume"]),
        "produced_water_volume": float(result.report["produced_water_volume"]),
        "material_balance_residual": float(balance["material_balance_residual"]),
        "relative_material_balance_error": float(balance["relative_material_balance_error"]),
        "material_balance_error": float(max(balance["relative_material_balance_error"], result.report["material_balance_error"])),
        "has_nan": bool(result.report["has_nan"]),
        "has_inf": bool(result.report["has_inf"]),
    }
    return _case(
        "material_balance_1d",
        grid.shape,
        metrics,
        abs(metrics["material_balance_residual"]) < 1.0e-14 and metrics["relative_material_balance_error"] < 1.0e-10,
        source="internal one-step material-balance case",
        limitations=["uses existing solver material-balance report and diagnostic recomputation"],
    )


def _areal_waterflood_case() -> dict:
    grid = Grid3D(nx=20, ny=10, nz=2, dx=1.0, dy=1.0, dz=1.0)
    phi = 0.2
    sw0 = np.full(grid.shape, 0.2)
    flux_x = np.full((grid.nz, grid.ny, grid.nx + 1), 2.0e-6)
    flux_y = np.zeros((grid.nz, grid.ny + 1, grid.nx))
    flux_z = np.zeros((grid.nz + 1, grid.ny, grid.nx))
    sw = Field3D(grid=grid, values=sw0, name="sw", unit="fraction")
    for _ in range(12):
        sw = advance_saturation_3d(grid, sw, phi, flux_x, flux_y, flux_z, 200.0, DEFAULT_RELPERM).sw
    final = sw.values
    metrics = {
        "injection_region_sw_initial": float(np.mean(sw0[:, :, 0])),
        "injection_region_sw_final": float(np.mean(final[:, :, 0])),
        "producer_region_sw_initial": float(np.mean(sw0[:, :, -1])),
        "producer_region_sw_final": float(np.mean(final[:, :, -1])),
        "front_direction_score": float(np.mean(final[:, :, :5] >= sw0[:, :, :5])),
        "sw_min": float(np.min(final)),
        "sw_max": float(np.max(final)),
        "has_nan": bool(np.isnan(final).any()),
        "has_inf": bool(np.isinf(final).any()),
    }
    success = (
        metrics["injection_region_sw_final"] > metrics["injection_region_sw_initial"]
        and metrics["sw_min"] >= DEFAULT_RELPERM["swi"] - 1.0e-12
        and metrics["sw_max"] <= 1.0 - DEFAULT_RELPERM["sor"] + 1.0e-12
        and not metrics["has_nan"]
        and not metrics["has_inf"]
    )
    return _case(
        "areal_waterflood_2d_qualitative",
        grid.shape,
        metrics,
        success,
        source="internal areal-like x-direction waterflood",
        limitations=[
            "implemented as a thin 3D Cartesian case because current 3D saturation API requires nz > 1",
            "qualitative direction and boundedness check only",
        ],
    )


def _opm_spe1_saturation_sanity_case() -> dict:
    reference = get_reference_case("opm_spe1_case1_layered_subset")
    arrays = load_reference_arrays()
    perm_md = np.asarray(arrays["spe1_permx_md"], dtype=float)
    porosity = np.full(perm_md.shape, float(reference["porosity_min"]))
    sw = np.full(perm_md.shape, DEFAULT_RELPERM["swi"])
    bounds = check_saturation_bounds(sw, DEFAULT_RELPERM["swi"], 1.0 - DEFAULT_RELPERM["sor"])
    metrics = {
        "porosity_min": float(np.min(porosity)),
        "porosity_max": float(np.max(porosity)),
        "permeability_min_md": float(np.min(perm_md)),
        "permeability_max_md": float(np.max(perm_md)),
        "permeability_contrast": float(np.max(perm_md) / np.min(perm_md)),
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "bounded": bool(bounds["bounded"]),
        "has_nan": bool(np.isnan(sw).any()),
        "has_inf": bool(np.isinf(sw).any()),
    }
    return _case(
        "opm_spe1case1_saturation_sanity_adapted",
        perm_md.shape,
        metrics,
        metrics["bounded"] and metrics["permeability_contrast"] >= 10.0,
        source="OPM/opm-tests spe1 SPE1CASE1.DATA",
        limitations=[
            "property and boundedness sanity only",
            "not exact SPE1 reproduction",
            "no OPM Flow equivalence, no black-oil, no PVT, no wells",
        ],
    )


def _x_flux_1d(grid: Grid3D, value: float) -> np.ndarray:
    return np.full((1, 1, grid.nx + 1), float(value), dtype=float)


def _step_x(grid: Grid3D, high: float, low: float) -> np.ndarray:
    values = np.full(grid.shape, float(low), dtype=float)
    values[:, :, : grid.nx // 2] = float(high)
    return values


def _front_position(sw: np.ndarray, threshold: float, dx: float) -> float | None:
    line = np.mean(np.asarray(sw, dtype=float), axis=(0, 1))
    wet = np.flatnonzero(line >= threshold)
    return None if wet.size == 0 else float((int(wet[-1]) + 0.5) * dx)


def _case(
    name: str,
    grid_shape: tuple[int, ...] | list[int],
    metrics: dict,
    success: bool,
    *,
    source: str,
    is_exact_reproduction: bool = False,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict:
    has_nan = bool(metrics.get("has_nan", False))
    has_inf = bool(metrics.get("has_inf", False))
    return {
        "case_name": name,
        "source": source,
        "is_exact_reproduction": bool(is_exact_reproduction),
        "success": bool(success and not has_nan and not has_inf),
        "grid_shape": list(grid_shape),
        "key_metrics": _jsonable(metrics),
        "warnings": [] if warnings is None else list(warnings),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "limitations": [] if limitations is None else list(limitations),
    }


def _metric(case: dict, *names: str, default: float) -> float:
    metrics = case["key_metrics"]
    for name in names:
        if name in metrics and metrics[name] is not None:
            return float(metrics[name])
    return float(default)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_reports(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "saturation_transport_benchmark_summary.json"
    md_path = output_dir / "saturation_transport_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Saturation Transport Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- overall_material_balance_error: {summary['overall_material_balance_error']:.6e}",
        f"- overall_max_cfl: {summary['overall_max_cfl']:.6e}",
        "",
        "## Open-source references used",
    ]
    for source in summary["open_source_references_used"]:
        lines.append(f"- {source}")
    lines.extend(["", "## Cases"])
    for case in summary["cases"]:
        lines.extend(["", f"### {case['case_name']}", "", f"- success: {case['success']}", f"- source: {case['source']}"])
        for key, value in case["key_metrics"].items():
            lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_saturation_transport_benchmark(), indent=2))
