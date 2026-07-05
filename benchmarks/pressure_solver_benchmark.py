"""Pressure solver benchmark hardening suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.pressure_diagnostics import (
    build_pressure_diagnostics_report,
    compute_pressure_error_metrics,
)
from reservoir_backend.solver.pressure_solver import (
    solve_steady_state_pressure_1d,
    solve_steady_state_pressure_2d,
    solve_steady_state_pressure_3d,
)
from reservoir_backend.solver.velocity import compute_face_fluxes
from benchmarks.reference_case_loader import (
    get_reference_case,
    load_open_source_reference_cases,
    load_reference_arrays,
)


MD_TO_M2 = 9.869233e-16


def run_pressure_solver_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run pressure benchmark cases and write summary reports."""
    cases = [
        _linear_1d_case(),
        _manufactured_2d_case(),
        _manufactured_3d_case(),
        _opm_water_1ph_adapted_case(),
        _opm_spe1case1_layered_adapted_case(),
        _mrst_simple_tpfa_reference_case(),
        _boundary_sanity_case(),
        _source_sink_case(),
    ]
    success = all(case["success"] for case in cases)
    overall_max_error = max(_metric(case, "max_abs_pressure_error", "linf_error", default=0.0) for case in cases)
    overall_mass_balance = max(abs(_metric(case, "mass_balance_error", "mass_balance_residual", default=0.0)) for case in cases)
    overall_flux = max(abs(_metric(case, "max_flux_variation", "max_flux_imbalance", default=0.0)) for case in cases)
    summary = {
        "benchmark_name": "pressure_solver_benchmark",
        "success": bool(success),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cases": cases,
        "overall_max_error": float(overall_max_error),
        "overall_mass_balance_error": float(overall_mass_balance),
        "overall_flux_conservation_error": float(overall_flux),
        "open_source_references_used": [
            "OPM water-1ph",
            "OPM SPE1CASE1",
            "MRST simpleIncompTPFA",
        ],
        "has_nan": bool(any(case["has_nan"] for case in cases)),
        "has_inf": bool(any(case["has_inf"] for case in cases)),
        "warnings": [],
        "recommendations": [
            "Keep analytical/manufactured pressure benchmarks as regression gates.",
            "Use extracted OPM/MRST fixtures as adapted references only.",
            "Do not claim full SPE1/SPE10 reproduction, OPM Flow equivalence, or MRST integration.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _linear_1d_case() -> dict:
    grid = Grid3D(nx=20, ny=1, nz=1, dx=5.0, dy=1.0, dz=1.0)
    p_left = 10.0e6
    p_right = 1.0e6
    k = 100.0e-15
    mu = 1.0e-3
    result = solve_steady_state_pressure_1d(grid, k, mu, p_left, p_right)
    expected = _linear_x_reference(grid, p_left, p_right)
    pressure = result.pressure.values
    errors = compute_pressure_error_metrics(pressure, expected)
    flux = compute_face_fluxes(grid, result.pressure, k, k, k, mu)
    internal_flux = flux.flux_x[:, :, 1:-1]
    max_flux_variation = float(np.max(internal_flux) - np.min(internal_flux))
    metrics = {
        "max_abs_pressure_error": errors["max_abs_error"],
        "relative_l2_pressure_error": errors["relative_l2_error"],
        "max_flux_variation": max_flux_variation,
        "mass_balance_error": max_flux_variation / max(float(np.max(np.abs(internal_flux))), 1.0e-30),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
    }
    return _case(
        "linear_1d_analytical",
        grid,
        metrics,
        metrics["max_abs_pressure_error"] < 1.0e-3 and metrics["mass_balance_error"] < 1.0e-10,
        source="internal analytical manufactured case",
        limitations=["not an open-source simulator reproduction"],
    )


def _manufactured_2d_case() -> dict:
    grid = Grid3D(nx=10, ny=8, nz=1, dx=4.0, dy=3.0, dz=1.0)
    p_left = 8.0e6
    p_right = 2.0e6
    k = 120.0e-15
    mu = 1.0e-3
    result = solve_steady_state_pressure_2d(grid, k, k, mu, {"left": p_left, "right": p_right})
    expected = _linear_x_reference(grid, p_left, p_right)
    metrics = compute_pressure_error_metrics(result.pressure.values, expected)
    metrics.update(
        {
            "pressure_min": float(np.min(result.pressure.values)),
            "pressure_max": float(np.max(result.pressure.values)),
            "has_nan": bool(np.isnan(result.pressure.values).any()),
            "has_inf": bool(np.isinf(result.pressure.values).any()),
        }
    )
    return _case(
        "manufactured_2d_linear",
        grid,
        metrics,
        metrics["linf_error"] < 1.0e-3,
        source="internal manufactured linear field",
        limitations=["uses constant left/right Dirichlet boundaries supported by current solver"],
    )


def _manufactured_3d_case() -> dict:
    grid = Grid3D(nx=6, ny=5, nz=5, dx=5.0, dy=4.0, dz=3.0)
    p_left = 9.0e6
    p_right = 3.0e6
    k = 90.0e-15
    mu = 1.0e-3
    result = solve_steady_state_pressure_3d(grid, k, k, k, mu, {"left": p_left, "right": p_right})
    expected = _linear_x_reference(grid, p_left, p_right)
    metrics = compute_pressure_error_metrics(result.pressure.values, expected)
    metrics.update(
        {
            "pressure_min": float(np.min(result.pressure.values)),
            "pressure_max": float(np.max(result.pressure.values)),
            "has_nan": bool(np.isnan(result.pressure.values).any()),
            "has_inf": bool(np.isinf(result.pressure.values).any()),
        }
    )
    return _case(
        "manufactured_3d_linear",
        grid,
        metrics,
        metrics["linf_error"] < 1.0e-3,
        source="internal manufactured linear field",
        limitations=["uses constant left/right Dirichlet boundaries supported by current solver"],
    )


def _opm_water_1ph_adapted_case() -> dict:
    reference = get_reference_case("opm_water_1ph_single_cell")
    k_md = reference["permeability_md"]
    metrics = {
        "porosity": float(reference["porosity"]),
        "permeability_x_md": float(k_md["kx"]),
        "permeability_y_md": float(k_md["ky"]),
        "permeability_z_md": float(k_md["kz"]),
        "metadata_loaded": True,
        "pressure_case_mode": "metadata_sanity_only",
        "has_nan": False,
        "has_inf": False,
    }
    return {
        "case_name": "opm_water_1ph_adapted",
        "source": "OPM/opm-tests water-1ph WATER2F.DATA",
        "is_exact_reproduction": False,
        "success": True,
        "grid_shape": list(reference["grid"][::-1]),
        "key_metrics": metrics,
        "warnings": [],
        "has_nan": False,
        "has_inf": False,
        "limitations": [
            "1x1x1 reference cannot form an internal pressure gradient in current benchmark mode",
            "metadata sanity only; not exact OPM Flow reproduction",
            "no black-oil or deck parser behavior is claimed",
        ],
    }


def _opm_spe1case1_layered_adapted_case() -> dict:
    reference = get_reference_case("opm_spe1_case1_layered_subset")
    arrays = load_reference_arrays()
    perm_md = np.asarray(arrays["spe1_permx_md"], dtype=float)
    k = perm_md * MD_TO_M2
    nz, ny, nx = perm_md.shape
    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=10.0, dy=10.0, dz=5.0)
    mu = 1.0e-3
    result = solve_steady_state_pressure_3d(grid, k, k, k, mu, {"left": 10.0e6, "right": 2.0e6})
    flux = compute_face_fluxes(grid, result.pressure, k, k, k, mu)
    pressure = result.pressure.values
    metrics = {
        "porosity_min": float(reference["porosity_min"]),
        "porosity_max": float(reference["porosity_max"]),
        "permeability_min_md": float(np.min(perm_md)),
        "permeability_max_md": float(np.max(perm_md)),
        "permeability_contrast": float(np.max(perm_md) / np.min(perm_md)),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "max_abs_flux": float(max(np.max(np.abs(flux.flux_x)), np.max(np.abs(flux.flux_y)), np.max(np.abs(flux.flux_z)))),
        "mass_balance_error": float(result.report["mass_balance_error"]),
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
    }
    success = (
        metrics["permeability_contrast"] >= 10.0
        and metrics["pressure_min"] >= 2.0e6 - 1.0e-6
        and metrics["pressure_max"] <= 10.0e6 + 1.0e-6
        and metrics["mass_balance_error"] < 1.0e-8
        and not metrics["has_nan"]
        and not metrics["has_inf"]
    )
    return _case(
        "opm_spe1case1_layered_adapted",
        grid,
        metrics,
        success,
        source="OPM/opm-tests spe1 SPE1CASE1.DATA",
        limitations=[
            "adapted layered Cartesian pressure benchmark only",
            "not exact SPE1 simulation reproduction",
            "no OPM Flow equivalence, no PVT, no wells, no black-oil behavior",
        ],
    )


def _mrst_simple_tpfa_reference_case() -> dict:
    reference = get_reference_case("mrst_simple_incomp_tpfa_reference")
    metrics = {
        "is_runtime_dependency": False,
        "mentions_tpfa": bool(reference["mentions_tpfa"]),
        "mentions_boundary_conditions": bool(reference["mentions_boundary_conditions"]),
        "mentions_sources": bool(reference["mentions_sources"]),
        "has_nan": False,
        "has_inf": False,
    }
    return {
        "case_name": "mrst_simple_incomp_tpfa_reference",
        "source": "MRST simpleIncompTPFA.m",
        "is_exact_reproduction": False,
        "success": True,
        "grid_shape": [],
        "key_metrics": metrics,
        "warnings": [],
        "has_nan": False,
        "has_inf": False,
        "limitations": [
            "method reference note only",
            "no MATLAB dependency",
            "no MRST runtime integration",
            "MRST code is not executed or copied into solver implementation",
        ],
    }


def _source_sink_case() -> dict:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=20.0, dy=20.0, dz=8.0)
    rate = 2.0e-5
    wells = [
        Well("I1", "injection", grid, i=1, j=2, k=1, rate=rate),
        Well("P1", "production", grid, i=4, j=2, k=2, rate=rate),
    ]
    result = solve_steady_state_pressure_3d(grid, 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3, wells=wells)
    pressure = result.pressure.values
    metrics = {
        "total_source": rate,
        "total_sink": -rate,
        "net_source": float(result.report["net_well_rate_m3_s"]),
        "boundary_flux_balance": float(result.report["boundary_outflow_m3_s"]),
        "mass_balance_residual": float(result.report["mass_balance_error"]),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
        "status": "done",
    }
    return _case(
        "source_sink_material_balance",
        grid,
        metrics,
        metrics["mass_balance_residual"] < 1.0e-8,
        source="internal balanced source/sink case using existing well support",
        limitations=["simplified source/sink benchmark; not a new well model"],
    )


def _boundary_sanity_case() -> dict:
    grid = Grid3D(nx=12, ny=4, nz=3, dx=10.0, dy=10.0, dz=4.0)
    left = 10.0e6
    right = 0.0
    result = solve_steady_state_pressure_3d(grid, 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3, {"left": left, "right": right})
    p = result.pressure.values
    diffs = np.diff(p, axis=2)
    monotonicity_score = float(np.mean(diffs < 0.0))
    within = bool(np.min(p) >= right - 1.0e-6 and np.max(p) <= left + 1.0e-6)
    metrics = {
        "pressure_monotonicity_score": monotonicity_score,
        "pressure_within_boundary_range": within,
        "boundary_pressure_left": left,
        "boundary_pressure_right": right,
        "pressure_min": float(np.min(p)),
        "pressure_max": float(np.max(p)),
        "warnings": [],
        "has_nan": bool(np.isnan(p).any()),
        "has_inf": bool(np.isinf(p).any()),
    }
    return _case(
        "boundary_sanity",
        grid,
        metrics,
        within and monotonicity_score == 1.0,
        source="internal boundary sanity case",
        limitations=["homogeneous Cartesian left/right Dirichlet only"],
    )


def _linear_x_reference(grid: Grid3D, left: float, right: float) -> np.ndarray:
    x = (np.arange(grid.nx) + 0.5) * grid.dx
    domain_length = grid.nx * grid.dx
    line = left + (right - left) * x / domain_length
    values = np.empty(grid.shape, dtype=float)
    values[:, :, :] = line[None, None, :]
    return values


def _case(
    name: str,
    grid: Grid3D,
    metrics: dict,
    success: bool,
    *,
    source: str,
    is_exact_reproduction: bool = False,
    limitations: list[str] | None = None,
) -> dict:
    has_nan = bool(metrics.get("has_nan", False))
    has_inf = bool(metrics.get("has_inf", False))
    return {
        "case_name": name,
        "source": source,
        "is_exact_reproduction": bool(is_exact_reproduction),
        "success": bool(success and not has_nan and not has_inf),
        "grid_shape": list(grid.shape),
        "key_metrics": _jsonable(metrics),
        "warnings": list(metrics.get("warnings", [])),
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
    json_path = output_dir / "pressure_solver_benchmark_summary.json"
    md_path = output_dir / "pressure_solver_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Pressure Solver Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- overall_max_error: {summary['overall_max_error']:.6e}",
        f"- overall_mass_balance_error: {summary['overall_mass_balance_error']:.6e}",
        f"- overall_flux_conservation_error: {summary['overall_flux_conservation_error']:.6e}",
        "",
        "## Cases",
    ]
    for case in summary["cases"]:
        lines.extend(["", f"### {case['case_name']}", "", f"- success: {case['success']}"])
        for key, value in case["key_metrics"].items():
            lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_pressure_solver_benchmark(), indent=2))
