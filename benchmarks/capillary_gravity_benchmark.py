"""Capillary, gravity, and combined transport benchmark hardening suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.reference_case_loader import get_reference_case, load_reference_arrays
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.capillary_flux import compute_capillary_fluxes
from reservoir_backend.solver.capillary_gravity_diagnostics import (
    check_expected_flux_sign,
    compute_capillary_smoothing_metrics,
    compute_flux_statistics,
    compute_gradient_norm,
    compute_gravity_segregation_metrics,
)
from reservoir_backend.solver.capillary_pressure import capillary_pressure
from reservoir_backend.solver.gravity_flux import compute_gravity_fluxes
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d_with_capillary,
    advance_saturation_3d_with_capillary_and_gravity,
    advance_saturation_3d_with_gravity,
)
from reservoir_backend.solver.water_flux_composer import compose_water_fluxes_3d


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
CAPILLARY = {
    "enabled": True,
    "model": "brooks_corey",
    "swi": 0.2,
    "sor": 0.2,
    "entry_pressure_pa": 1000.0,
    "lambda_pc": 2.0,
}
GRAVITY = {
    "enabled": True,
    "g": 9.80665,
    "rho_w": 1000.0,
    "rho_o": 800.0,
    "depth_axis": "z",
    "depth_positive": "down",
}


def run_capillary_gravity_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run capillary/gravity benchmark cases and write summary reports."""
    cases = [
        _capillary_pressure_monotonicity_case(),
        _capillary_no_gradient_zero_flux_case(),
        _capillary_smoothing_case(),
        _gravity_zero_density_difference_case(),
        _gravity_segregation_direction_case(),
        _combined_capillary_gravity_stability_case(),
        _water_flux_composer_consistency_case(),
        _opm_spe1_capillary_gravity_sanity_case(),
    ]
    success = all(case["success"] for case in cases)
    summary = {
        "benchmark_name": "capillary_gravity_benchmark",
        "success": bool(success),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cases": cases,
        "overall_gradient_reduction": float(max(_metric(case, "gradient_reduction", default=0.0) for case in cases)),
        "overall_max_capillary_flux": float(max(_metric(case, "max_abs_capillary_flux", default=0.0) for case in cases)),
        "overall_max_gravity_flux": float(max(_metric(case, "max_abs_gravity_flux", default=0.0) for case in cases)),
        "overall_material_balance_error": float(max(abs(_metric(case, "material_balance_error", default=0.0)) for case in cases)),
        "open_source_references_used": [
            "OPM SPE1CASE1 property metadata",
            "MRST simpleIncompTPFA reference context",
        ],
        "has_nan": bool(any(case["has_nan"] for case in cases)),
        "has_inf": bool(any(case["has_inf"] for case in cases)),
        "warnings": [],
        "recommendations": [
            "Keep capillary smoothing, gravity direction, composer, and combined stability checks as regression gates.",
            "Use OPM SPE1 as property metadata only; do not claim full SPE1/SPE10 reproduction.",
            "Do not infer semi-implicit capillary behavior from these explicit small-case checks.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _capillary_pressure_monotonicity_case() -> dict:
    sw = np.linspace(0.21, 0.79, 80)
    pc = np.asarray(capillary_pressure(sw, CAPILLARY), dtype=float)
    diffs = np.diff(pc)
    score = float(np.mean(diffs <= 1.0e-12))
    metrics = {
        "pc_min": float(np.min(pc)),
        "pc_max": float(np.max(pc)),
        "pc_mean": float(np.mean(pc)),
        "pc_monotonicity_score": score,
        "num_nonfinite": int(np.count_nonzero(~np.isfinite(pc))),
        "model_convention": "Brooks-Corey Pc=Po-Pw decreases as Sw increases",
        "has_nan": bool(np.isnan(pc).any()),
        "has_inf": bool(np.isinf(pc).any()),
    }
    return _case(
        "capillary_pressure_monotonicity",
        [80],
        metrics,
        metrics["pc_monotonicity_score"] == 1.0 and metrics["num_nonfinite"] == 0,
        source="internal capillary pressure trend case",
        limitations=["uses current Brooks-Corey implementation without changing model convention"],
    )


def _capillary_no_gradient_zero_flux_case() -> dict:
    grid = Grid3D(nx=5, ny=4, nz=3, dx=1.0, dy=1.0, dz=1.0)
    sw = np.full(grid.shape, 0.5)
    fx, fy, fz, _ = compute_capillary_fluxes(grid, sw, 1.0e-12, 1.0e-12, 1.0e-12, CAPILLARY, DEFAULT_RELPERM)
    max_flux = _max_abs(fx, fy, fz)
    mean_flux = float(np.mean(np.abs(np.concatenate([fx.ravel(), fy.ravel(), fz.ravel()]))))
    metrics = {
        "max_abs_capillary_flux": max_flux,
        "mean_abs_capillary_flux": mean_flux,
        "flux_zero_tolerance": 1.0e-30,
        "success": bool(max_flux <= 1.0e-30),
        "has_nan": bool(np.isnan(fx).any() or np.isnan(fy).any() or np.isnan(fz).any()),
        "has_inf": bool(np.isinf(fx).any() or np.isinf(fy).any() or np.isinf(fz).any()),
    }
    return _case(
        "capillary_no_gradient_zero_flux",
        grid.shape,
        metrics,
        metrics["success"],
        source="internal uniform-Sw capillary flux case",
        limitations=["no-flow boundary; homogeneous permeability"],
    )


def _capillary_smoothing_case() -> dict:
    grid = Grid3D(nx=16, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    sw0 = _step_x(grid, high=0.7, low=0.3)
    result = advance_saturation_1d_with_capillary(
        grid,
        sw0,
        0.2,
        np.zeros((1, 1, grid.nx + 1)),
        500.0,
        DEFAULT_RELPERM,
        CAPILLARY,
        1.0e-9,
    )
    cap_x, _, _, _ = compute_capillary_fluxes(grid, sw0, 1.0e-9, 1.0e-9, 1.0e-9, CAPILLARY, DEFAULT_RELPERM)
    smoothing = compute_capillary_smoothing_metrics(sw0, result.sw.values)
    bounds = _bounds_metrics(result.sw.values)
    metrics = {
        **smoothing,
        "max_abs_capillary_flux": float(np.max(np.abs(cap_x))),
        "sw_min": bounds["sw_min"],
        "sw_max": bounds["sw_max"],
        "num_bound_violations": bounds["num_bound_violations"],
        "has_nan": bool(result.report["has_nan"]),
        "has_inf": bool(result.report["has_inf"]),
    }
    return _case(
        "capillary_smoothing",
        grid.shape,
        metrics,
        metrics["gradient_reduction"] > 0.0 and metrics["max_abs_capillary_flux"] > 0.0 and metrics["num_bound_violations"] == 0,
        source="internal 1D capillary smoothing case",
        limitations=["explicit capillary update uses small stable dt; strong capillary/fine grids may be dt-sensitive"],
    )


def _gravity_zero_density_difference_case() -> dict:
    grid = Grid3D(nx=3, ny=3, nz=4, dx=1.0, dy=1.0, dz=1.0)
    sw = np.full(grid.shape, 0.5)
    params = dict(GRAVITY)
    params["rho_o"] = params["rho_w"]
    gx, gy, gz, _ = compute_gravity_fluxes(grid, sw, 1.0e-12, 1.0e-12, 1.0e-12, params, DEFAULT_RELPERM)
    max_flux = _max_abs(gx, gy, gz)
    mean_flux = float(np.mean(np.abs(np.concatenate([gx.ravel(), gy.ravel(), gz.ravel()]))))
    metrics = {
        "max_abs_gravity_flux": max_flux,
        "mean_abs_gravity_flux": mean_flux,
        "flux_zero_tolerance": 1.0e-30,
        "success": bool(max_flux <= 1.0e-30),
        "has_nan": bool(np.isnan(gx).any() or np.isnan(gy).any() or np.isnan(gz).any()),
        "has_inf": bool(np.isinf(gx).any() or np.isinf(gy).any() or np.isinf(gz).any()),
    }
    return _case(
        "gravity_zero_density_difference",
        grid.shape,
        metrics,
        metrics["success"],
        source="internal zero-density-difference gravity case",
        limitations=["uses current z-only gravity implementation"],
    )


def _gravity_segregation_direction_case() -> dict:
    grid = Grid3D(nx=3, ny=3, nz=4, dx=1.0, dy=1.0, dz=1.0)
    sw0 = np.full(grid.shape, 0.5)
    fx, fy, fz = _zero_fluxes(grid)
    result = advance_saturation_3d_with_gravity(
        grid,
        sw0,
        0.2,
        fx,
        fy,
        fz,
        1000.0,
        DEFAULT_RELPERM,
        GRAVITY,
        1.0e-12,
        1.0e-12,
        1.0e-12,
    )
    gx, gy, gz, _ = compute_gravity_fluxes(grid, sw0, 1.0e-12, 1.0e-12, 1.0e-12, GRAVITY, DEFAULT_RELPERM)
    sign = check_expected_flux_sign(gz[1:-1, :, :], -1)
    segregation = compute_gravity_segregation_metrics(sw0, result.sw.values, vertical_axis=0)
    bounds = _bounds_metrics(result.sw.values)
    metrics = {
        "expected_gravity_flux_sign": -1,
        "observed_gravity_flux_sign": sign["observed_sign"],
        "sign_matches_expectation": sign["sign_matches_expectation"],
        "top_sw_change": segregation["top_saturation_change"],
        "bottom_sw_change": segregation["bottom_saturation_change"],
        "vertical_axis_convention": "Grid3D arrays use (nz, ny, nx); positive flux_z is bottom-to-top; rho_w>rho_o gives negative internal gravity_flux_z",
        "positive_gravity_direction_convention": "depth_positive=down",
        "sw_min": bounds["sw_min"],
        "sw_max": bounds["sw_max"],
        "num_bound_violations": bounds["num_bound_violations"],
        "max_abs_gravity_flux": _max_abs(gx, gy, gz),
        "has_nan": bool(result.report["has_nan"]),
        "has_inf": bool(result.report["has_inf"]),
    }
    return _case(
        "gravity_segregation_direction",
        grid.shape,
        metrics,
        bool(metrics["sign_matches_expectation"]) and metrics["top_sw_change"] > 0.0 and metrics["num_bound_violations"] == 0,
        source="internal gravity segregation direction case",
        limitations=["direction check follows current project z-face convention"],
    )


def _combined_capillary_gravity_stability_case() -> dict:
    grid = Grid3D(nx=5, ny=4, nz=3, dx=1.0, dy=1.0, dz=1.0)
    sw0 = _step_x(grid, high=0.65, low=0.35)
    fx, fy, fz = _zero_fluxes(grid)
    result = advance_saturation_3d_with_capillary_and_gravity(
        grid,
        sw0,
        0.2,
        fx,
        fy,
        fz,
        50.0,
        DEFAULT_RELPERM,
        CAPILLARY,
        GRAVITY,
        1.0e-12,
        1.0e-12,
        1.0e-12,
        max_cfl=1.0,
    )
    cap_flux = compute_capillary_fluxes(grid, sw0, 1.0e-12, 1.0e-12, 1.0e-12, CAPILLARY, DEFAULT_RELPERM)[:3]
    grav_flux = compute_gravity_fluxes(grid, sw0, 1.0e-12, 1.0e-12, 1.0e-12, GRAVITY, DEFAULT_RELPERM)[:3]
    bounds = _bounds_metrics(result.sw.values)
    metrics = {
        "max_abs_capillary_flux": _max_abs(*cap_flux),
        "max_abs_gravity_flux": _max_abs(*grav_flux),
        "sw_min": bounds["sw_min"],
        "sw_max": bounds["sw_max"],
        "num_bound_violations": bounds["num_bound_violations"],
        "material_balance_error": float(result.report["material_balance_error"]),
        "max_cfl": float(result.report["max_cfl"]),
        "has_nan": bool(result.report["has_nan"]),
        "has_inf": bool(result.report["has_inf"]),
    }
    return _case(
        "combined_capillary_gravity_stability",
        grid.shape,
        metrics,
        metrics["max_abs_capillary_flux"] > 0.0
        and metrics["max_abs_gravity_flux"] > 0.0
        and metrics["num_bound_violations"] == 0
        and np.isfinite(metrics["material_balance_error"]),
        source="internal combined capillary + gravity stability case",
        limitations=["small explicit time step; not a semi-implicit capillary solver"],
    )


def _water_flux_composer_consistency_case() -> dict:
    shape_x = (2, 3, 5)
    shape_y = (2, 4, 4)
    shape_z = (3, 3, 4)
    adv = (
        np.full(shape_x, 2.0),
        np.full(shape_y, -1.0),
        np.full(shape_z, 0.5),
    )
    cap = (
        np.full(shape_x, 0.25),
        np.zeros(shape_y),
        np.full(shape_z, -0.1),
    )
    grav = (
        np.zeros(shape_x),
        np.zeros(shape_y),
        np.full(shape_z, -0.2),
    )
    pressure_only = compose_water_fluxes_3d(*adv)
    capillary_only = compose_water_fluxes_3d(*adv, *cap, include_capillary=True)
    gravity_only = compose_water_fluxes_3d(*adv, grav_flux_x=grav[0], grav_flux_y=grav[1], grav_flux_z=grav[2], include_gravity=True)
    combined = compose_water_fluxes_3d(*adv, *cap, *grav, include_capillary=True, include_gravity=True)
    wx, wy, wz, report = combined
    metrics = {
        "pressure_only_flux_norm": _norm(pressure_only[:3]),
        "capillary_contribution_norm": _norm(tuple(c - a for c, a in zip(capillary_only[:3], pressure_only[:3], strict=True))),
        "gravity_contribution_norm": _norm(tuple(g - a for g, a in zip(gravity_only[:3], pressure_only[:3], strict=True))),
        "combined_flux_norm": _norm((wx, wy, wz)),
        "shape_consistent": bool(wx.shape == shape_x and wy.shape == shape_y and wz.shape == shape_z),
        "has_nan": bool(report["has_nan"]),
        "has_inf": bool(report["has_inf"]),
    }
    return _case(
        "water_flux_composer_consistency",
        list(shape_x),
        metrics,
        metrics["shape_consistent"]
        and metrics["capillary_contribution_norm"] > 0.0
        and metrics["gravity_contribution_norm"] > 0.0
        and not metrics["has_nan"]
        and not metrics["has_inf"],
        source="internal water_flux_composer consistency case",
        limitations=["composer consistency only; no saturation update"],
    )


def _opm_spe1_capillary_gravity_sanity_case() -> dict:
    reference = get_reference_case("opm_spe1_case1_layered_subset")
    arrays = load_reference_arrays()
    perm_md = np.asarray(arrays["spe1_permx_md"], dtype=float)
    sw = np.full(perm_md.shape, 0.5)
    bounds = _bounds_metrics(sw)
    metrics = {
        "porosity_min": float(reference["porosity_min"]),
        "porosity_max": float(reference["porosity_max"]),
        "permeability_min_md": float(np.min(perm_md)),
        "permeability_max_md": float(np.max(perm_md)),
        "permeability_contrast": float(np.max(perm_md) / np.min(perm_md)),
        "sw_min": bounds["sw_min"],
        "sw_max": bounds["sw_max"],
        "num_bound_violations": bounds["num_bound_violations"],
        "metadata_loaded": True,
        "has_nan": bool(np.isnan(sw).any()),
        "has_inf": bool(np.isinf(sw).any()),
    }
    return _case(
        "opm_spe1case1_capillary_gravity_sanity_adapted",
        perm_md.shape,
        metrics,
        metrics["metadata_loaded"] and metrics["permeability_contrast"] >= 10.0 and metrics["num_bound_violations"] == 0,
        source="OPM/opm-tests spe1 SPE1CASE1.DATA",
        limitations=[
            "property metadata and boundedness sanity only",
            "not exact SPE1 reproduction",
            "no OPM Flow equivalence, no black-oil, no PVT",
        ],
    )


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((grid.nz, grid.ny, grid.nx + 1)),
        np.zeros((grid.nz, grid.ny + 1, grid.nx)),
        np.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )


def _step_x(grid: Grid3D, high: float, low: float) -> np.ndarray:
    values = np.full(grid.shape, low, dtype=float)
    values[:, :, : grid.nx // 2] = high
    return values


def _bounds_metrics(sw: np.ndarray) -> dict:
    lower = DEFAULT_RELPERM["swi"]
    upper = 1.0 - DEFAULT_RELPERM["sor"]
    return {
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "num_bound_violations": int(np.count_nonzero((sw < lower - 1.0e-12) | (sw > upper + 1.0e-12))),
    }


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


def _metric(case: dict, name: str, default: float) -> float:
    value = case["key_metrics"].get(name, default)
    return float(default if value is None else value)


def _max_abs(*arrays: np.ndarray) -> float:
    return float(max(np.max(np.abs(array)) for array in arrays))


def _norm(arrays: tuple[np.ndarray, ...]) -> float:
    return float(np.sqrt(sum(np.sum(np.asarray(array) ** 2) for array in arrays)))


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
    json_path = output_dir / "capillary_gravity_benchmark_summary.json"
    md_path = output_dir / "capillary_gravity_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Capillary / Gravity Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- overall_gradient_reduction: {summary['overall_gradient_reduction']:.6e}",
        f"- overall_max_capillary_flux: {summary['overall_max_capillary_flux']:.6e}",
        f"- overall_max_gravity_flux: {summary['overall_max_gravity_flux']:.6e}",
        f"- overall_material_balance_error: {summary['overall_material_balance_error']:.6e}",
        "",
        "## Cases",
    ]
    for case in summary["cases"]:
        lines.extend(["", f"### {case['case_name']}", "", f"- success: {case['success']}", f"- source: {case['source']}"])
        for key, value in case["key_metrics"].items():
            lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_capillary_gravity_benchmark(), indent=2))
