"""Benchmark hardening suite for simplified three-phase WOG modules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reservoir_backend.solver.three_phase_diagnostics import (
    build_three_phase_diagnostics_report,
    check_three_phase_bounds,
    compute_fractional_flow_closure_metrics,
    compute_phase_flux_statistics,
    compute_three_phase_closure_error,
    compute_three_phase_mobility_metrics,
    compute_three_phase_relperm_metrics,
    compute_three_phase_saturation_statistics,
    compute_three_phase_transport_metrics,
)
from reservoir_backend.solver.three_phase_flux import compute_three_phase_fluxes_3d
from reservoir_backend.solver.three_phase_relperm import (
    compute_oil_saturation,
    corey_three_phase_relative_permeability,
    fractional_flow_three_phase,
)
from reservoir_backend.solver.three_phase_transport import (
    advance_three_phase_saturation_1d,
    advance_three_phase_saturation_3d,
)


DEFAULT_PARAMS = {
    "swi": 0.2,
    "sor": 0.2,
    "sgc": 0.05,
    "krw0": 0.3,
    "kro0": 0.8,
    "krg0": 0.6,
    "nw": 2.0,
    "no": 2.0,
    "ng": 2.0,
    "mu_w": 1.0e-3,
    "mu_o": 5.0e-3,
    "mu_g": 1.0e-5,
}


def run_three_phase_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run three-phase WOG benchmark cases and write JSON/Markdown summaries."""
    cases = [
        _three_phase_saturation_closure_case(),
        _residual_saturation_bounds_case(),
        _three_phase_relperm_endpoint_sanity_case(),
        _phase_mobility_fractional_flow_consistency_case(),
        _phase_flux_finite_shape_consistency_case(),
        _three_phase_1d_transport_boundedness_case(),
        _three_phase_3d_transport_closure_case(),
        _production_summary_consistency_case(),
    ]
    summary = {
        "benchmark_name": "three_phase_benchmark",
        "success": bool(all(case["success"] for case in cases)),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cases": cases,
        "overall_max_closure_error": float(max(_metric(case, "closure_max_abs_error", default=0.0) for case in cases)),
        "overall_num_bound_violations": int(max(_metric(case, "num_bound_violations", default=0.0) for case in cases)),
        "overall_fractional_flow_sum_error": float(max(_metric(case, "fractional_flow_sum_error", default=0.0) for case in cases)),
        "overall_max_phase_flux": float(
            max(
                max(
                    _metric(case, "max_abs_water_flux", default=0.0),
                    _metric(case, "max_abs_oil_flux", default=0.0),
                    _metric(case, "max_abs_gas_flux", default=0.0),
                )
                for case in cases
            )
        ),
        "has_nan": bool(any(case["has_nan"] for case in cases)),
        "has_inf": bool(any(case["has_inf"] for case in cases)),
        "warnings": [],
        "recommendations": [
            "Keep WOG closure, residual bounds, relperm, mobility, fractional-flow, phase-flux, and transport checks as regression gates.",
            "Do not describe this simplified incompressible WOG model as black-oil or commercial simulator equivalent.",
            "Use stable explicit time steps for WOG transport benchmarks; do not infer implicit stability.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _three_phase_saturation_closure_case() -> dict:
    sw, sg = _smooth_state((2, 3, 4))
    so = np.asarray(compute_oil_saturation(sw, sg), dtype=float)
    stats = compute_three_phase_saturation_statistics(sw, so, sg)
    bounds = check_three_phase_bounds(sw, so, sg)
    metrics = {
        **stats,
        "num_bound_violations": bounds["num_bound_violations"],
    }
    return _case(
        "three_phase_saturation_closure",
        sw.shape,
        metrics,
        metrics["closure_max_abs_error"] <= 1.0e-12 and metrics["num_bound_violations"] == 0,
        limitations=["closure uses So = 1 - Sw - Sg in simplified incompressible WOG"],
    )


def _residual_saturation_bounds_case() -> dict:
    sw = np.array([DEFAULT_PARAMS["swi"], 0.3, 0.45, 0.6])
    sg = np.array([DEFAULT_PARAMS["sgc"], 0.12, 0.18, 0.15])
    so = np.asarray(compute_oil_saturation(sw, sg), dtype=float)
    bounds = check_three_phase_bounds(sw, so, sg)
    residual_violations = int(
        np.count_nonzero(
            (sw < DEFAULT_PARAMS["swi"] - 1.0e-12)
            | (sg < DEFAULT_PARAMS["sgc"] - 1.0e-12)
            | (so < DEFAULT_PARAMS["sor"] - 1.0e-12)
        )
    )
    metrics = {
        **compute_three_phase_saturation_statistics(sw, so, sg),
        "num_bound_violations": int(bounds["num_bound_violations"] + residual_violations),
        "residual_violations": residual_violations,
        "swi": DEFAULT_PARAMS["swi"],
        "sor": DEFAULT_PARAMS["sor"],
        "sgc": DEFAULT_PARAMS["sgc"],
    }
    return _case(
        "residual_saturation_bounds",
        list(sw.shape),
        metrics,
        metrics["num_bound_violations"] == 0,
        limitations=["checks current residual-bounded saturation triangle; no black-oil phase appearance logic"],
    )


def _three_phase_relperm_endpoint_sanity_case() -> dict:
    p = DEFAULT_PARAMS
    sw_path = np.linspace(p["swi"], 0.65, 16)
    sg_low = np.full_like(sw_path, p["sgc"])
    krw_path, _, _ = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw_path, sg_low, p))
    sg_path = np.linspace(p["sgc"], 0.45, 16)
    sw_low = np.full_like(sg_path, p["swi"])
    _, _, krg_path = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw_low, sg_path, p))
    sw_oil = np.linspace(p["swi"], 0.55, 16)
    sg_oil = np.full_like(sw_oil, p["sgc"])
    _, kro_path, _ = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw_oil, sg_oil, p))
    metrics = {
        "krw_endpoint": float(krw_path[0]),
        "krg_endpoint": float(krg_path[0]),
        "kro_low_oil_endpoint": float(kro_path[-1]),
        "krw_monotonicity_score": float(np.mean(np.diff(krw_path) >= -1.0e-14)),
        "krg_monotonicity_score": float(np.mean(np.diff(krg_path) >= -1.0e-14)),
        "kro_decreases_as_oil_saturation_decreases": bool(np.all(np.diff(kro_path) <= 1.0e-14)),
        "krw_min": float(np.min(krw_path)),
        "krw_max": float(np.max(krw_path)),
        "kro_min": float(np.min(kro_path)),
        "kro_max": float(np.max(kro_path)),
        "krg_min": float(np.min(krg_path)),
        "krg_max": float(np.max(krg_path)),
        "has_nan": bool(np.isnan(np.concatenate([krw_path, kro_path, krg_path])).any()),
        "has_inf": bool(np.isinf(np.concatenate([krw_path, kro_path, krg_path])).any()),
    }
    success = (
        abs(metrics["krw_endpoint"]) <= 1.0e-15
        and abs(metrics["krg_endpoint"]) <= 1.0e-15
        and metrics["krw_monotonicity_score"] == 1.0
        and metrics["krg_monotonicity_score"] == 1.0
        and metrics["kro_decreases_as_oil_saturation_decreases"]
    )
    return _case(
        "three_phase_relperm_endpoint_sanity",
        [16],
        metrics,
        success,
        limitations=["Corey-style WOG relperm only; Stone I/II and Baker models are not implemented"],
    )


def _phase_mobility_fractional_flow_consistency_case() -> dict:
    sw, sg = _smooth_state((3, 4, 5))
    mobility = compute_three_phase_mobility_metrics(sw, sg, DEFAULT_PARAMS)
    fw, fo, fg = fractional_flow_three_phase(sw, sg, DEFAULT_PARAMS)
    frac = compute_fractional_flow_closure_metrics(fw, fo, fg)
    relperm = compute_three_phase_relperm_metrics(sw, sg, DEFAULT_PARAMS)
    metrics = {**relperm, **mobility, **frac}
    return _case(
        "phase_mobility_fractional_flow_consistency",
        sw.shape,
        metrics,
        metrics["lambda_total_positive"] and metrics["fractional_flow_sum_error"] <= 1.0e-12,
        limitations=["fixed viscosity incompressible WOG; no PVT table or surface-volume conversion"],
    )


def _phase_flux_finite_shape_consistency_case() -> dict:
    sw, sg = _smooth_state((2, 3, 4))
    fx, fy, fz = _face_fluxes(sw.shape, scale=1.0e-5)
    water_x, water_y, water_z, oil_x, oil_y, oil_z, gas_x, gas_y, gas_z, flux_report = compute_three_phase_fluxes_3d(
        fx, fy, fz, sw, sg, DEFAULT_PARAMS
    )
    flux_stats = compute_phase_flux_statistics(
        water_flux=np.concatenate([water_x.ravel(), water_y.ravel(), water_z.ravel()]),
        oil_flux=np.concatenate([oil_x.ravel(), oil_y.ravel(), oil_z.ravel()]),
        gas_flux=np.concatenate([gas_x.ravel(), gas_y.ravel(), gas_z.ravel()]),
    )
    metrics = {
        **flux_stats,
        "phase_flux_closure_error_max": float(flux_report["phase_flux_closure_error_max"]),
        "flux_shape_x": list(flux_report["flux_shape_x"]),
        "flux_shape_y": list(flux_report["flux_shape_y"]),
        "flux_shape_z": list(flux_report["flux_shape_z"]),
    }
    success = (
        metrics["phase_flux_closure_error_max"] <= 1.0e-18
        and metrics["flux_shape_x"] == list(fx.shape)
        and metrics["flux_shape_y"] == list(fy.shape)
        and metrics["flux_shape_z"] == list(fz.shape)
    )
    return _case(
        "phase_flux_finite_shape_consistency",
        sw.shape,
        metrics,
        success,
        limitations=["advective phase flux only; no WOG capillary/gravity phase flux composition"],
    )


def _three_phase_1d_transport_boundedness_case() -> dict:
    nx = 30
    sw0 = np.full(nx, 0.3)
    sg0 = np.full(nx, 0.1)
    flux_x = np.full(nx + 1, 1.0e-5)
    sw1, sg1, so1, report = advance_three_phase_saturation_1d(
        flux_x,
        sw0,
        sg0,
        phi=0.2,
        cell_volume=1.0,
        dt=100.0,
        params=DEFAULT_PARAMS,
        max_cfl=1.0,
        injected_sw=0.65,
        injected_sg=0.05,
    )
    metrics = compute_three_phase_transport_metrics(
        {"sw": sw0, "sg": sg0},
        {"sw": sw1, "sg": sg1, "so": so1},
    )
    metrics.update(
        {
            "max_cfl": float(report["max_cfl"]),
            "water_balance_error": float(report["water_balance_error"]),
            "oil_balance_error": float(report["oil_balance_error"]),
            "gas_balance_error": float(report["gas_balance_error"]),
        }
    )
    success = metrics["num_bound_violations"] == 0 and metrics["closure_max_abs_error"] <= 1.0e-12 and metrics["max_cfl"] <= 1.0
    return _case(
        "three_phase_1d_transport_boundedness",
        [nx],
        metrics,
        success,
        limitations=["one explicit 1D step with stable dt; no black-oil production controls"],
    )


def _three_phase_3d_transport_closure_case() -> dict:
    shape = (3, 3, 4)
    sw0 = np.full(shape, 0.3)
    sg0 = np.full(shape, 0.1)
    fx, fy, fz = _face_fluxes(shape, scale=1.0e-6)
    sw1, sg1, so1, report = advance_three_phase_saturation_3d(
        fx,
        fy,
        fz,
        sw0,
        sg0,
        phi=0.2,
        cell_volume=1.0,
        dt=50.0,
        params=DEFAULT_PARAMS,
        max_cfl=1.0,
        injected_sw=0.65,
        injected_sg=0.05,
    )
    metrics = compute_three_phase_transport_metrics(
        {"sw": sw0, "sg": sg0},
        {"sw": sw1, "sg": sg1, "so": so1},
    )
    metrics.update(
        {
            "max_cfl": float(report["max_cfl"]),
            "water_balance_error": float(report["water_balance_error"]),
            "oil_balance_error": float(report["oil_balance_error"]),
            "gas_balance_error": float(report["gas_balance_error"]),
        }
    )
    success = metrics["num_bound_violations"] == 0 and metrics["closure_max_abs_error"] <= 1.0e-12 and metrics["max_cfl"] <= 1.0
    return _case(
        "three_phase_3d_transport_closure",
        shape,
        metrics,
        success,
        limitations=["small structured-grid 3D explicit WOG transport case"],
    )


def _production_summary_consistency_case() -> dict:
    nx = 20
    dt = 100.0
    sw0 = np.full(nx, 0.3)
    sg0 = np.full(nx, 0.1)
    flux_x = np.full(nx + 1, 1.0e-5)
    sw1, sg1, so1, report = advance_three_phase_saturation_1d(
        flux_x,
        sw0,
        sg0,
        phi=0.2,
        cell_volume=1.0,
        dt=dt,
        params=DEFAULT_PARAMS,
        max_cfl=1.0,
        injected_sw=0.65,
        injected_sg=0.05,
    )
    metrics = build_three_phase_diagnostics_report(sw1, sg1, DEFAULT_PARAMS)
    production_metrics = {
        "water_rate": float(report["water_outflow"] / dt),
        "oil_rate": float(report["oil_outflow"] / dt),
        "gas_rate": float(report["gas_outflow"] / dt),
        "water_cumulative": float(report["water_outflow"]),
        "oil_cumulative": float(report["oil_outflow"]),
        "gas_cumulative": float(report["gas_outflow"]),
        "summary_json_serializable": True,
        "production_summary_type": "pore-volume phase outflow diagnostic, not surface-volume black-oil rate",
    }
    metrics.update(production_metrics)
    success = all(metrics[key] >= 0.0 for key in ("water_rate", "oil_rate", "gas_rate")) and metrics["closure_max_abs_error"] <= 1.0e-12
    return _case(
        "production_summary_consistency",
        [nx],
        metrics,
        success,
        limitations=[
            "production summary is a simplified phase outflow diagnostic",
            "no surface-volume production rate, Bo/Bw/Bg, Rs/Rv, or black-oil material balance",
        ],
    )


def _smooth_state(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    total = int(np.prod(shape))
    sw = np.linspace(0.28, 0.46, total).reshape(shape)
    sg = np.linspace(0.08, 0.18, total).reshape(shape)
    return sw, sg


def _face_fluxes(shape: tuple[int, int, int], scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = shape
    fx = np.full((nz, ny, nx + 1), scale, dtype=float)
    fy = np.zeros((nz, ny + 1, nx), dtype=float)
    fz = np.zeros((nz + 1, ny, nx), dtype=float)
    return fx, fy, fz


def _case(
    name: str,
    grid_shape: tuple[int, ...] | list[int],
    metrics: dict,
    success: bool,
    *,
    source: str = "internal simplified incompressible WOG benchmark",
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


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_reports(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "three_phase_benchmark_summary.json"
    md_path = output_dir / "three_phase_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Three-Phase WOG Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- overall_max_closure_error: {summary['overall_max_closure_error']:.6e}",
        f"- overall_num_bound_violations: {summary['overall_num_bound_violations']}",
        f"- overall_fractional_flow_sum_error: {summary['overall_fractional_flow_sum_error']:.6e}",
        f"- overall_max_phase_flux: {summary['overall_max_phase_flux']:.6e}",
        "",
        "## Cases",
    ]
    for case in summary["cases"]:
        lines.extend(["", f"### {case['case_name']}", "", f"- success: {case['success']}", f"- source: {case['source']}"])
        for key, value in case["key_metrics"].items():
            lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_three_phase_benchmark(), indent=2))
