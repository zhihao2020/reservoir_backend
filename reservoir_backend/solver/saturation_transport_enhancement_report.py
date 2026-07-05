"""Saturation transport enhancement report for CFL/TVD/fallback diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.transport_diagnostics import build_transport_diagnostics, compute_front_sharpness
from reservoir_backend.solver.tvd_transport import (
    adapt_timestep,
    advance_saturation_1d_enhanced,
    compute_cfl,
    suggest_stable_timestep,
)


LIMITATIONS = [
    "Upwind baseline is preserved.",
    "TVD/MUSCL is optional and currently limited to 1D benchmark scenarios.",
    "Implicit saturation transport is deferred and not implemented as a full solver.",
    "No fully implicit reservoir simulator implemented.",
    "No black-oil transport implemented.",
    "No PVT table or phase behavior implemented.",
    "No commercial simulator equivalence.",
    "No history matching implemented.",
    "No front-end integration implemented.",
    "No UDP implementation.",
]


def run_saturation_transport_enhancement_report(output_dir: str | Path = "accuracy_reports") -> dict[str, Any]:
    """Run enhancement diagnostics and write JSON/Markdown reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = [
        _cfl_adaptive_case(),
        _upwind_tvd_comparison_case(),
        _fallback_case(),
    ]
    summary = {
        "report_name": "saturation_transport_enhancement",
        "success": bool(all(case["success"] for case in cases)),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cfl_cases": [case for case in cases if "cfl" in case["case_name"]],
        "upwind_tvd_comparison": [case for case in cases if "comparison" in case["case_name"]],
        "fallback_cases": [case for case in cases if "fallback" in case["case_name"]],
        "cases": cases,
        "limitations": LIMITATIONS,
        "warnings": [warning for case in cases for warning in case.get("warnings", [])],
    }
    json_path = output_path / "saturation_transport_enhancement_summary.json"
    markdown_path = output_path / "saturation_transport_enhancement_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _cfl_adaptive_case() -> dict[str, Any]:
    grid = _grid()
    phi = 0.2
    flux = _flux(grid, 2.0e-4)
    dt = 2000.0
    _, cfl_report = compute_cfl(
        grid,
        phi,
        flux,
        np.zeros((1, 2, grid.nx), dtype=float),
        np.zeros((2, 1, grid.nx), dtype=float),
        dt,
    )
    suggestion = suggest_stable_timestep(
        grid,
        phi,
        flux,
        np.zeros((1, 2, grid.nx), dtype=float),
        np.zeros((2, 1, grid.nx), dtype=float),
        dt,
        target_cfl=0.8,
    )
    adapted = adapt_timestep(
        grid,
        phi,
        flux,
        np.zeros((1, 2, grid.nx), dtype=float),
        np.zeros((2, 1, grid.nx), dtype=float),
        dt,
        target_cfl=0.8,
    )
    return {
        "case_name": "cfl_adaptive_timestep",
        "success": bool(adapted["dt_adapted"] < dt and adapted["num_limited_cells"] > 0),
        "grid_shape": list(grid.shape),
        "key_metrics": {
            "dt_original": dt,
            "dt_adapted": adapted["dt_adapted"],
            "max_cfl": cfl_report["max_cfl"],
            "target_cfl": adapted["target_cfl"],
            "num_limited_cells": adapted["num_limited_cells"],
        },
        "cfl_report": cfl_report,
        "suggestion": suggestion,
        "warnings": adapted["warnings"],
        "limitations": ["Adaptive timestep is reported; baseline solver behavior is unchanged."],
    }


def _upwind_tvd_comparison_case() -> dict[str, Any]:
    grid = _grid()
    sw0 = np.full(grid.shape, 0.2)
    sw0[0, 0, :5] = 0.65
    phi = 0.25
    flux = _flux(grid, 2.0e-5)
    params = _params()
    upwind = advance_saturation_1d_enhanced(grid, sw0, phi, flux, 500.0, params, max_cfl=0.8, method="upwind")
    tvd = advance_saturation_1d_enhanced(grid, sw0, phi, flux, 500.0, params, max_cfl=0.8, method="tvd", limiter="minmod")
    upwind_diag = build_transport_diagnostics(sw0, upwind.sw.values, lower=0.2, upper=0.8, dx=grid.dx)
    tvd_diag = build_transport_diagnostics(sw0, tvd.sw.values, lower=0.2, upper=0.8, dx=grid.dx)
    return {
        "case_name": "upwind_tvd_front_sharpness_comparison",
        "success": bool(tvd_diag["boundedness_passed"] and not tvd_diag["has_nan"] and not tvd_diag["has_inf"]),
        "grid_shape": list(grid.shape),
        "key_metrics": {
            "upwind_front_sharpness": upwind_diag["front_sharpness"],
            "tvd_front_sharpness": tvd_diag["front_sharpness"],
            "front_sharpness_delta": float(tvd_diag["front_sharpness"] - upwind_diag["front_sharpness"]),
            "upwind_total_variation": upwind_diag["total_variation"],
            "tvd_total_variation": tvd_diag["total_variation"],
            "tvd_material_balance_error": tvd.report["material_balance_error"],
            "tvd_max_cfl": tvd.report["max_cfl"],
        },
        "upwind_report": upwind.report,
        "tvd_report": tvd.report,
        "warnings": list(tvd.report.get("warnings", [])),
        "limitations": ["TVD/MUSCL comparison is a 1D benchmark diagnostic, not a default solver replacement."],
    }


def _fallback_case() -> dict[str, Any]:
    grid = _grid()
    sw0 = np.linspace(0.2, 0.8, grid.nx).reshape(grid.shape)
    result = advance_saturation_1d_enhanced(
        grid,
        sw0,
        0.25,
        _flux(grid, 1.0e-5),
        200.0,
        _params(),
        max_cfl=0.8,
        method="implicit",
        fallback="upwind",
    )
    return {
        "case_name": "implicit_deferred_fallback",
        "success": bool(result.report["fallback_used"] and result.report["implicit_deferred"]),
        "grid_shape": list(grid.shape),
        "key_metrics": {
            "method_requested": result.report["method_requested"],
            "method_used": result.report["method_used"],
            "fallback_used": result.report["fallback_used"],
            "implicit_deferred": result.report["implicit_deferred"],
            "front_sharpness": compute_front_sharpness(result.sw.values, dx=grid.dx),
        },
        "warnings": result.report["warnings"],
        "limitations": ["Implicit saturation transport is deferred; this case validates warning and fallback behavior."],
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Saturation Transport Enhancement Summary",
        "",
        f"- success: `{summary['success']}`",
        f"- num_cases: `{summary['num_cases']}`",
        f"- num_passed: `{summary['num_passed']}`",
        "",
        "## Cases",
        "",
        "| Case | Success | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for case in summary["cases"]:
        lines.append(f"| {case['case_name']} | {case['success']} | `{json.dumps(case['key_metrics'], sort_keys=True)}` |")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def _grid() -> Grid3D:
    return Grid3D(nx=20, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)


def _flux(grid: Grid3D, value: float) -> np.ndarray:
    flux = np.zeros((1, 1, grid.nx + 1), dtype=float)
    flux[0, 0, :] = float(value)
    return flux


def _params() -> dict[str, float]:
    return {
        "swi": 0.2,
        "sor": 0.2,
        "krw0": 1.0,
        "kro0": 1.0,
        "nw": 2.0,
        "no": 2.0,
        "mu_w": 1.0e-3,
        "mu_o": 5.0e-3,
    }


def main() -> None:
    summary = run_saturation_transport_enhancement_report()
    print(json.dumps({"success": summary["success"], "num_cases": summary["num_cases"]}, sort_keys=True))


if __name__ == "__main__":
    main()
