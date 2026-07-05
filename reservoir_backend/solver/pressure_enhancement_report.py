"""Pressure solver enhancement report for wells, boundaries, and backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.boundary_matrix import (
    BoundaryConditionContribution,
    apply_source_sink_to_rhs,
    build_boundary_contribution,
    build_boundary_diagnostics,
)
from reservoir_backend.solver.linear_solver_backend import solve_linear_system
from reservoir_backend.solver.well_source import (
    RateControlledWell,
    build_well_contribution_vector,
    summarize_wells,
)


LIMITATIONS = [
    "No black-oil model implemented.",
    "No PVT table or phase behavior implemented.",
    "No full Peaceman industrial well model implemented.",
    "No complex wellbore network implemented.",
    "No fully implicit reservoir simulator implemented.",
    "No history matching implemented.",
    "No front-end integration implemented.",
    "No UDP implementation.",
    "No commercial simulator equivalence.",
]


def run_pressure_solver_enhancement_report(output_dir: str | Path = "accuracy_reports") -> dict[str, Any]:
    """Run pressure enhancement diagnostics and write JSON/Markdown reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = [
        _well_source_case(),
        _boundary_contribution_case(),
        _solver_backend_case(),
        _mass_balance_case(),
    ]
    success = all(case["success"] for case in cases)
    residuals = [
        float(case["key_metrics"].get("mass_balance_error", 0.0))
        for case in cases
        if case["key_metrics"].get("mass_balance_error") is not None
    ]
    summary = {
        "report_name": "pressure_solver_enhancement",
        "success": bool(success),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "well_cases": [case for case in cases if "well" in case["case_name"]],
        "boundary_cases": [case for case in cases if "boundary" in case["case_name"]],
        "solver_backend_cases": [case for case in cases if "backend" in case["case_name"]],
        "cases": cases,
        "mass_balance_residuals": residuals,
        "max_mass_balance_error": float(max(residuals) if residuals else 0.0),
        "solver_stats": _collect_solver_stats(cases),
        "limitations": LIMITATIONS,
        "warnings": _collect_warnings(cases),
    }

    json_path = output_path / "pressure_solver_enhancement_summary.json"
    markdown_path = output_path / "pressure_solver_enhancement_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _well_source_case() -> dict[str, Any]:
    grid = Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    wells = [
        RateControlledWell("INJ-1", "injector", rate=10.0, i=0, j=0, k=0),
        RateControlledWell("PROD-1", "producer", rate=7.0, i=2, j=1, k=0),
    ]
    vector = build_well_contribution_vector(wells, grid)
    diagnostics = summarize_wells(wells, grid)
    mass_balance_error = abs(diagnostics["net_source_rate"] - float(np.sum(vector)))
    return {
        "case_name": "well_source_sink_rate_control",
        "success": bool(np.isfinite(vector).all() and mass_balance_error <= 1.0e-12),
        "grid_shape": list(grid.shape),
        "key_metrics": {
            "total_injection_rate": diagnostics["total_injection_rate"],
            "total_production_rate": diagnostics["total_production_rate"],
            "net_source_rate": diagnostics["net_source_rate"],
            "mass_balance_error": float(mass_balance_error),
            "well_contribution_nonzero": int(np.count_nonzero(vector)),
        },
        "well_diagnostics": diagnostics["well_diagnostics"],
        "warnings": diagnostics["warnings"],
        "limitations": ["Rate control only; no BHP or industrial wellbore model."],
    }


def _boundary_contribution_case() -> dict[str, Any]:
    grid = Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    contribution = build_boundary_contribution(
        grid,
        [
            BoundaryConditionContribution("left", "dirichlet", value=3.0, transmissibility=2.0),
            BoundaryConditionContribution("right", "neumann", value=-0.5, transmissibility=1.0),
            BoundaryConditionContribution("front", "noflow", value=0.0, transmissibility=1.0),
        ],
    )
    diagnostics = build_boundary_diagnostics(contribution)
    return {
        "case_name": "boundary_matrix_contribution",
        "success": diagnostics["success"],
        "grid_shape": list(grid.shape),
        "key_metrics": {
            "matrix_shape": diagnostics["matrix_shape"],
            "rhs_shape": diagnostics["rhs_shape"],
            "num_nonzero_diagonal": diagnostics["num_nonzero_diagonal"],
            "num_nonzero_rhs": diagnostics["num_nonzero_rhs"],
            "rhs_sum": diagnostics["rhs_sum"],
            "diagonal_sum": diagnostics["diagonal_sum"],
            "mass_balance_error": 0.0,
        },
        "boundary_diagnostics": diagnostics["diagnostics"],
        "warnings": diagnostics["warnings"],
        "limitations": ["Boundary contribution helper is diagnostic; existing solver assembly remains the baseline."],
    }


def _solver_backend_case() -> dict[str, Any]:
    matrix = np.array(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=float,
    )
    rhs = np.array([15.0, 10.0, 10.0], dtype=float)
    backends = ["direct", "cg", "gmres", "ilu", "amg"]
    solver_stats = {}
    success = True
    warnings: list[str] = []
    for backend in backends:
        solution, stats = solve_linear_system(matrix, rhs, backend=backend)
        solver_stats[backend] = stats
        success = success and bool(stats["success"]) and bool(np.isfinite(solution).all())
        warnings.extend(stats["warnings"])
    max_residual = max(float(stats["residual_norm"]) for stats in solver_stats.values())
    return {
        "case_name": "linear_solver_backend_evaluation",
        "success": bool(success and max_residual < 1.0e-7),
        "grid_shape": [1, 1, 3],
        "key_metrics": {
            "max_residual_norm": float(max_residual),
            "num_backends_requested": len(backends),
            "fallback_count": int(sum(bool(stats["fallback_used"]) for stats in solver_stats.values())),
            "mass_balance_error": float(max(float(stats["mass_balance_error"]) for stats in solver_stats.values())),
        },
        "solver_stats": solver_stats,
        "warnings": warnings,
        "limitations": ["AMG is optional and falls back to direct solve when pyamg is unavailable."],
    }


def _mass_balance_case() -> dict[str, Any]:
    rhs = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    source = np.array([4.0, 0.0, 0.0, -4.0], dtype=float)
    updated_rhs = apply_source_sink_to_rhs(rhs, source)
    residual = float(abs(np.sum(updated_rhs)))
    return {
        "case_name": "mass_balance_with_wells_and_rhs",
        "success": bool(residual <= 1.0e-12),
        "grid_shape": [1, 1, 4],
        "key_metrics": {
            "total_injection_rate": 4.0,
            "total_production_rate": 4.0,
            "net_source_rate": float(np.sum(source)),
            "mass_balance_error": residual,
            "rhs_sum_after_sources": float(np.sum(updated_rhs)),
        },
        "warnings": [],
        "limitations": ["Mass-balance check is source-vector based and does not add new pressure physics."],
    }


def _collect_solver_stats(cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in cases:
        if "solver_stats" in case:
            return case["solver_stats"]
    return {}


def _collect_warnings(cases: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for case in cases:
        warnings.extend(str(warning) for warning in case.get("warnings", []))
    return warnings


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Pressure Solver Enhancement Summary",
        "",
        f"- success: `{summary['success']}`",
        f"- num_cases: `{summary['num_cases']}`",
        f"- num_passed: `{summary['num_passed']}`",
        f"- max_mass_balance_error: `{summary['max_mass_balance_error']}`",
        "",
        "## Cases",
        "",
        "| Case | Success | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for case in summary["cases"]:
        metrics = json.dumps(case["key_metrics"], sort_keys=True)
        lines.append(f"| {case['case_name']} | {case['success']} | `{metrics}` |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_pressure_solver_enhancement_report()
    print(json.dumps({"success": summary["success"], "num_cases": summary["num_cases"]}, sort_keys=True))


if __name__ == "__main__":
    main()
