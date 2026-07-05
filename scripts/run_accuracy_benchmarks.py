"""Run numerical accuracy benchmarks and write summary reports."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import (  # noqa: E402
    buckley_leverett_1d,
    capillary_smoothing,
    combined_transport_stability,
    cross_scale_formula_check,
    gravity_segregation,
    pressure_linear_1d,
    pressure_manufactured_3d,
    three_phase_closure,
)


BENCHMARKS: list[Callable[[], dict]] = [
    pressure_linear_1d.run_benchmark,
    pressure_manufactured_3d.run_benchmark,
    buckley_leverett_1d.run_benchmark,
    capillary_smoothing.run_benchmark,
    gravity_segregation.run_benchmark,
    combined_transport_stability.run_benchmark,
    three_phase_closure.run_benchmark,
    cross_scale_formula_check.run_benchmark,
]


def run_all_benchmarks() -> dict:
    reports: list[dict] = []
    overall_warnings: list[str] = []
    for run in BENCHMARKS:
        try:
            report = run()
        except Exception as exc:  # keep the suite reportable even when one benchmark fails
            report = {
                "benchmark_name": run.__module__.split(".")[-1],
                "success": False,
                "has_nan": False,
                "has_inf": False,
                "warnings": [str(exc)],
                "error": str(exc),
            }
        reports.append(report)
        overall_warnings.extend(f"{report['benchmark_name']}: {warning}" for warning in report.get("warnings", []))

    num_passed = sum(1 for report in reports if report.get("success") is True)
    summary = {
        "success": num_passed == len(reports) and not any(report.get("has_nan") or report.get("has_inf") for report in reports),
        "num_benchmarks": len(reports),
        "num_passed": num_passed,
        "num_failed": len(reports) - num_passed,
        "benchmarks": [_compact_report(report) for report in reports],
        "overall_warnings": overall_warnings,
        "has_nan": any(bool(report.get("has_nan")) for report in reports),
        "has_inf": any(bool(report.get("has_inf")) for report in reports),
        "recommendations": {
            "current C++ recommendation": "No C++ migration is recommended from this MVP accuracy suite; benchmark cases are small and stable.",
            "current numerical risk": "Explicit transport remains the main risk for stronger capillary/gravity cases or finer grids.",
            "which module needs further validation": "Pressure, saturation, capillary, gravity, combined, three-phase, and cross-scale formulas need larger experimental/golden-case validation before production use.",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return summary


def write_reports(summary: dict, output_dir: Path | str = "accuracy_reports") -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "accuracy_benchmark_summary.json"
    md_path = output / "accuracy_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    summary = run_all_benchmarks()
    json_path, md_path = write_reports(summary)
    print(f"accuracy summary: {json_path}")
    print(f"accuracy markdown: {md_path}")
    print(f"success={summary['success']}")
    return 0 if summary["success"] else 1


def _compact_report(report: dict) -> dict:
    excluded = {"warnings", "benchmark_name", "success", "has_nan", "has_inf"}
    return {
        "benchmark_name": report["benchmark_name"],
        "success": bool(report.get("success")),
        "key_metrics": {key: value for key, value in report.items() if key not in excluded},
        "warnings": list(report.get("warnings", [])),
        "has_nan": bool(report.get("has_nan")),
        "has_inf": bool(report.get("has_inf")),
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Accuracy Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_benchmarks: {summary['num_benchmarks']}",
        f"- num_passed: {summary['num_passed']}",
        f"- num_failed: {summary['num_failed']}",
        f"- has_nan: {summary['has_nan']}",
        f"- has_inf: {summary['has_inf']}",
        "",
        "## Benchmarks",
        "",
    ]
    for report in summary["benchmarks"]:
        lines.append(f"### {report['benchmark_name']}")
        lines.append(f"- success: {report['success']}")
        lines.append(f"- has_nan: {report['has_nan']}")
        lines.append(f"- has_inf: {report['has_inf']}")
        for key, value in report["key_metrics"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Recommendations", ""])
    for key, value in summary["recommendations"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
