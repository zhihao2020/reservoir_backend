"""Performance baseline report runner for TASK-019."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from reservoir_backend.performance.profiler import (
    check_numerical_equivalence,
    run_stage_profiles,
    summarize_profiles,
)


LIMITATIONS = [
    "Synthetic small/medium/large cases are performance baselines, not production-scale capacity tests.",
    "The report does not implement C++, pybind11, numba kernels, or numerical algorithm changes.",
    "OPM/MRST and commercial simulator equivalence are not claimed.",
    "C++ or numba migration should start only after larger profiling proves a concrete hotspot.",
]


def run_performance_baseline(output_dir: str | Path = "accuracy_reports") -> dict[str, Any]:
    """Run the Python/NumPy/SciPy performance baseline and write reports."""
    profiles = run_stage_profiles()
    equivalence = check_numerical_equivalence()
    aggregate = summarize_profiles(profiles)
    summary: dict[str, Any] = {
        "benchmark_name": "performance_baseline",
        "source_task": "TASK-019",
        "success": bool(all(case["success"] for case in profiles) and equivalence["success"]),
        "runtime_environment": {
            "implementation": "Python / NumPy / SciPy",
            "numba_used": False,
            "cpp_used": False,
            "pybind11_used": False,
        },
        "case_profiles": profiles,
        "runtime_summary": aggregate["runtime_summary"],
        "memory_summary": aggregate["memory_summary"],
        "slowest_stage": aggregate["slowest_stage"],
        "numerical_equivalence": equivalence,
        "numba_recommended": aggregate["numba_recommended"],
        "cpp_recommended": aggregate["cpp_recommended"],
        "numba_recommendation": aggregate["numba_recommendation"],
        "cpp_recommendation": aggregate["cpp_recommendation"],
        "limitations": LIMITATIONS,
        "recommendations": [
            aggregate["numba_recommendation"],
            aggregate["cpp_recommendation"],
            "Repeat this baseline with larger field-like grids before any kernel migration decision.",
        ],
        "has_nan": bool(any(case["has_nan"] for case in profiles)),
        "has_inf": bool(any(case["has_inf"] for case in profiles)),
    }
    start = time.perf_counter()
    write_performance_reports(summary, output_dir)
    summary["report_generation_time_sec"] = float(time.perf_counter() - start)
    write_performance_reports(summary, output_dir)
    return summary


def write_performance_reports(summary: dict[str, Any], output_dir: str | Path = "accuracy_reports") -> dict[str, str]:
    """Write JSON and Markdown performance baseline reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "performance_baseline_summary.json"
    md_path = root / "performance_baseline_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Performance Baseline Summary",
        "",
        f"- success: {summary['success']}",
        f"- implementation: {summary['runtime_environment']['implementation']}",
        f"- slowest_stage: {summary['slowest_stage']['stage_name']}",
        f"- slowest_stage_runtime_sec: {summary['slowest_stage']['runtime_sec']:.6f}",
        f"- numba_recommended: {summary['numba_recommended']}",
        f"- cpp_recommended: {summary['cpp_recommended']}",
        f"- numerical_equivalence_max_abs_error: {summary['numerical_equivalence']['max_abs_error']:.6e}",
        "",
        "## Runtime Summary",
        "",
        "| Stage | Runtime sec | Peak memory MB |",
        "| --- | ---: | ---: |",
    ]
    for stage, runtime in summary["runtime_summary"].items():
        memory = summary["memory_summary"].get(stage, 0.0)
        lines.append(f"| {stage} | {runtime:.6f} | {memory:.6f} |")
    lines.extend(["", "## Synthetic Cases", ""])
    for case in summary["case_profiles"]:
        lines.append(f"### {case['case_id']}")
        lines.append("")
        lines.append(f"- total_cells: {case['total_cells']}")
        lines.append(f"- total_runtime_sec: {case['total_runtime_sec']:.6f}")
        lines.append(f"- total_memory_peak_mb: {case['total_memory_peak_mb']:.6f}")
        for stage in case["stages"]:
            lines.append(
                f"- {stage['stage_name']}: runtime={stage['runtime_sec']:.6f}s, "
                f"memory={stage['memory_peak_mb']:.6f}MB, success={stage['success']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Recommendations",
            "",
            f"- numba: {summary['numba_recommendation']}",
            f"- C++: {summary['cpp_recommendation']}",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_performance_baseline()
    print(json.dumps({"success": summary["success"], "slowest_stage": summary["slowest_stage"]}, indent=2))


if __name__ == "__main__":
    main()
