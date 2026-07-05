"""Project / case / run management report for TASK-056."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reservoir_backend.project.case_registry import CaseMetadata, CaseRegistry
from reservoir_backend.project.project_registry import ProjectMetadata, ProjectRegistry
from reservoir_backend.project.run_history import RunHistory, RunRecord
from reservoir_backend.results.report_index import build_report_path_index


LIMITATIONS = [
    "No database service.",
    "No frontend implementation.",
    "No UDP or REST API implementation.",
    "No Petrel-like full workflow.",
    "No solver, inversion, fusion, cross-scale, data, result, benchmark, reference, config, C++, CMake, or pybind11 changes.",
]


def build_example_project_case_state(root: str | Path = ".") -> dict[str, Any]:
    """Build a deterministic example state aligned with existing reports."""
    project_registry = ProjectRegistry(
        [
            ProjectMetadata(
                project_id="project_reservoir_backend_validation",
                name="Reservoir Backend Validation Project",
                description="Lightweight project wrapper for validated backend reports.",
                created_at="2026-07-06T00:00:00+00:00",
                metadata={"source_task": "TASK-056", "management_layer": "file_based"},
            )
        ]
    )
    case_registry = CaseRegistry(
        [
            CaseMetadata(
                case_id="case_benchmark_registry",
                project_id="project_reservoir_backend_validation",
                case_name="Benchmark Registry Evidence Case",
                input_paths=["accuracy_reports/benchmark_registry_summary.json"],
                output_paths=["accuracy_reports/result_manifest_summary.json"],
                module_tags=["M8", "benchmark_registry", "result_manifest"],
                status="validated",
                metadata={"source_task": "TASK-056", "case_type": "report_index"},
            ),
            CaseMetadata(
                case_id="case_performance_baseline",
                project_id="project_reservoir_backend_validation",
                case_name="Performance Baseline Evidence Case",
                input_paths=["accuracy_reports/performance_baseline_summary.json"],
                output_paths=["accuracy_reports/performance_baseline_summary.md"],
                module_tags=["M8", "performance", "report"],
                status="validated",
                metadata={"source_task": "TASK-019", "case_type": "performance_report"},
            ),
        ]
    )
    run_history = RunHistory(
        [
            RunRecord(
                run_id="run_project_case_management_summary",
                case_id="case_benchmark_registry",
                started_at="2026-07-06T00:00:00+00:00",
                finished_at="2026-07-06T00:00:01+00:00",
                status="validated",
                report_paths=[
                    "accuracy_reports/benchmark_registry_summary.json",
                    "accuracy_reports/result_manifest_summary.json",
                    "accuracy_reports/performance_baseline_summary.json",
                ],
                result_manifest_paths=["accuracy_reports/result_manifest_summary.json"],
                metrics={"num_registered_reports": 9, "success": True},
                warnings=[],
            )
        ]
    )
    report_index = build_report_path_index(root=root)
    return {
        "project_registry": project_registry,
        "case_registry": case_registry,
        "run_history": run_history,
        "report_index": report_index,
    }


def build_project_case_management_summary(root: str | Path = ".") -> dict[str, Any]:
    """Build the TASK-056 project / case / run management summary."""
    state = build_example_project_case_state(root)
    project_registry: ProjectRegistry = state["project_registry"]
    case_registry: CaseRegistry = state["case_registry"]
    run_history: RunHistory = state["run_history"]
    report_index = state["report_index"]
    case_paths = case_registry.validate_paths(root)
    run_paths = run_history.validate_report_paths(root)
    warnings = [*case_paths["warnings"], *run_paths["warnings"], *report_index["warnings"]]
    summary = {
        "summary_name": "project_case_management_summary",
        "source_task": "TASK-056",
        "success": not warnings,
        "project_registry": project_registry.to_dict(),
        "case_registry": case_registry.to_dict(),
        "run_history": run_history.to_dict(),
        "report_index": report_index,
        "path_validation": {
            "case_paths": case_paths,
            "run_paths": run_paths,
        },
        "capabilities": {
            "project_registry": ["add", "list", "find", "json_serialization"],
            "case_registry": ["add", "list", "find", "update_status", "validate_paths", "json_serialization"],
            "run_history": ["append", "list", "find", "validate_report_paths", "json_serialization"],
            "report_index": ["existing report path alignment", "missing path warnings"],
        },
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }
    return summary


def write_project_case_management_reports(
    summary: dict[str, Any],
    output_dir: str | Path = "accuracy_reports",
) -> dict[str, str]:
    """Write JSON and Markdown reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "project_case_management_summary.json"
    md_path = root / "project_case_management_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_project_case_management_report(output_dir: str | Path = "accuracy_reports", root: str | Path = ".") -> dict[str, Any]:
    """Build and write the TASK-056 summary report."""
    summary = build_project_case_management_summary(root)
    write_project_case_management_reports(summary, output_dir)
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Project / Case Management Summary",
        "",
        f"- success: {summary['success']}",
        f"- source_task: {summary['source_task']}",
        f"- num_projects: {summary['project_registry']['num_projects']}",
        f"- num_cases: {summary['case_registry']['num_cases']}",
        f"- num_runs: {summary['run_history']['num_runs']}",
        f"- report_index_existing: {summary['report_index']['num_existing_reports']}",
        "",
        "## Projects",
        "",
    ]
    for project in summary["project_registry"]["projects"]:
        lines.append(f"- {project['project_id']}: {project['name']}")
    lines.extend(["", "## Cases", ""])
    for case in summary["case_registry"]["cases"]:
        lines.append(f"- {case['case_id']}: {case['case_name']} ({case['status']})")
    lines.extend(["", "## Runs", ""])
    for run in summary["run_history"]["runs"]:
        lines.append(f"- {run['run_id']}: case={run['case_id']}, status={run['status']}")
    lines.extend(["", "## Report Index", ""])
    for report in summary["report_index"]["reports"]:
        lines.append(f"- {report['path']}: exists={report['exists']}, type={report['result_type']}")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    if summary["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_project_case_management_report()
    print(json.dumps({"success": summary["success"], "num_cases": summary["case_registry"]["num_cases"]}, indent=2))


if __name__ == "__main__":
    main()
