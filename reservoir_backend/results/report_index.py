from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORT_PATHS = (
    "accuracy_reports/experimental_data_qc_summary.json",
    "accuracy_reports/experimental_data_qc_summary.md",
    "accuracy_reports/saturation_inversion_benchmark_summary.json",
    "accuracy_reports/pressure_solver_benchmark_summary.json",
    "accuracy_reports/saturation_transport_benchmark_summary.json",
    "accuracy_reports/capillary_gravity_benchmark_summary.json",
    "accuracy_reports/three_phase_benchmark_summary.json",
    "accuracy_reports/parameter_fusion_benchmark_summary.json",
    "accuracy_reports/benchmark_registry_summary.json",
)


def infer_result_type(path: str) -> str:
    name = Path(path).name
    if "experimental_data_qc" in name:
        return "experimental_data_qc"
    if "benchmark_registry" in name:
        return "benchmark_registry"
    if "benchmark" in name:
        return "benchmark_summary"
    return "report"


def build_report_path_index(
    report_paths: Iterable[str] | None = None,
    root: str | Path = ".",
) -> dict[str, Any]:
    root_path = Path(root)
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in report_paths or DEFAULT_REPORT_PATHS:
        path_obj = Path(path)
        resolved = path_obj if path_obj.is_absolute() else root_path / path_obj
        exists = resolved.exists()
        entry_warnings = [] if exists else [f"missing report path: {path}"]
        warnings.extend(entry_warnings)
        entries.append(
            {
                "path": str(path),
                "format": path_obj.suffix.lstrip("."),
                "result_type": infer_result_type(str(path)),
                "exists": exists,
                "warnings": entry_warnings,
            }
        )
    return {
        "success": not warnings,
        "num_reports": len(entries),
        "num_existing_reports": sum(1 for item in entries if item["exists"]),
        "num_missing_reports": sum(1 for item in entries if not item["exists"]),
        "reports": entries,
        "warnings": warnings,
    }
