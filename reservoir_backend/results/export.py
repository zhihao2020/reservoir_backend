from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .catalog import ResultCatalog
from .manifest import ResultManifest, utc_timestamp, validate_result_manifest
from .report_index import DEFAULT_REPORT_PATHS, build_report_path_index


def export_manifest_json(catalog: ResultCatalog | Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = catalog.to_dict() if isinstance(catalog, ResultCatalog) else dict(catalog)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def export_summary_csv(catalog: ResultCatalog, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = catalog.list()
    columns = [
        "result_id",
        "case_id",
        "run_id",
        "module",
        "result_type",
        "field_name",
        "shape",
        "dtype",
        "unit",
        "path",
        "format",
        "source_task",
        "source_report",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            out["shape"] = "x".join(str(item) for item in row["shape"])
            writer.writerow(out)
    return output_path


def export_field_npz(fields: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in fields.items()}
    np.savez(output_path, **arrays)
    return output_path


def export_markdown_report_index(
    catalog: ResultCatalog,
    report_index: Mapping[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Result Manifest Summary",
        "",
        f"- success: {report_index['success']}",
        f"- num_results: {len(catalog.list())}",
        f"- num_reports: {report_index['num_reports']}",
        f"- num_missing_reports: {report_index['num_missing_reports']}",
        "",
        "## Result Catalog",
        "",
        "| result_id | module | result_type | field_name | format | path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in catalog.list():
        lines.append(
            f"| {item['result_id']} | {item['module']} | {item['result_type']} | "
            f"{item['field_name']} | {item['format']} | {item['path']} |"
        )
    lines.extend(
        [
            "",
            "## Report Path Index",
            "",
            "| path | format | result_type | exists |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report_index["reports"]:
        lines.append(f"| {item['path']} | {item['format']} | {item['result_type']} | {item['exists']} |")
    if report_index["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report_index["warnings"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_example_result_manifests(created_at: str | None = None) -> list[dict[str, Any]]:
    ts = created_at or utc_timestamp()
    examples = [
        ResultManifest(
            result_id="pressure_field_demo",
            case_id="demo_case",
            run_id="result-contract-example",
            module="M3",
            result_type="pressure_field",
            field_name="pressure",
            shape=[3, 4, 5],
            dtype="float64",
            unit="Pa",
            path="results/demo_case/pressure.npy",
            format="npy",
            created_at=ts,
            source_task="TASK-020",
            source_report="",
            metadata={"shape_convention": "(nz, ny, nx)", "csv_exports_metadata_only": True},
            warnings=[],
            limitations=["example manifest only; does not generate pressure field"],
        ),
        ResultManifest(
            result_id="saturation_field_demo",
            case_id="demo_case",
            run_id="result-contract-example",
            module="M4",
            result_type="saturation_field",
            field_name="sw",
            shape=[3, 4, 5],
            dtype="float64",
            unit="fraction",
            path="results/demo_case/sw_simulated.npy",
            format="npy",
            created_at=ts,
            source_task="TASK-020",
            source_report="",
            metadata={"bounds": [0.0, 1.0], "shape_convention": "(nz, ny, nx)"},
            warnings=[],
            limitations=["example manifest only; does not run saturation solver"],
        ),
        ResultManifest(
            result_id="parameter_fusion_report",
            case_id="benchmark",
            run_id="result-contract-example",
            module="M5",
            result_type="parameter_fusion_report",
            field_name="fusion_summary",
            shape=[],
            dtype="json",
            unit="dimensionless",
            path="accuracy_reports/parameter_fusion_benchmark_summary.json",
            format="json",
            created_at=ts,
            source_task="TASK-051",
            source_report="accuracy_reports/parameter_fusion_benchmark_summary.json",
            metadata={"contains": ["mae", "rmse", "bounds", "mask"]},
            warnings=[],
            limitations=["summary report, not a full 3D field export"],
        ),
        ResultManifest(
            result_id="experimental_data_qc_report",
            case_id="experimental_data",
            run_id="result-contract-example",
            module="M1",
            result_type="experimental_data_qc",
            field_name="qc_summary",
            shape=[],
            dtype="json",
            unit="dimensionless",
            path="accuracy_reports/experimental_data_qc_summary.json",
            format="json",
            created_at=ts,
            source_task="TASK-008",
            source_report="accuracy_reports/experimental_data_qc_summary.json",
            metadata={"contains": ["fields_detected", "bounds_violations", "warnings"]},
            warnings=[],
            limitations=["QC summary only; no database service"],
        ),
        ResultManifest(
            result_id="benchmark_registry_report",
            case_id="benchmark_registry",
            run_id="result-contract-example",
            module="M8",
            result_type="benchmark_registry",
            field_name="registry_summary",
            shape=[],
            dtype="json",
            unit="dimensionless",
            path="accuracy_reports/benchmark_registry_summary.json",
            format="json",
            created_at=ts,
            source_task="TASK-052",
            source_report="accuracy_reports/benchmark_registry_summary.json",
            metadata={"contains": ["modules_covered", "validation_levels", "open_source_references"]},
            warnings=[],
            limitations=["registry indexes benchmark reports; it does not rerun solvers"],
        ),
    ]
    return [validate_result_manifest(item) for item in examples]


def generate_result_manifest_summary(
    output_dir: str | Path = "accuracy_reports",
    root: str | Path = ".",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    catalog = ResultCatalog(build_example_result_manifests())
    report_index = build_report_path_index(DEFAULT_REPORT_PATHS, root=root)
    path_validation = catalog.validate_paths(root=root)
    warnings = report_index["warnings"] + path_validation["warnings"]
    summary = {
        "result_manifest_summary_name": "result_manifest_summary",
        "success": report_index["num_missing_reports"] == 0,
        "num_results": len(catalog.list()),
        "catalog": catalog.to_dict(),
        "report_path_index": report_index,
        "path_validation": path_validation,
        "export_formats_supported": ["json", "csv", "npz", "markdown"],
        "frontend_field_contract": "docs/frontend_field_contract.md",
        "warnings": warnings,
        "limitations": [
            "No frontend implementation.",
            "No UDP implementation.",
            "No REST API implementation.",
            "No database service.",
            "No solver rewrite.",
        ],
    }
    export_manifest_json(summary, output_path / "result_manifest_summary.json")
    export_markdown_report_index(catalog, report_index, output_path / "result_manifest_summary.md")
    return summary


def main() -> None:
    summary = generate_result_manifest_summary()
    print(json.dumps({"success": summary["success"], "num_results": summary["num_results"]}, indent=2))


if __name__ == "__main__":
    main()
