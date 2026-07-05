"""QC report writer for experimental data pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.data.qc import run_qc_pipeline
from reservoir_backend.data.schema import dataset_from_arrays


def write_qc_report(report: dict, output_dir: str | Path = "accuracy_reports", stem: str = "experimental_data_qc_summary") -> tuple[Path, Path]:
    """Write QC report as JSON and Markdown."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{stem}.json"
    md_path = root / f"{stem}.md"
    json_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return json_path, md_path


def generate_demo_qc_report(output_dir: str | Path = "accuracy_reports") -> dict:
    """Generate a small synthetic QC report for manual smoke testing."""
    dataset = dataset_from_arrays(
        {
            "time_s": np.array([0.0, 10.0, 20.0]),
            "porosity_fraction": np.array([0.2, 0.25, 0.3]),
            "permeability_md": np.array([100.0, 120.0, 140.0]),
            "pressure_mpa": np.array([10.0, 9.8, 9.6]),
            "resistivity_ohm_m": np.array([20.0, 22.0, 24.0]),
        },
        source_name="demo_qc_dataset",
        input_file="synthetic",
        input_format="synthetic",
    )
    result = run_qc_pipeline(dataset, required_fields=["time", "porosity", "permeability"])
    write_qc_report(result.report, output_dir)
    return result.report


def _markdown(report: dict) -> str:
    lines = [
        "# Experimental Data QC Summary",
        "",
        f"- success: {report.get('success')}",
        f"- input_file: {report.get('input_file')}",
        f"- format: {report.get('format')}",
        f"- num_rows: {report.get('num_rows')}",
        f"- shape: {report.get('shape')}",
        f"- fields_detected: {', '.join(report.get('fields_detected', []))}",
        f"- fields_missing: {report.get('fields_missing', [])}",
        f"- num_nan: {report.get('num_nan')}",
        f"- num_inf: {report.get('num_inf')}",
        f"- num_missing: {report.get('num_missing')}",
        f"- num_outliers: {report.get('num_outliers')}",
        f"- bounds_violations: {report.get('bounds_violations', {})}",
        "",
        "## Warnings",
        "",
    ]
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.get("recommendations", []))
    return "\n".join(lines) + "\n"


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


if __name__ == "__main__":
    print(json.dumps(generate_demo_qc_report(), indent=2))
