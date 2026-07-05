from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_markdown_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-Scale Benchmark Summary",
        "",
        f"- success: {report.get('success')}",
        f"- case_id: {report.get('case_id')}",
        f"- similarity_score: {report.get('similarity_report', {}).get('overall_similarity_score')}",
        f"- regime_shift_detected: {report.get('scale_effect_report', {}).get('regime_shift_detected')}",
        f"- num_curves: {report.get('lab_field_validation_report', {}).get('num_curves')}",
        "",
        "## Similarity Criteria",
        "",
        "| criterion | lab | field | score |",
        "| --- | --- | --- | --- |",
    ]
    similarity = report.get("similarity_report", {})
    lab_numbers = similarity.get("dimensionless_numbers_lab", {})
    field_numbers = similarity.get("dimensionless_numbers_field", {})
    scores = similarity.get("criterion_scores", {})
    for key in ("reynolds", "capillary", "peclet", "mobility_ratio", "gravity_number", "dimensionless_pressure", "dimensionless_time"):
        lines.append(f"| {key} | {lab_numbers.get(key)} | {field_numbers.get(key)} | {scores.get(key)} |")

    scale = report.get("scale_effect_report", {})
    lines.extend(["", "## Scale Effect", "", "| metric | value |", "| --- | --- |"])
    for key, value in scale.get("scale_ratios", {}).items():
        lines.append(f"| {key} | {value} |")
    lines.append(f"| regime_lab | {scale.get('regime_lab', {}).get('flow_regime')} |")
    lines.append(f"| regime_field | {scale.get('regime_field', {}).get('flow_regime')} |")
    lines.append(f"| regime_shift_detected | {scale.get('regime_shift_detected')} |")

    validation = report.get("lab_field_validation_report", {})
    lines.extend(["", "## Lab-Field Validation", "", "| metric | value |", "| --- | --- |"])
    for key in ("rmse", "mae", "mape", "r2", "nrmse", "max_absolute_error", "num_matched_samples"):
        lines.append(f"| {key} | {validation.get(key)} |")

    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report.get("limitations"):
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {limitation}" for limitation in report["limitations"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
