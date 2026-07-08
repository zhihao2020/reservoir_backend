"""Field data ingestion helpers for industrial workflow inputs.

The module provides lightweight file-based schemas for common field inputs:
well table, production history, pressure history, schedule CSV, and property
fields. It does not implement a database service, commercial data platform, LAS
parser, Eclipse deck parser, or RESQML reader.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from reservoir_backend.data.qc import run_qc_pipeline
from reservoir_backend.data.reader import read_experimental_data
from reservoir_backend.project.project_registry import json_safe


TABLE_SCHEMAS: dict[str, dict[str, Any]] = {
    "well_table": {
        "required": ("well_id", "well_type", "i", "j", "k"),
        "numeric": ("i", "j", "k"),
        "allowed": {"well_type": {"injector", "producer"}, "status": {"open", "shut"}},
    },
    "production_history": {
        "required": ("well_id", "time", "oil_rate", "water_rate"),
        "numeric": ("time", "oil_rate", "water_rate", "gas_rate"),
        "allowed": {},
    },
    "pressure_history": {
        "required": ("well_id", "time", "pressure"),
        "numeric": ("time", "pressure"),
        "allowed": {},
    },
    "schedule": {
        "required": ("well_id", "time", "control_type", "target", "unit", "status"),
        "numeric": ("time", "target"),
        "allowed": {"control_type": {"rate", "bhp"}, "status": {"open", "shut"}},
    },
}


def read_field_csv(path: str | Path, schema_name: str) -> list[dict[str, Any]]:
    """Read a field CSV table and validate it against a lightweight schema."""
    if schema_name not in TABLE_SCHEMAS:
        raise ValueError(f"unsupported field table schema: {schema_name}")
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    with target.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV table has no header")
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError("CSV table has no rows")
    validate_field_records(records, schema_name)
    return _normalize_records(records, schema_name)


def validate_field_records(records: Iterable[Mapping[str, Any]], schema_name: str) -> dict[str, Any]:
    """Validate records for a known field-data schema."""
    if schema_name not in TABLE_SCHEMAS:
        raise ValueError(f"unsupported field table schema: {schema_name}")
    schema = TABLE_SCHEMAS[schema_name]
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("records must be non-empty")
    fields = set(rows[0])
    missing = [name for name in schema["required"] if name not in fields]
    if missing:
        raise ValueError(f"{schema_name} missing required fields: {missing}")
    errors: list[str] = []
    for row_index, row in enumerate(rows):
        for name in schema["required"]:
            if str(row.get(name, "")).strip() == "":
                errors.append(f"row {row_index} missing value for {name}")
        for name in schema["numeric"]:
            if name in row and str(row.get(name, "")).strip() != "":
                try:
                    value = float(row[name])
                except ValueError:
                    errors.append(f"row {row_index} invalid numeric value for {name}")
                    continue
                if not np.isfinite(value):
                    errors.append(f"row {row_index} nonfinite value for {name}")
        for name, allowed in schema["allowed"].items():
            if name in row and str(row[name]).strip():
                value = str(row[name]).strip().lower()
                if value not in allowed:
                    errors.append(f"row {row_index} invalid {name}: {value}")
    if schema_name == "well_table":
        duplicates = duplicate_values([str(row["well_id"]).strip() for row in rows])
        if duplicates:
            errors.append(f"duplicate well id(s): {duplicates}")
    if schema_name in {"production_history", "pressure_history", "schedule"}:
        ordering = validate_time_ordering(rows)
        if not ordering["success"]:
            errors.extend(ordering["errors"])
    if errors:
        raise ValueError("; ".join(errors))
    return {"success": True, "num_records": len(rows), "schema": schema_name}


def validate_time_ordering(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check nondecreasing time ordering per well id."""
    errors: list[str] = []
    last_by_well: dict[str, float] = {}
    for index, record in enumerate(records):
        well_id = str(record.get("well_id", "")).strip()
        time_value = float(record.get("time", 0.0))
        if well_id in last_by_well and time_value < last_by_well[well_id]:
            errors.append(f"time ordering violation for {well_id} at row {index}")
        last_by_well[well_id] = time_value
    return {"success": not errors, "errors": errors}


def duplicate_values(values: Iterable[str]) -> list[str]:
    """Return duplicated string values in insertion order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def read_well_table(path: str | Path) -> list[dict[str, Any]]:
    return read_field_csv(path, "well_table")


def read_production_history(path: str | Path) -> list[dict[str, Any]]:
    return read_field_csv(path, "production_history")


def read_pressure_history(path: str | Path) -> list[dict[str, Any]]:
    return read_field_csv(path, "pressure_history")


def read_schedule_csv(path: str | Path) -> list[dict[str, Any]]:
    return read_field_csv(path, "schedule")


def read_property_field(path: str | Path, required_fields: list[str] | None = None) -> dict[str, Any]:
    """Read and QC a property field using the existing experimental-data pipeline."""
    dataset = read_experimental_data(path, required_fields=required_fields)
    qc = run_qc_pipeline(dataset, required_fields=required_fields)
    return {"dataset": qc.dataset, "qc_report": qc.report}


def build_case_input_summary(
    *,
    well_table: str | Path,
    production_history: str | Path,
    pressure_history: str | Path,
    schedule: str | Path,
    property_field: str | Path,
) -> dict[str, Any]:
    """Read field inputs and return a JSON-serializable case-input summary."""
    wells = read_well_table(well_table)
    production = read_production_history(production_history)
    pressure = read_pressure_history(pressure_history)
    schedule_records = read_schedule_csv(schedule)
    property_result = read_property_field(property_field, required_fields=["porosity", "permeability"])
    qc_report = property_result["qc_report"]
    warnings = list(qc_report.get("warnings", []))
    summary = {
        "summary_name": "field_data_ingestion_summary",
        "source_task": "IND-002",
        "success": bool(qc_report["success"]),
        "inputs": {
            "well_table": str(well_table),
            "production_history": str(production_history),
            "pressure_history": str(pressure_history),
            "schedule": str(schedule),
            "property_field": str(property_field),
        },
        "well_table": {
            "num_wells": len(wells),
            "well_ids": [row["well_id"] for row in wells],
            "well_types": sorted({row["well_type"] for row in wells}),
        },
        "production_history": {
            "num_records": len(production),
            "well_ids": sorted({row["well_id"] for row in production}),
            "time_min": min(float(row["time"]) for row in production),
            "time_max": max(float(row["time"]) for row in production),
        },
        "pressure_history": {
            "num_records": len(pressure),
            "pressure_min": min(float(row["pressure"]) for row in pressure),
            "pressure_max": max(float(row["pressure"]) for row in pressure),
        },
        "schedule": {
            "num_records": len(schedule_records),
            "control_types": sorted({row["control_type"] for row in schedule_records}),
            "statuses": sorted({row["status"] for row in schedule_records}),
        },
        "property_field_qc": qc_report,
        "warnings": warnings,
        "limitations": [
            "File-based CSV/JSON/NPZ ingestion only.",
            "No database service.",
            "No commercial data platform.",
            "LAS, Eclipse deck, and RESQML are roadmap items only.",
        ],
    }
    return json_safe(summary)


def write_field_data_ingestion_report(summary: Mapping[str, Any], output_dir: str | Path = "accuracy_reports") -> dict[str, str]:
    """Write field-data ingestion JSON and Markdown reports."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "field_data_ingestion_summary.json"
    md_path = root / "field_data_ingestion_summary.md"
    json_path.write_text(json.dumps(json_safe(dict(summary)), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _normalize_records(records: list[dict[str, Any]], schema_name: str) -> list[dict[str, Any]]:
    schema = TABLE_SCHEMAS[schema_name]
    normalized: list[dict[str, Any]] = []
    for row in records:
        item = {key: str(value).strip() for key, value in row.items()}
        for name in schema["numeric"]:
            if name in item and item[name] != "":
                item[name] = float(item[name])
        for name in schema["allowed"]:
            if name in item:
                item[name] = str(item[name]).strip().lower()
        normalized.append(item)
    return normalized


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Field Data Ingestion Summary",
        "",
        "## Implemented Scope",
        "",
        f"- success: {summary['success']}",
        f"- num_wells: {summary['well_table']['num_wells']}",
        f"- production_records: {summary['production_history']['num_records']}",
        f"- pressure_records: {summary['pressure_history']['num_records']}",
        f"- schedule_records: {summary['schedule']['num_records']}",
        "",
        "## Test Results",
        "",
        "- See `tests/test_field_data_ingestion.py` and `pytest -q`.",
        "",
        "## Known Limitations",
        "",
    ]
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(
        [
            "- No database service.",
            "- No commercial data platform.",
            "- No LAS, Eclipse deck, or RESQML parser is implemented.",
        ]
    )
    lines.extend(["", "## Next Steps", "", "- Connect validated field inputs to schedule v0 in IND-003."])
    return "\n".join(lines) + "\n"
