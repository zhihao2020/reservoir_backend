"""Lightweight readers for experimental data files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.data.schema import (
    ExperimentalDataset,
    dataset_from_arrays,
    infer_unit_from_name,
    normalize_field_name,
)


SUPPORTED_FORMATS = {".csv", ".json", ".npz"}


def read_experimental_data(
    path: str | Path,
    *,
    required_fields: list[str] | None = None,
    source_name: str | None = None,
) -> ExperimentalDataset:
    """Read CSV, JSON, or NPZ into the standard experimental dataset."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.stat().st_size == 0:
        raise ValueError("input file is empty")
    suffix = target.suffix.lower()
    if suffix == ".csv":
        dataset = _read_csv(target, source_name=source_name)
    elif suffix == ".json":
        dataset = _read_json(target, source_name=source_name)
    elif suffix == ".npz":
        dataset = _read_npz(target, source_name=source_name)
    else:
        raise ValueError(f"unsupported experimental data format: {suffix}")
    missing = _missing_required(dataset, required_fields)
    if missing:
        dataset.metadata.setdefault("reader_warnings", []).append(f"missing required fields: {missing}")
    return dataset


def _read_csv(path: Path, *, source_name: str | None) -> ExperimentalDataset:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV input has no header")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("CSV input has no rows")
    arrays: dict[str, list[float]] = {}
    units: dict[str, str] = {}
    metadata: dict[str, Any] = {"raw_columns": list(rows[0].keys())}
    unit_columns = {name[:-5]: name for name in rows[0] if name.lower().endswith("_unit")}
    for raw_name in rows[0]:
        if raw_name.lower().endswith("_unit"):
            continue
        values = []
        for row in rows:
            text = (row.get(raw_name) or "").strip()
            if text == "":
                values.append(np.nan)
            else:
                try:
                    values.append(float(text))
                except ValueError as exc:
                    raise ValueError(f"non-numeric value in column {raw_name!r}: {text!r}") from exc
        field_name = normalize_field_name(raw_name)
        arrays[field_name] = values
        unit_column = unit_columns.get(raw_name)
        if unit_column is not None:
            unit_values = {(row.get(unit_column) or "").strip() for row in rows if (row.get(unit_column) or "").strip()}
            if len(unit_values) == 1:
                units[field_name] = next(iter(unit_values))
        units.setdefault(field_name, infer_unit_from_name(raw_name))
    return dataset_from_arrays(
        arrays,
        units=units,
        metadata=metadata,
        source_name=source_name or path.stem,
        input_file=path,
        input_format="csv",
    )


def _read_json(path: Path, *, source_name: str | None) -> ExperimentalDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError("JSON record list is empty")
        columns = sorted({key for row in payload for key in row})
        arrays = {column: [row.get(column, np.nan) for row in payload] for column in columns}
        return dataset_from_arrays(
            arrays,
            metadata={"json_layout": "records"},
            source_name=source_name or path.stem,
            input_file=path,
            input_format="json",
        )
    if not isinstance(payload, dict):
        raise ValueError("JSON input must be an object or record list")
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("JSON object must contain non-empty fields")
    arrays: dict[str, Any] = {}
    units: dict[str, str] = {}
    for raw_name, value in fields.items():
        if isinstance(value, dict):
            arrays[raw_name] = value.get("values")
            units[normalize_field_name(raw_name)] = value.get("unit", infer_unit_from_name(raw_name))
        else:
            arrays[raw_name] = value
            units[normalize_field_name(raw_name)] = infer_unit_from_name(raw_name)
    return dataset_from_arrays(
        arrays,
        units=units,
        metadata=dict(payload.get("metadata", {})),
        source_name=source_name or payload.get("source_name", path.stem),
        input_file=path,
        input_format="json",
    )


def _read_npz(path: Path, *, source_name: str | None) -> ExperimentalDataset:
    archive = np.load(path, allow_pickle=False)
    arrays: dict[str, Any] = {}
    units: dict[str, str] = {}
    metadata: dict[str, Any] = {"npz_arrays": list(archive.files)}
    for name in archive.files:
        if name == "metadata_json":
            metadata.update(json.loads(str(archive[name].item())))
            continue
        if name.endswith("_unit"):
            continue
        arrays[name] = archive[name]
        unit_name = f"{name}_unit"
        if unit_name in archive.files:
            units[normalize_field_name(name)] = str(archive[unit_name].item())
        else:
            units[normalize_field_name(name)] = infer_unit_from_name(name)
    if not arrays:
        raise ValueError("NPZ input has no data arrays")
    return dataset_from_arrays(
        arrays,
        units=units,
        metadata=metadata,
        source_name=source_name or metadata.get("source_name", path.stem),
        input_file=path,
        input_format="npz",
    )


def _missing_required(dataset: ExperimentalDataset, required_fields: list[str] | None) -> list[str]:
    if not required_fields:
        return []
    present = set(dataset.fields)
    return [normalize_field_name(name) for name in required_fields if normalize_field_name(name) not in present]
