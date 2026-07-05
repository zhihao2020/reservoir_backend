"""Quality-control pipeline for experimental datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reservoir_backend.core.units import convert
from reservoir_backend.data.schema import (
    ExperimentalDataset,
    ExperimentalField,
    canonical_unit,
    field_spec,
    normalize_field_name,
)


@dataclass
class QCPipelineResult:
    """Result of experimental data QC."""

    dataset: ExperimentalDataset
    report: dict[str, Any]


def run_qc_pipeline(
    dataset: ExperimentalDataset,
    *,
    required_fields: list[str] | None = None,
    normalize_units: bool = True,
    outlier_zscore: float = 3.0,
) -> QCPipelineResult:
    """Run schema and physical-quality checks on a dataset."""
    working = normalize_dataset_units(dataset) if normalize_units else dataset
    fields_missing = _missing_fields(working, required_fields)
    shape_report = check_shape_consistency(working)
    unit_warnings = _unit_warnings(dataset)
    nan_count = 0
    inf_count = 0
    missing_count = 0
    outlier_count = 0
    bounds_violations: dict[str, int] = {}
    warnings: list[str] = []
    for name, field in working.fields.items():
        values = np.asarray(field.values, dtype=float)
        nan = int(np.count_nonzero(np.isnan(values)))
        inf = int(np.count_nonzero(np.isinf(values)))
        nan_count += nan
        inf_count += inf
        missing_count += nan
        if nan:
            warnings.append(f"{name} contains NaN/missing values")
        if inf:
            warnings.append(f"{name} contains Inf values")
        violations = _bounds_violations(name, values)
        if violations:
            bounds_violations[name] = violations
            warnings.append(f"{name} violates physical bounds")
        outliers = _outlier_count(values, zscore=outlier_zscore)
        outlier_count += outliers
        if outliers:
            warnings.append(f"{name} has {outliers} statistical outlier flag(s)")
    duplicate_report = detect_duplicate_time_or_coordinates(working)
    if duplicate_report["duplicate_time_count"]:
        warnings.append("duplicate time values detected")
    if duplicate_report["duplicate_coordinate_count"]:
        warnings.append("duplicate coordinate tuples detected")
    if fields_missing:
        warnings.append(f"missing required fields: {fields_missing}")
    warnings.extend(unit_warnings)
    success = (
        not fields_missing
        and inf_count == 0
        and sum(bounds_violations.values()) == 0
        and shape_report["shape_consistent"]
    )
    report = {
        "success": bool(success),
        "input_file": working.input_file,
        "format": working.input_format,
        "num_rows": _num_rows(working),
        "shape": list(working.shape) if working.shape is not None else None,
        "fields_detected": working.field_names,
        "fields_missing": fields_missing,
        "unit_warnings": unit_warnings,
        "num_nan": int(nan_count),
        "num_inf": int(inf_count),
        "num_missing": int(missing_count),
        "num_outliers": int(outlier_count),
        "bounds_violations": bounds_violations,
        "shape_consistency": shape_report,
        "duplicate_time_count": duplicate_report["duplicate_time_count"],
        "duplicate_coordinate_count": duplicate_report["duplicate_coordinate_count"],
        "resample_summary": {},
        "warnings": warnings,
        "recommendations": _recommendations(fields_missing, bounds_violations, inf_count),
        "metadata": working.metadata,
        "source_name": working.source_name,
    }
    return QCPipelineResult(dataset=working, report=report)


def normalize_dataset_units(dataset: ExperimentalDataset) -> ExperimentalDataset:
    """Return a copy with known fields converted to canonical units."""
    fields: dict[str, ExperimentalField] = {}
    unit_warnings: list[str] = []
    for name, field in dataset.fields.items():
        canonical = canonical_unit(name)
        if canonical == "unknown" or field.unit in {"unknown", ""}:
            unit_warnings.append(f"missing or unknown unit for {name}")
            fields[name] = ExperimentalField(name, field.values.copy(), field.unit, field.source_name, field.metadata.copy())
            continue
        try:
            converted = _convert_array(name, field.values, field.unit, canonical)
            unit = canonical
        except Exception:
            unit_warnings.append(f"unsupported unit conversion for {name}: {field.unit} -> {canonical}")
            converted = field.values.copy()
            unit = field.unit
        fields[name] = ExperimentalField(name, converted, unit, field.source_name, field.metadata.copy())
    metadata = dataset.metadata.copy()
    if unit_warnings:
        metadata["unit_warnings"] = unit_warnings
    return ExperimentalDataset(fields, metadata, dataset.source_name, dataset.input_file, dataset.input_format)


def check_shape_consistency(dataset: ExperimentalDataset) -> dict[str, Any]:
    """Check that all field arrays have the same shape."""
    shapes = {name: list(field.shape) for name, field in dataset.fields.items()}
    unique = {tuple(shape) for shape in shapes.values()}
    return {
        "shape_consistent": bool(len(unique) <= 1),
        "field_shapes": shapes,
        "warnings": [] if len(unique) <= 1 else ["field shapes are inconsistent"],
    }


def detect_duplicate_time_or_coordinates(dataset: ExperimentalDataset) -> dict[str, int]:
    """Detect duplicate time values and duplicate x/y/z coordinate tuples."""
    duplicate_time = 0
    if "time" in dataset.fields:
        values = dataset.fields["time"].values.ravel()
        finite = values[np.isfinite(values)]
        duplicate_time = int(finite.size - np.unique(finite).size)
    duplicate_coordinates = 0
    if {"x", "y", "z"} <= set(dataset.fields):
        coords = np.column_stack([dataset.fields[name].values.ravel() for name in ("x", "y", "z")])
        coords = coords[np.all(np.isfinite(coords), axis=1)]
        if coords.size:
            duplicate_coordinates = int(coords.shape[0] - np.unique(coords, axis=0).shape[0])
    return {
        "duplicate_time_count": duplicate_time,
        "duplicate_coordinate_count": duplicate_coordinates,
    }


def _convert_array(name: str, values: np.ndarray, unit: str, target: str) -> np.ndarray:
    if unit == target:
        return np.asarray(values, dtype=float).copy()
    vectorized = np.vectorize(lambda item: _convert_value(name, item, unit, target), otypes=[float])
    return vectorized(values)


def _convert_value(name: str, value: float, unit: str, target: str) -> float:
    if not np.isfinite(value):
        return float(value)
    if name in {"pressure", "permeability", "porosity", "saturation", "confidence"}:
        return convert(float(value), unit, target)
    if name == "time":
        return _time_to_seconds(float(value), unit)
    if name in {"x", "y", "z"}:
        return _length_to_meter(float(value), unit)
    if name == "temperature":
        return _temperature_to_kelvin(float(value), unit)
    return float(value)


def _time_to_seconds(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        return value
    if normalized in {"min", "minute", "minutes"}:
        return value * 60.0
    if normalized in {"h", "hr", "hour", "hours"}:
        return value * 3600.0
    if normalized in {"day", "days", "d"}:
        return value * 86400.0
    raise ValueError(f"unsupported time unit: {unit}")


def _length_to_meter(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized == "m":
        return value
    if normalized == "cm":
        return value / 100.0
    if normalized == "mm":
        return value / 1000.0
    raise ValueError(f"unsupported length unit: {unit}")


def _temperature_to_kelvin(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"k", "kelvin"}:
        return value
    if normalized in {"c", "degc", "celsius"}:
        return value + 273.15
    raise ValueError(f"unsupported temperature unit: {unit}")


def _missing_fields(dataset: ExperimentalDataset, required_fields: list[str] | None) -> list[str]:
    if not required_fields:
        return []
    present = set(dataset.fields)
    return [normalize_field_name(name) for name in required_fields if normalize_field_name(name) not in present]


def _unit_warnings(dataset: ExperimentalDataset) -> list[str]:
    warnings = list(dataset.metadata.get("unit_warnings", []))
    for name, field in dataset.fields.items():
        if field.unit in {"", "unknown"}:
            warnings.append(f"missing unit for {name}")
    return sorted(set(warnings))


def _bounds_violations(name: str, values: np.ndarray) -> int:
    spec = field_spec(name)
    if spec is None:
        return 0
    finite = np.isfinite(values)
    violations = np.zeros(values.shape, dtype=bool)
    if spec.lower is not None:
        if spec.strict_lower:
            violations |= finite & (values <= spec.lower)
        else:
            violations |= finite & (values < spec.lower)
    if spec.upper is not None:
        violations |= finite & (values > spec.upper)
    if name == "pressure":
        violations |= ~np.isfinite(values)
    return int(np.count_nonzero(violations))


def _outlier_count(values: np.ndarray, zscore: float) -> int:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return 0
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if std <= 0.0:
        return 0
    return int(np.count_nonzero(np.abs((finite - mean) / std) > zscore))


def _num_rows(dataset: ExperimentalDataset) -> int:
    shape = dataset.shape
    if shape is None:
        return 0
    if len(shape) == 0:
        return 1
    return int(shape[0])


def _recommendations(fields_missing: list[str], bounds_violations: dict[str, int], inf_count: int) -> list[str]:
    recommendations: list[str] = []
    if fields_missing:
        recommendations.append("Provide required columns before running inversion or simulation modules.")
    if bounds_violations:
        recommendations.append("Review physical bounds violations before using data as solver input.")
    if inf_count:
        recommendations.append("Remove or replace Inf values before downstream computation.")
    if not recommendations:
        recommendations.append("Dataset passed core QC checks for the current lightweight schema.")
    return recommendations
