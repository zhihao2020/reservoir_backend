"""Dynamic field fusion utilities for lightweight synthetic twins."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.fusion.fusion_diagnostics import compute_fusion_error
from reservoir_backend.fusion.synthetic_twin import (
    DynamicFieldRecord,
    DynamicFusionSummary,
    ProductionSeriesRecord,
    StaticFieldRecord,
    SyntheticTwinMetadata,
)
from reservoir_backend.fusion.uncertainty import uncertainty_weighted_fusion


STATIC_BOUNDS = {
    "porosity": (0.0, 1.0),
    "permeability": (0.0, None),
}
DYNAMIC_BOUNDS = {
    "saturation": (0.0, 1.0),
    "water_cut": (0.0, 1.0),
}


def check_shape_time_consistency(
    metadata: SyntheticTwinMetadata,
    static_records: list[StaticFieldRecord],
    dynamic_records: list[DynamicFieldRecord],
    production_records: list[ProductionSeriesRecord],
) -> dict[str, Any]:
    """Check static shape, dynamic shape, and time-step consistency."""
    warnings: list[str] = []
    success = True
    for record in static_records:
        if tuple(record.shape) != tuple(metadata.grid_shape):
            success = False
            warnings.append(f"static field {record.field_name} shape mismatch")
    expected_dynamic_shape = (len(metadata.time_steps),) + tuple(metadata.grid_shape)
    for record in dynamic_records:
        if tuple(record.shape) != expected_dynamic_shape:
            success = False
            warnings.append(f"dynamic field {record.field_name} shape mismatch")
        if tuple(record.time_steps) != tuple(metadata.time_steps):
            success = False
            warnings.append(f"dynamic field {record.field_name} time-step mismatch")
    for record in production_records:
        if tuple(float(v) for v in record.time) != tuple(metadata.time_steps):
            success = False
            warnings.append(f"production series {record.series_name} time-step mismatch")
    return {
        "success": bool(success),
        "grid_shape": list(metadata.grid_shape),
        "time_steps": list(metadata.time_steps),
        "num_static_records": len(static_records),
        "num_dynamic_records": len(dynamic_records),
        "num_production_records": len(production_records),
        "warnings": warnings,
    }


def fuse_static_field_records(records: list[StaticFieldRecord]) -> dict[str, dict[str, Any]]:
    """Fuse static records grouped by field name."""
    grouped = _group(records, "field_name")
    return {name: _fuse_record_group(group, name=name, is_series=False, bounds=_bounds_for_static(name)) for name, group in grouped.items()}


def fuse_dynamic_field_records(records: list[DynamicFieldRecord]) -> dict[str, dict[str, Any]]:
    """Fuse dynamic field records grouped by field name."""
    grouped = _group(records, "field_name")
    return {name: _fuse_record_group(group, name=name, is_series=False, bounds=_bounds_for_dynamic(name)) for name, group in grouped.items()}


def fuse_production_series_records(records: list[ProductionSeriesRecord]) -> dict[str, dict[str, Any]]:
    """Fuse production and water-cut time series grouped by series name."""
    grouped = _group(records, "series_name")
    return {name: _fuse_record_group(group, name=name, is_series=True, bounds=_bounds_for_dynamic(name)) for name, group in grouped.items()}


def build_synthetic_twin_fusion_summary(
    *,
    metadata: SyntheticTwinMetadata,
    static_records: list[StaticFieldRecord],
    dynamic_records: list[DynamicFieldRecord],
    production_records: list[ProductionSeriesRecord],
) -> DynamicFusionSummary:
    """Fuse static, dynamic, and production records into a summary object."""
    consistency = check_shape_time_consistency(metadata, static_records, dynamic_records, production_records)
    if not consistency["success"]:
        raise ValueError("; ".join(consistency["warnings"]))
    static_fields = fuse_static_field_records(static_records)
    dynamic_fields = fuse_dynamic_field_records(dynamic_records)
    production_series = fuse_production_series_records(production_records)
    all_reports = list(static_fields.values()) + list(dynamic_fields.values()) + list(production_series.values())
    has_nan = any(bool(report["diagnostics"].get("has_nan", False)) for report in all_reports)
    has_inf = any(bool(report["diagnostics"].get("has_inf", False)) for report in all_reports)
    total_bound_violations = int(sum(int(report["diagnostics"].get("bounds_violations", 0)) for report in all_reports))
    rmse_values = [
        float(report["truth_error"]["rmse"])
        for report in all_reports
        if report.get("truth_error", {}).get("rmse") is not None
    ]
    provenance = {
        "sources": sorted(
            {
                record.source
                for record in [*static_records, *dynamic_records, *production_records]
            }
        ),
        "static_record_count": len(static_records),
        "dynamic_record_count": len(dynamic_records),
        "production_record_count": len(production_records),
        "source_task": "F4-04",
    }
    limitations = [
        "No history matching is performed.",
        "No EnKF or ES-MDA update is performed.",
        "No automatic geological model update is performed.",
        "No closed-loop digital twin control is implemented.",
    ]
    diagnostics = {
        "success": bool(not has_inf and total_bound_violations == 0),
        "shape_time_consistency": consistency,
        "num_static_fields": len(static_fields),
        "num_dynamic_fields": len(dynamic_fields),
        "num_production_series": len(production_series),
        "overall_rmse": None if not rmse_values else float(np.mean(rmse_values)),
        "overall_max_rmse": None if not rmse_values else float(np.max(rmse_values)),
        "total_bound_violations": total_bound_violations,
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
    }
    warnings = list(consistency["warnings"])
    for report in all_reports:
        warnings.extend(str(item) for item in report["diagnostics"].get("warnings", []))
    return DynamicFusionSummary(
        metadata=metadata,
        static_fields=static_fields,
        dynamic_fields=dynamic_fields,
        production_series=production_series,
        diagnostics=diagnostics,
        provenance=provenance,
        warnings=warnings,
        limitations=limitations,
    )


def _fuse_record_group(records: list[Any], *, name: str, is_series: bool, bounds: tuple[float | None, float | None] | None) -> dict[str, Any]:
    values = [_masked_values(record) for record in records]
    variances = [record.variance for record in records if getattr(record, "variance", None) is not None]
    confidences = [record.confidence for record in records if getattr(record, "confidence", None) is not None]
    kwargs: dict[str, Any] = {}
    if len(variances) == len(records):
        kwargs["variances"] = variances
    elif len(confidences) == len(records):
        kwargs["confidences"] = confidences
    if bounds is not None and bounds[0] is not None and bounds[1] is not None:
        kwargs["bounds"] = (float(bounds[0]), float(bounds[1]))
    fused, fused_variance, report = uncertainty_weighted_fusion(values, **kwargs)
    if bounds is not None:
        report["bounds_violations"] = _count_bound_violations(fused, bounds)
    truth = _first_truth(records)
    truth_error = compute_fusion_error(truth, fused) if truth is not None else _empty_truth_error()
    provenance = {
        "field_or_series": name,
        "sources": [record.source for record in records],
        "provenance": [dict(record.provenance) for record in records],
        "units": sorted({record.unit for record in records}),
        "record_count": len(records),
    }
    if is_series:
        provenance["time"] = np.asarray(records[0].time, dtype=float).tolist()
    elif hasattr(records[0], "time_steps"):
        provenance["time_steps"] = list(records[0].time_steps)
    return {
        "name": name,
        "shape": list(fused.shape),
        "unit": records[0].unit,
        "source_count": len(records),
        "fused_min": _finite_stat(fused, np.nanmin),
        "fused_max": _finite_stat(fused, np.nanmax),
        "variance_min": _finite_stat(fused_variance, np.nanmin),
        "variance_max": _finite_stat(fused_variance, np.nanmax),
        "diagnostics": _json_ready(report),
        "truth_error": _json_ready(truth_error),
        "provenance": provenance,
    }


def _group(records: list[Any], attr: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, attr))].append(record)
    return dict(grouped)


def _masked_values(record: Any) -> NDArray[np.float64]:
    values = np.asarray(record.values, dtype=float).copy()
    mask = getattr(record, "mask", None)
    if mask is not None:
        values = np.where(mask, values, np.nan)
    return values


def _first_truth(records: list[Any]) -> NDArray[np.float64] | None:
    for record in records:
        truth = getattr(record, "truth", None)
        if truth is not None:
            return np.asarray(truth, dtype=float)
    return None


def _bounds_for_static(name: str) -> tuple[float | None, float | None] | None:
    normalized = name.lower()
    if normalized in STATIC_BOUNDS:
        return STATIC_BOUNDS[normalized]
    return None


def _bounds_for_dynamic(name: str) -> tuple[float | None, float | None] | None:
    normalized = name.lower()
    if normalized in DYNAMIC_BOUNDS:
        return DYNAMIC_BOUNDS[normalized]
    return None


def _count_bound_violations(values: NDArray[np.float64], bounds: tuple[float | None, float | None]) -> int:
    finite = np.isfinite(values)
    count = 0
    if bounds[0] is not None:
        count += int(np.count_nonzero(finite & (values < float(bounds[0]))))
    if bounds[1] is not None:
        count += int(np.count_nonzero(finite & (values > float(bounds[1]))))
    return count


def _empty_truth_error() -> dict[str, Any]:
    return {
        "success": True,
        "shape_consistent": True,
        "mae": None,
        "rmse": None,
        "max_abs_error": None,
        "num_compared": 0,
        "warnings": ["synthetic truth not provided"],
    }


def _finite_stat(values: NDArray[np.float64], fn) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(fn(finite))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value
