"""Diagnostics for parameter field fusion benchmarks.

This module is intentionally side-effect free: it does not modify fusion
inputs, rewrite field values, or save files. It reports shape, finite-value,
weight, NaN/mask, bound, and error metrics used by benchmark runners.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from reservoir_backend.core.field import Field3D


def compute_field_statistics(field) -> dict[str, object]:
    """Return finite-value statistics for an array-like field or ``Field3D``."""
    values = _values(field)
    finite = np.isfinite(values)
    warnings: list[str] = []
    if np.isnan(values).any():
        warnings.append("field contains NaN values")
    if np.isinf(values).any():
        warnings.append("field contains Inf values")
    if not finite.any():
        warnings.append("field has no finite values")
    finite_values = values[finite]
    return {
        "field_min": _finite_stat(finite_values, np.min),
        "field_max": _finite_stat(finite_values, np.max),
        "field_mean": _finite_stat(finite_values, np.mean),
        "field_std": _finite_stat(finite_values, np.std),
        "shape": list(values.shape),
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
        "num_nan": int(np.count_nonzero(np.isnan(values))),
        "num_inf": int(np.count_nonzero(np.isinf(values))),
        "warnings": warnings,
    }


def check_field_finite(field) -> dict[str, object]:
    """Report whether a field has NaN or Inf values."""
    values = _values(field)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    return {
        "success": bool(not has_nan and not has_inf),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "num_nan": int(np.count_nonzero(np.isnan(values))),
        "num_inf": int(np.count_nonzero(np.isinf(values))),
        "warnings": _finite_warnings(has_nan, has_inf),
    }


def check_shape_consistency(fields: Iterable, target_shape: tuple[int, ...] | None = None) -> dict[str, object]:
    """Check that all fields have the same shape and optionally match target."""
    arrays = [_values(field) for field in fields]
    shapes = [tuple(array.shape) for array in arrays]
    warnings: list[str] = []
    if not shapes:
        warnings.append("no fields provided")
        return {
            "success": False,
            "shape_consistent": False,
            "shapes": [],
            "target_shape": list(target_shape) if target_shape is not None else None,
            "warnings": warnings,
        }
    shape_consistent = all(shape == shapes[0] for shape in shapes)
    if target_shape is not None:
        shape_consistent = shape_consistent and shapes[0] == tuple(target_shape)
    if not shape_consistent:
        warnings.append("field shapes are inconsistent")
    return {
        "success": bool(shape_consistent),
        "shape_consistent": bool(shape_consistent),
        "shapes": [list(shape) for shape in shapes],
        "target_shape": list(target_shape) if target_shape is not None else None,
        "warnings": warnings,
    }


def check_bounds(field, lower: float | None = None, upper: float | None = None, tolerance: float = 1.0e-12) -> dict[str, object]:
    """Count finite values outside optional lower/upper bounds."""
    values = _values(field)
    finite = np.isfinite(values)
    below = np.zeros(values.shape, dtype=bool)
    above = np.zeros(values.shape, dtype=bool)
    if lower is not None:
        below = finite & (values < float(lower) - float(tolerance))
    if upper is not None:
        above = finite & (values > float(upper) + float(tolerance))
    num_below = int(np.count_nonzero(below))
    num_above = int(np.count_nonzero(above))
    return {
        "success": bool(num_below == 0 and num_above == 0 and not np.isnan(values).any() and not np.isinf(values).any()),
        "lower": None if lower is None else float(lower),
        "upper": None if upper is None else float(upper),
        "tolerance": float(tolerance),
        "num_below_lower": num_below,
        "num_above_upper": num_above,
        "num_bound_violations": int(num_below + num_above),
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
        "warnings": [] if num_below + num_above == 0 else ["field values violate requested bounds"],
    }


def compute_weight_statistics(weights) -> dict[str, object]:
    """Return finite and cell-wise sum statistics for fusion weights."""
    values = np.asarray(weights, dtype=float).copy()
    finite = np.isfinite(values)
    warnings: list[str] = []
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        warnings.extend(_finite_warnings(has_nan, has_inf))
    negative = int(np.count_nonzero(finite & (values < 0.0)))
    if negative:
        warnings.append("weights contain negative values")
    finite_values = values[finite]
    if values.ndim >= 2:
        weight_sum = np.sum(values, axis=0)
    else:
        weight_sum = np.asarray([np.sum(values)])
    return {
        "success": bool(not has_nan and not has_inf and negative == 0 and np.any(weight_sum > 0.0)),
        "weight_min": _finite_stat(finite_values, np.min),
        "weight_max": _finite_stat(finite_values, np.max),
        "weight_sum_min": _finite_stat(weight_sum[np.isfinite(weight_sum)], np.min),
        "weight_sum_max": _finite_stat(weight_sum[np.isfinite(weight_sum)], np.max),
        "num_zero_weight_cells": int(np.count_nonzero(np.isfinite(weight_sum) & (weight_sum <= 0.0))),
        "num_negative_weights": negative,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": warnings,
    }


def compute_nan_mask_report(fields: Iterable) -> dict[str, object]:
    """Report source NaNs and cells where every source is NaN."""
    arrays = [_values(field) for field in fields]
    shape_report = check_shape_consistency(arrays)
    if not shape_report["shape_consistent"]:
        return {
            "success": False,
            "shape_consistent": False,
            "num_source_nan_values": 0,
            "num_masked_cells": 0,
            "num_partially_masked_cells": 0,
            "has_nan": False,
            "has_inf": False,
            "warnings": shape_report["warnings"],
        }
    stack = np.stack(arrays, axis=0)
    nan_mask = np.isnan(stack)
    all_nan = np.all(nan_mask, axis=0)
    any_nan = np.any(nan_mask, axis=0)
    return {
        "success": True,
        "shape_consistent": True,
        "num_source_nan_values": int(np.count_nonzero(nan_mask)),
        "num_masked_cells": int(np.count_nonzero(all_nan)),
        "num_partially_masked_cells": int(np.count_nonzero(any_nan & ~all_nan)),
        "has_nan": bool(np.isnan(stack).any()),
        "has_inf": bool(np.isinf(stack).any()),
        "warnings": ["all-source NaN cells require mask/warning"] if np.any(all_nan) else [],
    }


def compute_fusion_error(reference, fused) -> dict[str, object]:
    """Compute MAE, RMSE, and max absolute error between reference and fused."""
    ref = _values(reference)
    out = _values(fused)
    if ref.shape != out.shape:
        return {
            "success": False,
            "shape_consistent": False,
            "mae": None,
            "rmse": None,
            "max_abs_error": None,
            "num_compared": 0,
            "has_nan": bool(np.isnan(ref).any() or np.isnan(out).any()),
            "has_inf": bool(np.isinf(ref).any() or np.isinf(out).any()),
            "warnings": ["reference and fused shapes differ"],
        }
    valid = np.isfinite(ref) & np.isfinite(out)
    if not valid.any():
        return {
            "success": False,
            "shape_consistent": True,
            "mae": None,
            "rmse": None,
            "max_abs_error": None,
            "num_compared": 0,
            "has_nan": bool(np.isnan(ref).any() or np.isnan(out).any()),
            "has_inf": bool(np.isinf(ref).any() or np.isinf(out).any()),
            "warnings": ["no finite overlapping values"],
        }
    diff = out[valid] - ref[valid]
    return {
        "success": bool(not np.isnan(out).any() and not np.isinf(out).any()),
        "shape_consistent": True,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_error": float(np.max(np.abs(diff))),
        "num_compared": int(np.count_nonzero(valid)),
        "has_nan": bool(np.isnan(ref).any() or np.isnan(out).any()),
        "has_inf": bool(np.isinf(ref).any() or np.isinf(out).any()),
        "warnings": [],
    }


def compute_confidence_weighting_metrics(low_conf_field, high_conf_field, fused) -> dict[str, object]:
    """Measure whether fused values are closer to the high-confidence source."""
    low = _values(low_conf_field)
    high = _values(high_conf_field)
    out = _values(fused)
    shape_report = check_shape_consistency([low, high, out])
    if not shape_report["shape_consistent"]:
        return {
            "success": False,
            "shape_consistent": False,
            "mean_distance_to_low_confidence": None,
            "mean_distance_to_high_confidence": None,
            "closer_to_high_confidence": False,
            "high_confidence_influence_ratio": None,
            "warnings": shape_report["warnings"],
        }
    valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(out)
    low_dist = float(np.mean(np.abs(out[valid] - low[valid]))) if valid.any() else None
    high_dist = float(np.mean(np.abs(out[valid] - high[valid]))) if valid.any() else None
    ratio = None
    if low_dist is not None and high_dist is not None and high_dist > 0.0:
        ratio = float(low_dist / high_dist)
    return {
        "success": bool(valid.any() and high_dist is not None and low_dist is not None and high_dist < low_dist),
        "shape_consistent": True,
        "mean_distance_to_low_confidence": low_dist,
        "mean_distance_to_high_confidence": high_dist,
        "closer_to_high_confidence": bool(high_dist is not None and low_dist is not None and high_dist < low_dist),
        "high_confidence_influence_ratio": ratio,
        "warnings": [],
    }


def build_fusion_diagnostics_report(
    field,
    *,
    reference=None,
    weights=None,
    fields: Iterable | None = None,
    lower: float | None = None,
    upper: float | None = None,
    target_shape: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Build a compact JSON-safe diagnostics report for a fused field."""
    stats = compute_field_statistics(field)
    finite = check_field_finite(field)
    bounds = check_bounds(field, lower=lower, upper=upper)
    shape = check_shape_consistency([field] if fields is None else list(fields), target_shape=target_shape)
    weight_stats = compute_weight_statistics(np.asarray([1.0])) if weights is None else compute_weight_statistics(weights)
    nan_report = compute_nan_mask_report([field] if fields is None else list(fields))
    if reference is None:
        error = {"mae": None, "rmse": None, "max_abs_error": None}
    else:
        error = compute_fusion_error(reference, field)
    warnings = (
        list(stats["warnings"])
        + list(finite["warnings"])
        + list(bounds["warnings"])
        + list(shape["warnings"])
        + list(weight_stats["warnings"])
        + list(nan_report["warnings"])
        + list(error.get("warnings", []))
    )
    return {
        "success": bool(
            finite["success"]
            and bounds["success"]
            and shape["shape_consistent"]
            and weight_stats["success"]
        ),
        "field_min": stats["field_min"],
        "field_max": stats["field_max"],
        "field_mean": stats["field_mean"],
        "field_std": stats["field_std"],
        "has_nan": bool(stats["has_nan"]),
        "has_inf": bool(stats["has_inf"]),
        "num_nan": int(stats["num_nan"]),
        "num_inf": int(stats["num_inf"]),
        "num_below_lower": int(bounds["num_below_lower"]),
        "num_above_upper": int(bounds["num_above_upper"]),
        "shape_consistent": bool(shape["shape_consistent"]),
        "weight_min": weight_stats["weight_min"],
        "weight_max": weight_stats["weight_max"],
        "weight_sum_min": weight_stats["weight_sum_min"],
        "weight_sum_max": weight_stats["weight_sum_max"],
        "num_zero_weight_cells": int(weight_stats["num_zero_weight_cells"]),
        "num_masked_cells": int(nan_report["num_masked_cells"]),
        "mae": error.get("mae"),
        "rmse": error.get("rmse"),
        "max_abs_error": error.get("max_abs_error"),
        "warnings": warnings,
    }


def _values(field) -> np.ndarray:
    if isinstance(field, Field3D):
        return np.asarray(field.values, dtype=float).copy()
    return np.asarray(field, dtype=float).copy()


def _finite_stat(values: np.ndarray, func) -> float | None:
    if values.size == 0:
        return None
    return float(func(values))


def _finite_warnings(has_nan: bool, has_inf: bool) -> list[str]:
    warnings: list[str] = []
    if has_nan:
        warnings.append("values contain NaN")
    if has_inf:
        warnings.append("values contain Inf")
    return warnings
