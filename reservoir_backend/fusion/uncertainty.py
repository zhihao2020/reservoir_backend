"""Uncertainty-aware field fusion utilities.

This module adds variance/std/confidence-aware weighting without changing the
existing IDW or confidence fusion baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D


EPSILON_VARIANCE = 1.0e-12


def uncertainty_weighted_fusion(
    fields: list[ArrayLike | Field3D],
    *,
    variances: list[ArrayLike | Field3D] | None = None,
    stds: list[ArrayLike | Field3D] | None = None,
    confidences: list[ArrayLike | Field3D] | None = None,
    weights: list[float] | None = None,
    mask: ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Fuse same-shape fields and return `(mean, variance, report)`.

    Weighting priority is variance, standard deviation, confidence, explicit
    weights, then equal weights. NaN field values are ignored cell-wise.
    """
    arrays = [_array(field) for field in fields]
    if not arrays:
        raise ValueError("fields must not be empty")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all fields must have matching shapes")
    stack = np.stack(arrays, axis=0)
    policy, weight_stack, warnings = _build_weights(
        shape,
        len(arrays),
        variances=variances,
        stds=stds,
        confidences=confidences,
        weights=weights,
    )
    valid = np.isfinite(stack)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != shape:
            raise ValueError("mask shape must match field shape")
        valid &= mask_array.reshape((1,) + shape)
    else:
        mask_array = np.ones(shape, dtype=bool)
    effective = np.where(valid, weight_stack, 0.0)
    total_weight = np.sum(effective, axis=0)
    numerator = np.nansum(np.where(valid, stack * effective, 0.0), axis=0)
    fused = np.divide(numerator, total_weight, out=np.full(shape, np.nan), where=total_weight > 0.0)
    fused_variance = np.divide(1.0, total_weight, out=np.full(shape, np.nan), where=total_weight > 0.0)
    masked_cells = int(np.count_nonzero(~mask_array | (total_weight <= 0.0)))
    if np.isnan(fused).any():
        warnings.append("some cells have no valid source value")
    clipped_cells = 0
    if bounds is not None:
        lower, upper = float(bounds[0]), float(bounds[1])
        if lower > upper:
            raise ValueError("bounds lower must be <= upper")
        finite_fused = np.isfinite(fused)
        clipped_cells = int(np.count_nonzero(finite_fused & ((fused < lower) | (fused > upper))))
        source_violations = int(np.count_nonzero(np.isfinite(stack) & ((stack < lower) | (stack > upper))))
        clipped_cells = max(clipped_cells, source_violations)
        fused = np.clip(fused, lower, upper)
    dominant_source = _dominant_source(effective)
    report = {
        "success": bool(not np.isinf(fused).any() and not np.isinf(fused_variance).any()),
        "weighting_policy": policy,
        "dominant_source": dominant_source,
        "fallback_used": policy in {"explicit_weight", "equal_weight"},
        "num_sources": len(arrays),
        "num_nan": int(np.count_nonzero(np.isnan(stack))),
        "num_inf": int(np.count_nonzero(np.isinf(stack))),
        "num_masked_cells": masked_cells,
        "num_clipped_cells": clipped_cells,
        "bounds_violations": clipped_cells,
        "variance_min": _finite_min(fused_variance),
        "variance_max": _finite_max(fused_variance),
        "variance_mean": _finite_mean(fused_variance),
        "uncertainty_nonnegative": bool(np.nanmin(fused_variance) >= 0.0) if np.isfinite(fused_variance).any() else False,
        "has_nan": bool(np.isnan(fused).any() or np.isnan(fused_variance).any()),
        "has_inf": bool(np.isinf(fused).any() or np.isinf(fused_variance).any()),
        "warnings": warnings,
    }
    return fused, fused_variance, report


def deferred_ensemble_update(method: str) -> dict[str, object]:
    """Return a deferred warning for EnKF / ES-MDA style requests."""
    normalized = method.lower().replace("_", "-")
    if normalized not in {"enkf", "en-kf", "esmda", "es-mda"}:
        raise ValueError("method must be EnKF or ES-MDA")
    return {
        "success": True,
        "method_requested": method,
        "method_used": "deferred",
        "deferred": True,
        "fallback_used": True,
        "warnings": [f"{method} is deferred; no history matching or ensemble update was performed"],
    }


def _build_weights(
    shape: tuple[int, ...],
    count: int,
    *,
    variances: list[ArrayLike | Field3D] | None,
    stds: list[ArrayLike | Field3D] | None,
    confidences: list[ArrayLike | Field3D] | None,
    weights: list[float] | None,
) -> tuple[str, NDArray[np.float64], list[str]]:
    warnings: list[str] = []
    if variances is not None:
        arrays = [_uncertainty_array(v, shape, "variance") for v in variances]
        _validate_count(arrays, count, "variances")
        stack = np.stack(arrays, axis=0)
        zero_count = int(np.count_nonzero(stack == 0.0))
        if zero_count:
            warnings.append("zero variance values were floored for numerical stability")
        return "variance", 1.0 / np.maximum(stack, EPSILON_VARIANCE), warnings
    if stds is not None:
        arrays = [_uncertainty_array(v, shape, "std") for v in stds]
        _validate_count(arrays, count, "stds")
        stack = np.stack(arrays, axis=0)
        zero_count = int(np.count_nonzero(stack == 0.0))
        if zero_count:
            warnings.append("zero standard deviation values were floored for numerical stability")
        return "std", 1.0 / np.maximum(stack, EPSILON_VARIANCE) ** 2, warnings
    if confidences is not None:
        arrays = [_confidence_array(v, shape) for v in confidences]
        _validate_count(arrays, count, "confidences")
        return "confidence", np.stack(arrays, axis=0), warnings
    if weights is not None:
        values = np.asarray(weights, dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all() or (values < 0.0).any() or np.sum(values) <= 0.0:
            raise ValueError("weights must be finite, nonnegative, and match field count")
        return "explicit_weight", values.reshape((count,) + (1,) * len(shape)) * np.ones((count,) + shape), warnings
    return "equal_weight", np.ones((count,) + shape, dtype=float), warnings


def _array(value: ArrayLike | Field3D) -> NDArray[np.float64]:
    array = np.asarray(value.values if isinstance(value, Field3D) else value, dtype=float)
    return array.copy()


def _uncertainty_array(value: ArrayLike | Field3D, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = _array(value)
    if array.shape == ():
        array = np.full(shape, float(array), dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape must match field shape")
    if np.isnan(array).any() or np.isinf(array).any():
        raise ValueError(f"{name} must not contain NaN or Inf")
    if (array < 0.0).any():
        raise ValueError(f"{name} must be nonnegative")
    return array


def _confidence_array(value: ArrayLike | Field3D, shape: tuple[int, ...]) -> NDArray[np.float64]:
    array = _array(value)
    if array.shape == ():
        array = np.full(shape, float(array), dtype=float)
    if array.shape != shape:
        raise ValueError("confidence shape must match field shape")
    if np.isnan(array).any() or np.isinf(array).any() or (array < 0.0).any() or (array > 1.0).any():
        raise ValueError("confidence must be finite and in [0, 1]")
    return array


def _validate_count(arrays: list[NDArray[np.float64]], count: int, name: str) -> None:
    if len(arrays) != count:
        raise ValueError(f"{name} length must match fields length")


def _dominant_source(weights: NDArray[np.float64]) -> int | None:
    totals = np.sum(weights, axis=tuple(range(1, weights.ndim)))
    if totals.size == 0 or not np.isfinite(totals).any() or np.max(totals) <= 0.0:
        return None
    return int(np.argmax(totals))


def _finite_min(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.min(finite))


def _finite_max(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


def _finite_mean(values: NDArray[np.float64]) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.mean(finite))
