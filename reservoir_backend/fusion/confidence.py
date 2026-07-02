"""Confidence field utilities for lightweight field fusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D


def normalize_confidence(confidence: float | ArrayLike | Field3D) -> float | NDArray[np.float64] | Field3D:
    """Normalize confidence values to `[0, 1]`.

    Values already in `[0, 1]` are preserved. If the maximum finite value is
    greater than 1, all values are divided by that maximum before clipping.
    """
    if isinstance(confidence, Field3D):
        values = _normalize_array(confidence.values)
        return Field3D(confidence.grid, values, name=confidence.name, unit="fraction")
    values = _normalize_array(np.asarray(confidence, dtype=float))
    if values.shape == ():
        return float(values)
    return values


def combine_confidence(
    confidence_fields: list[Field3D | ArrayLike],
    weights: list[float] | None = None,
) -> NDArray[np.float64]:
    """Combine confidence fields with optional source weights."""
    if not confidence_fields:
        raise ValueError("confidence_fields must not be empty")
    normalized = []
    for confidence in confidence_fields:
        values = normalize_confidence(confidence)
        if isinstance(values, Field3D):
            values = values.values
        normalized.append(np.asarray(values, dtype=float))

    if weights is None:
        weight_values = np.ones(len(normalized), dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float)
        if weight_values.shape != (len(normalized),) or (weight_values < 0.0).any():
            raise ValueError("weights must be non-negative and match confidence count")

    stacked = np.stack(normalized, axis=0)
    weighted = stacked * weight_values.reshape((-1,) + (1,) * stacked[0].ndim)
    denominator = np.sum(weight_values)
    if denominator <= 0.0:
        raise ValueError("total confidence weight must be positive")
    return np.clip(np.sum(weighted, axis=0) / denominator, 0.0, 1.0)


def confidence_from_error(error: float | ArrayLike, scale: float) -> float | NDArray[np.float64]:
    """Convert error magnitude to confidence using `exp(-abs(error) / scale)`."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    values = np.exp(-np.abs(np.asarray(error, dtype=float)) / scale)
    if values.shape == ():
        return float(values)
    return values


def confidence_from_distance(distance: float | ArrayLike, range_scale: float) -> float | NDArray[np.float64]:
    """Convert distance to confidence using `exp(-distance / range_scale)`."""
    range_scale = float(range_scale)
    if not np.isfinite(range_scale) or range_scale <= 0.0:
        raise ValueError("range_scale must be positive and finite")
    values = np.exp(-np.maximum(np.asarray(distance, dtype=float), 0.0) / range_scale)
    if values.shape == ():
        return float(values)
    return values


def _normalize_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("confidence values must be finite")
    if values.size == 0:
        return values.astype(float)
    scale = max(float(np.max(values)), 1.0)
    return np.clip(values / scale, 0.0, 1.0)
