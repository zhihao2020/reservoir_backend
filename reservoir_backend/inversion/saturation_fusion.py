"""Utilities for fusing saturation estimates from multiple inversion sources."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError


def fuse_saturation_estimates(
    estimates: dict[str, float | ArrayLike],
    weights: dict[str, float] | None = None,
    uncertainties: dict[str, float] | None = None,
    confidence: dict[str, float] | None = None,
    clip: bool = True,
    return_report: bool = False,
) -> float | NDArray[np.float64] | tuple[float | NDArray[np.float64], dict]:
    """Fuse saturation estimates with uncertainty, confidence, user, or equal weights."""
    if not estimates:
        raise InvalidPhysicalValueError("at least one saturation estimate is required")

    arrays: dict[str, NDArray[np.float64]] = {}
    dropped: list[str] = []
    warnings: list[str] = []
    target_shape: tuple[int, ...] | None = None
    for name, value in estimates.items():
        arr = np.asarray(value, dtype=float)
        if target_shape is None:
            target_shape = arr.shape
        elif arr.shape != target_shape:
            raise InvalidPhysicalValueError("all saturation estimates must have the same shape")
        if (~np.isfinite(arr)).any():
            dropped.append(name)
            warnings.append(f"dropped {name}: non-finite estimate")
            continue
        arrays[name] = arr

    if not arrays:
        raise InvalidPhysicalValueError("no valid saturation estimates remain after validation")

    mode, raw_weights = _select_weights(arrays.keys(), weights, uncertainties, confidence)
    total = sum(raw_weights.values())
    if total <= 0.0:
        raise InvalidPhysicalValueError("total fusion weight must be positive")
    normalized = {name: value / total for name, value in raw_weights.items()}

    fused = np.zeros(next(iter(arrays.values())).shape, dtype=float)
    for name, arr in arrays.items():
        fused += normalized[name] * arr

    raw = fused.copy()
    if clip:
        fused = np.clip(fused, 0.0, 1.0)

    report = {
        "method": "saturation_estimate_fusion",
        "success": True,
        "fusion_mode": mode,
        "used_signals": list(arrays.keys()),
        "dropped_signals": dropped,
        "normalized_weights": normalized,
        "saturation": _to_scalar_if_needed(fused),
        "saturation_min": float(np.min(fused)),
        "saturation_max": float(np.max(fused)),
        "num_clipped_low": int(np.sum(raw < 0.0)),
        "num_clipped_high": int(np.sum(raw > 1.0)),
        "warnings": warnings,
        "has_nan": bool(np.isnan(fused).any()),
        "has_inf": bool(np.isinf(fused).any()),
    }
    result = _to_scalar_if_needed(fused)
    if return_report:
        return result, report
    return result


def _select_weights(
    names: ArrayLike,
    weights: dict[str, float] | None,
    uncertainties: dict[str, float] | None,
    confidence: dict[str, float] | None,
) -> tuple[str, dict[str, float]]:
    names = list(names)
    if uncertainties is not None:
        selected = {}
        for name in names:
            sigma = float(uncertainties.get(name, np.inf))
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise InvalidPhysicalValueError("uncertainties must be positive and finite")
            selected[name] = 1.0 / (sigma * sigma)
        return "uncertainty_inverse_variance", selected

    if confidence is not None:
        selected = {name: float(confidence.get(name, 0.0)) for name in names}
        _validate_nonnegative_weights(selected, "confidence")
        return "confidence", selected

    if weights is not None:
        selected = {name: float(weights.get(name, 0.0)) for name in names}
        _validate_nonnegative_weights(selected, "weights")
        return "user_weights", selected

    return "equal_weights", {name: 1.0 for name in names}


def _validate_nonnegative_weights(values: dict[str, float], label: str) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value < 0.0:
            raise InvalidPhysicalValueError(f"{label} for {name} must be non-negative and finite")


def _to_scalar_if_needed(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr
