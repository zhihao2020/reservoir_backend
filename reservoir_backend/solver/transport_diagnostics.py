"""Diagnostics for saturation transport enhancement experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_total_variation(saturation: ArrayLike) -> float:
    """Return 1D total variation after flattening non-x axes by averaging."""
    line = _line(saturation)
    if line.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(line))))


def estimate_front_position(saturation: ArrayLike, threshold: float = 0.5, dx: float = 1.0) -> float | None:
    """Return farthest downstream x-position where saturation exceeds threshold."""
    line = _line(saturation)
    if not np.isfinite(line).all():
        return None
    wet = np.flatnonzero(line >= float(threshold))
    if wet.size == 0:
        return None
    return float((int(wet[-1]) + 0.5) * float(dx))


def compute_front_sharpness(saturation: ArrayLike, dx: float = 1.0) -> float:
    """Return max absolute x-gradient magnitude as a front-sharpness proxy."""
    line = _line(saturation)
    if line.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(line))) / float(dx))


def compute_overshoot_undershoot(saturation: ArrayLike, lower: float = 0.0, upper: float = 1.0) -> dict:
    """Report overshoot and undershoot relative to physical bounds."""
    values = np.asarray(saturation, dtype=float)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "overshoot": None,
            "undershoot": None,
            "num_overshoot_cells": None,
            "num_undershoot_cells": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
    overshoot = np.maximum(values - float(upper), 0.0)
    undershoot = np.maximum(float(lower) - values, 0.0)
    return {
        "overshoot": float(np.max(overshoot)),
        "undershoot": float(np.max(undershoot)),
        "num_overshoot_cells": int(np.count_nonzero(overshoot > 0.0)),
        "num_undershoot_cells": int(np.count_nonzero(undershoot > 0.0)),
        "has_nan": False,
        "has_inf": False,
    }


def build_boundedness_diagnostics(
    saturation: ArrayLike,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1.0e-12,
) -> dict:
    """Build JSON-serializable boundedness diagnostics."""
    values = np.asarray(saturation, dtype=float)
    over = compute_overshoot_undershoot(values, lower=lower, upper=upper)
    if over["has_nan"] or over["has_inf"]:
        return {
            "boundedness_passed": False,
            "saturation_min": None,
            "saturation_max": None,
            "num_clipped_cells": None,
            **over,
            "warnings": ["saturation contains NaN or Inf"],
        }
    below = values < float(lower) - float(tolerance)
    above = values > float(upper) + float(tolerance)
    return {
        "boundedness_passed": bool(not below.any() and not above.any()),
        "saturation_min": float(np.min(values)),
        "saturation_max": float(np.max(values)),
        "num_clipped_cells": int(np.count_nonzero(below | above)),
        **over,
        "warnings": [] if not below.any() and not above.any() else ["saturation bounds violation detected"],
    }


def build_transport_diagnostics(
    initial_saturation: ArrayLike,
    final_saturation: ArrayLike,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    threshold: float = 0.5,
    dx: float = 1.0,
    max_cfl: float | None = None,
    material_balance_error: float | None = None,
) -> dict:
    """Build transport diagnostics for upwind/TVD comparison."""
    initial = np.asarray(initial_saturation, dtype=float)
    final = np.asarray(final_saturation, dtype=float)
    if initial.shape != final.shape:
        raise ValueError("initial_saturation and final_saturation must have matching shapes")
    bounded = build_boundedness_diagnostics(final, lower=lower, upper=upper)
    return {
        "success": bool(bounded["boundedness_passed"] and not bounded["has_nan"] and not bounded["has_inf"]),
        "front_position": estimate_front_position(final, threshold=threshold, dx=dx),
        "initial_front_position": estimate_front_position(initial, threshold=threshold, dx=dx),
        "front_sharpness": compute_front_sharpness(final, dx=dx),
        "initial_front_sharpness": compute_front_sharpness(initial, dx=dx),
        "total_variation": compute_total_variation(final),
        "initial_total_variation": compute_total_variation(initial),
        "mass_balance_error": None if material_balance_error is None else float(material_balance_error),
        "max_cfl": None if max_cfl is None else float(max_cfl),
        **bounded,
    }


def _line(saturation: ArrayLike) -> np.ndarray:
    values = np.asarray(saturation, dtype=float)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.mean(values, axis=0)
    if values.ndim == 3:
        return np.mean(values, axis=(0, 1))
    return values.reshape(-1)
