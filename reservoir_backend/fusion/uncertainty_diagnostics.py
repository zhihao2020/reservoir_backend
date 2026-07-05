"""Diagnostics for uncertainty-aware parameter fusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_uncertainty_statistics(variance: ArrayLike) -> dict[str, object]:
    """Return JSON-safe variance / uncertainty statistics."""
    values = np.asarray(variance, dtype=float)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    finite = values[np.isfinite(values)]
    negative = int(np.count_nonzero(finite < 0.0))
    return {
        "variance_min": None if finite.size == 0 else float(np.min(finite)),
        "variance_max": None if finite.size == 0 else float(np.max(finite)),
        "variance_mean": None if finite.size == 0 else float(np.mean(finite)),
        "uncertainty_nonnegative": bool(negative == 0 and finite.size > 0),
        "num_negative_variance": negative,
        "num_nan": int(np.count_nonzero(np.isnan(values))),
        "num_inf": int(np.count_nonzero(np.isinf(values))),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": [] if negative == 0 and not has_nan and not has_inf else ["invalid uncertainty values detected"],
    }


def compute_confidence_range(confidence: ArrayLike | None) -> dict[str, object]:
    """Return confidence range diagnostics."""
    if confidence is None:
        return {
            "confidence_min": None,
            "confidence_max": None,
            "confidence_range": None,
            "confidence_valid": None,
            "warnings": ["confidence field not provided"],
        }
    values = np.asarray(confidence, dtype=float)
    valid = bool(np.isfinite(values).all() and (values >= 0.0).all() and (values <= 1.0).all())
    return {
        "confidence_min": float(np.nanmin(values)),
        "confidence_max": float(np.nanmax(values)),
        "confidence_range": [float(np.nanmin(values)), float(np.nanmax(values))],
        "confidence_valid": valid,
        "warnings": [] if valid else ["confidence must be finite and in [0, 1]"],
    }


def build_uncertainty_diagnostics_report(
    fused_field: ArrayLike,
    variance: ArrayLike,
    *,
    confidence: ArrayLike | None = None,
    mask: ArrayLike | None = None,
    bounds: tuple[float, float] | None = None,
    dominant_source: int | None = None,
    weighting_policy: str | None = None,
    fallback_used: bool = False,
) -> dict[str, object]:
    """Build a compact diagnostics report for uncertainty-aware fusion."""
    field = np.asarray(fused_field, dtype=float)
    var_stats = compute_uncertainty_statistics(variance)
    conf = compute_confidence_range(confidence)
    if mask is None:
        num_masked = int(np.count_nonzero(np.isnan(field)))
    else:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != field.shape:
            raise ValueError("mask shape must match fused field")
        num_masked = int(np.count_nonzero(~mask_array))
    violations = 0
    if bounds is not None:
        lower, upper = float(bounds[0]), float(bounds[1])
        finite = np.isfinite(field)
        violations = int(np.count_nonzero(finite & ((field < lower) | (field > upper))))
    warnings = list(var_stats["warnings"]) + list(conf["warnings"])
    if violations:
        warnings.append("bounds violations detected")
    return {
        "success": bool(not var_stats["has_inf"] and var_stats["uncertainty_nonnegative"] and violations == 0),
        "variance_min": var_stats["variance_min"],
        "variance_max": var_stats["variance_max"],
        "variance_mean": var_stats["variance_mean"],
        "uncertainty_nonnegative": var_stats["uncertainty_nonnegative"],
        "confidence_range": conf["confidence_range"],
        "num_nan": int(np.count_nonzero(np.isnan(field)) + var_stats["num_nan"]),
        "num_inf": int(np.count_nonzero(np.isinf(field)) + var_stats["num_inf"]),
        "num_masked_cells": num_masked,
        "bounds_violations": violations,
        "dominant_source": dominant_source,
        "weighting_policy": weighting_policy,
        "fallback_used": bool(fallback_used),
        "has_nan": bool(np.isnan(field).any() or var_stats["has_nan"]),
        "has_inf": bool(np.isinf(field).any() or var_stats["has_inf"]),
        "warnings": warnings,
    }
