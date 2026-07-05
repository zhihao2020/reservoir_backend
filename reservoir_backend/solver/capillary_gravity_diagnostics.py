"""Diagnostics for capillary, gravity, and combined water transport.

These helpers inspect arrays produced by the existing capillary, gravity, and
combined transport modules. They do not modify saturation or flux arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D


def compute_gradient_norm(field: Field3D | ArrayLike) -> float:
    """Return an L2-like norm of finite differences over all array axes."""
    values = _values(field)
    if np.isnan(values).any() or np.isinf(values).any():
        return float("nan")
    total = 0.0
    for axis in range(values.ndim):
        if values.shape[axis] > 1:
            diff = np.diff(values, axis=axis)
            total += float(np.sum(diff * diff))
    return float(np.sqrt(total))


def compute_flux_statistics(flux: Field3D | ArrayLike) -> dict:
    """Return finite summary statistics for one flux component."""
    values = _values(flux)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "min_flux": None,
            "max_flux": None,
            "mean_flux": None,
            "mean_abs_flux": None,
            "max_abs_flux": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["flux contains NaN or Inf"],
        }
    return {
        "min_flux": float(np.min(values)),
        "max_flux": float(np.max(values)),
        "mean_flux": float(np.mean(values)),
        "mean_abs_flux": float(np.mean(np.abs(values))),
        "max_abs_flux": float(np.max(np.abs(values))),
        "has_nan": False,
        "has_inf": False,
        "warnings": [],
    }


def check_flux_finite(flux: Field3D | ArrayLike) -> dict:
    """Report whether a flux array contains NaN or Inf."""
    values = _values(flux)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    return {"finite": not (has_nan or has_inf), "has_nan": has_nan, "has_inf": has_inf}


def check_expected_flux_sign(flux_component: Field3D | ArrayLike, expected_sign: int) -> dict:
    """Check nonzero flux signs against ``expected_sign`` (-1, 0, or 1)."""
    expected = int(expected_sign)
    if expected not in {-1, 0, 1}:
        raise ValueError("expected_sign must be -1, 0, or 1")
    values = _values(flux_component)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "expected_sign": expected,
            "observed_sign": None,
            "sign_matches_expectation": False,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["cannot check sign for flux containing NaN or Inf"],
        }
    nonzero = values[np.abs(values) > 0.0]
    if nonzero.size == 0:
        observed = 0
        matches = expected == 0
    else:
        signs = np.sign(nonzero)
        observed = int(np.sign(np.mean(signs)))
        matches = bool(np.all(signs == expected)) if expected != 0 else False
    return {
        "expected_sign": expected,
        "observed_sign": observed,
        "sign_matches_expectation": matches,
        "has_nan": False,
        "has_inf": False,
        "warnings": [] if matches else ["flux sign does not match expectation"],
    }


def compute_capillary_smoothing_metrics(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
) -> dict:
    """Return gradient-reduction metrics for capillary smoothing checks."""
    initial = _values(initial_saturation)
    final = _values(final_saturation)
    if initial.shape != final.shape:
        raise ValueError("initial_saturation and final_saturation must have matching shapes")
    initial_norm = compute_gradient_norm(initial)
    final_norm = compute_gradient_norm(final)
    return {
        "initial_gradient_norm": float(initial_norm),
        "final_gradient_norm": float(final_norm),
        "gradient_reduction": float(initial_norm - final_norm),
        "has_nan": bool(np.isnan(initial).any() or np.isnan(final).any()),
        "has_inf": bool(np.isinf(initial).any() or np.isinf(final).any()),
    }


def compute_gravity_segregation_metrics(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
    vertical_axis: int = 2,
) -> dict:
    """Return top/bottom saturation changes along a NumPy vertical axis.

    ``vertical_axis`` is the axis index in the supplied NumPy array. Project
    `Grid3D` saturation fields are stored as `(nz, ny, nx)`, so callers should
    pass `vertical_axis=0` for grid-aligned z direction.
    """
    initial = _values(initial_saturation)
    final = _values(final_saturation)
    if initial.shape != final.shape:
        raise ValueError("initial_saturation and final_saturation must have matching shapes")
    axis = int(vertical_axis)
    top_initial = np.take(initial, 0, axis=axis)
    top_final = np.take(final, 0, axis=axis)
    bottom_initial = np.take(initial, initial.shape[axis] - 1, axis=axis)
    bottom_final = np.take(final, final.shape[axis] - 1, axis=axis)
    return {
        "top_saturation_change": float(np.mean(top_final - top_initial)),
        "bottom_saturation_change": float(np.mean(bottom_final - bottom_initial)),
        "vertical_axis": axis,
        "has_nan": bool(np.isnan(initial).any() or np.isnan(final).any()),
        "has_inf": bool(np.isinf(initial).any() or np.isinf(final).any()),
    }


def compute_combined_transport_metrics(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
    capillary_flux: ArrayLike | tuple[ArrayLike, ...] | None = None,
    gravity_flux: ArrayLike | tuple[ArrayLike, ...] | None = None,
) -> dict:
    """Return combined transport trend and flux-magnitude metrics."""
    smoothing = compute_capillary_smoothing_metrics(initial_saturation, final_saturation)
    values = _values(final_saturation)
    max_cap = _max_abs_flux(capillary_flux)
    max_grav = _max_abs_flux(gravity_flux)
    return {
        "initial_gradient_norm": smoothing["initial_gradient_norm"],
        "final_gradient_norm": smoothing["final_gradient_norm"],
        "gradient_reduction": smoothing["gradient_reduction"],
        "max_abs_capillary_flux": max_cap,
        "max_abs_gravity_flux": max_grav,
        "saturation_min": float(np.nanmin(values)),
        "saturation_max": float(np.nanmax(values)),
        "has_nan": bool(smoothing["has_nan"] or np.isnan(values).any()),
        "has_inf": bool(smoothing["has_inf"] or np.isinf(values).any()),
    }


def build_capillary_gravity_diagnostics_report(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
    *,
    capillary_flux: ArrayLike | tuple[ArrayLike, ...] | None = None,
    gravity_flux: ArrayLike | tuple[ArrayLike, ...] | None = None,
    expected_gravity_flux_sign: int | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    vertical_axis: int = 0,
) -> dict:
    """Build a JSON-serializable capillary/gravity diagnostics report."""
    initial = _values(initial_saturation)
    final = _values(final_saturation)
    combined = compute_combined_transport_metrics(initial, final, capillary_flux, gravity_flux)
    gravity = compute_gravity_segregation_metrics(initial, final, vertical_axis=vertical_axis)
    if gravity_flux is None or expected_gravity_flux_sign is None:
        sign = {
            "expected_sign": expected_gravity_flux_sign,
            "observed_sign": None,
            "sign_matches_expectation": None,
            "warnings": [],
        }
    else:
        sign_flux = gravity_flux[-1] if isinstance(gravity_flux, tuple) else gravity_flux
        sign = check_expected_flux_sign(sign_flux, expected_gravity_flux_sign)
    violations = int(np.count_nonzero((final < lower) | (final > upper)))
    warnings = []
    warnings.extend(sign.get("warnings", []))
    has_nan = bool(combined["has_nan"])
    has_inf = bool(combined["has_inf"])
    success = bool(violations == 0 and not has_nan and not has_inf)
    return {
        "success": success,
        "initial_gradient_norm": combined["initial_gradient_norm"],
        "final_gradient_norm": combined["final_gradient_norm"],
        "gradient_reduction": combined["gradient_reduction"],
        "max_abs_capillary_flux": combined["max_abs_capillary_flux"],
        "max_abs_gravity_flux": combined["max_abs_gravity_flux"],
        "expected_gravity_flux_sign": sign["expected_sign"],
        "observed_gravity_flux_sign": sign["observed_sign"],
        "top_saturation_change": gravity["top_saturation_change"],
        "bottom_saturation_change": gravity["bottom_saturation_change"],
        "saturation_min": float(np.nanmin(final)),
        "saturation_max": float(np.nanmax(final)),
        "num_bound_violations": violations,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": warnings,
    }


def _values(value: Field3D | ArrayLike) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        return np.asarray(value.values, dtype=float)
    return np.asarray(value, dtype=float)


def _max_abs_flux(value: ArrayLike | tuple[ArrayLike, ...] | None) -> float:
    if value is None:
        return 0.0
    arrays = value if isinstance(value, tuple) else (value,)
    return float(max(np.max(np.abs(np.asarray(array, dtype=float))) for array in arrays))
