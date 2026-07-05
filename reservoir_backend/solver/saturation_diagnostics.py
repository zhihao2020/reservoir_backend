"""Saturation transport diagnostic utilities.

These helpers inspect saturation arrays and transport reports produced by the
existing solvers. They do not update saturation, flux, or solver state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D


def compute_saturation_statistics(saturation: Field3D | ArrayLike) -> dict:
    """Return summary statistics for a saturation field."""
    values = _values(saturation)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "saturation_min": None,
            "saturation_max": None,
            "saturation_mean": None,
            "saturation_std": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["saturation contains NaN or Inf"],
        }
    return {
        "saturation_min": float(np.min(values)),
        "saturation_max": float(np.max(values)),
        "saturation_mean": float(np.mean(values)),
        "saturation_std": float(np.std(values)),
        "has_nan": False,
        "has_inf": False,
        "warnings": [],
    }


def check_saturation_finite(saturation: Field3D | ArrayLike) -> dict:
    """Report whether a saturation field contains NaN or Inf."""
    values = _values(saturation)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    return {"finite": not (has_nan or has_inf), "has_nan": has_nan, "has_inf": has_inf}


def check_saturation_bounds(
    saturation: Field3D | ArrayLike,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1.0e-12,
) -> dict:
    """Check saturation boundedness with a small numerical tolerance."""
    values = _values(saturation)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "bounded": False,
            "num_below_lower": None,
            "num_above_upper": None,
            "lower": float(lower),
            "upper": float(upper),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["cannot check bounds for saturation containing NaN or Inf"],
        }
    below = values < float(lower) - float(tolerance)
    above = values > float(upper) + float(tolerance)
    num_below = int(np.count_nonzero(below))
    num_above = int(np.count_nonzero(above))
    return {
        "bounded": bool(num_below == 0 and num_above == 0),
        "num_below_lower": num_below,
        "num_above_upper": num_above,
        "lower": float(lower),
        "upper": float(upper),
        "has_nan": False,
        "has_inf": False,
        "warnings": [] if num_below == 0 and num_above == 0 else ["saturation bounds violation detected"],
    }


def estimate_front_position_1d(
    saturation: Field3D | ArrayLike,
    threshold: float = 0.5,
    dx: float = 1.0,
) -> float | None:
    """Estimate the farthest downstream x-position above a threshold.

    For 2D/3D inputs, the field is averaged over non-x axes first. If no cell
    exceeds the threshold, ``None`` is returned.
    """
    values = _values(saturation)
    if np.isnan(values).any() or np.isinf(values).any():
        return None
    if values.ndim == 1:
        line = values
    elif values.ndim == 2:
        line = np.mean(values, axis=0)
    elif values.ndim == 3:
        line = np.mean(values, axis=(0, 1))
    else:
        line = values.reshape(-1)
    wet = np.flatnonzero(line >= float(threshold))
    if wet.size == 0:
        return None
    return float((int(wet[-1]) + 0.5) * float(dx))


def compute_saturation_change_norm(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
) -> dict:
    """Return L1 and L2 norms of saturation change."""
    initial = _values(initial_saturation)
    final = _values(final_saturation)
    if initial.shape != final.shape:
        raise ValueError("initial_saturation and final_saturation must have matching shapes")
    diff = final - initial
    return {
        "saturation_change_l1": float(np.sum(np.abs(diff))),
        "saturation_change_l2": float(np.linalg.norm(diff.ravel())),
        "max_abs_saturation_change": float(np.max(np.abs(diff))),
        "has_nan": bool(np.isnan(diff).any()),
        "has_inf": bool(np.isinf(diff).any()),
    }


def compute_material_balance_error(
    initial_saturation: Field3D | ArrayLike,
    final_saturation: Field3D | ArrayLike,
    injected_volume: float | None = None,
    produced_volume: float | None = None,
    pore_volume: float | ArrayLike | None = None,
) -> dict:
    """Compute approximate material-balance residual from storage and flows."""
    initial = _values(initial_saturation)
    final = _values(final_saturation)
    if initial.shape != final.shape:
        raise ValueError("initial_saturation and final_saturation must have matching shapes")
    if pore_volume is None:
        pore = np.ones_like(final, dtype=float)
    else:
        pore = np.asarray(pore_volume, dtype=float)
        if pore.shape == ():
            pore = np.full(final.shape, float(pore), dtype=float)
        if pore.shape != final.shape:
            raise ValueError("pore_volume must be scalar or match saturation shape")
    storage_change = float(np.sum((final - initial) * pore))
    injected = 0.0 if injected_volume is None else float(injected_volume)
    produced = 0.0 if produced_volume is None else float(produced_volume)
    residual = injected - produced - storage_change
    scale = max(abs(injected), abs(produced), abs(storage_change), 1.0e-30)
    return {
        "storage_change": storage_change,
        "injected_volume": injected,
        "produced_volume": produced,
        "material_balance_residual": float(residual),
        "relative_material_balance_error": float(abs(residual) / scale),
        "material_balance_error": float(abs(residual) / scale),
        "has_nan": bool(np.isnan(storage_change) or np.isnan(residual)),
        "has_inf": bool(np.isinf(storage_change) or np.isinf(residual)),
    }


def compute_cfl_statistics(cfl_values: Field3D | ArrayLike) -> dict:
    """Return summary statistics for CFL numbers."""
    values = _values(cfl_values)
    has_nan = bool(np.isnan(values).any())
    has_inf = bool(np.isinf(values).any())
    if has_nan or has_inf:
        return {
            "max_cfl": None,
            "mean_cfl": None,
            "min_cfl": None,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "warnings": ["CFL values contain NaN or Inf"],
        }
    return {
        "max_cfl": float(np.max(values)),
        "mean_cfl": float(np.mean(values)),
        "min_cfl": float(np.min(values)),
        "has_nan": False,
        "has_inf": False,
        "warnings": [],
    }


def build_saturation_diagnostics_report(
    saturation: Field3D | ArrayLike,
    *,
    initial_saturation: Field3D | ArrayLike | None = None,
    lower: float = 0.0,
    upper: float = 1.0,
    threshold: float = 0.5,
    dx: float = 1.0,
    injected_volume: float | None = None,
    produced_volume: float | None = None,
    pore_volume: float | ArrayLike | None = None,
    cfl_values: Field3D | ArrayLike | None = None,
) -> dict:
    """Build a JSON-serializable saturation diagnostics report."""
    stats = compute_saturation_statistics(saturation)
    bounds = check_saturation_bounds(saturation, lower=lower, upper=upper)
    if initial_saturation is None:
        change = {"saturation_change_l1": None, "saturation_change_l2": None}
        balance = {"material_balance_error": None}
    else:
        change = compute_saturation_change_norm(initial_saturation, saturation)
        balance = compute_material_balance_error(
            initial_saturation,
            saturation,
            injected_volume=injected_volume,
            produced_volume=produced_volume,
            pore_volume=pore_volume,
        )
    cfl = {"max_cfl": None, "mean_cfl": None} if cfl_values is None else compute_cfl_statistics(cfl_values)
    has_nan = bool(stats["has_nan"] or bounds["has_nan"] or change.get("has_nan", False) or cfl.get("has_nan", False))
    has_inf = bool(stats["has_inf"] or bounds["has_inf"] or change.get("has_inf", False) or cfl.get("has_inf", False))
    warnings = []
    warnings.extend(stats.get("warnings", []))
    warnings.extend(bounds.get("warnings", []))
    warnings.extend(cfl.get("warnings", []))
    return {
        "success": bool(bounds["bounded"] and not has_nan and not has_inf),
        "saturation_min": stats["saturation_min"],
        "saturation_max": stats["saturation_max"],
        "saturation_mean": stats["saturation_mean"],
        "saturation_std": stats["saturation_std"],
        "num_below_lower": bounds["num_below_lower"],
        "num_above_upper": bounds["num_above_upper"],
        "front_position": estimate_front_position_1d(saturation, threshold=threshold, dx=dx),
        "saturation_change_l1": change["saturation_change_l1"],
        "saturation_change_l2": change["saturation_change_l2"],
        "material_balance_error": balance["material_balance_error"],
        "max_cfl": cfl["max_cfl"],
        "mean_cfl": cfl["mean_cfl"],
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": warnings,
    }


def _values(value: Field3D | ArrayLike) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        return np.asarray(value.values, dtype=float)
    return np.asarray(value, dtype=float)
