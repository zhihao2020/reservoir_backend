"""Diagnostics for simplified incompressible three-phase WOG checks.

This module reports saturation closure, bounds, relperm, mobility,
fractional-flow, phase-flux, and transport consistency. It does not update
transport state or modify solver inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.solver.three_phase_relperm import (
    compute_oil_saturation,
    corey_three_phase_relative_permeability,
    fractional_flow_three_phase,
    three_phase_mobility,
)


def compute_three_phase_saturation_statistics(sw: ArrayLike, so: ArrayLike, sg: ArrayLike) -> dict[str, object]:
    """Return basic WOG saturation statistics and finite/closure diagnostics."""
    sw_array, so_array, sg_array = _arrays(sw, so, sg)
    closure = sw_array + so_array + sg_array - 1.0
    all_values = np.concatenate([sw_array.ravel(), so_array.ravel(), sg_array.ravel()])
    return {
        "sw_min": _safe_min(sw_array),
        "sw_max": _safe_max(sw_array),
        "so_min": _safe_min(so_array),
        "so_max": _safe_max(so_array),
        "sg_min": _safe_min(sg_array),
        "sg_max": _safe_max(sg_array),
        "sw_mean": _safe_mean(sw_array),
        "so_mean": _safe_mean(so_array),
        "sg_mean": _safe_mean(sg_array),
        "closure_max_abs_error": _safe_max_abs(closure),
        "closure_l2_error": _safe_l2(closure),
        "has_nan": bool(np.isnan(all_values).any()),
        "has_inf": bool(np.isinf(all_values).any()),
        "warnings": _finite_warnings(all_values, "saturation"),
    }


def compute_three_phase_closure_error(sw: ArrayLike, so: ArrayLike, sg: ArrayLike) -> dict[str, object]:
    """Return closure errors for `Sw + So + Sg = 1`."""
    sw_array, so_array, sg_array = _arrays(sw, so, sg)
    closure = sw_array + so_array + sg_array - 1.0
    return {
        "closure_max_abs_error": _safe_max_abs(closure),
        "closure_l2_error": _safe_l2(closure),
        "closure_mean_abs_error": _safe_mean_abs(closure),
        "has_nan": bool(np.isnan(closure).any()),
        "has_inf": bool(np.isinf(closure).any()),
        "warnings": _finite_warnings(closure, "closure"),
    }


def check_three_phase_bounds(
    sw: ArrayLike,
    so: ArrayLike,
    sg: ArrayLike,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Check phase saturation bounds without changing the arrays."""
    sw_array, so_array, sg_array = _arrays(sw, so, sg)
    lower_value = float(lower)
    upper_value = float(upper)
    tol = float(tolerance)
    all_values = np.concatenate([sw_array.ravel(), so_array.ravel(), sg_array.ravel()])
    finite = np.isfinite(all_values)
    violations = np.count_nonzero((all_values[finite] < lower_value - tol) | (all_values[finite] > upper_value + tol))
    return {
        "lower": lower_value,
        "upper": upper_value,
        "tolerance": tol,
        "num_bound_violations": int(violations),
        "num_below_lower": int(np.count_nonzero(all_values[finite] < lower_value - tol)),
        "num_above_upper": int(np.count_nonzero(all_values[finite] > upper_value + tol)),
        "has_nan": bool(np.isnan(all_values).any()),
        "has_inf": bool(np.isinf(all_values).any()),
        "success": bool(violations == 0 and np.isfinite(all_values).all()),
        "warnings": _finite_warnings(all_values, "bounds"),
    }


def compute_three_phase_relperm_metrics(sw: ArrayLike, sg: ArrayLike, params: dict[str, float]) -> dict[str, object]:
    """Return Corey three-phase relperm diagnostics."""
    krw, kro, krg = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw, sg, params))
    arrays = [krw, kro, krg]
    return {
        "krw_min": _safe_min(krw),
        "krw_max": _safe_max(krw),
        "kro_min": _safe_min(kro),
        "kro_max": _safe_max(kro),
        "krg_min": _safe_min(krg),
        "krg_max": _safe_max(krg),
        "krw_nonnegative": bool(np.all(krw >= 0.0)),
        "kro_nonnegative": bool(np.all(kro >= 0.0)),
        "krg_nonnegative": bool(np.all(krg >= 0.0)),
        "has_nan": any(bool(np.isnan(array).any()) for array in arrays),
        "has_inf": any(bool(np.isinf(array).any()) for array in arrays),
        "warnings": _multi_warnings(arrays, "relperm"),
    }


def compute_three_phase_mobility_metrics(sw: ArrayLike, sg: ArrayLike, params: dict[str, float]) -> dict[str, object]:
    """Return phase mobility diagnostics."""
    lw, lo, lg, lt = (np.asarray(value, dtype=float) for value in three_phase_mobility(sw, sg, params))
    arrays = [lw, lo, lg, lt]
    return {
        "lambda_w_min": _safe_min(lw),
        "lambda_w_max": _safe_max(lw),
        "lambda_o_min": _safe_min(lo),
        "lambda_o_max": _safe_max(lo),
        "lambda_g_min": _safe_min(lg),
        "lambda_g_max": _safe_max(lg),
        "lambda_total_min": _safe_min(lt),
        "lambda_total_max": _safe_max(lt),
        "lambda_total_positive": bool(np.all(lt > 0.0)),
        "has_nan": any(bool(np.isnan(array).any()) for array in arrays),
        "has_inf": any(bool(np.isinf(array).any()) for array in arrays),
        "warnings": _multi_warnings(arrays, "mobility"),
    }


def compute_fractional_flow_closure_metrics(fw: ArrayLike, fo: ArrayLike, fg: ArrayLike) -> dict[str, object]:
    """Return diagnostics for `fw + fo + fg = 1`."""
    fw_array, fo_array, fg_array = _arrays(fw, fo, fg)
    total = fw_array + fo_array + fg_array
    all_values = np.concatenate([fw_array.ravel(), fo_array.ravel(), fg_array.ravel(), total.ravel()])
    return {
        "fw_min": _safe_min(fw_array),
        "fw_max": _safe_max(fw_array),
        "fo_min": _safe_min(fo_array),
        "fo_max": _safe_max(fo_array),
        "fg_min": _safe_min(fg_array),
        "fg_max": _safe_max(fg_array),
        "fractional_flow_sum_error": _safe_max_abs(total - 1.0),
        "has_nan": bool(np.isnan(all_values).any()),
        "has_inf": bool(np.isinf(all_values).any()),
        "warnings": _finite_warnings(all_values, "fractional_flow"),
    }


def compute_phase_flux_statistics(
    water_flux: ArrayLike | None = None,
    oil_flux: ArrayLike | None = None,
    gas_flux: ArrayLike | None = None,
) -> dict[str, object]:
    """Return finite, shape, and magnitude diagnostics for optional phase flux arrays."""
    arrays = {
        "water": None if water_flux is None else np.asarray(water_flux, dtype=float),
        "oil": None if oil_flux is None else np.asarray(oil_flux, dtype=float),
        "gas": None if gas_flux is None else np.asarray(gas_flux, dtype=float),
    }
    present = [array for array in arrays.values() if array is not None]
    values = np.concatenate([array.ravel() for array in present]) if present else np.array([], dtype=float)
    return {
        "max_abs_water_flux": 0.0 if arrays["water"] is None else _safe_max_abs(arrays["water"]),
        "max_abs_oil_flux": 0.0 if arrays["oil"] is None else _safe_max_abs(arrays["oil"]),
        "max_abs_gas_flux": 0.0 if arrays["gas"] is None else _safe_max_abs(arrays["gas"]),
        "water_flux_shape": None if arrays["water"] is None else list(arrays["water"].shape),
        "oil_flux_shape": None if arrays["oil"] is None else list(arrays["oil"].shape),
        "gas_flux_shape": None if arrays["gas"] is None else list(arrays["gas"].shape),
        "has_nan": bool(np.isnan(values).any()) if values.size else False,
        "has_inf": bool(np.isinf(values).any()) if values.size else False,
        "warnings": _finite_warnings(values, "phase_flux") if values.size else [],
    }


def compute_three_phase_transport_metrics(initial_state: dict[str, ArrayLike], final_state: dict[str, ArrayLike]) -> dict[str, object]:
    """Return closure, bounds, and change metrics between initial and final WOG states."""
    sw0, so0, sg0 = _state_arrays(initial_state)
    sw1, so1, sg1 = _state_arrays(final_state)
    stats = compute_three_phase_saturation_statistics(sw1, so1, sg1)
    bounds = check_three_phase_bounds(sw1, so1, sg1)
    all_delta = np.concatenate([(sw1 - sw0).ravel(), (so1 - so0).ravel(), (sg1 - sg0).ravel()])
    return {
        **stats,
        "num_bound_violations": bounds["num_bound_violations"],
        "saturation_change_l1": float(np.sum(np.abs(all_delta))),
        "saturation_change_l2": float(np.sqrt(np.sum(all_delta * all_delta))),
    }


def build_three_phase_diagnostics_report(
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
    *,
    water_flux: ArrayLike | None = None,
    oil_flux: ArrayLike | None = None,
    gas_flux: ArrayLike | None = None,
) -> dict[str, object]:
    """Build a consolidated WOG diagnostics report for benchmark summaries."""
    sw_array = np.asarray(sw, dtype=float)
    sg_array = np.asarray(sg, dtype=float)
    so_array = np.asarray(compute_oil_saturation(sw_array, sg_array), dtype=float)
    stats = compute_three_phase_saturation_statistics(sw_array, so_array, sg_array)
    bounds = check_three_phase_bounds(sw_array, so_array, sg_array)
    relperm = compute_three_phase_relperm_metrics(sw_array, sg_array, params)
    mobility = compute_three_phase_mobility_metrics(sw_array, sg_array, params)
    fw, fo, fg = fractional_flow_three_phase(sw_array, sg_array, params)
    frac = compute_fractional_flow_closure_metrics(fw, fo, fg)
    flux = compute_phase_flux_statistics(water_flux, oil_flux, gas_flux)
    warnings = [*stats["warnings"], *bounds["warnings"], *relperm["warnings"], *mobility["warnings"], *frac["warnings"], *flux["warnings"]]
    report = {
        "success": bool(
            bounds["success"]
            and stats["closure_max_abs_error"] is not None
            and float(stats["closure_max_abs_error"]) <= 1.0e-12
            and not any([stats["has_nan"], stats["has_inf"], relperm["has_nan"], relperm["has_inf"], mobility["has_nan"], mobility["has_inf"], frac["has_nan"], frac["has_inf"], flux["has_nan"], flux["has_inf"]])
        ),
        "sw_min": stats["sw_min"],
        "sw_max": stats["sw_max"],
        "so_min": stats["so_min"],
        "so_max": stats["so_max"],
        "sg_min": stats["sg_min"],
        "sg_max": stats["sg_max"],
        "closure_max_abs_error": stats["closure_max_abs_error"],
        "closure_l2_error": stats["closure_l2_error"],
        "num_bound_violations": bounds["num_bound_violations"],
        "krw_min": relperm["krw_min"],
        "krw_max": relperm["krw_max"],
        "kro_min": relperm["kro_min"],
        "kro_max": relperm["kro_max"],
        "krg_min": relperm["krg_min"],
        "krg_max": relperm["krg_max"],
        "lambda_total_min": mobility["lambda_total_min"],
        "lambda_total_max": mobility["lambda_total_max"],
        "fractional_flow_sum_error": frac["fractional_flow_sum_error"],
        "max_abs_water_flux": flux["max_abs_water_flux"],
        "max_abs_oil_flux": flux["max_abs_oil_flux"],
        "max_abs_gas_flux": flux["max_abs_gas_flux"],
        "has_nan": bool(any([stats["has_nan"], relperm["has_nan"], mobility["has_nan"], frac["has_nan"], flux["has_nan"]])),
        "has_inf": bool(any([stats["has_inf"], relperm["has_inf"], mobility["has_inf"], frac["has_inf"], flux["has_inf"]])),
        "warnings": warnings,
    }
    return _jsonable(report)


def _state_arrays(state: dict[str, ArrayLike]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sw = np.asarray(state["sw"], dtype=float)
    sg = np.asarray(state["sg"], dtype=float)
    so = np.asarray(state.get("so", compute_oil_saturation(sw, sg)), dtype=float)
    return sw, so, sg


def _arrays(*values: ArrayLike) -> tuple[NDArray[np.float64], ...]:
    return tuple(np.asarray(value, dtype=float) for value in values)


def _finite_values(array: NDArray[np.float64]) -> NDArray[np.float64]:
    return array[np.isfinite(array)]


def _safe_min(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.min(values))


def _safe_max(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.max(values))


def _safe_mean(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.mean(values))


def _safe_mean_abs(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.mean(np.abs(values)))


def _safe_max_abs(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.max(np.abs(values)))


def _safe_l2(array: NDArray[np.float64]) -> float | None:
    values = _finite_values(array)
    return None if values.size == 0 else float(np.sqrt(np.sum(values * values)))


def _finite_warnings(array: NDArray[np.float64], name: str) -> list[str]:
    warnings: list[str] = []
    if np.isnan(array).any():
        warnings.append(f"{name} contains NaN")
    if np.isinf(array).any():
        warnings.append(f"{name} contains Inf")
    return warnings


def _multi_warnings(arrays: list[NDArray[np.float64]], name: str) -> list[str]:
    warnings: list[str] = []
    if any(np.isnan(array).any() for array in arrays):
        warnings.append(f"{name} contains NaN")
    if any(np.isinf(array).any() for array in arrays):
        warnings.append(f"{name} contains Inf")
    return warnings


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
