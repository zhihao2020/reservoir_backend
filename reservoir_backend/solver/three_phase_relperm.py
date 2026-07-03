"""Corey-style water-oil-gas relative permeability and fractional flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D


def compute_oil_saturation(sw: float | ArrayLike | Field3D, sg: float | ArrayLike | Field3D) -> float | NDArray[np.float64]:
    """Return oil saturation from closure: `So = 1 - Sw - Sg`."""
    sw_values, sg_values = _broadcast_saturations(sw, sg)
    so = 1.0 - sw_values - sg_values
    return _scalar_or_array(so)


def validate_three_phase_params(params: dict[str, float]) -> None:
    """Validate three-phase residuals, endpoints, exponents, and viscosities."""
    p = _params(params)
    for name in ["swi", "sor", "sgc"]:
        value = p[name]
        if not np.isfinite(value) or value < 0.0 or value >= 1.0:
            raise InvalidPhysicalValueError(f"{name} must be finite and in [0, 1)")
    if p["swi"] + p["sor"] + p["sgc"] >= 1.0:
        raise InvalidPhysicalValueError("swi + sor + sgc must be less than 1")

    for name in ["krw0", "kro0", "krg0"]:
        value = p[name]
        if not np.isfinite(value) or value < 0.0:
            raise InvalidPhysicalValueError(f"{name} must be finite and non-negative")
    for name in ["nw", "no", "ng", "mu_w", "mu_o", "mu_g"]:
        value = p[name]
        if not np.isfinite(value) or value <= 0.0:
            raise InvalidPhysicalValueError(f"{name} must be a positive finite value")


def validate_three_phase_saturations(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> None:
    """Validate three-phase saturation state against residual bounds."""
    validate_three_phase_params(params)
    p = _params(params)
    sw_values, sg_values = _broadcast_saturations(sw, sg)
    so_values = 1.0 - sw_values - sg_values
    tolerance = 1.0e-12
    if (sw_values < p["swi"] - tolerance).any():
        raise InvalidPhysicalValueError("sw must be greater than or equal to swi")
    if (sg_values < p["sgc"] - tolerance).any():
        raise InvalidPhysicalValueError("sg must be greater than or equal to sgc")
    if (so_values < p["sor"] - tolerance).any():
        raise InvalidPhysicalValueError("so must be greater than or equal to sor")
    if (sw_values + sg_values > 1.0 - p["sor"] + tolerance).any():
        raise InvalidPhysicalValueError("sw + sg must be less than or equal to 1 - sor")


def effective_saturations_three_phase(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return clipped effective saturations `(Sew, Seo, Seg)`."""
    validate_three_phase_params(params)
    p = _params(params)
    sw_values, sg_values = _broadcast_saturations(sw, sg)
    so_values = 1.0 - sw_values - sg_values
    denominator = _denominator(p)
    sew = np.clip((sw_values - p["swi"]) / denominator, 0.0, 1.0)
    seg = np.clip((sg_values - p["sgc"]) / denominator, 0.0, 1.0)
    seo = np.clip((so_values - p["sor"]) / denominator, 0.0, 1.0)
    return _scalar_or_array(sew), _scalar_or_array(seo), _scalar_or_array(seg)


def corey_three_phase_relative_permeability(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return Corey `(krw, kro, krg)` for a valid three-phase state."""
    validate_three_phase_saturations(sw, sg, params)
    p = _params(params)
    sew, seo, seg = (np.asarray(value, dtype=float) for value in effective_saturations_three_phase(sw, sg, params))
    krw = p["krw0"] * sew ** p["nw"]
    kro = p["kro0"] * seo ** p["no"]
    krg = p["krg0"] * seg ** p["ng"]
    return _scalar_or_array(krw), _scalar_or_array(kro), _scalar_or_array(krg)


def three_phase_mobility(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> tuple[
    float | NDArray[np.float64],
    float | NDArray[np.float64],
    float | NDArray[np.float64],
    float | NDArray[np.float64],
]:
    """Return `(lambda_w, lambda_o, lambda_g, lambda_t)`."""
    validate_three_phase_params(params)
    p = _params(params)
    krw, kro, krg = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw, sg, params))
    lambda_w = krw / p["mu_w"]
    lambda_o = kro / p["mu_o"]
    lambda_g = krg / p["mu_g"]
    lambda_t = lambda_w + lambda_o + lambda_g
    if np.isnan(lambda_t).any() or np.isinf(lambda_t).any():
        raise InvalidPhysicalValueError("three-phase mobility must be finite")
    return (
        _scalar_or_array(lambda_w),
        _scalar_or_array(lambda_o),
        _scalar_or_array(lambda_g),
        _scalar_or_array(lambda_t),
    )


def fractional_flow_three_phase(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Return three-phase fractional flow `(fw, fo, fg)`."""
    lambda_w, lambda_o, lambda_g, lambda_t = (
        np.asarray(value, dtype=float) for value in three_phase_mobility(sw, sg, params)
    )
    if (lambda_t <= 0.0).any():
        raise InvalidPhysicalValueError("total mobility lambda_t must be positive")
    fw = lambda_w / lambda_t
    fo = lambda_o / lambda_t
    fg = lambda_g / lambda_t
    fw = np.clip(fw, 0.0, 1.0)
    fo = np.clip(fo, 0.0, 1.0)
    fg = np.clip(fg, 0.0, 1.0)
    total = fw + fo + fg
    fw = fw / total
    fo = fo / total
    fg = fg / total
    return _scalar_or_array(fw), _scalar_or_array(fo), _scalar_or_array(fg)


def clip_three_phase_saturations(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]]:
    """Clip `(Sw, Sg)` into the residual-bounded three-phase saturation triangle."""
    validate_three_phase_params(params)
    p = _params(params)
    sw_values, sg_values = _broadcast_saturations(sw, sg)
    sw_clip = np.maximum(sw_values, p["swi"])
    sg_clip = np.maximum(sg_values, p["sgc"])
    water_excess = sw_clip - p["swi"]
    gas_excess = sg_clip - p["sgc"]
    total_excess = water_excess + gas_excess
    denominator = _denominator(p)
    scale = np.divide(
        denominator,
        total_excess,
        out=np.ones_like(total_excess, dtype=float),
        where=total_excess > denominator,
    )
    scale = np.minimum(scale, 1.0)
    sw_result = p["swi"] + water_excess * scale
    sg_result = p["sgc"] + gas_excess * scale
    return _scalar_or_array(sw_result), _scalar_or_array(sg_result)


def build_three_phase_relperm_report(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
    params: dict[str, float],
) -> dict[str, float | bool]:
    """Build a compact report for three-phase relperm and fractional flow."""
    sw_values, sg_values = _broadcast_saturations(sw, sg)
    so_values = 1.0 - sw_values - sg_values
    krw, kro, krg = (np.asarray(value, dtype=float) for value in corey_three_phase_relative_permeability(sw, sg, params))
    fw, fo, fg = (np.asarray(value, dtype=float) for value in fractional_flow_three_phase(sw, sg, params))
    arrays = [sw_values, sg_values, so_values, krw, kro, krg, fw, fo, fg]
    closure_error = np.abs(sw_values + so_values + sg_values - 1.0)
    return {
        "sw_min": float(np.min(sw_values)),
        "sw_max": float(np.max(sw_values)),
        "sg_min": float(np.min(sg_values)),
        "sg_max": float(np.max(sg_values)),
        "so_min": float(np.min(so_values)),
        "so_max": float(np.max(so_values)),
        "krw_min": float(np.min(krw)),
        "krw_max": float(np.max(krw)),
        "kro_min": float(np.min(kro)),
        "kro_max": float(np.max(kro)),
        "krg_min": float(np.min(krg)),
        "krg_max": float(np.max(krg)),
        "fw_min": float(np.min(fw)),
        "fw_max": float(np.max(fw)),
        "fo_min": float(np.min(fo)),
        "fo_max": float(np.max(fo)),
        "fg_min": float(np.min(fg)),
        "fg_max": float(np.max(fg)),
        "has_nan": any(np.isnan(array).any() for array in arrays),
        "has_inf": any(np.isinf(array).any() for array in arrays),
        "closure_error_max": float(np.max(closure_error)),
    }


def _params(params: dict[str, float]) -> dict[str, float]:
    required = ["swi", "sor", "sgc", "krw0", "kro0", "krg0", "nw", "no", "ng", "mu_w", "mu_o", "mu_g"]
    missing = [name for name in required if name not in params]
    if missing:
        raise InvalidPhysicalValueError(f"missing three-phase parameters: {', '.join(missing)}")
    return {name: float(params[name]) for name in required}


def _denominator(params: dict[str, float]) -> float:
    return 1.0 - params["swi"] - params["sor"] - params["sgc"]


def _broadcast_saturations(
    sw: float | ArrayLike | Field3D,
    sg: float | ArrayLike | Field3D,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(sw, Field3D) and isinstance(sg, Field3D):
        if sw.grid != sg.grid:
            raise GridMismatchError("sw and sg Field3D inputs must use the same grid")
    sw_values = _values(sw, "sw")
    sg_values = _values(sg, "sg")
    try:
        sw_broadcast, sg_broadcast = np.broadcast_arrays(sw_values, sg_values)
    except ValueError as exc:
        raise InvalidPhysicalValueError("sw and sg shapes must be broadcast-compatible") from exc
    return sw_broadcast.astype(float, copy=False), sg_broadcast.astype(float, copy=False)


def _values(value: float | ArrayLike | Field3D, name: str) -> NDArray[np.float64]:
    raw = value.values if isinstance(value, Field3D) else value
    array = np.asarray(raw, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError(f"{name} must be finite")
    return array


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if value.shape == ():
        return float(value)
    return np.asarray(value, dtype=float)
