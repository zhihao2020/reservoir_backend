"""Oil-water capillary pressure models.

This module is intentionally standalone: it evaluates Pcow(Sw) and numeric
derivatives, but does not couple capillary pressure into transport solvers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D


def effective_saturation_for_pc(
    sw: float | ArrayLike | Field3D,
    swi: float,
    sor: float,
    eps: float = 1.0e-8,
) -> float | NDArray[np.float64]:
    """Return effective saturation clipped to ``[eps, 1]`` for Pc models."""
    _validate_saturation_params(swi, sor)
    eps_value = _positive_finite(eps, "eps")
    if eps_value > 1.0:
        raise InvalidPhysicalValueError("eps must be <= 1")
    values = _values(sw)
    se = (values - float(swi)) / (1.0 - float(swi) - float(sor))
    return _scalar_or_array(np.clip(se, eps_value, 1.0))


def brooks_corey_pc(
    sw: float | ArrayLike | Field3D,
    swi: float,
    sor: float,
    entry_pressure: float,
    lambda_pc: float,
    eps: float = 1.0e-8,
) -> float | NDArray[np.float64] | Field3D:
    """Evaluate Brooks-Corey capillary pressure ``Pc = Pe * Se^(-1/lambda)``."""
    entry = _positive_finite(entry_pressure, "entry_pressure")
    lambda_value = _positive_finite(lambda_pc, "lambda_pc")
    se = np.asarray(effective_saturation_for_pc(sw, swi, sor, eps), dtype=float)
    pc = entry * se ** (-1.0 / lambda_value)
    _validate_pc_values(pc)
    return _wrap_like(sw, pc)


def van_genuchten_pc(
    sw: float | ArrayLike | Field3D,
    swi: float,
    sor: float,
    p0: float,
    m: float,
    n: float,
    eps: float = 1.0e-8,
) -> float | NDArray[np.float64] | Field3D:
    """Evaluate van Genuchten capillary pressure."""
    p0_value = _positive_finite(p0, "p0")
    m_value = _positive_finite(m, "m")
    n_value = _positive_finite(n, "n")
    se = np.asarray(effective_saturation_for_pc(sw, swi, sor, eps), dtype=float)
    term = np.maximum(se ** (-1.0 / m_value) - 1.0, 0.0)
    pc = p0_value * term ** (1.0 / n_value)
    _validate_pc_values(pc)
    return _wrap_like(sw, pc)


def no_capillary_pressure(sw: float | ArrayLike | Field3D) -> float | NDArray[np.float64] | Field3D:
    """Return zero capillary pressure with the same shape as ``sw``."""
    values = _values(sw)
    return _wrap_like(sw, np.zeros_like(values, dtype=float))


def capillary_pressure(
    sw: float | ArrayLike | Field3D,
    model_params: dict[str, Any],
) -> float | NDArray[np.float64] | Field3D:
    """Evaluate capillary pressure from a model parameter dictionary."""
    params = validate_capillary_params(model_params)
    model = params["model"]
    if model == "none":
        return no_capillary_pressure(sw)
    if model == "brooks_corey":
        return brooks_corey_pc(
            sw,
            params["swi"],
            params["sor"],
            params["entry_pressure_pa"],
            params["lambda_pc"],
            params["eps"],
        )
    if model == "van_genuchten":
        return van_genuchten_pc(
            sw,
            params["swi"],
            params["sor"],
            params["p0_pa"],
            params["m"],
            params["n"],
            params["eps"],
        )
    raise ValueError(f"unsupported capillary pressure model: {model}")


def capillary_pressure_derivative_numeric(
    sw: float | ArrayLike | Field3D,
    model_params: dict[str, Any],
    delta: float = 1.0e-6,
) -> float | NDArray[np.float64] | Field3D:
    """Return a central finite-difference derivative ``dPc/dSw``."""
    delta_value = _positive_finite(delta, "delta")
    values = _values(sw)
    params = validate_capillary_params(model_params)
    upper = np.asarray(capillary_pressure(values + delta_value, params), dtype=float)
    lower = np.asarray(capillary_pressure(values - delta_value, params), dtype=float)
    derivative = (upper - lower) / (2.0 * delta_value)
    if np.isnan(derivative).any() or np.isinf(derivative).any():
        raise InvalidPhysicalValueError("capillary pressure derivative must be finite")
    return _wrap_like(sw, derivative, name="capillary_pressure_derivative", unit="Pa/fraction")


def validate_capillary_params(model_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize capillary pressure model parameters."""
    params = dict(model_params or {})
    enabled = bool(params.get("enabled", True))
    raw_model = str(params.get("model", "none")).lower()
    model_aliases = {
        "no_capillary": "none",
        "no-capillary": "none",
        "disabled": "none",
        "brooks-corey": "brooks_corey",
        "bc": "brooks_corey",
        "van-genuchten": "van_genuchten",
        "vg": "van_genuchten",
    }
    model = "none" if not enabled else model_aliases.get(raw_model, raw_model)
    if model not in {"none", "brooks_corey", "van_genuchten"}:
        raise ValueError("capillary_pressure.model must be none, brooks_corey, or van_genuchten")

    normalized = {
        "enabled": enabled,
        "model": model,
        "swi": float(params.get("swi", 0.0)),
        "sor": float(params.get("sor", 0.0)),
        "entry_pressure_pa": float(params.get("entry_pressure_pa", 1000.0)),
        "lambda_pc": float(params.get("lambda_pc", 2.0)),
        "p0_pa": float(params.get("p0_pa", 1000.0)),
        "m": float(params.get("m", 0.5)),
        "n": float(params.get("n", 2.0)),
        "eps": float(params.get("eps", 1.0e-8)),
    }
    _validate_saturation_params(normalized["swi"], normalized["sor"])
    _positive_finite(normalized["eps"], "eps")
    if normalized["eps"] > 1.0:
        raise InvalidPhysicalValueError("eps must be <= 1")
    if model == "brooks_corey":
        _positive_finite(normalized["entry_pressure_pa"], "entry_pressure_pa")
        _positive_finite(normalized["lambda_pc"], "lambda_pc")
    elif model == "van_genuchten":
        _positive_finite(normalized["p0_pa"], "p0_pa")
        _positive_finite(normalized["m"], "m")
        _positive_finite(normalized["n"], "n")
    return normalized


def build_capillary_model_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build capillary model params from a full case config or section config."""
    section = config.get("capillary_pressure", config)
    params = dict(section or {})
    saturation = config.get("saturation", {}) if "capillary_pressure" in config else {}
    params.setdefault("swi", saturation.get("swi", 0.0))
    params.setdefault("sor", saturation.get("sor", 0.0))
    return validate_capillary_params(params)


def _validate_saturation_params(swi: float, sor: float) -> None:
    swi_value = float(swi)
    sor_value = float(sor)
    if (
        not np.isfinite(swi_value)
        or not np.isfinite(sor_value)
        or swi_value < 0.0
        or sor_value < 0.0
        or swi_value >= 1.0
        or sor_value >= 1.0
        or swi_value + sor_value >= 1.0
    ):
        raise InvalidPhysicalValueError("swi and sor must be finite, non-negative, and sum to less than 1")


def _positive_finite(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise InvalidPhysicalValueError(f"{name} must be a positive finite value")
    return numeric


def _values(sw: float | ArrayLike | Field3D) -> NDArray[np.float64]:
    values = sw.values if isinstance(sw, Field3D) else sw
    array = np.asarray(values, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError("sw must be finite")
    return array


def _validate_pc_values(pc: NDArray[np.float64]) -> None:
    if np.isnan(pc).any() or np.isinf(pc).any() or (pc < 0.0).any():
        raise InvalidPhysicalValueError("capillary pressure must be finite and non-negative")


def _wrap_like(
    original: float | ArrayLike | Field3D,
    values: NDArray[np.float64],
    *,
    name: str = "capillary_pressure",
    unit: str = "Pa",
) -> float | NDArray[np.float64] | Field3D:
    if isinstance(original, Field3D):
        return Field3D(original.grid, values, name=name, unit=unit)
    return _scalar_or_array(values)


def _scalar_or_array(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    if value.shape == ():
        return float(value)
    return value
