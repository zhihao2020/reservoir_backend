"""Lightweight empirical electromagnetic saturation inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D

InvalidPolicy = Literal["raise", "low_confidence"]


def invert_saturation_em(
    signal: float | ArrayLike,
    coefficients: dict | ArrayLike,
    clip: bool = True,
    return_report: bool = False,
) -> float | NDArray[np.float64] | tuple[float | NDArray[np.float64], dict]:
    """Invert saturation from an empirical EM signal mapping."""
    values = np.asarray(signal, dtype=float)
    if (~np.isfinite(values)).any():
        raise InvalidPhysicalValueError("signal must be finite")
    raw = _evaluate_empirical_mapping(values, coefficients)
    if (~np.isfinite(raw)).any():
        raise InvalidPhysicalValueError("EM inversion produced non-finite saturation")
    saturation = np.clip(raw, 0.0, 1.0) if clip else raw.copy()
    report = _build_signal_report("em_empirical", values, raw, saturation)
    result = _to_scalar_if_needed(saturation)
    if return_report:
        report["saturation"] = result
        return result, report
    return result


@dataclass(frozen=True)
class ElectromagneticInverter:
    """Empirical EM signal to water saturation inverter."""

    def invert(
        self,
        signal: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None = None,
        invalid_policy: InvalidPolicy = "raise",
    ) -> float | NDArray[np.float64] | Field3D:
        """Invert EM signal to saturation."""
        sw, confidence, template = self._compute(signal, model_params, grid, invalid_policy)
        if template is not None:
            return Field3D(template.grid, sw, name="sw_em", unit="fraction", confidence=confidence)
        if sw.shape == ():
            return float(sw)
        return sw

    def invert_with_confidence(
        self,
        signal: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None = None,
        invalid_policy: InvalidPolicy = "raise",
    ):
        """Invert EM signal and return saturation plus confidence."""
        sw, confidence, template = self._compute(signal, model_params, grid, invalid_policy)
        if template is not None:
            return (
                Field3D(template.grid, sw, name="sw_em", unit="fraction", confidence=confidence),
                Field3D(template.grid, confidence, name="sw_em_confidence", unit="fraction"),
            )
        if sw.shape == ():
            return float(sw), float(confidence)
        return sw, confidence

    def calibrate_linear(self, signal_values: ArrayLike, sw_values: ArrayLike) -> dict:
        """Fit `Sw = a * signal + b`."""
        signal_arr, sw_arr = _calibration_arrays(signal_values, sw_values)
        a, b = np.polyfit(signal_arr, sw_arr, deg=1)
        return {"model": "linear", "a": float(a), "b": float(b), "calibration_range": [float(signal_arr.min()), float(signal_arr.max())]}

    def calibrate_polynomial(self, signal_values: ArrayLike, sw_values: ArrayLike, degree: int = 2) -> dict:
        """Fit a polynomial model with coefficients in ascending order."""
        signal_arr, sw_arr = _calibration_arrays(signal_values, sw_values)
        if degree < 1:
            raise ValueError("degree must be >= 1")
        descending = np.polyfit(signal_arr, sw_arr, deg=degree)
        return {
            "model": "polynomial",
            "coefficients": [float(v) for v in descending[::-1]],
            "calibration_range": [float(signal_arr.min()), float(signal_arr.max())],
        }

    def predict_saturation(self, signal: float | ArrayLike | Field3D, model_params: dict):
        """Predict unclipped saturation from an empirical model."""
        values = _values(signal)
        model = model_params.get("model", "linear")
        if model == "linear":
            result = float(model_params["a"]) * values + float(model_params["b"])
        elif model == "polynomial":
            result = np.zeros_like(values, dtype=float)
            for power, coefficient in enumerate(model_params["coefficients"]):
                result += float(coefficient) * values**power
        elif model in {"maxwell", "physics", "full_physics"}:
            raise NotImplementedError("Full Maxwell electromagnetic inversion is not implemented")
        else:
            raise ValueError(f"unsupported EM model: {model}")
        if result.shape == ():
            return float(result)
        return result

    def compute_confidence(self, signal: float | ArrayLike | Field3D, model_params: dict) -> float | NDArray[np.float64]:
        """Compute confidence from calibration range distance."""
        values = _values(signal)
        confidence = np.ones(values.shape, dtype=float)
        invalid = ~np.isfinite(values)
        confidence[invalid] = 0.0
        if "calibration_range" in model_params:
            low, high = [float(v) for v in model_params["calibration_range"]]
            span = max(high - low, 1.0e-12)
            below = values < low
            above = values > high
            distance = np.where(below, low - values, np.where(above, values - high, 0.0))
            confidence = np.where(invalid, 0.0, np.exp(-distance / span))
        confidence = np.clip(confidence, 0.0, 1.0)
        if confidence.shape == ():
            return float(confidence)
        return confidence

    def invert_complex_physics(self, *args, **kwargs):
        """Reserved full-physics EM inversion interface."""
        raise NotImplementedError("Full Maxwell electromagnetic inversion is not implemented")

    def _compute(
        self,
        signal: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None,
        invalid_policy: InvalidPolicy,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Field3D | None]:
        template = signal if isinstance(signal, Field3D) else None
        values = _values(signal)
        if template is None and grid is not None:
            if values.shape != grid.shape:
                raise ValueError("grid was provided but signal shape does not match")
            template = Field3D(grid, np.zeros(grid.shape))
        invalid = ~np.isfinite(values)
        if invalid.any() and invalid_policy == "raise":
            raise InvalidPhysicalValueError("signal must be finite")
        raw = np.asarray(self.predict_saturation(values, model_params), dtype=float)
        swi = float(model_params.get("swi", 0.0))
        sor = float(model_params.get("sor", 0.0))
        if swi < 0.0 or sor < 0.0 or swi + sor >= 1.0:
            raise InvalidPhysicalValueError("invalid saturation bounds")
        sw = np.clip(raw, swi, 1.0 - sor)
        sw = np.where(invalid, np.nan, sw)
        confidence = np.asarray(self.compute_confidence(values, model_params), dtype=float)
        confidence = np.where(invalid, 0.0, confidence)
        return sw, confidence, template


def _values(value: float | ArrayLike | Field3D) -> NDArray[np.float64]:
    source = value.values if isinstance(value, Field3D) else value
    return np.asarray(source, dtype=float)


def _calibration_arrays(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape or x_arr.size < 2:
        raise ValueError("calibration arrays must have matching shape and at least two values")
    if np.isnan(x_arr).any() or np.isnan(y_arr).any() or np.isinf(x_arr).any() or np.isinf(y_arr).any():
        raise InvalidPhysicalValueError("calibration arrays must be finite")
    return x_arr.ravel(), y_arr.ravel()


def _evaluate_empirical_mapping(values: NDArray[np.float64], coefficients: dict | ArrayLike) -> NDArray[np.float64]:
    if isinstance(coefficients, dict):
        model = coefficients.get("model", "polynomial")
        if model == "linear":
            if "coefficients" in coefficients:
                coeffs = coefficients["coefficients"]
                if len(coeffs) != 2:
                    raise InvalidPhysicalValueError("linear coefficients must contain c0 and c1")
                c0, c1 = [float(v) for v in coeffs]
            else:
                c0 = float(coefficients.get("c0", coefficients.get("b", 0.0)))
                c1 = float(coefficients.get("c1", coefficients.get("a", 0.0)))
            return c0 + c1 * values
        if model == "polynomial":
            coeffs = coefficients.get("coefficients")
        elif model in {"maxwell", "physics", "full_physics"}:
            raise NotImplementedError("Full Maxwell electromagnetic inversion is not implemented")
        else:
            raise ValueError(f"unsupported EM model: {model}")
    else:
        coeffs = coefficients

    coeff_arr = np.asarray(coeffs, dtype=float)
    if coeff_arr.size == 0:
        raise InvalidPhysicalValueError("coefficients must be non-empty")
    if (~np.isfinite(coeff_arr)).any():
        raise InvalidPhysicalValueError("coefficients must be finite")
    result = np.zeros_like(values, dtype=float)
    for power, coefficient in enumerate(coeff_arr.ravel()):
        result += float(coefficient) * values**power
    return result


def _build_signal_report(
    method: str,
    signal: NDArray[np.float64],
    raw: NDArray[np.float64],
    saturation: NDArray[np.float64],
) -> dict:
    return {
        "method": method,
        "success": True,
        "saturation": _to_scalar_if_needed(saturation),
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
        "saturation_min": float(np.min(saturation)),
        "saturation_max": float(np.max(saturation)),
        "num_clipped_low": int(np.sum(raw < 0.0)),
        "num_clipped_high": int(np.sum(raw > 1.0)),
        "warnings": [],
        "has_nan": bool(np.isnan(saturation).any()),
        "has_inf": bool(np.isinf(saturation).any()),
    }


def _to_scalar_if_needed(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr
