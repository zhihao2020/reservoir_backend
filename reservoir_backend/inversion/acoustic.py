"""Lightweight empirical acoustic saturation inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D

InvalidPolicy = Literal["raise", "low_confidence"]


def invert_saturation_acoustic(
    signal: float | ArrayLike,
    coefficients: dict | ArrayLike,
    clip: bool = True,
    return_report: bool = False,
) -> float | NDArray[np.float64] | tuple[float | NDArray[np.float64], dict]:
    """Invert saturation from an empirical acoustic signal mapping."""
    values = np.asarray(signal, dtype=float)
    if (~np.isfinite(values)).any():
        raise InvalidPhysicalValueError("signal must be finite")
    raw = _evaluate_empirical_mapping(values, coefficients)
    if (~np.isfinite(raw)).any():
        raise InvalidPhysicalValueError("acoustic inversion produced non-finite saturation")
    saturation = np.clip(raw, 0.0, 1.0) if clip else raw.copy()
    report = _build_signal_report("acoustic_empirical", values, raw, saturation)
    result = _to_scalar_if_needed(saturation)
    if return_report:
        report["saturation"] = result
        return result, report
    return result


@dataclass(frozen=True)
class AcousticInverter:
    """Empirical acoustic velocity to water saturation inverter."""

    def invert(
        self,
        velocity: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None = None,
        invalid_policy: InvalidPolicy = "raise",
    ) -> float | NDArray[np.float64] | Field3D:
        """Invert acoustic velocity to saturation."""
        sw, confidence, template = self._compute(velocity, model_params, grid, invalid_policy)
        if template is not None:
            return Field3D(template.grid, sw, name="sw_acoustic", unit="fraction", confidence=confidence)
        if sw.shape == ():
            return float(sw)
        return sw

    def invert_with_confidence(
        self,
        velocity: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None = None,
        invalid_policy: InvalidPolicy = "raise",
    ):
        """Invert acoustic velocity and return saturation plus confidence."""
        sw, confidence, template = self._compute(velocity, model_params, grid, invalid_policy)
        if template is not None:
            return (
                Field3D(template.grid, sw, name="sw_acoustic", unit="fraction", confidence=confidence),
                Field3D(template.grid, confidence, name="sw_acoustic_confidence", unit="fraction"),
            )
        if sw.shape == ():
            return float(sw), float(confidence)
        return sw, confidence

    def calibrate_linear(self, velocity_values: ArrayLike, sw_values: ArrayLike) -> dict:
        """Fit `Sw = a * Vp + b`."""
        x, y = _calibration_arrays(velocity_values, sw_values)
        a, b = np.polyfit(x, y, deg=1)
        return {"model": "linear", "a": float(a), "b": float(b), "calibration_range": [float(x.min()), float(x.max())]}

    def calibrate_polynomial(self, velocity_values: ArrayLike, sw_values: ArrayLike, degree: int = 2) -> dict:
        """Fit polynomial model with ascending coefficients."""
        x, y = _calibration_arrays(velocity_values, sw_values)
        if degree < 1:
            raise ValueError("degree must be >= 1")
        descending = np.polyfit(x, y, deg=degree)
        return {
            "model": "polynomial",
            "coefficients": [float(v) for v in descending[::-1]],
            "calibration_range": [float(x.min()), float(x.max())],
        }

    def predict_saturation(self, velocity: float | ArrayLike | Field3D, model_params: dict):
        """Predict unclipped saturation from an empirical model."""
        values = _values(velocity)
        model = model_params.get("model", "linear")
        if model == "linear":
            result = float(model_params["a"]) * values + float(model_params["b"])
        elif model == "polynomial":
            result = np.zeros_like(values, dtype=float)
            for power, coefficient in enumerate(model_params["coefficients"]):
                result += float(coefficient) * values**power
        elif model in {"gassmann", "rock_physics", "full_physics"}:
            raise NotImplementedError("Full Gassmann acoustic inversion is not implemented")
        else:
            raise ValueError(f"unsupported acoustic model: {model}")
        if result.shape == ():
            return float(result)
        return result

    def compute_confidence(self, velocity: float | ArrayLike | Field3D, model_params: dict) -> float | NDArray[np.float64]:
        """Compute confidence from calibration range distance."""
        values = _values(velocity)
        invalid = (~np.isfinite(values)) | (values <= 0.0)
        confidence = np.ones(values.shape, dtype=float)
        confidence[invalid] = 0.0
        if "calibration_range" in model_params:
            low, high = [float(v) for v in model_params["calibration_range"]]
            span = max(high - low, 1.0e-12)
            distance = np.where(values < low, low - values, np.where(values > high, values - high, 0.0))
            confidence = np.where(invalid, 0.0, np.exp(-distance / span))
        confidence = np.clip(confidence, 0.0, 1.0)
        if confidence.shape == ():
            return float(confidence)
        return confidence

    def invert_gassmann(self, *args, **kwargs):
        """Reserved full Gassmann inversion interface."""
        raise NotImplementedError("Full Gassmann acoustic inversion is not implemented")

    def _compute(
        self,
        velocity: float | ArrayLike | Field3D,
        model_params: dict,
        grid: Grid3D | None,
        invalid_policy: InvalidPolicy,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Field3D | None]:
        template = velocity if isinstance(velocity, Field3D) else None
        values = _values(velocity)
        if template is None and grid is not None:
            if values.shape != grid.shape:
                raise ValueError("grid was provided but velocity shape does not match")
            template = Field3D(grid, np.zeros(grid.shape))
        invalid = (~np.isfinite(values)) | (values <= 0.0)
        if invalid.any() and invalid_policy == "raise":
            raise InvalidPhysicalValueError("velocity must be positive and finite")
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
    if (x_arr <= 0.0).any():
        raise InvalidPhysicalValueError("velocity calibration values must be positive")
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
        elif model in {"gassmann", "rock_physics", "full_physics"}:
            raise NotImplementedError("Full Gassmann acoustic inversion is not implemented")
        else:
            raise ValueError(f"unsupported acoustic model: {model}")
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
