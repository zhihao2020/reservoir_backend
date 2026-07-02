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
