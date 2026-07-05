"""Lightweight Kriging / Gaussian-process style prediction interface.

The implementation prefers optional sklearn GaussianProcessRegressor when
available and requested. Otherwise it falls back to deterministic IDW prediction
with a distance-based uncertainty proxy. No hard geostatistics dependency is
introduced.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def predict_spatial_field(
    sample_points: ArrayLike,
    sample_values: ArrayLike,
    target_points: ArrayLike,
    *,
    method: str = "auto",
    power: float = 2.0,
    length_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Predict values and uncertainty at target points."""
    points = _points(sample_points, "sample_points")
    targets = _points(target_points, "target_points")
    values = np.asarray(sample_values, dtype=float).reshape(-1)
    if values.shape != (points.shape[0],):
        raise ValueError("sample_values length must match sample_points")
    if not np.isfinite(values).all():
        raise ValueError("sample_values must be finite")
    requested = method.lower().replace("_", "-")
    warnings: list[str] = []
    if requested in {"auto", "gp", "gaussian-process", "kriging"}:
        try:
            pred, std = _predict_sklearn_gp(points, values, targets, length_scale=length_scale)
            variance = np.maximum(std**2, 0.0)
            return pred, variance, {
                "success": True,
                "method_requested": method,
                "method_used": "sklearn_gaussian_process",
                "fallback_used": False,
                "warnings": warnings,
            }
        except Exception as exc:
            warnings.append(f"optional GP/Kriging backend unavailable; used IDW uncertainty fallback: {exc}")
    elif requested != "idw":
        raise ValueError("method must be auto, kriging, gp, gaussian_process, or idw")
    pred, variance = idw_uncertainty_fallback(points, values, targets, power=power)
    return pred, variance, {
        "success": True,
        "method_requested": method,
        "method_used": "idw_uncertainty_fallback",
        "fallback_used": requested != "idw",
        "warnings": warnings,
    }


def idw_uncertainty_fallback(
    sample_points: ArrayLike,
    sample_values: ArrayLike,
    target_points: ArrayLike,
    *,
    power: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict by IDW and estimate uncertainty from weighted local variance."""
    points = _points(sample_points, "sample_points")
    targets = _points(target_points, "target_points")
    values = np.asarray(sample_values, dtype=float).reshape(-1)
    if values.shape != (points.shape[0],):
        raise ValueError("sample_values length must match sample_points")
    distances = np.linalg.norm(targets[:, None, :] - points[None, :, :], axis=2)
    exact = distances <= 1.0e-12
    weights = 1.0 / np.maximum(distances, 1.0e-12) ** float(power)
    predictions = np.empty(targets.shape[0], dtype=float)
    variances = np.empty(targets.shape[0], dtype=float)
    for row in range(targets.shape[0]):
        if np.any(exact[row]):
            value = float(values[np.argmax(exact[row])])
            predictions[row] = value
            variances[row] = 0.0
            continue
        w = weights[row]
        w_sum = float(np.sum(w))
        prediction = float(np.sum(w * values) / w_sum)
        predictions[row] = prediction
        variances[row] = float(np.sum(w * (values - prediction) ** 2) / w_sum)
    return predictions, np.maximum(variances, 0.0)


def deferred_assimilation_request(method: str) -> dict[str, object]:
    """Return a deferred warning for EnKF/ES-MDA requests."""
    normalized = method.lower().replace("_", "-")
    if normalized not in {"enkf", "en-kf", "esmda", "es-mda"}:
        raise ValueError("method must be EnKF or ES-MDA")
    return {
        "success": True,
        "method_requested": method,
        "method_used": "deferred",
        "deferred": True,
        "fallback_used": True,
        "warnings": [f"{method} is deferred; no ensemble history matching was performed"],
    }


def _predict_sklearn_gp(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    length_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel  # type: ignore

    kernel = RBF(length_scale=float(length_scale)) + WhiteKernel(noise_level=1.0e-8)
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(points, values)
        pred, std = model.predict(targets, return_std=True)
    return np.asarray(pred, dtype=float), np.asarray(std, dtype=float)


def _points(points: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array
