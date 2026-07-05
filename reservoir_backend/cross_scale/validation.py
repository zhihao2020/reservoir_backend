"""Curve-to-curve lab-field validation metrics for cross-scale analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import InvalidPhysicalValueError


@dataclass(frozen=True)
class CurveData:
    """One one-dimensional time-series curve used for lab-field validation."""

    name: str
    time: NDArray[np.float64]
    values: NDArray[np.float64]
    unit: str | None = None
    curve_type: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", np.asarray(self.time, dtype=float))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))
        validate_curve_data(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurveData":
        """Build a curve from a dictionary."""
        required = ["name", "time", "values"]
        missing = [name for name in required if name not in data]
        if missing:
            raise InvalidPhysicalValueError(f"missing curve fields: {', '.join(missing)}")
        return cls(
            name=str(data["name"]),
            time=np.asarray(data["time"], dtype=float),
            values=np.asarray(data["values"], dtype=float),
            unit=data.get("unit"),
            curve_type=data.get("curve_type"),
            source=data.get("source"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable curve dictionary."""
        return {
            "name": self.name,
            "time": self.time.tolist(),
            "values": self.values.tolist(),
            "unit": self.unit,
            "curve_type": self.curve_type,
            "source": self.source,
        }


def validate_curve_data(curve: CurveData) -> None:
    """Validate curve shape, finite values, and strictly increasing time."""
    if not curve.name:
        raise InvalidPhysicalValueError("curve name must be non-empty")
    time = np.asarray(curve.time, dtype=float)
    values = np.asarray(curve.values, dtype=float)
    if time.ndim != 1:
        raise InvalidPhysicalValueError("curve time must be one-dimensional")
    if values.ndim != 1:
        raise InvalidPhysicalValueError("curve values must be one-dimensional")
    if time.shape != values.shape:
        raise InvalidPhysicalValueError("curve time and values must have the same shape")
    if time.size < 2:
        raise InvalidPhysicalValueError("curve must contain at least two points")
    if np.isnan(time).any() or np.isinf(time).any():
        raise InvalidPhysicalValueError("curve time must be finite")
    if np.isnan(values).any() or np.isinf(values).any():
        raise InvalidPhysicalValueError("curve values must be finite")
    if not np.all(np.diff(time) > 0.0):
        raise InvalidPhysicalValueError("curve time must be strictly increasing")


def align_curves_to_common_time(
    reference_curve: CurveData,
    target_curve: CurveData,
    method: str = "linear",
    overlap_only: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Align target curve to reference timestamps inside the common overlap."""
    validate_curve_data(reference_curve)
    validate_curve_data(target_curve)
    if method != "linear":
        raise InvalidPhysicalValueError("only linear interpolation is supported")
    if not overlap_only:
        raise InvalidPhysicalValueError("extrapolation is not supported; overlap_only must be True")

    overlap_start = max(float(reference_curve.time[0]), float(target_curve.time[0]))
    overlap_end = min(float(reference_curve.time[-1]), float(target_curve.time[-1]))
    if overlap_start > overlap_end:
        raise InvalidPhysicalValueError("curves do not have overlapping time ranges")

    mask = (reference_curve.time >= overlap_start) & (reference_curve.time <= overlap_end)
    common_time = reference_curve.time[mask].astype(float, copy=True)
    if common_time.size == 0:
        raise InvalidPhysicalValueError("reference curve has no samples inside the overlap interval")
    reference_values = reference_curve.values[mask].astype(float, copy=True)
    target_values = np.interp(common_time, target_curve.time, target_curve.values).astype(float, copy=False)
    report = {
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "num_points": int(common_time.size),
        "method": method,
        "overlap_only": overlap_only,
        "warnings": [],
    }
    return common_time, reference_values, target_values, report


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def compute_mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1.0e-12) -> float:
    true, pred = _paired_arrays(y_true, y_pred)
    if epsilon <= 0.0 or not np.isfinite(float(epsilon)):
        raise InvalidPhysicalValueError("epsilon must be positive and finite")
    denominator = np.maximum(np.abs(true), epsilon)
    return float(np.mean(np.abs((pred - true) / denominator)) * 100.0)


def compute_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float | None:
    true, pred = _paired_arrays(y_true, y_pred)
    ss_res = float(np.sum((pred - true) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return None
    return float(1.0 - ss_res / ss_tot)


def compute_normalized_rmse(y_true: ArrayLike, y_pred: ArrayLike, normalization: str = "range") -> float | None:
    true, _ = _paired_arrays(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    if normalization == "range":
        denominator = float(np.max(true) - np.min(true))
    elif normalization == "mean":
        denominator = float(np.mean(np.abs(true)))
    elif normalization == "std":
        denominator = float(np.std(true))
    else:
        raise InvalidPhysicalValueError("normalization must be one of: range, mean, std")
    if denominator == 0.0:
        return None
    return float(rmse / denominator)


def compute_max_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _paired_arrays(y_true, y_pred)
    return float(np.max(np.abs(pred - true)))


def validate_curve_pair(
    reference_curve: CurveData,
    target_curve: CurveData,
    interpolation_method: str = "linear",
    normalization: str = "range",
) -> dict[str, Any]:
    """Validate one measured/reference curve against a target curve."""
    common_time, reference_values, target_values, alignment_report = align_curves_to_common_time(
        reference_curve, target_curve, method=interpolation_method, overlap_only=True
    )
    warnings: list[str] = list(alignment_report["warnings"])
    r2 = compute_r2(reference_values, target_values)
    if r2 is None:
        warnings.append("r2 is undefined because reference values are constant")
    normalized_rmse = compute_normalized_rmse(reference_values, target_values, normalization=normalization)
    if normalized_rmse is None:
        warnings.append(f"normalized_rmse is undefined because {normalization} denominator is zero")
    zero_reference_count = int(np.sum(np.abs(reference_values) <= 1.0e-12))
    if zero_reference_count:
        warnings.append(f"mape used epsilon for {zero_reference_count} near-zero reference values")

    arrays = [common_time, reference_values, target_values]
    has_nan = any(np.isnan(array).any() for array in arrays)
    has_inf = any(np.isinf(array).any() for array in arrays)
    return {
        "curve_name": reference_curve.name,
        "curve_type": reference_curve.curve_type,
        "reference_source": reference_curve.source,
        "target_source": target_curve.source,
        "unit": reference_curve.unit,
        "num_points": int(common_time.size),
        "time_start": float(common_time[0]),
        "time_end": float(common_time[-1]),
        "rmse": compute_rmse(reference_values, target_values),
        "mae": compute_mae(reference_values, target_values),
        "mape": compute_mape(reference_values, target_values),
        "r2": r2,
        "normalized_rmse": normalized_rmse,
        "max_absolute_error": compute_max_absolute_error(reference_values, target_values),
        "alignment_report": alignment_report,
        "warnings": warnings,
        "success": not has_nan and not has_inf,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "zero_reference_count": zero_reference_count,
    }


def validate_multiple_curve_pairs(
    curve_pairs: Iterable[tuple[CurveData, CurveData]],
    interpolation_method: str = "linear",
    normalization: str = "range",
) -> dict[str, Any]:
    """Validate multiple curve pairs and aggregate successful metrics."""
    pairs = list(curve_pairs)
    reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise InvalidPhysicalValueError("curve_pairs must contain (reference_curve, target_curve) tuples")
        reference_curve, target_curve = pair
        try:
            report = validate_curve_pair(reference_curve, target_curve, interpolation_method, normalization)
        except Exception as exc:  # Deliberately record per-curve failures instead of aborting the full summary.
            report = _failed_curve_report(reference_curve, target_curve, exc)
        reports.append(report)
        for warning in report.get("warnings", []):
            warnings.append(f"curve {index}: {warning}")

    successful = [report for report in reports if report.get("success")]
    aggregate = _aggregate_metrics(successful, len(reports))
    has_nan = any(bool(report.get("has_nan")) for report in reports)
    has_inf = any(bool(report.get("has_inf")) for report in reports)
    return {
        "success": aggregate["num_successful_curves"] > 0 and not has_nan and not has_inf,
        "num_curves": len(reports),
        "curve_reports": reports,
        "aggregate_metrics": aggregate,
        "warnings": warnings,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _paired_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise InvalidPhysicalValueError("metric arrays must have matching shapes")
    if true.ndim != 1 or pred.ndim != 1:
        raise InvalidPhysicalValueError("metric arrays must be one-dimensional")
    if true.size == 0:
        raise InvalidPhysicalValueError("metric arrays must not be empty")
    if np.isnan(true).any() or np.isnan(pred).any() or np.isinf(true).any() or np.isinf(pred).any():
        raise InvalidPhysicalValueError("metric arrays must be finite")
    return true, pred


def _failed_curve_report(reference_curve: CurveData, target_curve: CurveData, exc: Exception) -> dict[str, Any]:
    return {
        "curve_name": getattr(reference_curve, "name", None),
        "curve_type": getattr(reference_curve, "curve_type", None),
        "reference_source": getattr(reference_curve, "source", None),
        "target_source": getattr(target_curve, "source", None),
        "unit": getattr(reference_curve, "unit", None),
        "num_points": 0,
        "time_start": None,
        "time_end": None,
        "rmse": None,
        "mae": None,
        "mape": None,
        "r2": None,
        "normalized_rmse": None,
        "max_absolute_error": None,
        "alignment_report": None,
        "warnings": [str(exc)],
        "success": False,
        "has_nan": False,
        "has_inf": False,
    }


def _aggregate_metrics(reports: list[dict[str, Any]], total_count: int) -> dict[str, float | int | None]:
    if not reports:
        return {
            "mean_rmse": None,
            "mean_mae": None,
            "mean_mape": None,
            "mean_normalized_rmse": None,
            "max_absolute_error": None,
            "num_successful_curves": 0,
            "num_failed_curves": total_count,
        }
    normalized = [report["normalized_rmse"] for report in reports if report["normalized_rmse"] is not None]
    return {
        "mean_rmse": float(np.mean([report["rmse"] for report in reports])),
        "mean_mae": float(np.mean([report["mae"] for report in reports])),
        "mean_mape": float(np.mean([report["mape"] for report in reports])),
        "mean_normalized_rmse": float(np.mean(normalized)) if normalized else None,
        "max_absolute_error": float(np.max([report["max_absolute_error"] for report in reports])),
        "num_successful_curves": len(reports),
        "num_failed_curves": total_count - len(reports),
    }
