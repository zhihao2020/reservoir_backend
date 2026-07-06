"""Lightweight synthetic twin data structures for dynamic field fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SyntheticTwinMetadata:
    """Metadata for a synthetic twin fusion run."""

    twin_id: str
    case_id: str
    run_id: str
    created_at: str
    grid_shape: tuple[int, ...]
    time_steps: tuple[float, ...]
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.twin_id or not self.case_id or not self.run_id:
            raise ValueError("twin_id, case_id, and run_id must be non-empty")
        shape = tuple(int(v) for v in self.grid_shape)
        if not shape or any(v <= 0 for v in shape):
            raise ValueError("grid_shape entries must be positive")
        times = tuple(float(v) for v in self.time_steps)
        if not times or not np.isfinite(times).all():
            raise ValueError("time_steps must be non-empty and finite")
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "time_steps", times)

    def to_dict(self) -> dict[str, Any]:
        return {
            "twin_id": self.twin_id,
            "case_id": self.case_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "grid_shape": list(self.grid_shape),
            "time_steps": list(self.time_steps),
            "source_name": self.source_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StaticFieldRecord:
    """Static property field record with provenance and optional truth."""

    field_name: str
    values: ArrayLike
    unit: str
    source: str
    confidence: ArrayLike | float | None = None
    variance: ArrayLike | float | None = None
    mask: ArrayLike | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    truth: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not self.field_name or not self.source:
            raise ValueError("field_name and source must be non-empty")
        values = np.asarray(self.values, dtype=float)
        if values.ndim == 0:
            raise ValueError("static field values must be array-like")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "confidence", _optional_array(self.confidence, values.shape, "confidence"))
        object.__setattr__(self, "variance", _optional_array(self.variance, values.shape, "variance"))
        object.__setattr__(self, "mask", _optional_bool_array(self.mask, values.shape))
        object.__setattr__(self, "truth", _optional_array(self.truth, values.shape, "truth"))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.values).shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "shape": list(self.shape),
            "unit": self.unit,
            "source": self.source,
            "has_confidence": self.confidence is not None,
            "has_variance": self.variance is not None,
            "has_mask": self.mask is not None,
            "has_truth": self.truth is not None,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class DynamicFieldRecord:
    """Time-indexed dynamic field record."""

    field_name: str
    values: ArrayLike
    time_steps: tuple[float, ...] | list[float]
    unit: str
    source: str
    confidence: ArrayLike | float | None = None
    variance: ArrayLike | float | None = None
    mask: ArrayLike | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    truth: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not self.field_name or not self.source:
            raise ValueError("field_name and source must be non-empty")
        values = np.asarray(self.values, dtype=float)
        if values.ndim < 2:
            raise ValueError("dynamic field values must have time plus spatial dimensions")
        times = tuple(float(v) for v in self.time_steps)
        if len(times) != values.shape[0] or not np.isfinite(times).all():
            raise ValueError("time_steps length must match dynamic field time dimension")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "time_steps", times)
        object.__setattr__(self, "confidence", _optional_array(self.confidence, values.shape, "confidence"))
        object.__setattr__(self, "variance", _optional_array(self.variance, values.shape, "variance"))
        object.__setattr__(self, "mask", _optional_bool_array(self.mask, values.shape))
        object.__setattr__(self, "truth", _optional_array(self.truth, values.shape, "truth"))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.values).shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "shape": list(self.shape),
            "time_steps": list(self.time_steps),
            "unit": self.unit,
            "source": self.source,
            "has_confidence": self.confidence is not None,
            "has_variance": self.variance is not None,
            "has_mask": self.mask is not None,
            "has_truth": self.truth is not None,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ProductionSeriesRecord:
    """Time-series production or water-cut record."""

    series_name: str
    time: ArrayLike
    values: ArrayLike
    unit: str
    source: str
    confidence: ArrayLike | float | None = None
    mask: ArrayLike | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    truth: ArrayLike | None = None

    def __post_init__(self) -> None:
        if not self.series_name or not self.source:
            raise ValueError("series_name and source must be non-empty")
        time = np.asarray(self.time, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if time.ndim != 1 or values.ndim != 1 or time.shape != values.shape:
            raise ValueError("production time and values must be 1D arrays with matching shape")
        if not np.isfinite(time).all():
            raise ValueError("production time values must be finite")
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "confidence", _optional_array(self.confidence, values.shape, "confidence"))
        object.__setattr__(self, "mask", _optional_bool_array(self.mask, values.shape))
        object.__setattr__(self, "truth", _optional_array(self.truth, values.shape, "truth"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_name": self.series_name,
            "num_samples": int(np.asarray(self.values).size),
            "unit": self.unit,
            "source": self.source,
            "has_confidence": self.confidence is not None,
            "has_mask": self.mask is not None,
            "has_truth": self.truth is not None,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class DynamicFusionSummary:
    """Container for synthetic twin fusion outputs and diagnostics."""

    metadata: SyntheticTwinMetadata
    static_fields: dict[str, dict[str, Any]]
    dynamic_fields: dict[str, dict[str, Any]]
    production_series: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.diagnostics.get("success", False)),
            "metadata": self.metadata.to_dict(),
            "static_fields": self.static_fields,
            "dynamic_fields": self.dynamic_fields,
            "production_series": self.production_series,
            "diagnostics": self.diagnostics,
            "provenance": self.provenance,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def _optional_array(value: ArrayLike | float | None, shape: tuple[int, ...], name: str) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        array = np.full(shape, float(array), dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if name != "truth" and np.isinf(array).any():
        raise ValueError(f"{name} must not contain Inf")
    return array.copy()


def _optional_bool_array(value: ArrayLike | None, shape: tuple[int, ...]) -> NDArray[np.bool_] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"mask shape {array.shape} does not match {shape}")
    return array.copy()
