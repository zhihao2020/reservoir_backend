"""Same-grid field fusion utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.exceptions import GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.fusion.confidence import combine_confidence, normalize_confidence


def weighted_average_fields(
    fields: list[Field3D],
    weights: list[float] | None = None,
    confidence_fields: list[Field3D | NDArray[np.float64]] | None = None,
    clip_range: tuple[float, float] | None = None,
) -> tuple[Field3D, dict[str, object]]:
    """Fuse same-grid fields with optional source weights and confidence."""
    validate_same_grid(fields)
    weight_values = validate_weights(weights, len(fields))
    grid = fields[0].grid
    values_stack = np.stack([field.values for field in fields], axis=0)
    finite_mask = ~np.isnan(values_stack)

    effective_weights = weight_values.reshape((-1,) + (1,) * len(grid.shape)) * np.ones_like(values_stack)
    if confidence_fields is not None:
        if len(confidence_fields) != len(fields):
            raise ValueError("confidence_fields must match fields length")
        confidence_arrays = []
        for confidence in confidence_fields:
            normalized = normalize_confidence(confidence)
            confidence_arrays.append(normalized.values if isinstance(normalized, Field3D) else normalized)
        effective_weights *= np.stack(confidence_arrays, axis=0)
    else:
        confidence_arrays = [
            field.confidence if field.confidence is not None else np.ones(grid.shape, dtype=float)
            for field in fields
        ]
        effective_weights *= np.stack(confidence_arrays, axis=0)

    effective_weights = np.where(finite_mask, effective_weights, 0.0)
    total_weight = np.sum(effective_weights, axis=0)
    numerator = np.nansum(values_stack * effective_weights, axis=0)
    fused_values = np.divide(
        numerator,
        total_weight,
        out=np.full(grid.shape, np.nan, dtype=float),
        where=total_weight > 0.0,
    )

    zero_weight_cells = int(np.count_nonzero(total_weight == 0.0))
    nan_cells_count = int(np.count_nonzero(np.isnan(fused_values)))
    clipped_cells = 0
    if clip_range is not None:
        lower, upper = clip_range
        if lower > upper:
            raise ValueError("clip_range lower must be <= upper")
        clipped_cells = int(np.count_nonzero((fused_values < lower) | (fused_values > upper)))
        fused_values = np.clip(fused_values, lower, upper)

    confidence = np.divide(
        total_weight,
        np.sum(weight_values),
        out=np.zeros(grid.shape, dtype=float),
        where=np.sum(weight_values) > 0.0,
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    fused = Field3D(grid=grid, values=fused_values, name="fused_field", unit=fields[0].unit, confidence=confidence)
    report = build_fusion_report(
        field_count=len(fields),
        used_weights=weight_values.tolist(),
        fused_values=fused_values,
        confidence=confidence,
        nan_cells_count=nan_cells_count,
        zero_weight_cells=zero_weight_cells,
        clipped_cells=clipped_cells,
    )
    return fused, report


def fuse_saturation_fields(
    sw_fields: list[Field3D],
    confidence_fields: list[Field3D | NDArray[np.float64]] | None = None,
    swi: float = 0.0,
    sor: float = 0.0,
) -> tuple[Field3D, dict[str, object]]:
    """Fuse saturation fields and clip to `[swi, 1 - sor]`."""
    if swi < 0.0 or sor < 0.0 or swi + sor >= 1.0:
        raise InvalidPhysicalValueError("swi and sor must be valid residual saturations")
    fused, report = weighted_average_fields(
        sw_fields,
        confidence_fields=confidence_fields,
        clip_range=(float(swi), 1.0 - float(sor)),
    )
    fused.name = "sw_fused"
    return fused, report


def update_simulated_with_observed(
    sw_sim: Field3D,
    sw_obs: Field3D,
    alpha: float,
    swi: float = 0.0,
    sor: float = 0.0,
) -> tuple[Field3D, dict[str, object]]:
    """Blend simulated and observed saturation fields."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be within [0, 1]")
    sw_sim.assert_same_grid(sw_obs)
    values = float(alpha) * sw_sim.values + (1.0 - float(alpha)) * sw_obs.values
    lower, upper = float(swi), 1.0 - float(sor)
    clipped_cells = int(np.count_nonzero((values < lower) | (values > upper)))
    values = np.clip(values, lower, upper)
    field = Field3D(sw_sim.grid, values, name="updated_sw", unit=sw_sim.unit)
    report = build_fusion_report(
        field_count=2,
        used_weights=[float(alpha), 1.0 - float(alpha)],
        fused_values=values,
        confidence=np.ones(sw_sim.grid.shape),
        nan_cells_count=int(np.count_nonzero(np.isnan(values))),
        zero_weight_cells=0,
        clipped_cells=clipped_cells,
    )
    return field, report


def fuse_dynamic_state(simulated_state, observed_fields: dict[str, Field3D], config: dict) -> dict[str, object]:
    """Fuse a minimal dynamic state dictionary with observed fields."""
    alpha = float(config.get("alpha", 0.5))
    result: dict[str, object] = {}
    if "sw_simulated" in simulated_state and "sw_observed" in observed_fields:
        result["updated_sw"] = update_simulated_with_observed(
            simulated_state["sw_simulated"],
            observed_fields["sw_observed"],
            alpha,
            config.get("swi", 0.0),
            config.get("sor", 0.0),
        )
    return result


def validate_same_grid(fields: list[Field3D]) -> None:
    """Validate that all fields are on the same grid."""
    if not fields:
        raise ValueError("fields must not be empty")
    first = fields[0]
    for field in fields[1:]:
        if field.grid != first.grid:
            raise GridMismatchError("all fields must share the same grid")


def validate_weights(weights: list[float] | None, count: int) -> NDArray[np.float64]:
    """Validate non-negative source weights."""
    if weights is None:
        return np.ones(count, dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,):
        raise ValueError("weights length must match field count")
    if np.isnan(values).any() or np.isinf(values).any() or (values < 0.0).any():
        raise ValueError("weights must be finite and non-negative")
    if float(np.sum(values)) <= 0.0:
        raise ValueError("at least one weight must be positive")
    return values


def build_fusion_report(
    field_count: int,
    used_weights: list[float],
    fused_values: NDArray[np.float64],
    confidence: NDArray[np.float64],
    nan_cells_count: int,
    zero_weight_cells: int,
    clipped_cells: int,
) -> dict[str, object]:
    """Build a standard fusion report."""
    return {
        "field_count": field_count,
        "used_weights": used_weights,
        "nan_cells_count": nan_cells_count,
        "zero_weight_cells": zero_weight_cells,
        "clipped_cells": clipped_cells,
        "fused_min": float(np.nanmin(fused_values)) if not np.all(np.isnan(fused_values)) else np.nan,
        "fused_max": float(np.nanmax(fused_values)) if not np.all(np.isnan(fused_values)) else np.nan,
        "confidence_min": float(np.nanmin(confidence)),
        "confidence_max": float(np.nanmax(confidence)),
        "has_nan": bool(np.isnan(fused_values).any()),
        "has_inf": bool(np.isinf(fused_values).any()),
    }
