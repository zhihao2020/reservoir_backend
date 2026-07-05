"""Lightweight resampling and alignment helpers for experimental data."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from reservoir_backend.data.schema import ExperimentalDataset, ExperimentalField


def resample_time_series(
    dataset: ExperimentalDataset,
    target_times: Iterable[float],
    *,
    time_field: str = "time",
) -> tuple[ExperimentalDataset, dict[str, object]]:
    """Linearly resample 1D fields that align with the time axis."""
    if time_field not in dataset.fields:
        raise ValueError(f"time field {time_field!r} is missing")
    source_time = np.asarray(dataset.fields[time_field].values, dtype=float).ravel()
    target = np.asarray(list(target_times), dtype=float)
    if source_time.size == 0 or target.size == 0:
        raise ValueError("source and target time arrays must be non-empty")
    order = np.argsort(source_time)
    sorted_time = source_time[order]
    fields: dict[str, ExperimentalField] = {}
    resampled_fields: list[str] = []
    skipped_fields: list[str] = []
    for name, field in dataset.fields.items():
        values = np.asarray(field.values, dtype=float)
        if name == time_field:
            fields[name] = ExperimentalField(name, target, field.unit, field.source_name, field.metadata.copy())
            resampled_fields.append(name)
        elif values.ndim == 1 and values.size == source_time.size:
            sorted_values = values[order]
            fields[name] = ExperimentalField(
                name,
                np.interp(target, sorted_time, sorted_values),
                field.unit,
                field.source_name,
                field.metadata.copy(),
            )
            resampled_fields.append(name)
        else:
            fields[name] = ExperimentalField(name, values.copy(), field.unit, field.source_name, field.metadata.copy())
            skipped_fields.append(name)
    output = dataset.with_fields(fields)
    summary = {
        "method": "linear_time_interpolation",
        "source_count": int(source_time.size),
        "target_count": int(target.size),
        "resampled_fields": resampled_fields,
        "skipped_fields": skipped_fields,
    }
    return output, summary


def align_fields_to_grid_shape(
    dataset: ExperimentalDataset,
    target_shape: tuple[int, ...],
    *,
    field_names: list[str] | None = None,
) -> tuple[ExperimentalDataset, dict[str, object]]:
    """Reshape compatible fields to a target structured-grid shape.

    This is shape alignment only. It does not perform corner-point regridding,
    geological upscaling, kriging, or spatial interpolation.
    """
    names = dataset.field_names if field_names is None else field_names
    target_size = int(np.prod(target_shape))
    fields: dict[str, ExperimentalField] = {}
    aligned: list[str] = []
    skipped: list[str] = []
    for name, field in dataset.fields.items():
        values = np.asarray(field.values, dtype=float)
        if name not in names:
            fields[name] = ExperimentalField(name, values.copy(), field.unit, field.source_name, field.metadata.copy())
            skipped.append(name)
            continue
        if values.shape == tuple(target_shape):
            aligned_values = values.copy()
        elif values.size == target_size:
            aligned_values = values.reshape(target_shape)
        elif values.size == 1:
            aligned_values = np.full(target_shape, float(values.ravel()[0]))
        else:
            raise ValueError(f"field {name!r} with shape {values.shape} cannot align to {target_shape}")
        fields[name] = ExperimentalField(name, aligned_values, field.unit, field.source_name, field.metadata.copy())
        aligned.append(name)
    return dataset.with_fields(fields), {
        "method": "shape_alignment",
        "target_shape": list(target_shape),
        "aligned_fields": aligned,
        "skipped_fields": skipped,
    }
