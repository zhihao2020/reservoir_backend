"""Point-to-grid mapping and simple field resampling utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D


def map_points_to_grid_nearest(grid: Grid3D, points: ArrayLike, values: ArrayLike) -> Field3D:
    """Map point values to nearest grid cells."""
    points_array, values_array = _validate_points(points, values)
    output = np.full(grid.shape, np.nan, dtype=float)
    for point, value in zip(points_array, values_array):
        i, j, k = _point_to_ijk(grid, point)
        output[k, j, i] = value
    return Field3D(grid, output, name="mapped_nearest")


def map_points_to_grid_idw(grid: Grid3D, points: ArrayLike, values: ArrayLike, power: float = 2.0) -> Field3D:
    """Map point values to all grid cells using inverse-distance weighting."""
    points_array, values_array = _validate_points(points, values)
    power = float(power)
    if power <= 0.0:
        raise ValueError("power must be positive")
    output = np.empty(grid.shape, dtype=float)
    centers = _cell_centers(grid)
    for index in np.ndindex(grid.shape):
        center = centers[index]
        distances = np.linalg.norm(points_array - center, axis=1)
        if np.any(distances == 0.0):
            output[index] = values_array[np.argmin(distances)]
        else:
            weights = 1.0 / distances**power
            output[index] = np.sum(weights * values_array) / np.sum(weights)
    return Field3D(grid, output, name="mapped_idw")


def resample_field_same_grid(field: Field3D, target_grid: Grid3D) -> Field3D:
    """Return a copy if source and target grids are identical."""
    if field.grid != target_grid:
        raise NotImplementedError("only same-grid resampling is implemented")
    return field.copy()


def fill_missing_values(field: Field3D, method: str = "nearest", value: float = 0.0) -> Field3D:
    """Fill NaN values using a constant or nearest valid cell."""
    values = field.values.copy()
    missing = np.isnan(values)
    if not missing.any():
        return field.copy()
    if method == "constant":
        values[missing] = float(value)
    elif method == "nearest":
        valid_indices = np.argwhere(~missing)
        if valid_indices.size == 0:
            raise InvalidPhysicalValueError("cannot nearest-fill a field with no valid values")
        missing_indices = np.argwhere(missing)
        for idx in missing_indices:
            distances = np.sum((valid_indices - idx) ** 2, axis=1)
            nearest = valid_indices[int(np.argmin(distances))]
            values[tuple(idx)] = values[tuple(nearest)]
    else:
        raise ValueError("method must be 'nearest' or 'constant'")
    return Field3D(field.grid, values, name=field.name, unit=field.unit, confidence=field.confidence)


def _validate_points(points: ArrayLike, values: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points_array = np.asarray(points, dtype=float)
    values_array = np.asarray(values, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if values_array.shape != (points_array.shape[0],):
        raise ValueError("values must have shape (n,)")
    if np.isnan(points_array).any() or np.isnan(values_array).any():
        raise InvalidPhysicalValueError("points and values must be finite")
    if np.isinf(points_array).any() or np.isinf(values_array).any():
        raise InvalidPhysicalValueError("points and values must be finite")
    return points_array, values_array


def _point_to_ijk(grid: Grid3D, point: NDArray[np.float64]) -> tuple[int, int, int]:
    x, y, z = point
    if x < 0.0 or y < 0.0 or z < 0.0 or x >= grid.nx * grid.dx or y >= grid.ny * grid.dy or z >= grid.nz * grid.dz:
        raise InvalidPhysicalValueError("point lies outside grid domain")
    i = min(int(x / grid.dx), grid.nx - 1)
    j = min(int(y / grid.dy), grid.ny - 1)
    k = min(int(z / grid.dz), grid.nz - 1)
    return i, j, k


def _cell_centers(grid: Grid3D) -> NDArray[np.float64]:
    centers = np.empty(grid.shape + (3,), dtype=float)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                centers[k, j, i] = ((i + 0.5) * grid.dx, (j + 0.5) * grid.dy, (k + 0.5) * grid.dz)
    return centers
