from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.fusion.field_mapper import (
    fill_missing_values,
    map_points_to_grid_idw,
    map_points_to_grid_nearest,
    resample_field_same_grid,
)


def test_point_to_grid_nearest_single_point() -> None:
    grid = _grid()
    mapped = map_points_to_grid_nearest(grid, points=[[0.2, 0.2, 0.2]], values=[10.0])
    assert mapped.values[0, 0, 0] == pytest.approx(10.0)


def test_point_to_grid_nearest_multiple_points() -> None:
    grid = _grid()
    mapped = map_points_to_grid_nearest(
        grid,
        points=[[0.2, 0.2, 0.2], [1.2, 1.2, 0.2]],
        values=[10.0, 20.0],
    )
    assert mapped.values[0, 0, 0] == pytest.approx(10.0)
    assert mapped.values[0, 1, 1] == pytest.approx(20.0)


def test_point_to_grid_idw_basic() -> None:
    grid = _grid()
    mapped = map_points_to_grid_idw(
        grid,
        points=[[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]],
        values=[10.0, 20.0],
    )
    assert np.nanmin(mapped.values) >= 10.0
    assert np.nanmax(mapped.values) <= 20.0


def test_point_to_grid_outside_domain_raises_or_ignored() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        map_points_to_grid_nearest(grid, points=[[5.0, 0.0, 0.0]], values=[1.0])


def test_resample_same_grid_no_change() -> None:
    grid = _grid()
    field = Field3D(grid, np.arange(grid.total_cells).reshape(grid.shape))
    resampled = resample_field_same_grid(field, grid)
    assert np.allclose(resampled.values, field.values)


def test_fill_missing_values_constant() -> None:
    grid = _grid()
    values = np.ones(grid.shape)
    values[0, 0, 0] = np.nan
    field = Field3D(grid, values)
    filled = fill_missing_values(field, method="constant", value=7.0)
    assert filled.values[0, 0, 0] == pytest.approx(7.0)


def test_fill_missing_values_nearest() -> None:
    grid = _grid()
    values = np.array([[[np.nan, 2.0], [3.0, 4.0]]])
    field = Field3D(grid, values)
    filled = fill_missing_values(field, method="nearest")
    assert not np.isnan(filled.values).any()
    assert filled.values[0, 0, 0] in {2.0, 3.0}


def test_mapper_output_shape() -> None:
    grid = _grid()
    mapped = map_points_to_grid_idw(grid, points=[[0.5, 0.5, 0.5]], values=[10.0])
    assert mapped.values.shape == grid.shape


def _grid() -> Grid3D:
    return Grid3D(nx=2, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
