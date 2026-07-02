from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D


def test_field_shape_match_grid(small_grid: Grid3D) -> None:
    values = np.zeros(small_grid.shape)
    field = Field3D(small_grid, values, name="pressure", unit="Pa")
    assert field.values.shape == small_grid.shape
    assert field.name == "pressure"
    assert field.unit == "Pa"


def test_field_wrong_shape_raises(small_grid: Grid3D) -> None:
    with pytest.raises(FieldShapeError):
        Field3D(small_grid, np.zeros((3, 4, 2)))


def test_field_from_constant(small_grid: Grid3D) -> None:
    field = Field3D.from_constant(small_grid, 0.25, name="phi")
    assert np.allclose(field.values, 0.25)
    assert field.values.shape == small_grid.shape


def test_field_clip(small_grid: Grid3D) -> None:
    values = np.linspace(0.0, 1.0, small_grid.total_cells).reshape(small_grid.shape)
    field = Field3D(small_grid, values, name="sw")
    clipped = field.clip(0.2, 0.8)
    assert clipped.values.min() >= 0.2
    assert clipped.values.max() <= 0.8
    assert field.values.min() == 0.0


def test_field_fill_nan(small_grid: Grid3D) -> None:
    values = np.ones(small_grid.shape)
    values[0, 0, 0] = np.nan
    field = Field3D(small_grid, values)
    filled = field.fill_nan(7.0)
    assert filled.values[0, 0, 0] == 7.0
    assert not np.isnan(filled.values).any()


def test_field_confidence_shape(small_grid: Grid3D) -> None:
    confidence = np.full(small_grid.shape, 0.8)
    field = Field3D(small_grid, np.ones(small_grid.shape), confidence=confidence)
    assert np.allclose(field.confidence, 0.8)


def test_field_confidence_wrong_shape_raises(small_grid: Grid3D) -> None:
    with pytest.raises(FieldShapeError):
        Field3D(small_grid, np.ones(small_grid.shape), confidence=np.ones((1, 1, 1)))


def test_assert_same_grid(small_grid: Grid3D) -> None:
    field = Field3D.from_constant(small_grid, 1.0)
    same = Field3D.from_constant(Grid3D(4, 3, 2, 2.0, 3.0, 4.0), 2.0)
    field.assert_same_grid(same)

    different = Field3D.from_constant(Grid3D(4, 3, 1, 2.0, 3.0, 4.0), 1.0)
    with pytest.raises(GridMismatchError):
        field.assert_same_grid(different)


def test_copy_is_deep(small_grid: Grid3D) -> None:
    field = Field3D.from_constant(small_grid, 1.0, confidence=0.5)
    copied = field.copy(name="copy")
    copied.values[0, 0, 0] = 9.0
    assert field.values[0, 0, 0] == 1.0
    assert copied.name == "copy"


def test_to_numpy_returns_copy_by_default(small_grid: Grid3D) -> None:
    field = Field3D.from_constant(small_grid, 3.0)
    array = field.to_numpy()
    array[0, 0, 0] = 0.0
    assert field.values[0, 0, 0] == 3.0
