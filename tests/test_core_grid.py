from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import GridIndexError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D


def test_grid_total_cells() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    assert grid.total_cells == 24


def test_grid_cell_volume() -> None:
    grid = Grid3D(nx=1, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    assert float(grid.cell_volume) == 24.0


def test_index_and_ijk_roundtrip() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                assert grid.ijk(grid.index(i, j, k)) == (i, j, k)


def test_neighbors_center_cell() -> None:
    grid = Grid3D(nx=3, ny=3, nz=3, dx=1.0, dy=1.0, dz=1.0)
    neighbors = grid.get_neighbors(grid.index(1, 1, 1))
    assert len(neighbors) == 6


def test_neighbors_corner_cell() -> None:
    grid = Grid3D(nx=3, ny=3, nz=3, dx=1.0, dy=1.0, dz=1.0)
    neighbors = grid.get_neighbors(grid.index(0, 0, 0))
    assert len(neighbors) == 3


def test_invalid_index_raises() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(GridIndexError):
        grid.index(4, 0, 0)
    with pytest.raises(GridIndexError):
        grid.ijk(grid.total_cells)


def test_active_mask() -> None:
    mask = np.ones((1, 3, 3), dtype=bool)
    mask[0, 1, 2] = False
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0, active_mask=mask)
    neighbors = grid.get_neighbors(grid.index(1, 1, 0))
    assert grid.index(2, 1, 0) not in neighbors
    assert len(neighbors) == 3


def test_invalid_grid_dimensions_raise() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        Grid3D(nx=0, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)


def test_active_mask_wrong_shape_raises() -> None:
    with pytest.raises(GridIndexError):
        Grid3D(
            nx=2,
            ny=2,
            nz=1,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            active_mask=np.ones((2, 2), dtype=bool),
        )
