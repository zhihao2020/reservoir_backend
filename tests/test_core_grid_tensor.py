"""Tests for non-uniform orthogonal spacing on Grid3D."""

from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.transmissibility import compute_directional_transmissibility


def test_uniform_grid_broadcasts_scalar_spacing() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=2.0, dy=3.0, dz=4.0)
    assert grid.is_uniform
    assert np.allclose(float(grid.dx[0]), 2.0)
    assert np.allclose(float(grid.dy[0]), 3.0)
    assert np.allclose(float(grid.dz[0]), 4.0)
    assert float(np.unique(grid.cell_volume)[0]) == 24.0


def test_nonuniform_spacing_volumes_and_centers() -> None:
    grid = Grid3D(nx=3, ny=1, nz=2, dx=[1.0, 2.0, 3.0], dy=1.0, dz=[4.0, 6.0])
    assert not grid.is_uniform
    assert np.allclose(grid.spacing_i, [1.0, 2.0, 3.0])
    assert np.allclose(grid.cell_volumes[0, 0, :], [4.0, 8.0, 12.0])
    centers = grid.cell_centers()
    assert np.allclose(centers[0, 0, :, 0], [0.5, 2.0, 4.5])
    assert np.allclose(centers[:, 0, 0, 2], [2.0, 7.0])


def test_face_areas_and_center_distances() -> None:
    grid = Grid3D(nx=3, ny=2, nz=2, dx=[1.0, 2.0, 3.0], dy=[4.0, 5.0], dz=[6.0, 7.0])
    dist_i = grid.center_distances_i()
    assert np.allclose(dist_i, [1.5, 2.5])
    areas_x = grid.x_face_areas()
    assert areas_x.shape == (2, 2, 2)
    assert np.allclose(areas_x[0, 0, :], 4.0 * 6.0)
    assert np.allclose(areas_x[0, 1, :], 5.0 * 6.0)


def test_directional_transmissibility_matches_uniform_formula() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=2.0, dy=3.0, dz=4.0)
    tx = compute_directional_transmissibility(grid, 1.0e-12, 1.0e-3, "x")
    expected = 1.0e-12 * 3.0 * 4.0 / (1.0e-3 * 2.0)
    assert np.allclose(tx, expected)


def test_directional_transmissibility_nonuniform() -> None:
    grid = Grid3D(nx=3, ny=1, nz=1, dx=[1.0, 3.0, 1.0], dy=2.0, dz=4.0)
    tx = compute_directional_transmissibility(grid, 1.0, 1.0, "x")
    # T = k * (dy*dz) / dist, dist = 0.5*(dx_i+dx_{i+1})
    assert tx.shape == (1, 1, 2)
    assert np.isclose(tx[0, 0, 0], 2.0 * 4.0 / 2.0)
    assert np.isclose(tx[0, 0, 1], 2.0 * 4.0 / 2.0)


def test_invalid_spacing_length_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        Grid3D(nx=3, ny=1, nz=1, dx=[1.0, 2.0], dy=1.0, dz=1.0)
