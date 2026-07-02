from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError, NonNeighborCellError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.transmissibility import (
    compute_directional_transmissibility,
    compute_transmissibility_between_cells,
    harmonic_average,
)


def test_harmonic_average_equal_values() -> None:
    assert harmonic_average(10.0, 10.0) == pytest.approx(10.0)


def test_harmonic_average_different_values() -> None:
    assert harmonic_average(10.0, 30.0) == pytest.approx(15.0)


def test_harmonic_average_zero_perm() -> None:
    assert harmonic_average(0.0, 30.0) == pytest.approx(0.0)
    assert harmonic_average(10.0, 0.0) == pytest.approx(0.0)


def test_harmonic_average_negative_perm_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        harmonic_average(-1.0, 30.0)
    with pytest.raises(InvalidPhysicalValueError):
        harmonic_average(10.0, -1.0)


def test_transmissibility_x_direction() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    mu = 2.0
    kx = np.array([[[10.0, 30.0]]])
    t = compute_transmissibility_between_cells(
        grid, kx=kx, ky=1.0, kz=1.0, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(1, 0, 0)
    )
    assert t == pytest.approx(15.0 * grid.dy * grid.dz / (mu * grid.dx))


def test_transmissibility_y_direction() -> None:
    grid = Grid3D(nx=1, ny=2, nz=1, dx=2.0, dy=3.0, dz=4.0)
    mu = 2.0
    ky = np.array([[[10.0], [30.0]]])
    t = compute_transmissibility_between_cells(
        grid, kx=1.0, ky=ky, kz=1.0, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(0, 1, 0)
    )
    assert t == pytest.approx(15.0 * grid.dx * grid.dz / (mu * grid.dy))


def test_transmissibility_z_direction() -> None:
    grid = Grid3D(nx=1, ny=1, nz=2, dx=2.0, dy=3.0, dz=4.0)
    mu = 2.0
    kz = np.array([[[10.0]], [[30.0]]])
    t = compute_transmissibility_between_cells(
        grid, kx=1.0, ky=1.0, kz=kz, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(0, 0, 1)
    )
    assert t == pytest.approx(15.0 * grid.dx * grid.dy / (mu * grid.dz))


def test_anisotropic_permeability() -> None:
    grid = Grid3D(nx=2, ny=2, nz=2, dx=2.0, dy=3.0, dz=5.0)
    mu = 2.0
    tx = compute_transmissibility_between_cells(
        grid, kx=10.0, ky=20.0, kz=40.0, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(1, 0, 0)
    )
    ty = compute_transmissibility_between_cells(
        grid, kx=10.0, ky=20.0, kz=40.0, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(0, 1, 0)
    )
    tz = compute_transmissibility_between_cells(
        grid, kx=10.0, ky=20.0, kz=40.0, mu=mu, cell_a=grid.index(0, 0, 0), cell_b=grid.index(0, 0, 1)
    )
    assert len({tx, ty, tz}) == 3


def test_invalid_viscosity_raises() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(InvalidPhysicalValueError):
        compute_transmissibility_between_cells(
            grid, kx=1.0, ky=1.0, kz=1.0, mu=0.0, cell_a=0, cell_b=1
        )
    with pytest.raises(InvalidPhysicalValueError):
        compute_directional_transmissibility(grid, k_field=1.0, mu=-1.0, direction="x")


def test_non_neighbor_cells_raise() -> None:
    grid = Grid3D(nx=3, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(NonNeighborCellError):
        compute_transmissibility_between_cells(
            grid, kx=1.0, ky=1.0, kz=1.0, mu=1.0, cell_a=grid.index(0, 0, 0), cell_b=grid.index(2, 0, 0)
        )


def test_diagonal_cells_raise() -> None:
    grid = Grid3D(nx=2, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(NonNeighborCellError):
        compute_transmissibility_between_cells(
            grid, kx=1.0, ky=1.0, kz=1.0, mu=1.0, cell_a=grid.index(0, 0, 0), cell_b=grid.index(1, 1, 0)
        )


def test_field3d_permeability_input() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    kx = Field3D(grid, np.array([[[10.0, 30.0]]]), name="kx", unit="m2")
    t = compute_transmissibility_between_cells(
        grid, kx=kx, ky=1.0, kz=1.0, mu=2.0, cell_a=grid.index(0, 0, 0), cell_b=grid.index(1, 0, 0)
    )
    assert t == pytest.approx(15.0 * grid.dy * grid.dz / (2.0 * grid.dx))


def test_ndarray_permeability_input() -> None:
    grid = Grid3D(nx=3, ny=1, nz=1, dx=1.0, dy=2.0, dz=3.0)
    kx = np.array([[[10.0, 30.0, 30.0]]])
    tx = compute_directional_transmissibility(grid, k_field=kx, mu=2.0, direction="x")
    assert tx.shape == (1, 1, 2)
    assert tx[0, 0, 0] == pytest.approx(15.0 * grid.dy * grid.dz / (2.0 * grid.dx))


def test_scalar_permeability_input() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    tx = compute_directional_transmissibility(grid, k_field=10.0, mu=2.0, direction="x")
    assert tx.shape == (1, 1, 1)
    assert tx[0, 0, 0] == pytest.approx(10.0 * grid.dy * grid.dz / (2.0 * grid.dx))
