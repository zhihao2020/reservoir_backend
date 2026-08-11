from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.velocity import compute_darcy_velocity, compute_face_fluxes


def test_flux_x_shape() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert fluxes.flux_x.shape == (3, 4, 6)


def test_flux_y_shape() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert fluxes.flux_y.shape == (3, 5, 5)


def test_flux_z_shape() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert fluxes.flux_z.shape == (4, 4, 5)


def test_velocity_shape() -> None:
    grid = _grid()
    result = compute_darcy_velocity(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert result.velocity_x.values.shape == (3, 4, 5)
    assert result.velocity_y.values.shape == (3, 4, 5)
    assert result.velocity_z.values.shape == (3, 4, 5)


def test_velocity_no_nan_inf() -> None:
    grid = _grid()
    result = compute_darcy_velocity(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    for array in (
        result.face_fluxes.flux_x,
        result.face_fluxes.flux_y,
        result.face_fluxes.flux_z,
        result.velocity_x.values,
        result.velocity_y.values,
        result.velocity_z.values,
    ):
        assert not np.isnan(array).any()
        assert not np.isinf(array).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_x_direction_flow_positive() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert np.all(fluxes.flux_x[:, :, 1:-1] > 0.0)


def test_y_direction_flow_positive() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _y_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert np.all(fluxes.flux_y[:, 1:-1, :] > 0.0)


def test_z_direction_flow_positive() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _z_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert np.all(fluxes.flux_z[1:-1, :, :] > 0.0)


def test_zero_pressure_gradient_zero_flux() -> None:
    grid = _grid()
    pressure = Field3D.from_constant(grid, 5.0e6, name="pressure", unit="Pa")
    fluxes = compute_face_fluxes(grid, pressure, 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert np.allclose(fluxes.flux_x[:, :, 1:-1], 0.0)
    assert np.allclose(fluxes.flux_y[:, 1:-1, :], 0.0)
    assert np.allclose(fluxes.flux_z[1:-1, :, :], 0.0)


def test_no_flow_boundary_flux_zero() -> None:
    grid = _grid()
    fluxes = compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)
    assert np.allclose(fluxes.flux_x[:, :, 0], 0.0)
    assert np.allclose(fluxes.flux_x[:, :, -1], 0.0)
    assert np.allclose(fluxes.flux_y[:, 0, :], 0.0)
    assert np.allclose(fluxes.flux_y[:, -1, :], 0.0)
    assert np.allclose(fluxes.flux_z[0, :, :], 0.0)
    assert np.allclose(fluxes.flux_z[-1, :, :], 0.0)


def test_darcy_formula_x() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    pressure = Field3D(grid, np.array([[[10.0e6, 8.0e6]]]), name="pressure", unit="Pa")
    fluxes = compute_face_fluxes(grid, pressure, 10.0e-15, 10.0e-15, 10.0e-15, 2.0e-3)
    transmissibility = 10.0e-15 * float(grid.dy[0]) * float(grid.dz[0]) / (2.0e-3 * float(grid.dx[0]))
    expected = -transmissibility * (8.0e6 - 10.0e6)
    assert fluxes.flux_x[0, 0, 1] == pytest.approx(expected)


def test_darcy_formula_y() -> None:
    grid = Grid3D(nx=1, ny=2, nz=1, dx=2.0, dy=3.0, dz=4.0)
    pressure = Field3D(grid, np.array([[[10.0e6], [8.0e6]]]), name="pressure", unit="Pa")
    fluxes = compute_face_fluxes(grid, pressure, 10.0e-15, 10.0e-15, 10.0e-15, 2.0e-3)
    transmissibility = 10.0e-15 * float(grid.dx[0]) * float(grid.dz[0]) / (2.0e-3 * float(grid.dy[0]))
    expected = -transmissibility * (8.0e6 - 10.0e6)
    assert fluxes.flux_y[0, 1, 0] == pytest.approx(expected)


def test_darcy_formula_z() -> None:
    grid = Grid3D(nx=1, ny=1, nz=2, dx=2.0, dy=3.0, dz=4.0)
    pressure = Field3D(grid, np.array([[[10.0e6]], [[8.0e6]]]), name="pressure", unit="Pa")
    fluxes = compute_face_fluxes(grid, pressure, 10.0e-15, 10.0e-15, 10.0e-15, 2.0e-3)
    transmissibility = 10.0e-15 * float(grid.dx[0]) * float(grid.dy[0]) / (2.0e-3 * float(grid.dz[0]))
    expected = -transmissibility * (8.0e6 - 10.0e6)
    assert fluxes.flux_z[1, 0, 0] == pytest.approx(expected)


def test_anisotropic_permeability_velocity() -> None:
    grid = _grid()
    pressure = Field3D(grid, 10.0e6 - np.indices(grid.shape).sum(axis=0) * 1.0e6, name="pressure", unit="Pa")
    fluxes = compute_face_fluxes(grid, pressure, 300.0e-15, 100.0e-15, 10.0e-15, 1.0e-3)
    assert np.mean(np.abs(fluxes.flux_x[:, :, 1:-1])) > np.mean(np.abs(fluxes.flux_y[:, 1:-1, :]))
    assert np.mean(np.abs(fluxes.flux_y[:, 1:-1, :])) > np.mean(np.abs(fluxes.flux_z[1:-1, :, :]))


def test_negative_permeability_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_face_fluxes(grid, _x_pressure(grid), -1.0, 100.0e-15, 100.0e-15, 1.0e-3)


def test_invalid_viscosity_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_face_fluxes(grid, _x_pressure(grid), 100.0e-15, 100.0e-15, 100.0e-15, 0.0)


def test_velocity_consistent_with_pressure_solver() -> None:
    grid = Grid3D(nx=6, ny=4, nz=3, dx=10.0, dy=10.0, dz=5.0)
    pressure = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": 10.0e6, "right": 0.0},
    ).pressure
    result = compute_darcy_velocity(grid, pressure, 100.0e-15, 100.0e-15, 100.0e-15, 1.0e-3)

    assert np.all(result.face_fluxes.flux_x[:, :, 1:-1] > 0.0)
    assert np.mean(np.abs(result.face_fluxes.flux_x[:, :, 1:-1])) > 0.0
    assert np.allclose(result.face_fluxes.flux_y[:, 1:-1, :], 0.0, atol=1e-16)
    assert np.allclose(result.face_fluxes.flux_z[1:-1, :, :], 0.0, atol=1e-16)


def _grid() -> Grid3D:
    return Grid3D(nx=5, ny=4, nz=3, dx=10.0, dy=8.0, dz=6.0)


def _x_pressure(grid: Grid3D) -> Field3D:
    values = 10.0e6 - np.arange(grid.nx, dtype=float).reshape(1, 1, grid.nx) * 1.0e6
    return Field3D(grid, np.broadcast_to(values, grid.shape), name="pressure", unit="Pa")


def _y_pressure(grid: Grid3D) -> Field3D:
    values = 10.0e6 - np.arange(grid.ny, dtype=float).reshape(1, grid.ny, 1) * 1.0e6
    return Field3D(grid, np.broadcast_to(values, grid.shape), name="pressure", unit="Pa")


def _z_pressure(grid: Grid3D) -> Field3D:
    values = 10.0e6 - np.arange(grid.nz, dtype=float).reshape(grid.nz, 1, 1) * 1.0e6
    return Field3D(grid, np.broadcast_to(values, grid.shape), name="pressure", unit="Pa")
