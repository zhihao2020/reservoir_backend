from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_2d


def test_2d_pressure_shape() -> None:
    grid = Grid3D(nx=10, ny=8, nz=1, dx=10.0, dy=10.0, dz=2.0)
    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": 10.0e6, "right": 0.0},
    )
    assert result.pressure.values.shape == (1, 8, 10)
    assert result.pressure.unit == "Pa"
    assert result.report["solver"] == "scipy.sparse.linalg.spsolve"


def test_2d_no_nan_inf() -> None:
    pressure = _left_right_solution().pressure.values
    assert not np.isnan(pressure).any()
    assert not np.isinf(pressure).any()


def test_2d_left_right_dirichlet_gradient() -> None:
    pressure = _left_right_solution().pressure.values[0]
    row_differences = np.diff(pressure, axis=1)
    assert np.all(row_differences < 0.0)


def test_2d_pressure_between_boundaries() -> None:
    pressure = _left_right_solution().pressure.values
    assert pressure.min() >= -1.0e-6
    assert pressure.max() <= 10.0e6 + 1.0e-6


def test_2d_injector_producer_gradient() -> None:
    grid = Grid3D(nx=7, ny=5, nz=1, dx=20.0, dy=20.0, dz=5.0)
    injector = Well("I1", "injection", grid, i=1, j=2, k=0, rate=1.0e-5)
    producer = Well("P1", "production", grid, i=5, j=2, k=0, rate=1.0e-5)

    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        wells=[injector, producer],
        reference_pressure=0.0,
    )
    pressure = result.pressure.values

    assert pressure[0, 2, 1] > pressure[0, 2, 5]
    assert result.report["pressure_reference_applied"] == 1


def test_2d_homogeneous_symmetry() -> None:
    grid = Grid3D(nx=7, ny=5, nz=1, dx=20.0, dy=20.0, dz=5.0)
    wells = [
        Well("I1", "injection", grid, i=2, j=1, k=0, rate=1.0e-5),
        Well("I2", "injection", grid, i=2, j=3, k=0, rate=1.0e-5),
        Well("P1", "production", grid, i=4, j=1, k=0, rate=1.0e-5),
        Well("P2", "production", grid, i=4, j=3, k=0, rate=1.0e-5),
    ]
    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        wells=wells,
        reference_pressure=0.0,
    )
    pressure = result.pressure.values[0]

    assert np.allclose(pressure[0, :], pressure[-1, :], rtol=1e-10, atol=1e-8)
    assert np.allclose(pressure[1, :], pressure[-2, :], rtol=1e-10, atol=1e-8)


def test_2d_mass_balance_source_sink() -> None:
    grid = Grid3D(nx=7, ny=5, nz=1, dx=20.0, dy=20.0, dz=5.0)
    wells = [
        Well("I1", "injection", grid, i=1, j=2, k=0, rate=2.0e-5),
        Well("P1", "production", grid, i=5, j=2, k=0, rate=2.0e-5),
    ]
    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        wells=wells,
    )

    assert result.report["mass_balance_error"] < 1.0e-8
    assert result.report["net_well_rate_m3_s"] == pytest.approx(0.0)


def test_2d_dirichlet_boundary_values() -> None:
    grid = Grid3D(nx=20, ny=4, nz=1, dx=10.0, dy=10.0, dz=2.0)
    left_pressure = 10.0e6
    right_pressure = 0.0
    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": left_pressure, "right": right_pressure},
    )
    pressure = result.pressure.values[0]
    cell_drop = (left_pressure - right_pressure) / grid.nx

    assert np.allclose(pressure[:, 0], left_pressure - 0.5 * cell_drop, rtol=0.0, atol=1e-3)
    assert np.allclose(pressure[:, -1], right_pressure + 0.5 * cell_drop, rtol=0.0, atol=1e-3)


def test_2d_no_flow_boundary_flux() -> None:
    result = _left_right_solution()
    pressure = result.pressure.values[0]
    assert np.allclose(pressure[0, :], pressure[1, :], rtol=0.0, atol=1e-6)
    assert np.allclose(pressure[-1, :], pressure[-2, :], rtol=0.0, atol=1e-6)
    assert result.report["mass_balance_error"] < 1.0e-8


def test_2d_anisotropic_permeability() -> None:
    grid = Grid3D(nx=10, ny=8, nz=1, dx=10.0, dy=10.0, dz=2.0)
    wells = [
        Well("I1", "injection", grid, i=2, j=2, k=0, rate=1.0e-5),
        Well("P1", "production", grid, i=7, j=5, k=0, rate=1.0e-5),
    ]
    isotropic = solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        wells=wells,
    )
    anisotropic = solve_steady_state_pressure_2d(
        grid=grid,
        kx=200.0e-15,
        ky=20.0e-15,
        mu=1.0e-3,
        wells=wells,
    )

    assert not np.allclose(anisotropic.pressure.values, isotropic.pressure.values)
    assert not np.isnan(anisotropic.pressure.values).any()
    assert anisotropic.report["mass_balance_error"] < 1.0e-8


def _left_right_solution():
    grid = Grid3D(nx=10, ny=8, nz=1, dx=10.0, dy=10.0, dz=2.0)
    return solve_steady_state_pressure_2d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": 10.0e6, "right": 0.0},
    )
