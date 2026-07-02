from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d


def test_3d_pressure_shape() -> None:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=10.0, dy=10.0, dz=5.0)
    result = _left_right_solution(grid)
    assert result.pressure.values.shape == (4, 5, 6)
    assert result.pressure.unit == "Pa"
    assert result.report["solver"] == "scipy.sparse.linalg.spsolve"


def test_3d_no_nan_inf() -> None:
    pressure = _left_right_solution().pressure.values
    assert not np.isnan(pressure).any()
    assert not np.isinf(pressure).any()


def test_3d_left_right_dirichlet_gradient() -> None:
    pressure = _left_right_solution().pressure.values
    assert np.all(np.diff(pressure, axis=2) < 0.0)


def test_3d_pressure_between_boundaries() -> None:
    pressure = _left_right_solution().pressure.values
    assert pressure.min() >= -1.0e-6
    assert pressure.max() <= 10.0e6 + 1.0e-6


def test_3d_injector_pressure_higher_than_producer() -> None:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=20.0, dy=20.0, dz=8.0)
    injector = Well("I1", "injection", grid, i=1, j=2, k=1, rate=1.0e-5)
    producer = Well("P1", "production", grid, i=4, j=2, k=2, rate=1.0e-5)

    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        wells=[injector, producer],
    )

    pressure = result.pressure.values
    assert pressure[1, 2, 1] > pressure[2, 2, 4]
    assert result.report["pressure_reference_applied"] is True


def test_3d_mass_balance_source_sink() -> None:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=20.0, dy=20.0, dz=8.0)
    wells = [
        Well("I1", "injection", grid, i=1, j=2, k=1, rate=2.0e-5),
        Well("P1", "production", grid, i=4, j=2, k=2, rate=2.0e-5),
    ]

    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        wells=wells,
    )

    assert result.report["mass_balance_error"] < 1.0e-8
    assert result.report["net_well_rate_m3_s"] == pytest.approx(0.0)


def test_3d_dirichlet_boundary_values() -> None:
    grid = Grid3D(nx=12, ny=4, nz=3, dx=10.0, dy=10.0, dz=5.0)
    left_pressure = 10.0e6
    right_pressure = 0.0
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": left_pressure, "right": right_pressure},
    )
    pressure = result.pressure.values
    cell_drop = (left_pressure - right_pressure) / grid.nx

    assert np.allclose(pressure[:, :, 0], left_pressure - 0.5 * cell_drop, rtol=0.0, atol=1e-3)
    assert np.allclose(pressure[:, :, -1], right_pressure + 0.5 * cell_drop, rtol=0.0, atol=1e-3)


def test_3d_no_flow_boundary_flux() -> None:
    pressure = _left_right_solution().pressure.values
    assert np.allclose(pressure[:, 0, :], pressure[:, 1, :], rtol=0.0, atol=1e-6)
    assert np.allclose(pressure[:, -1, :], pressure[:, -2, :], rtol=0.0, atol=1e-6)
    assert np.allclose(pressure[0, :, :], pressure[1, :, :], rtol=0.0, atol=1e-6)
    assert np.allclose(pressure[-1, :, :], pressure[-2, :, :], rtol=0.0, atol=1e-6)


def test_3d_anisotropic_permeability() -> None:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=20.0, dy=20.0, dz=8.0)
    wells = [
        Well("I1", "injection", grid, i=1, j=1, k=1, rate=1.0e-5),
        Well("P1", "production", grid, i=4, j=3, k=2, rate=1.0e-5),
    ]
    isotropic = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        wells=wells,
    )
    anisotropic = solve_steady_state_pressure_3d(
        grid=grid,
        kx=250.0e-15,
        ky=50.0e-15,
        kz=10.0e-15,
        mu=1.0e-3,
        wells=wells,
    )

    assert not np.allclose(anisotropic.pressure.values, isotropic.pressure.values)
    assert not np.isnan(anisotropic.pressure.values).any()
    assert anisotropic.report["mass_balance_error"] < 1.0e-8


def test_3d_vertical_transmissibility_effect() -> None:
    grid = Grid3D(nx=4, ny=4, nz=5, dx=10.0, dy=10.0, dz=5.0)
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=200.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"bottom": 10.0e6, "top": 0.0},
    )
    pressure = result.pressure.values

    assert np.all(np.diff(pressure[:, 2, 2]) < 0.0)
    assert np.allclose(pressure[:, :, :], pressure[:, 0:1, 0:1], rtol=0.0, atol=1e-6)
    assert result.report["mass_balance_error"] < 1.0e-8


def test_3d_homogeneous_symmetry() -> None:
    grid = Grid3D(nx=7, ny=5, nz=5, dx=20.0, dy=20.0, dz=10.0)
    wells = [
        Well("I1", "injection", grid, i=2, j=1, k=1, rate=1.0e-5),
        Well("I2", "injection", grid, i=2, j=3, k=1, rate=1.0e-5),
        Well("I3", "injection", grid, i=2, j=1, k=3, rate=1.0e-5),
        Well("I4", "injection", grid, i=2, j=3, k=3, rate=1.0e-5),
        Well("P1", "production", grid, i=4, j=1, k=1, rate=1.0e-5),
        Well("P2", "production", grid, i=4, j=3, k=1, rate=1.0e-5),
        Well("P3", "production", grid, i=4, j=1, k=3, rate=1.0e-5),
        Well("P4", "production", grid, i=4, j=3, k=3, rate=1.0e-5),
    ]

    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        wells=wells,
    )
    pressure = result.pressure.values

    assert np.allclose(pressure[:, 0, :], pressure[:, -1, :], rtol=1e-10, atol=1e-8)
    assert np.allclose(pressure[:, 1, :], pressure[:, -2, :], rtol=1e-10, atol=1e-8)
    assert np.allclose(pressure[0, :, :], pressure[-1, :, :], rtol=1e-10, atol=1e-8)
    assert np.allclose(pressure[1, :, :], pressure[-2, :, :], rtol=1e-10, atol=1e-8)


def test_3d_grid_refinement_trend() -> None:
    coarse = Grid3D(nx=6, ny=3, nz=2, dx=10.0, dy=10.0, dz=5.0)
    fine = Grid3D(nx=12, ny=3, nz=2, dx=5.0, dy=10.0, dz=5.0)
    coarse_pressure = _left_right_solution(coarse).pressure.values[0, 1, :]
    fine_pressure = _left_right_solution(fine).pressure.values[0, 1, :]

    assert np.all(np.diff(coarse_pressure) < 0.0)
    assert np.all(np.diff(fine_pressure) < 0.0)
    assert coarse_pressure[0] > fine_pressure[len(fine_pressure) // 2] > coarse_pressure[-1]


def test_3d_singular_system_or_reference_pressure() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=10.0, dy=10.0, dz=5.0)
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        reference_pressure=12345.0,
    )

    assert result.report["pressure_reference_applied"] is True
    assert np.allclose(result.pressure.values, 12345.0)
    assert result.report["mass_balance_error"] < 1.0e-12


def _left_right_solution(grid: Grid3D | None = None):
    grid = Grid3D(nx=6, ny=5, nz=4, dx=10.0, dy=10.0, dz=5.0) if grid is None else grid
    return solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": 10.0e6, "right": 0.0},
    )
