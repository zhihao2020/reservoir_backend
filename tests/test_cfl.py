from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.cfl import (
    check_cfl_condition,
    compute_cfl_number,
    estimate_stable_dt,
)


def test_cfl_zero_flux() -> None:
    grid = _grid()
    cfl, report = compute_cfl_number(grid, 0.2, *_zero_fluxes(grid), dt=10.0)
    assert np.allclose(cfl, 0.0)
    assert report["max_cfl"] == pytest.approx(0.0)


def test_cfl_shape() -> None:
    grid = _grid()
    cfl, _ = compute_cfl_number(grid, 0.2, *_sample_fluxes(grid), dt=1.0)
    assert cfl.shape == grid.shape


def test_cfl_valid_dt() -> None:
    grid = _grid()
    report = check_cfl_condition(grid, 0.2, *_sample_fluxes(grid), dt=1.0, max_cfl=1.0)
    assert report["stable"] is True


def test_cfl_invalid_dt_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        check_cfl_condition(grid, 0.2, *_sample_fluxes(grid), dt=1.0e6, max_cfl=0.1)


def test_cfl_report_keys() -> None:
    grid = _grid()
    _, report = compute_cfl_number(grid, 0.2, *_sample_fluxes(grid), dt=1.0)
    keys = {
        "max_cfl",
        "mean_cfl",
        "min_cfl",
        "max_cfl_location",
        "dt",
        "max_cfl_allowed",
        "stable",
        "has_nan",
        "has_inf",
    }
    assert keys.issubset(report)


def test_estimate_stable_dt_positive() -> None:
    grid = _grid()
    stable_dt = estimate_stable_dt(grid, 0.2, *_sample_fluxes(grid), max_cfl=0.5)
    assert stable_dt > 0.0
    assert np.isfinite(stable_dt)


def test_estimate_stable_dt_zero_flux() -> None:
    grid = _grid()
    stable_dt = estimate_stable_dt(grid, 0.2, *_zero_fluxes(grid), max_cfl=0.5)
    assert stable_dt == np.inf


def test_invalid_dt_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_cfl_number(grid, 0.2, *_sample_fluxes(grid), dt=0.0)


def test_invalid_porosity_zero_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_cfl_number(grid, 0.0, *_sample_fluxes(grid), dt=1.0)


def test_invalid_porosity_negative_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_cfl_number(grid, -0.1, *_sample_fluxes(grid), dt=1.0)


def test_phi_scalar_input() -> None:
    grid = _grid()
    cfl, _ = compute_cfl_number(grid, 0.2, *_sample_fluxes(grid), dt=1.0)
    assert cfl.shape == grid.shape


def test_phi_ndarray_input() -> None:
    grid = _grid()
    phi = np.full(grid.shape, 0.2)
    cfl, _ = compute_cfl_number(grid, phi, *_sample_fluxes(grid), dt=1.0)
    assert cfl.shape == grid.shape


def test_phi_field3d_input() -> None:
    grid = _grid()
    phi = Field3D.from_constant(grid, 0.2, name="phi", unit="fraction")
    cfl, _ = compute_cfl_number(grid, phi, *_sample_fluxes(grid), dt=1.0)
    assert cfl.shape == grid.shape


def test_flux_shape_mismatch_raises() -> None:
    grid = _grid()
    fx, fy, fz = _sample_fluxes(grid)
    with pytest.raises(FieldShapeError):
        compute_cfl_number(grid, 0.2, fx[:, :, :-1], fy, fz, dt=1.0)


def test_cfl_location() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    fx[1, 2, 3] = 10.0
    cfl, report = compute_cfl_number(grid, 0.5, fx, fy, fz, dt=2.0)
    assert report["max_cfl_location"] == (1, 2, 2)
    assert cfl[1, 2, 2] == pytest.approx(report["max_cfl"])


def test_cfl_matches_manual_calculation() -> None:
    grid = Grid3D(nx=1, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    fx = np.array([[[1.0, -2.0]]])
    fy = np.array([[[3.0], [-4.0]]])
    fz = np.array([[[5.0]], [[-6.0]]])
    cfl, _ = compute_cfl_number(grid, phi=0.5, flux_x=fx, flux_y=fy, flux_z=fz, dt=10.0)
    expected = 10.0 / (0.5 * grid.cell_volume) * (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0)
    assert cfl[0, 0, 0] == pytest.approx(expected)


def test_check_cfl_condition_returns_stable_report() -> None:
    grid = _grid()
    report = check_cfl_condition(grid, 0.2, *_sample_fluxes(grid), dt=1.0, max_cfl=1.0)
    assert report["stable"] is True
    assert report["max_cfl_allowed"] == pytest.approx(1.0)


def test_check_cfl_condition_exception_message() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError, match="max_cfl=.*max_cfl_allowed="):
        check_cfl_condition(grid, 0.2, *_sample_fluxes(grid), dt=1.0e6, max_cfl=0.1)


def _grid() -> Grid3D:
    return Grid3D(nx=3, ny=4, nz=2, dx=10.0, dy=8.0, dz=6.0)


def _zero_fluxes(grid: Grid3D):
    return (
        np.zeros((grid.nz, grid.ny, grid.nx + 1)),
        np.zeros((grid.nz, grid.ny + 1, grid.nx)),
        np.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )


def _sample_fluxes(grid: Grid3D):
    fx, fy, fz = _zero_fluxes(grid)
    fx[:, :, 1:-1] = 1.0e-5
    fy[:, 1:-1, :] = -2.0e-5
    fz[1:-1, :, :] = 3.0e-5
    return fx, fy, fz
