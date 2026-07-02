from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    advance_saturation_1d_vertical_with_gravity,
    advance_saturation_1d_with_capillary,
    advance_saturation_3d,
    advance_saturation_3d_with_gravity,
    compute_total_water_flux_3d_with_gravity,
)


def test_3d_gravity_disabled_matches_existing_solver() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    base = advance_saturation_3d(grid, sw, 0.2, fx, fy, fz, 100.0, _relperm())
    result = advance_saturation_3d_with_gravity(
        grid, sw, 0.2, fx, fy, fz, 100.0, _relperm(), _gravity_disabled(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    assert np.allclose(result.sw.values, base.sw.values)
    assert result.report["material_balance_error"] == pytest.approx(base.report["material_balance_error"])
    assert result.report["gravity_flux_included"] is False


def test_3d_gravity_zero_density_difference_matches_existing_solver() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    params = _gravity_enabled()
    params["rho_o"] = params["rho_w"]
    base = advance_saturation_3d(grid, sw, 0.2, fx, fy, fz, 100.0, _relperm())
    result = advance_saturation_3d_with_gravity(
        grid, sw, 0.2, fx, fy, fz, 100.0, _relperm(), params, 1.0e-12, 1.0e-12, 1.0e-12
    )
    assert np.allclose(result.sw.values, base.sw.values)
    assert result.report["max_abs_gravity_flux"] == pytest.approx(0.0)


def test_3d_gravity_changes_solution() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    base = advance_saturation_3d(grid, sw, 0.2, fx, fy, fz, 1000.0, _relperm())
    result = advance_saturation_3d_with_gravity(
        grid, sw, 0.2, fx, fy, fz, 1000.0, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    assert not np.allclose(result.sw.values, base.sw.values)
    assert result.report["max_abs_gravity_flux"] > 0.0


def test_3d_gravity_water_heavier_moves_down() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    result = _advance_gravity(grid, sw, _gravity_enabled(), dt=1000.0)
    assert result.sw.values[0, :, :].mean() > sw[0, :, :].mean()
    assert result.sw.values[-1, :, :].mean() < sw[-1, :, :].mean()


def test_3d_gravity_oil_heavier_reverses_direction() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    params = _gravity_enabled()
    params["rho_w"] = 700.0
    params["rho_o"] = 1000.0
    result = _advance_gravity(grid, sw, params, dt=1000.0)
    assert result.sw.values[0, :, :].mean() < sw[0, :, :].mean()
    assert result.sw.values[-1, :, :].mean() > sw[-1, :, :].mean()


def test_3d_gravity_x_y_flux_zero_for_regular_grid() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    _, gravity_flux, _, _ = compute_total_water_flux_3d_with_gravity(
        grid, _uniform_sw(grid), fx, fy, fz, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    gx, gy, gz = gravity_flux
    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)
    assert np.max(np.abs(gz)) > 0.0


def test_3d_gravity_z_flux_sign_water_heavier() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    _, gravity_flux, _, _ = compute_total_water_flux_3d_with_gravity(
        grid, _uniform_sw(grid), fx, fy, fz, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    assert np.all(gravity_flux[2][1:-1, :, :] <= 0.0)
    assert np.any(gravity_flux[2][1:-1, :, :] < 0.0)


def test_3d_gravity_z_flux_sign_oil_heavier() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    params = _gravity_enabled()
    params["rho_w"] = 700.0
    params["rho_o"] = 1000.0
    _, gravity_flux, _, _ = compute_total_water_flux_3d_with_gravity(
        grid, _uniform_sw(grid), fx, fy, fz, _relperm(), params, 1.0e-12, 1.0e-12, 1.0e-12
    )
    assert np.all(gravity_flux[2][1:-1, :, :] >= 0.0)
    assert np.any(gravity_flux[2][1:-1, :, :] > 0.0)


def test_3d_gravity_endpoint_mobility_zero() -> None:
    grid = _grid()
    for endpoint in (_relperm()["swi"], 1.0 - _relperm()["sor"]):
        result = _advance_gravity(grid, np.full(grid.shape, endpoint), _gravity_enabled(), dt=1000.0)
        assert result.report["max_abs_gravity_flux"] == pytest.approx(0.0)
        assert np.allclose(result.sw.values, endpoint)


def test_3d_gravity_report_keys() -> None:
    grid = _grid()
    result = _advance_gravity(grid, _uniform_sw(grid), _gravity_enabled(), dt=100.0)
    keys = {
        "gravity_enabled",
        "gravity_flux_included",
        "rho_w",
        "rho_o",
        "density_difference",
        "max_abs_gravity_flux",
        "max_total_water_flux",
        "material_balance_error",
        "max_cfl",
    }
    assert keys.issubset(result.report)


def test_3d_gravity_saturation_bounds() -> None:
    grid = _grid()
    result = _advance_gravity(grid, _uniform_sw(grid), _gravity_enabled(), dt=5000.0)
    assert result.sw.values.min() >= _relperm()["swi"]
    assert result.sw.values.max() <= 1.0 - _relperm()["sor"]


def test_3d_gravity_no_nan_inf() -> None:
    grid = _grid()
    result = _advance_gravity(grid, _uniform_sw(grid), _gravity_enabled(), dt=1000.0)
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_3d_gravity_material_balance() -> None:
    grid = _grid()
    result = _advance_gravity(grid, _uniform_sw(grid), _gravity_enabled(), dt=1000.0)
    assert result.report["material_balance_error"] < 1.0e-10


def test_3d_gravity_cfl_violation_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    with pytest.raises(CFLViolationError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 1.0e9, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, 1.0e-12
        )


def test_3d_gravity_invalid_density_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    params = _gravity_enabled()
    params["rho_w"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 100.0, _relperm(), params, 1.0e-12, 1.0e-12, 1.0e-12
        )
    params = _gravity_enabled()
    params["rho_o"] = -1.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 100.0, _relperm(), params, 1.0e-12, 1.0e-12, 1.0e-12
        )


def test_3d_gravity_invalid_k_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 100.0, _relperm(), _gravity_enabled(), -1.0, 1.0e-12, 1.0e-12
        )
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 100.0, _relperm(), _gravity_enabled(), 1.0e-12, -1.0, 1.0e-12
        )
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_gravity(
            grid, _uniform_sw(grid), 0.2, fx, fy, fz, 100.0, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, -1.0
        )


def test_3d_gravity_repeatability() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    args = (grid, _uniform_sw(grid), 0.2, fx, fy, fz, 1000.0, _relperm(), _gravity_enabled(), 1.0e-12, 1.0e-12, 1.0e-12)
    first = advance_saturation_3d_with_gravity(*args)
    second = advance_saturation_3d_with_gravity(*args)
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def test_existing_1d_gravity_tests_still_pass() -> None:
    grid = Grid3D(nx=1, ny=1, nz=5, dx=1.0, dy=1.0, dz=1.0)
    result = advance_saturation_1d_vertical_with_gravity(
        grid,
        np.full(grid.shape, 0.5),
        0.2,
        np.zeros((grid.nz + 1, 1, 1)),
        100.0,
        _relperm(),
        _gravity_enabled(),
        1.0e-12,
    )
    assert result.report["stable"] is True
    assert result.sw.values.shape == grid.shape


def test_existing_1d_3d_saturation_tests_still_pass() -> None:
    grid_1d = Grid3D(nx=6, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    result_1d = advance_saturation_1d(
        grid_1d, 0.2, 0.2, np.full((1, 1, grid_1d.nx + 1), 1.0e-5), 100.0, _relperm()
    )
    assert result_1d.report["stable"] is True

    grid_3d = _grid()
    fx, fy, fz = _zero_fluxes(grid_3d)
    result_3d = advance_saturation_3d(grid_3d, _uniform_sw(grid_3d), 0.2, fx, fy, fz, 100.0, _relperm())
    assert np.allclose(result_3d.sw.values, _uniform_sw(grid_3d))


def test_existing_capillary_tests_still_pass() -> None:
    grid = Grid3D(nx=6, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    sw = np.full(grid.shape, 0.45)
    result = advance_saturation_1d_with_capillary(
        grid, sw, 0.2, np.zeros((1, 1, grid.nx + 1)), 100.0, _relperm(), _capillary_disabled(), 1.0e-12
    )
    assert np.allclose(result.sw.values, sw)


def _grid() -> Grid3D:
    return Grid3D(nx=4, ny=3, nz=4, dx=1.0, dy=1.0, dz=1.0)


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float),
        np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float),
        np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float),
    )


def _uniform_sw(grid: Grid3D) -> np.ndarray:
    return np.full(grid.shape, 0.5)


def _relperm() -> dict[str, float]:
    return {
        "swi": 0.2,
        "sor": 0.2,
        "krw0": 1.0,
        "kro0": 1.0,
        "nw": 2.0,
        "no": 2.0,
        "mu_w": 1.0e-3,
        "mu_o": 5.0e-3,
    }


def _gravity_enabled() -> dict[str, float | bool | str]:
    return {
        "enabled": True,
        "g": 9.80665,
        "rho_w": 1000.0,
        "rho_o": 800.0,
        "depth_axis": "z",
        "depth_positive": "down",
    }


def _gravity_disabled() -> dict[str, float | bool | str]:
    params = _gravity_enabled()
    params["enabled"] = False
    return params


def _capillary_disabled() -> dict[str, float | bool | str]:
    return {
        "enabled": False,
        "model": "none",
        "swi": 0.2,
        "sor": 0.2,
        "entry_pressure_pa": 1000.0,
        "lambda_pc": 2.0,
    }


def _advance_gravity(
    grid: Grid3D,
    sw: np.ndarray,
    gravity_params: dict[str, float | bool | str],
    dt: float,
):
    fx, fy, fz = _zero_fluxes(grid)
    return advance_saturation_3d_with_gravity(
        grid, sw, 0.2, fx, fy, fz, dt, _relperm(), gravity_params, 1.0e-12, 1.0e-12, 1.0e-12
    )
