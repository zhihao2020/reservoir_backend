from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.gravity_flux import gravity_mobility
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    advance_saturation_1d_vertical_with_gravity,
    advance_saturation_1d_with_capillary,
    advance_saturation_3d,
    compute_gravity_saturation_update_1d_vertical,
    compute_upwind_water_flux_1d_vertical,
)


def test_1d_vertical_gravity_disabled_zero_effect() -> None:
    grid = _grid()
    sw = _gradient_sw(grid)
    flux_z = _vertical_flux(grid, 1.0e-5)
    adv = compute_upwind_water_flux_1d_vertical(sw, flux_z, _relperm())
    expected_field, expected_report = compute_gravity_saturation_update_1d_vertical(
        grid, sw, 0.2, adv, 100.0, _relperm()
    )

    result = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, flux_z, 100.0, _relperm(), _gravity_disabled(), 1.0e-12
    )

    assert np.allclose(result.sw.values, expected_field.values)
    assert result.report["material_balance_error"] == pytest.approx(expected_report["material_balance_error"])
    assert result.report["gravity_flux_included"] is False
    assert result.report["max_abs_gravity_flux"] == pytest.approx(0.0)


def test_1d_vertical_zero_density_difference_no_effect() -> None:
    grid = _grid()
    sw = _gradient_sw(grid)
    params = _gravity_enabled()
    params["rho_o"] = params["rho_w"]
    disabled = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 100.0, _relperm(), _gravity_disabled(), 1.0e-12
    )
    result = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 100.0, _relperm(), params, 1.0e-12
    )
    assert np.allclose(result.sw.values, disabled.sw.values)
    assert result.report["max_abs_gravity_flux"] == pytest.approx(0.0)


def test_1d_vertical_water_heavier_moves_down() -> None:
    grid = _grid()
    sw = _top_high_sw(grid)
    result = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert result.sw.values[0, 0, 0] > sw[0, 0, 0]
    assert result.sw.values[-1, 0, 0] < sw[-1, 0, 0]
    assert result.report["max_abs_gravity_flux"] > 0.0


def test_1d_vertical_oil_heavier_moves_up_equivalent() -> None:
    grid = _grid()
    sw = _bottom_high_sw(grid)
    params = _gravity_enabled()
    params["rho_w"] = 700.0
    params["rho_o"] = 1000.0
    result = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 1000.0, _relperm(), params, 1.0e-12
    )
    assert result.sw.values[0, 0, 0] < sw[0, 0, 0]
    assert result.sw.values[-1, 0, 0] > sw[-1, 0, 0]


def test_1d_vertical_gravity_changes_solution() -> None:
    grid = _grid()
    sw = _gradient_sw(grid)
    disabled = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_disabled(), 1.0e-12
    )
    enabled = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert not np.allclose(enabled.sw.values, disabled.sw.values)


def test_1d_vertical_uniform_sw_gravity_mobility_effect() -> None:
    grid = _grid()
    sw = np.full(grid.shape, 0.5)
    result = advance_saturation_1d_vertical_with_gravity(
        grid, sw, 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert result.report["max_abs_gravity_flux"] > 0.0
    assert result.sw.values[0, 0, 0] > sw[0, 0, 0]
    endpoint = np.full(grid.shape, _relperm()["swi"])
    endpoint_result = advance_saturation_1d_vertical_with_gravity(
        grid, endpoint, 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert endpoint_result.report["max_abs_gravity_flux"] == pytest.approx(0.0)
    assert np.allclose(endpoint_result.sw.values, endpoint)


def test_1d_vertical_endpoint_mobility_zero() -> None:
    grid = _grid()
    low = gravity_mobility(np.full(grid.shape, _relperm()["swi"]), _relperm())
    high = gravity_mobility(np.full(grid.shape, 1.0 - _relperm()["sor"]), _relperm())
    assert np.asarray(low).max() == pytest.approx(0.0)
    assert np.asarray(high).max() == pytest.approx(0.0)


def test_1d_vertical_report_keys() -> None:
    grid = _grid()
    result = advance_saturation_1d_vertical_with_gravity(
        grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 100.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
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


def test_1d_vertical_saturation_bounds() -> None:
    grid = _grid()
    result = advance_saturation_1d_vertical_with_gravity(
        grid, _top_high_sw(grid), 0.2, _zero_flux(grid), 5000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert result.sw.values.min() >= _relperm()["swi"]
    assert result.sw.values.max() <= 1.0 - _relperm()["sor"]


def test_1d_vertical_no_nan_inf() -> None:
    grid = _grid()
    result = advance_saturation_1d_vertical_with_gravity(
        grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_1d_vertical_material_balance() -> None:
    grid = _grid()
    result = advance_saturation_1d_vertical_with_gravity(
        grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12
    )
    assert result.report["material_balance_error"] < 1.0e-10


def test_1d_vertical_cfl_violation_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        advance_saturation_1d_vertical_with_gravity(
            grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 1.0e9, _relperm(), _gravity_enabled(), 1.0e-12
        )


def test_1d_vertical_invalid_density_raises() -> None:
    grid = _grid()
    params = _gravity_enabled()
    params["rho_w"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_1d_vertical_with_gravity(
            grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 100.0, _relperm(), params, 1.0e-12
        )
    params = _gravity_enabled()
    params["rho_o"] = -1.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_1d_vertical_with_gravity(
            grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 100.0, _relperm(), params, 1.0e-12
        )


def test_1d_vertical_invalid_kz_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_1d_vertical_with_gravity(
            grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 100.0, _relperm(), _gravity_enabled(), -1.0
        )


def test_1d_vertical_repeatability() -> None:
    grid = _grid()
    args = (grid, _gradient_sw(grid), 0.2, _zero_flux(grid), 1000.0, _relperm(), _gravity_enabled(), 1.0e-12)
    first = advance_saturation_1d_vertical_with_gravity(*args)
    second = advance_saturation_1d_vertical_with_gravity(*args)
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def _grid(nz: int = 6) -> Grid3D:
    return Grid3D(nx=1, ny=1, nz=nz, dx=1.0, dy=1.0, dz=1.0)


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


def _zero_flux(grid: Grid3D) -> np.ndarray:
    return np.zeros((grid.nz + 1, 1, 1), dtype=float)


def _vertical_flux(grid: Grid3D, value: float) -> np.ndarray:
    return np.full((grid.nz + 1, 1, 1), value, dtype=float)


def _gradient_sw(grid: Grid3D) -> np.ndarray:
    return np.linspace(0.3, 0.6, grid.nz).reshape(grid.shape)


def _top_high_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[-grid.nz // 2 :, 0, 0] = 0.65
    return sw


def _bottom_high_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[: grid.nz // 2, 0, 0] = 0.65
    return sw
