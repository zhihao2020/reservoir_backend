from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    advance_saturation_1d_with_capillary,
    advance_saturation_3d,
    advance_saturation_3d_with_capillary,
)


def test_3d_capillary_disabled_matches_existing_solver() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    fluxes = _x_flux(grid, 1.0e-5)
    base = advance_saturation_3d(grid, sw, 0.2, *fluxes, 10.0, _relperm())
    coupled = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *fluxes, 10.0, _relperm(), _cap_disabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert np.allclose(coupled.sw.values, base.sw.values)
    assert coupled.report["material_balance_error"] == pytest.approx(base.report["material_balance_error"])
    assert coupled.report["capillary_flux_included"] is False


def test_3d_capillary_no_pc_model_matches_existing_solver() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    fluxes = _x_flux(grid, 1.0e-5)
    base = advance_saturation_3d(grid, sw, 0.2, *fluxes, 10.0, _relperm())
    coupled = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *fluxes, 10.0, _relperm(), _cap_none(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert np.allclose(coupled.sw.values, base.sw.values)
    assert coupled.report["capillary_enabled"] is False


def test_3d_capillary_flux_changes_solution() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    base = advance_saturation_3d(grid, sw, 0.2, *_zero_fluxes(grid), 100.0, _relperm())
    coupled = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *_zero_fluxes(grid), 100.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert not np.allclose(coupled.sw.values, base.sw.values)
    assert coupled.report["max_abs_capillary_flux"] > 0.0


def test_3d_capillary_zero_sw_gradient_no_effect() -> None:
    grid = _grid()
    sw = np.full(grid.shape, 0.5)
    result = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert np.allclose(result.sw.values, sw)
    assert result.report["max_abs_capillary_flux"] == pytest.approx(0.0)


def test_3d_capillary_smooths_x_saturation_front() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    before = _max_jump(sw, axis=2)
    result = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert _max_jump(result.sw.values, axis=2) < before


def test_3d_capillary_smooths_y_saturation_front() -> None:
    grid = _grid()
    sw = _y_step_sw(grid)
    before = _max_jump(sw, axis=1)
    result = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert _max_jump(result.sw.values, axis=1) < before


def test_3d_capillary_smooths_z_saturation_front() -> None:
    grid = _grid()
    sw = _z_step_sw(grid)
    before = _max_jump(sw, axis=0)
    result = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert _max_jump(result.sw.values, axis=0) < before


def test_3d_capillary_report_keys() -> None:
    grid = _grid()
    result = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 100.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    keys = {
        "capillary_enabled",
        "capillary_model",
        "max_abs_capillary_flux",
        "max_total_water_flux",
        "capillary_flux_included",
        "material_balance_error",
    }
    assert keys.issubset(result.report)


def test_3d_capillary_saturation_bounds() -> None:
    grid = _grid()
    result = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert result.sw.values.min() >= _relperm()["swi"]
    assert result.sw.values.max() <= 1.0 - _relperm()["sor"]


def test_3d_capillary_no_nan_inf() -> None:
    grid = _grid()
    result = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_3d_capillary_material_balance() -> None:
    grid = _grid()
    result = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert result.report["material_balance_error"] < 1.0e-10


def test_3d_capillary_cfl_violation_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        advance_saturation_3d_with_capillary(
            grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1.0e8, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
        )


def test_3d_invalid_capillary_params_raises() -> None:
    grid = _grid()
    params = _cap_enabled()
    params["entry_pressure_pa"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_capillary(
            grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 100.0, _relperm(), params, 1.0e-9, 1.0e-9, 1.0e-9
        )
    params = _cap_enabled()
    params["lambda_pc"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_capillary(
            grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 100.0, _relperm(), params, 1.0e-9, 1.0e-9, 1.0e-9
        )


def test_3d_invalid_permeability_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d_with_capillary(
            grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 100.0, _relperm(), _cap_enabled(), -1.0, 1.0e-9, 1.0e-9
        )


def test_3d_capillary_repeatability() -> None:
    grid = _grid()
    first = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    second = advance_saturation_3d_with_capillary(
        grid, _x_step_sw(grid), 0.2, *_zero_fluxes(grid), 1000.0, _relperm(), _cap_enabled(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def _grid() -> Grid3D:
    return Grid3D(nx=6, ny=5, nz=4, dx=1.0, dy=1.0, dz=1.0)


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


def _cap_enabled() -> dict[str, float | bool | str]:
    return {
        "enabled": True,
        "model": "brooks_corey",
        "swi": 0.2,
        "sor": 0.2,
        "entry_pressure_pa": 1000.0,
        "lambda_pc": 2.0,
    }


def _cap_disabled() -> dict[str, float | bool | str]:
    params = _cap_enabled()
    params["enabled"] = False
    return params


def _cap_none() -> dict[str, float | bool | str]:
    params = _cap_enabled()
    params["model"] = "none"
    return params


def _zero_fluxes(grid: Grid3D):
    return (
        np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float),
        np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float),
        np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float),
    )


def _x_flux(grid: Grid3D, value: float):
    fx, fy, fz = _zero_fluxes(grid)
    fx[:, :, :] = value
    return fx, fy, fz


def _x_step_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[:, :, : grid.nx // 2] = 0.65
    return sw


def _y_step_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[:, : grid.ny // 2, :] = 0.65
    return sw


def _z_step_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[: grid.nz // 2, :, :] = 0.65
    return sw


def _max_jump(sw: np.ndarray, axis: int) -> float:
    return float(np.max(np.abs(np.diff(sw, axis=axis))))
