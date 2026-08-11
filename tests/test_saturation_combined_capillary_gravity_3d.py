from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.core.exceptions import CFLViolationError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.io.config_loader import load_case_config
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d_with_capillary,
    advance_saturation_1d_vertical_with_gravity,
    advance_saturation_3d,
    advance_saturation_3d_with_capillary,
    advance_saturation_3d_with_capillary_and_gravity,
    advance_saturation_3d_with_gravity,
)


def test_combined_disabled_matches_existing_solver() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    fluxes = _zero_fluxes(grid)
    base = advance_saturation_3d(grid, sw, 0.2, *fluxes, 100.0, _relperm())
    combined = _advance(grid, sw, _cap_disabled(), _gravity_disabled(), dt=100.0)
    assert np.allclose(combined.sw.values, base.sw.values)
    assert combined.report["capillary_flux_included"] is False
    assert combined.report["gravity_flux_included"] is False


def test_combined_capillary_only_matches_capillary_solver() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    fluxes = _zero_fluxes(grid)
    capillary = advance_saturation_3d_with_capillary(
        grid, sw, 0.2, *fluxes, 100.0, _relperm(), _cap_enabled(), _k(), _k(), _k()
    )
    combined = _advance(grid, sw, _cap_enabled(), _gravity_disabled(), dt=100.0)
    assert np.allclose(combined.sw.values, capillary.sw.values)
    assert combined.report["capillary_enabled"] is True
    assert combined.report["gravity_enabled"] is False


def test_combined_gravity_only_matches_gravity_solver() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    fluxes = _zero_fluxes(grid)
    gravity = advance_saturation_3d_with_gravity(
        grid, sw, 0.2, *fluxes, 100.0, _relperm(), _gravity_enabled(), _k(), _k(), _k()
    )
    combined = _advance(grid, sw, _cap_disabled(), _gravity_enabled(), dt=100.0)
    assert np.allclose(combined.sw.values, gravity.sw.values)
    assert combined.report["capillary_enabled"] is False
    assert combined.report["gravity_enabled"] is True


def test_combined_capillary_and_gravity_changes_solution() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    base = advance_saturation_3d(grid, sw, 0.2, *_zero_fluxes(grid), 100.0, _relperm())
    combined = _advance(grid, sw, _cap_enabled(), _gravity_enabled(), dt=100.0)
    assert not np.allclose(combined.sw.values, base.sw.values)
    assert combined.report["max_capillary_flux"] > 0.0
    assert combined.report["max_gravity_flux"] > 0.0


def test_combined_contains_capillary_smoothing() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    before = _max_jump(sw, axis=2)
    combined = _advance(grid, sw, _cap_strong(), _gravity_enabled(), dt=100.0)
    assert _max_jump(combined.sw.values, axis=2) < before


def test_combined_contains_gravity_segregation() -> None:
    grid = _grid()
    sw = _uniform_sw(grid)
    combined = _advance(grid, sw, _cap_enabled(), _gravity_enabled(), dt=1000.0)
    assert combined.sw.values[0, :, :].mean() > sw[0, :, :].mean()
    assert combined.sw.values[-1, :, :].mean() < sw[-1, :, :].mean()


def test_combined_report_keys() -> None:
    report = _advance(_grid(), _x_step_sw(_grid()), _cap_enabled(), _gravity_enabled(), dt=100.0).report
    keys = {
        "capillary_enabled",
        "gravity_enabled",
        "capillary_model",
        "rho_w",
        "rho_o",
        "density_difference",
        "max_advective_flux",
        "max_capillary_flux",
        "max_gravity_flux",
        "max_total_water_flux",
        "max_effective_flux",
        "max_cfl",
        "material_balance_error",
        "capillary_flux_included",
        "gravity_flux_included",
        "has_nan",
        "has_inf",
    }
    assert keys.issubset(report)


def test_combined_composer_report_included() -> None:
    report = _advance(_grid(), _x_step_sw(_grid()), _cap_enabled(), _gravity_enabled(), dt=100.0).report
    assert "composer_report" in report
    composer = report["composer_report"]
    assert composer["include_capillary"] is True
    assert composer["include_gravity"] is True
    assert composer["max_effective_flux"] == pytest.approx(report["max_effective_flux"])


def test_combined_saturation_bounds() -> None:
    result = _advance(_grid(), _x_step_sw(_grid()), _cap_enabled(), _gravity_enabled(), dt=1000.0)
    assert result.sw.values.min() >= _relperm()["swi"]
    assert result.sw.values.max() <= 1.0 - _relperm()["sor"]


def test_combined_no_nan_inf() -> None:
    result = _advance(_grid(), _x_step_sw(_grid()), _cap_enabled(), _gravity_enabled(), dt=100.0)
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_combined_material_balance() -> None:
    result = _advance(_grid(), _x_step_sw(_grid()), _cap_enabled(), _gravity_enabled(), dt=100.0)
    assert result.report["material_balance_error"] < 1.0e-10


def test_combined_cfl_violation_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=1.0e9)


def test_combined_invalid_capillary_params_raises() -> None:
    grid = _grid()
    params = _cap_enabled()
    params["entry_pressure_pa"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), params, _gravity_enabled(), dt=100.0)
    params = _cap_enabled()
    params["lambda_pc"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), params, _gravity_enabled(), dt=100.0)


def test_combined_invalid_gravity_params_raises() -> None:
    grid = _grid()
    params = _gravity_enabled()
    params["rho_w"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), params, dt=100.0)
    params = _gravity_enabled()
    params["rho_o"] = -1.0
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), params, dt=100.0)


def test_combined_invalid_permeability_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=100.0, kx=-1.0)
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=100.0, ky=-1.0)
    with pytest.raises(InvalidPhysicalValueError):
        _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=100.0, kz=-1.0)


def test_combined_repeatability() -> None:
    grid = _grid()
    first = _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=100.0)
    second = _advance(grid, _x_step_sw(grid), _cap_enabled(), _gravity_enabled(), dt=100.0)
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def test_combined_does_not_modify_input_sw() -> None:
    grid = _grid()
    sw = _x_step_sw(grid)
    original = sw.copy()
    _advance(grid, sw, _cap_enabled(), _gravity_enabled(), dt=100.0)
    assert np.allclose(sw, original)


def test_config_accepts_capillary_gravity_together_when_flags_consistent() -> None:
    config = load_case_config("config/combined_case.yaml")
    assert config["capillary_pressure"]["enabled"] is True
    assert config["gravity"]["enabled"] is True


def test_config_rejects_inconsistent_combined_flags(tmp_path: Path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_capillary"] = False
    path = tmp_path / "combined_inconsistent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="capillary_pressure.enabled=true requires saturation.use_capillary=true"):
        load_case_config(path)


def _grid() -> Grid3D:
    return Grid3D(nx=6, ny=5, nz=4, dx=1.0, dy=1.0, dz=1.0)


def _k() -> float:
    return 1.0e-12


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


def _cap_strong() -> dict[str, float | bool | str]:
    params = _cap_enabled()
    params["entry_pressure_pa"] = 1.0e5
    return params


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


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float),
        np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float),
        np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float),
    )


def _x_step_sw(grid: Grid3D) -> np.ndarray:
    sw = np.full(grid.shape, 0.35)
    sw[:, :, : grid.nx // 2] = 0.65
    return sw


def _uniform_sw(grid: Grid3D) -> np.ndarray:
    return np.full(grid.shape, 0.5)


def _max_jump(sw: np.ndarray, axis: int) -> float:
    return float(np.max(np.abs(np.diff(sw, axis=axis))))


def _advance(
    grid: Grid3D,
    sw: np.ndarray,
    capillary_params: dict[str, float | bool | str],
    gravity_params: dict[str, float | bool | str],
    *,
    dt: float,
    kx: float | None = None,
    ky: float | None = None,
    kz: float | None = None,
):
    return advance_saturation_3d_with_capillary_and_gravity(
        grid=grid,
        sw=sw,
        phi=0.2,
        flux_x=_zero_fluxes(grid)[0],
        flux_y=_zero_fluxes(grid)[1],
        flux_z=_zero_fluxes(grid)[2],
        dt=dt,
        relperm_params=_relperm(),
        capillary_params=capillary_params,
        gravity_params=gravity_params,
        kx=_k() if kx is None else kx,
        ky=_k() if ky is None else ky,
        kz=_k() if kz is None else kz,
    )
