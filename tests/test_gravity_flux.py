from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.io.config_loader import load_case_config
from reservoir_backend.solver.gravity_flux import (
    compute_gravity_fluxes,
    compute_gravity_water_flux_1d_vertical,
    gravity_mobility,
    validate_gravity_params,
)
from reservoir_backend.solver.transmissibility import harmonic_average


def test_gravity_mobility_shape() -> None:
    sw = np.full((3, 4, 5), 0.5)
    mobility = gravity_mobility(sw, _relperm_params())
    assert isinstance(mobility, np.ndarray)
    assert mobility.shape == sw.shape


def test_gravity_mobility_nonnegative() -> None:
    mobility = np.asarray(gravity_mobility(np.linspace(0.2, 0.8, 11), _relperm_params()))
    assert np.all(mobility >= 0.0)


def test_gravity_mobility_zero_at_endpoints() -> None:
    mobility = np.asarray(gravity_mobility(np.array([0.2, 0.8]), _relperm_params()))
    assert np.allclose(mobility, 0.0)


def test_gravity_flux_shapes_3d() -> None:
    grid = _grid()
    fx, fy, fz, _ = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, _gravity_params(), _relperm_params())
    assert fx.shape == (3, 4, 6)
    assert fy.shape == (3, 5, 5)
    assert fz.shape == (4, 4, 5)


def test_gravity_disabled_zero_flux() -> None:
    grid = _grid()
    params = _gravity_params()
    params["enabled"] = False
    fx, fy, fz, report = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)
    assert np.allclose(fz, 0.0)
    assert report["enabled"] is False


def test_zero_density_difference_zero_flux() -> None:
    grid = _grid()
    params = _gravity_params()
    params["rho_o"] = params["rho_w"]
    fx, fy, fz, report = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)
    assert np.allclose(fz, 0.0)
    assert report["density_difference"] == pytest.approx(0.0)


def test_gravity_flux_x_y_zero_for_horizontal_grid() -> None:
    grid = _grid()
    fx, fy, _, _ = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, _gravity_params(), _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)


def test_gravity_flux_z_direction_water_heavier() -> None:
    grid = _grid()
    _, _, fz, _ = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, _gravity_params(), _relperm_params())
    assert np.all(fz[1:-1, :, :] < 0.0)


def test_gravity_flux_z_direction_oil_heavier() -> None:
    grid = _grid()
    params = _gravity_params()
    params["rho_w"] = 700.0
    params["rho_o"] = 900.0
    _, _, fz, _ = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    assert np.all(fz[1:-1, :, :] > 0.0)


def test_gravity_flux_boundary_zero() -> None:
    grid = _grid()
    fx, fy, fz, _ = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, 100.0e-15, _gravity_params(), _relperm_params())
    assert np.allclose(fx[:, :, 0], 0.0)
    assert np.allclose(fx[:, :, -1], 0.0)
    assert np.allclose(fy[:, 0, :], 0.0)
    assert np.allclose(fy[:, -1, :], 0.0)
    assert np.allclose(fz[0, :, :], 0.0)
    assert np.allclose(fz[-1, :, :], 0.0)


def test_gravity_flux_manual_z() -> None:
    grid = Grid3D(nx=1, ny=1, nz=2, dx=2.0, dy=3.0, dz=4.0)
    sw = np.array([[[0.5]], [[0.5]]])
    flux_z, _ = compute_gravity_water_flux_1d_vertical(grid, sw, 10.0e-15, _gravity_params(), _relperm_params())
    mobility = np.asarray(gravity_mobility(sw, _relperm_params()))
    t_abs = 10.0e-15 * grid.dx * grid.dy / grid.dz
    m_face = harmonic_average(mobility[0, 0, 0], mobility[1, 0, 0])
    expected = -t_abs * m_face * (1000.0 - 800.0) * 9.80665 * grid.dz
    assert flux_z[1, 0, 0] == pytest.approx(expected)


def test_gravity_flux_no_nan_inf() -> None:
    grid = _grid()
    fx, fy, fz, report = compute_gravity_fluxes(grid, 0.5, 100.0e-15, 90.0e-15, 80.0e-15, _gravity_params(), _relperm_params())
    for array in (fx, fy, fz):
        assert np.isfinite(array).all()
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_invalid_density_raises() -> None:
    params = _gravity_params()
    params["rho_w"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_gravity_params(params)
    params = _gravity_params()
    params["rho_o"] = -1.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_gravity_params(params)


def test_invalid_gravity_acceleration_raises() -> None:
    params = _gravity_params()
    params["g"] = -1.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_gravity_params(params)


def test_invalid_permeability_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_gravity_fluxes(grid, 0.5, 100.0e-15, 100.0e-15, -1.0, _gravity_params(), _relperm_params())


def test_field3d_input() -> None:
    grid = _grid()
    sw = Field3D.from_constant(grid, 0.5, name="sw")
    k = Field3D.from_constant(grid, 100.0e-15, name="k")
    fx, fy, fz, report = compute_gravity_fluxes(grid, sw, k, k, k, _gravity_params(), _relperm_params())
    assert fx.shape == (3, 4, 6)
    assert fy.shape == (3, 5, 5)
    assert fz.shape == (4, 4, 5)
    assert report["max_abs_gravity_flux"] > 0.0


def test_config_loader_accepts_gravity_section() -> None:
    config = load_case_config("config/demo_case.yaml")
    assert "gravity" in config
    assert config["gravity"]["enabled"] is False
    assert config["gravity"]["rho_w"] == pytest.approx(1000.0)


def test_existing_cli_dry_run_still_passes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_case.py", "--config", "config/demo_case.yaml", "--dry-run"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert '"success": true' in result.stdout.lower()


def test_existing_full_pipeline_unchanged_when_disabled(tmp_path) -> None:
    result = run_demo(case_id="gravity_disabled", results_root=tmp_path)
    assert result["summary"]["success"] is True
    assert "gravity_enabled" not in result["summary"]
    assert not (result["case_dir"] / "gravity_flux_z.npy").exists()


def _grid() -> Grid3D:
    return Grid3D(nx=5, ny=4, nz=3, dx=10.0, dy=8.0, dz=6.0)


def _gravity_params() -> dict[str, float | bool | str]:
    return {
        "enabled": True,
        "g": 9.80665,
        "rho_w": 1000.0,
        "rho_o": 800.0,
        "depth_axis": "z",
        "depth_positive": "down",
    }


def _relperm_params() -> dict[str, float]:
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
