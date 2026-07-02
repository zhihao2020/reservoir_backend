from __future__ import annotations

import numpy as np
import pytest

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.capillary_flux import (
    capillary_mobility,
    compute_absolute_transmissibility_between_cells,
    compute_capillary_fluxes,
)
from reservoir_backend.solver.capillary_pressure import capillary_pressure
from reservoir_backend.solver.transmissibility import harmonic_average


def test_capillary_mobility_shape() -> None:
    sw = np.full((3, 4, 5), 0.5)
    mobility = capillary_mobility(sw, _relperm_params())
    assert isinstance(mobility, np.ndarray)
    assert mobility.shape == sw.shape


def test_capillary_mobility_nonnegative() -> None:
    mobility = np.asarray(capillary_mobility(np.linspace(0.2, 0.8, 11), _relperm_params()))
    assert np.all(mobility >= 0.0)


def test_capillary_mobility_zero_at_endpoints() -> None:
    mobility = np.asarray(capillary_mobility(np.array([0.2, 0.8]), _relperm_params()))
    assert np.allclose(mobility, 0.0)


def test_capillary_flux_shapes_3d() -> None:
    grid = _grid()
    fx, fy, fz, _ = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert fx.shape == (3, 4, 6)
    assert fy.shape == (3, 5, 5)
    assert fz.shape == (4, 4, 5)


def test_no_capillary_flux_disabled() -> None:
    grid = _grid()
    params = _cap_params()
    params["enabled"] = False
    fx, fy, fz, report = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)
    assert np.allclose(fz, 0.0)
    assert report["enabled"] is False


def test_no_capillary_flux_model_none() -> None:
    grid = _grid()
    params = _cap_params()
    params["model"] = "none"
    fx, fy, fz, report = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)
    assert np.allclose(fz, 0.0)
    assert report["enabled"] is False
    assert report["model"] == "none"


def test_zero_pc_gradient_zero_capillary_flux() -> None:
    grid = _grid()
    sw = Field3D.from_constant(grid, 0.5, name="sw")
    fx, fy, fz, _ = compute_capillary_fluxes(grid, sw, 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert np.allclose(fx, 0.0)
    assert np.allclose(fy, 0.0)
    assert np.allclose(fz, 0.0)


def test_x_capillary_flux_direction() -> None:
    grid = _grid()
    fx, _, _, _ = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert np.all(fx[:, :, 1:-1] > 0.0)


def test_y_capillary_flux_direction() -> None:
    grid = _grid()
    _, fy, _, _ = compute_capillary_fluxes(grid, _y_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert np.all(fy[:, 1:-1, :] > 0.0)


def test_z_capillary_flux_direction() -> None:
    grid = _grid()
    _, _, fz, _ = compute_capillary_fluxes(grid, _z_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert np.all(fz[1:-1, :, :] > 0.0)


def test_capillary_flux_no_nan_inf() -> None:
    grid = _grid()
    fx, fy, fz, report = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 90.0e-15, 80.0e-15, _cap_params(), _relperm_params())
    for array in (fx, fy, fz):
        assert not np.isnan(array).any()
        assert not np.isinf(array).any()
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_capillary_flux_boundary_zero() -> None:
    grid = _grid()
    fx, fy, fz, _ = compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())
    assert np.allclose(fx[:, :, 0], 0.0)
    assert np.allclose(fx[:, :, -1], 0.0)
    assert np.allclose(fy[:, 0, :], 0.0)
    assert np.allclose(fy[:, -1, :], 0.0)
    assert np.allclose(fz[0, :, :], 0.0)
    assert np.allclose(fz[-1, :, :], 0.0)


def test_capillary_flux_manual_x() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    sw = np.array([[[0.6, 0.4]]])
    fx, _, _, _ = compute_capillary_fluxes(grid, sw, 10.0e-15, 10.0e-15, 10.0e-15, _cap_params(), _relperm_params())
    expected = _manual_expected(grid, sw, 10.0e-15, "x")
    assert fx[0, 0, 1] == pytest.approx(expected)


def test_capillary_flux_manual_y() -> None:
    grid = Grid3D(nx=1, ny=2, nz=1, dx=2.0, dy=3.0, dz=4.0)
    sw = np.array([[[0.6], [0.4]]])
    _, fy, _, _ = compute_capillary_fluxes(grid, sw, 10.0e-15, 10.0e-15, 10.0e-15, _cap_params(), _relperm_params())
    expected = _manual_expected(grid, sw, 10.0e-15, "y")
    assert fy[0, 1, 0] == pytest.approx(expected)


def test_capillary_flux_manual_z() -> None:
    grid = Grid3D(nx=1, ny=1, nz=2, dx=2.0, dy=3.0, dz=4.0)
    sw = np.array([[[0.6]], [[0.4]]])
    _, _, fz, _ = compute_capillary_fluxes(grid, sw, 10.0e-15, 10.0e-15, 10.0e-15, _cap_params(), _relperm_params())
    expected = _manual_expected(grid, sw, 10.0e-15, "z")
    assert fz[1, 0, 0] == pytest.approx(expected)


def test_invalid_permeability_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        compute_capillary_fluxes(grid, _x_sw(grid), -1.0, 100.0e-15, 100.0e-15, _cap_params(), _relperm_params())


def test_invalid_saturation_params_raises() -> None:
    params = _relperm_params()
    params["swi"] = 0.7
    params["sor"] = 0.4
    with pytest.raises(InvalidPhysicalValueError):
        capillary_mobility(np.array([0.4]), params)


def test_invalid_capillary_params_raises() -> None:
    grid = _grid()
    params = _cap_params()
    params["entry_pressure_pa"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())
    params = _cap_params()
    params["lambda_pc"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        compute_capillary_fluxes(grid, _x_sw(grid), 100.0e-15, 100.0e-15, 100.0e-15, params, _relperm_params())


def test_field3d_input() -> None:
    grid = _grid()
    sw = Field3D(grid, _x_sw(grid), name="sw")
    k = Field3D.from_constant(grid, 100.0e-15, name="kx")
    fx, fy, fz, report = compute_capillary_fluxes(grid, sw, k, k, k, _cap_params(), _relperm_params())
    assert fx.shape == (3, 4, 6)
    assert fy.shape == (3, 5, 5)
    assert fz.shape == (4, 4, 5)
    assert report["max_abs_capillary_flux"] > 0.0


def test_existing_full_pipeline_unchanged(tmp_path) -> None:
    result = run_demo(case_id="cap_flux_disabled", results_root=tmp_path)
    assert result["summary"]["success"] is True
    assert not (result["case_dir"] / "capillary_flux_x.npy").exists()


def test_absolute_transmissibility_between_cells() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=2.0, dy=3.0, dz=4.0)
    t_abs = compute_absolute_transmissibility_between_cells(grid, 10.0e-15, 10.0e-15, 10.0e-15, 0, 1)
    assert t_abs == pytest.approx(10.0e-15 * grid.dy * grid.dz / grid.dx)


def _grid() -> Grid3D:
    return Grid3D(nx=5, ny=4, nz=3, dx=10.0, dy=8.0, dz=6.0)


def _cap_params() -> dict:
    return {
        "enabled": True,
        "model": "brooks_corey",
        "swi": 0.2,
        "sor": 0.2,
        "entry_pressure_pa": 1000.0,
        "lambda_pc": 2.0,
    }


def _relperm_params() -> dict:
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


def _x_sw(grid: Grid3D) -> np.ndarray:
    values = np.linspace(0.7, 0.3, grid.nx).reshape(1, 1, grid.nx)
    return np.broadcast_to(values, grid.shape).copy()


def _y_sw(grid: Grid3D) -> np.ndarray:
    values = np.linspace(0.7, 0.3, grid.ny).reshape(1, grid.ny, 1)
    return np.broadcast_to(values, grid.shape).copy()


def _z_sw(grid: Grid3D) -> np.ndarray:
    values = np.linspace(0.7, 0.3, grid.nz).reshape(grid.nz, 1, 1)
    return np.broadcast_to(values, grid.shape).copy()


def _manual_expected(grid: Grid3D, sw: np.ndarray, permeability: float, direction: str) -> float:
    pc = np.asarray(capillary_pressure(sw, _cap_params()))
    mobility = np.asarray(capillary_mobility(sw, _relperm_params()))
    if direction == "x":
        t_abs = permeability * grid.dy * grid.dz / grid.dx
        m_face = harmonic_average(mobility[0, 0, 0], mobility[0, 0, 1])
        return t_abs * m_face * (pc[0, 0, 1] - pc[0, 0, 0])
    if direction == "y":
        t_abs = permeability * grid.dx * grid.dz / grid.dy
        m_face = harmonic_average(mobility[0, 0, 0], mobility[0, 1, 0])
        return t_abs * m_face * (pc[0, 1, 0] - pc[0, 0, 0])
    t_abs = permeability * grid.dx * grid.dy / grid.dz
    m_face = harmonic_average(mobility[0, 0, 0], mobility[1, 0, 0])
    return t_abs * m_face * (pc[1, 0, 0] - pc[0, 0, 0])
