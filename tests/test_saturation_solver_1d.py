from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.relperm import fractional_flow_water
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    compute_upwind_water_flux_1d,
    compute_water_cut_1d,
)


def test_1d_saturation_shape() -> None:
    grid = _grid(nx=10)
    result = advance_saturation_1d(grid, 0.2, 0.2, _positive_flux(grid), 100.0, _params())
    assert result.sw.values.shape == (1, 1, 10)


def test_1d_saturation_no_nan_inf() -> None:
    grid = _grid()
    result = advance_saturation_1d(grid, 0.2, 0.2, _positive_flux(grid), 100.0, _params())
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_1d_saturation_bounds() -> None:
    grid = _grid()
    result = advance_saturation_1d(grid, 0.79, 0.2, _positive_flux(grid), 100.0, _params())
    assert result.sw.values.min() >= _params()["swi"]
    assert result.sw.values.max() <= 1.0 - _params()["sor"]


def test_1d_no_velocity_no_change() -> None:
    grid = _grid()
    sw = np.linspace(0.2, 0.6, grid.nx).reshape(grid.shape)
    result = advance_saturation_1d(grid, sw, 0.2, _zero_flux(grid), 1000.0, _params())
    assert np.allclose(result.sw.values, sw)


def test_1d_positive_flux_front_moves_right() -> None:
    grid = _grid(nx=30, dx=1.0)
    sw = Field3D.from_constant(grid, 0.2, name="sw", unit="fraction")
    for _ in range(80):
        sw = advance_saturation_1d(grid, sw, 0.2, _positive_flux(grid, 1.0e-5), 200.0, _params()).sw
    wet_cells = np.flatnonzero(sw.values[0, 0, :] > 0.2001)
    assert wet_cells.size > 1
    assert wet_cells[-1] > 0


def test_1d_negative_flux_front_moves_left() -> None:
    grid = _grid(nx=30, dx=1.0)
    sw = Field3D.from_constant(grid, 0.2, name="sw", unit="fraction")
    for _ in range(80):
        sw = advance_saturation_1d(grid, sw, 0.2, _negative_flux(grid, -1.0e-5), 200.0, _params()).sw
    wet_cells = np.flatnonzero(sw.values[0, 0, :] > 0.2001)
    assert wet_cells.size > 1
    assert wet_cells[0] < grid.nx - 1


def test_1d_upwind_positive_flux() -> None:
    grid = _grid(nx=3)
    sw = np.array([0.2, 0.5, 0.8]).reshape(grid.shape)
    flux = _positive_flux(grid, 2.0)
    water_flux = compute_upwind_water_flux_1d(sw, flux, _params())[0, 0, :]
    fw_left_cell = fractional_flow_water(0.2, **_fractional_kwargs())
    assert water_flux[1] == pytest.approx(2.0 * fw_left_cell)


def test_1d_upwind_negative_flux() -> None:
    grid = _grid(nx=3)
    sw = np.array([0.2, 0.5, 0.8]).reshape(grid.shape)
    flux = _negative_flux(grid, -2.0)
    water_flux = compute_upwind_water_flux_1d(sw, flux, _params())[0, 0, :]
    fw_right_cell = fractional_flow_water(0.5, **_fractional_kwargs())
    assert water_flux[1] == pytest.approx(-2.0 * fw_right_cell)


def test_1d_cfl_valid_dt() -> None:
    grid = _grid()
    result = advance_saturation_1d(grid, 0.2, 0.2, _positive_flux(grid), 100.0, _params())
    assert result.report["stable"] is True
    assert result.report["max_cfl"] <= 1.0


def test_1d_cfl_violation_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        advance_saturation_1d(grid, 0.2, 0.2, _positive_flux(grid, 1.0e-3), 1.0e6, _params())


def test_1d_water_cut_range() -> None:
    grid = _grid()
    water_cut = compute_water_cut_1d(np.full(grid.shape, 0.5), _positive_flux(grid), _params())
    assert 0.0 <= water_cut <= 1.0


def test_1d_water_cut_low_before_breakthrough() -> None:
    grid = _grid()
    water_cut = compute_water_cut_1d(np.full(grid.shape, 0.2), _positive_flux(grid), _params())
    assert water_cut == pytest.approx(0.0)


def test_1d_water_cut_increases_after_front_reaches_producer() -> None:
    grid = _grid()
    low = compute_water_cut_1d(np.full(grid.shape, 0.2), _positive_flux(grid), _params())
    high_sw = np.full(grid.shape, 0.2)
    high_sw[0, 0, -1] = 0.8
    high = compute_water_cut_1d(high_sw, _positive_flux(grid), _params())
    assert high > low


def test_1d_material_balance_no_flux() -> None:
    grid = _grid()
    result = advance_saturation_1d(grid, 0.4, 0.2, _zero_flux(grid), 1000.0, _params())
    assert result.report["storage_change"] == pytest.approx(0.0)
    assert result.report["injected_water_volume"] == pytest.approx(0.0)
    assert result.report["produced_water_volume"] == pytest.approx(0.0)


def test_1d_material_balance_positive_flux() -> None:
    grid = _grid(nx=20)
    result = advance_saturation_1d(grid, 0.2, 0.25, _positive_flux(grid, 1.0e-5), 1000.0, _params())
    assert result.report["storage_change"] == pytest.approx(
        result.report["injected_water_volume"] - result.report["produced_water_volume"],
        rel=1.0e-12,
    )
    assert result.report["material_balance_error"] < 1.0e-12


def test_1d_invalid_grid_dimension_raises() -> None:
    grid = Grid3D(nx=4, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(NotImplementedError):
        advance_saturation_1d(grid, 0.2, 0.2, np.zeros((1, 2, 5)), 1.0, _params())


def test_1d_invalid_flux_shape_raises() -> None:
    grid = _grid()
    with pytest.raises(FieldShapeError):
        advance_saturation_1d(grid, 0.2, 0.2, np.zeros((grid.nx + 1,)), 1.0, _params())


def test_1d_invalid_saturation_params_raises() -> None:
    grid = _grid()
    params = _params()
    params["swi"] = 0.7
    params["sor"] = 0.4
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_1d(grid, 0.2, 0.2, _positive_flux(grid), 1.0, params)


def test_1d_invalid_porosity_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_1d(grid, 0.2, 0.0, _positive_flux(grid), 1.0, _params())


def test_1d_repeatability() -> None:
    grid = _grid()
    args = (grid, 0.2, 0.2, _positive_flux(grid), 100.0, _params())
    first = advance_saturation_1d(*args)
    second = advance_saturation_1d(*args)
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def _grid(nx: int = 10, dx: float = 1.0) -> Grid3D:
    return Grid3D(nx=nx, ny=1, nz=1, dx=dx, dy=1.0, dz=1.0)


def _params() -> dict[str, float]:
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


def _fractional_kwargs() -> dict[str, float]:
    return _params()


def _positive_flux(grid: Grid3D, value: float = 1.0e-5) -> np.ndarray:
    return np.full((1, 1, grid.nx + 1), value, dtype=float)


def _negative_flux(grid: Grid3D, value: float = -1.0e-5) -> np.ndarray:
    return np.full((1, 1, grid.nx + 1), value, dtype=float)


def _zero_flux(grid: Grid3D) -> np.ndarray:
    return np.zeros((1, 1, grid.nx + 1), dtype=float)
