from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.relperm import fractional_flow_water
from reservoir_backend.solver.saturation_solver import (
    advance_saturation_1d,
    advance_saturation_3d,
    compute_upwind_water_flux_3d,
    compute_water_cut_3d,
)


def test_3d_saturation_shape() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.2, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert result.sw.values.shape == (3, 4, 5)


def test_3d_saturation_no_nan_inf() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.2, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert not np.isnan(result.sw.values).any()
    assert not np.isinf(result.sw.values).any()
    assert result.report["has_nan"] is False
    assert result.report["has_inf"] is False


def test_3d_saturation_bounds() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.79, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert result.sw.values.min() >= 0.2
    assert result.sw.values.max() <= 0.8


def test_3d_zero_flux_no_change() -> None:
    grid = _grid()
    sw = _pattern_sw(grid)
    result = advance_saturation_3d(grid, sw, 0.2, *_zero_fluxes(grid), 100.0, _params())
    assert np.allclose(result.sw.values, sw)


def test_3d_positive_x_flux_front_moves_right() -> None:
    sw = _advance_many(_x_flux, 1.0e-5)
    wet = np.flatnonzero(sw[1, 2, :] > 0.2001)
    assert wet.size > 1
    assert wet[-1] > 0


def test_3d_negative_x_flux_front_moves_left() -> None:
    sw = _advance_many(_x_flux, -1.0e-5)
    wet = np.flatnonzero(sw[1, 2, :] > 0.2001)
    assert wet.size > 1
    assert wet[0] < sw.shape[2] - 1


def test_3d_positive_y_flux_front_moves_back() -> None:
    sw = _advance_many(_y_flux, 1.0e-5)
    wet = np.flatnonzero(sw[1, :, 2] > 0.2001)
    assert wet.size > 1
    assert wet[-1] > 0


def test_3d_negative_y_flux_front_moves_front() -> None:
    sw = _advance_many(_y_flux, -1.0e-5)
    wet = np.flatnonzero(sw[1, :, 2] > 0.2001)
    assert wet.size > 1
    assert wet[0] < sw.shape[1] - 1


def test_3d_positive_z_flux_front_moves_up() -> None:
    sw = _advance_many(_z_flux, 1.0e-5)
    wet = np.flatnonzero(sw[:, 2, 2] > 0.2001)
    assert wet.size > 1
    assert wet[-1] > 0


def test_3d_negative_z_flux_front_moves_down() -> None:
    sw = _advance_many(_z_flux, -1.0e-5)
    wet = np.flatnonzero(sw[:, 2, 2] > 0.2001)
    assert wet.size > 1
    assert wet[0] < sw.shape[0] - 1


def test_3d_upwind_x_positive_flux() -> None:
    grid = _grid(nx=3, ny=2, nz=2)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fx[0, 0, 1] = 2.0
    water_x, _, _ = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_x[0, 0, 1] == pytest.approx(2.0 * _fw(sw[0, 0, 0]))


def test_3d_upwind_x_negative_flux() -> None:
    grid = _grid(nx=3, ny=2, nz=2)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fx[0, 0, 1] = -2.0
    water_x, _, _ = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_x[0, 0, 1] == pytest.approx(-2.0 * _fw(sw[0, 0, 1]))


def test_3d_upwind_y_positive_flux() -> None:
    grid = _grid(nx=2, ny=3, nz=2)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fy[0, 1, 0] = 2.0
    _, water_y, _ = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_y[0, 1, 0] == pytest.approx(2.0 * _fw(sw[0, 0, 0]))


def test_3d_upwind_y_negative_flux() -> None:
    grid = _grid(nx=2, ny=3, nz=2)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fy[0, 1, 0] = -2.0
    _, water_y, _ = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_y[0, 1, 0] == pytest.approx(-2.0 * _fw(sw[0, 1, 0]))


def test_3d_upwind_z_positive_flux() -> None:
    grid = _grid(nx=2, ny=2, nz=3)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fz[1, 0, 0] = 2.0
    _, _, water_z = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_z[1, 0, 0] == pytest.approx(2.0 * _fw(sw[0, 0, 0]))


def test_3d_upwind_z_negative_flux() -> None:
    grid = _grid(nx=2, ny=2, nz=3)
    sw = _pattern_sw(grid)
    fx, fy, fz = _zero_fluxes(grid)
    fz[1, 0, 0] = -2.0
    _, _, water_z = compute_upwind_water_flux_3d(sw, fx, fy, fz, _params())
    assert water_z[1, 0, 0] == pytest.approx(-2.0 * _fw(sw[1, 0, 0]))


def test_3d_cfl_valid_dt() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.2, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert result.report["stable"] is True


def test_3d_cfl_violation_raises() -> None:
    grid = _grid()
    with pytest.raises(CFLViolationError):
        advance_saturation_3d(grid, 0.2, 0.2, *_x_flux(grid, 1.0e-3), 1.0e6, _params())


def test_3d_water_cut_range() -> None:
    grid = _grid()
    water_cut = compute_water_cut_3d(np.full(grid.shape, 0.5), *_x_flux(grid, 1.0e-5), _params())
    assert 0.0 <= water_cut <= 1.0


def test_3d_material_balance_zero_flux() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.4, 0.2, *_zero_fluxes(grid), 10.0, _params())
    assert result.report["storage_change"] == pytest.approx(0.0)
    assert result.report["injected_water_volume"] == pytest.approx(0.0)
    assert result.report["produced_water_volume"] == pytest.approx(0.0)


def test_3d_material_balance_positive_flux() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.2, 0.25, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert result.report["storage_change"] == pytest.approx(
        result.report["injected_water_volume"] - result.report["produced_water_volume"],
        rel=1.0e-12,
    )
    assert result.report["material_balance_error"] < 1.0e-12


def test_3d_invalid_flux_x_shape_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    with pytest.raises(FieldShapeError):
        advance_saturation_3d(grid, 0.2, 0.2, fx[:, :, :-1], fy, fz, 1.0, _params())


def test_3d_invalid_flux_y_shape_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    with pytest.raises(FieldShapeError):
        advance_saturation_3d(grid, 0.2, 0.2, fx, fy[:, :-1, :], fz, 1.0, _params())


def test_3d_invalid_flux_z_shape_raises() -> None:
    grid = _grid()
    fx, fy, fz = _zero_fluxes(grid)
    with pytest.raises(FieldShapeError):
        advance_saturation_3d(grid, 0.2, 0.2, fx, fy, fz[:-1, :, :], 1.0, _params())


def test_3d_invalid_porosity_raises() -> None:
    grid = _grid()
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d(grid, 0.2, 0.0, *_zero_fluxes(grid), 1.0, _params())


def test_3d_invalid_saturation_params_raises() -> None:
    grid = _grid()
    params = _params()
    params["swi"] = 0.7
    params["sor"] = 0.4
    with pytest.raises(InvalidPhysicalValueError):
        advance_saturation_3d(grid, 0.2, 0.2, *_zero_fluxes(grid), 1.0, params)


def test_3d_invalid_grid_dimension_raises() -> None:
    grid = Grid3D(nx=5, ny=1, nz=3, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(NotImplementedError):
        advance_saturation_3d(grid, 0.2, 0.2, *_zero_fluxes(grid), 1.0, _params())


def test_3d_repeatability() -> None:
    grid = _grid()
    args = (grid, 0.2, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    first = advance_saturation_3d(*args)
    second = advance_saturation_3d(*args)
    assert np.allclose(first.sw.values, second.sw.values)
    assert first.report["material_balance_error"] == pytest.approx(second.report["material_balance_error"])


def test_3d_injection_region_sw_increases() -> None:
    grid = _grid()
    result = advance_saturation_3d(grid, 0.2, 0.2, *_x_flux(grid, 1.0e-5), 10.0, _params())
    assert np.all(result.sw.values[:, :, 0] > 0.2)


def test_3d_preserve_existing_1d_behavior() -> None:
    grid = Grid3D(nx=8, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    flux_x = np.full((1, 1, grid.nx + 1), 1.0e-5)
    result = advance_saturation_1d(grid, 0.2, 0.2, flux_x, 100.0, _params())
    assert result.sw.values.shape == grid.shape
    assert result.sw.values[0, 0, 0] > 0.2


def _grid(nx: int = 5, ny: int = 4, nz: int = 3) -> Grid3D:
    return Grid3D(nx=nx, ny=ny, nz=nz, dx=1.0, dy=1.0, dz=1.0)


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


def _fw(sw: float) -> float:
    return float(fractional_flow_water(sw, **_params()))


def _pattern_sw(grid: Grid3D) -> np.ndarray:
    values = np.zeros(grid.shape, dtype=float)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                values[k, j, i] = 0.2 + 0.6 * (i + j + k) / (grid.nx + grid.ny + grid.nz - 3)
    return values


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


def _y_flux(grid: Grid3D, value: float):
    fx, fy, fz = _zero_fluxes(grid)
    fy[:, :, :] = value
    return fx, fy, fz


def _z_flux(grid: Grid3D, value: float):
    fx, fy, fz = _zero_fluxes(grid)
    fz[:, :, :] = value
    return fx, fy, fz


def _advance_many(flux_factory, value: float) -> np.ndarray:
    grid = _grid()
    sw = np.full(grid.shape, 0.2, dtype=float)
    for _ in range(80):
        sw = advance_saturation_3d(grid, sw, 0.2, *flux_factory(grid, value), 10.0, _params()).sw.values
    return sw
