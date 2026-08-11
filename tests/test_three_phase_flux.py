from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.core.exceptions import FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.solver.relperm import fractional_flow_water
from reservoir_backend.solver.three_phase_flux import (
    compute_three_phase_flux_1d,
    compute_three_phase_fluxes_3d,
    compute_upwind_fractional_flow_1d,
    compute_upwind_fractional_flow_3d,
    validate_three_phase_flux_inputs,
)
from reservoir_backend.solver.three_phase_relperm import fractional_flow_three_phase


def test_three_phase_flux_1d_shapes() -> None:
    flux, sw, sg = _case_1d()
    water, oil, gas, _ = compute_three_phase_flux_1d(flux, sw, sg, _params())
    assert water.shape == flux.shape
    assert oil.shape == flux.shape
    assert gas.shape == flux.shape


def test_three_phase_flux_1d_closure() -> None:
    flux, sw, sg = _case_1d()
    water, oil, gas, _ = compute_three_phase_flux_1d(flux, sw, sg, _params())
    assert np.allclose(water + oil + gas, flux)


def test_three_phase_flux_1d_positive_upwind() -> None:
    flux = np.array([0.0, 2.0, 0.0])
    sw = np.array([0.25, 0.45])
    sg = np.array([0.10, 0.20])
    fw, fo, fg = compute_upwind_fractional_flow_1d(flux, sw, sg, _params())
    expected = fractional_flow_three_phase(sw[0], sg[0], _params())
    assert fw[1] == pytest.approx(expected[0])
    assert fo[1] == pytest.approx(expected[1])
    assert fg[1] == pytest.approx(expected[2])


def test_three_phase_flux_1d_negative_upwind() -> None:
    flux = np.array([0.0, -2.0, 0.0])
    sw = np.array([0.25, 0.45])
    sg = np.array([0.10, 0.20])
    fw, fo, fg = compute_upwind_fractional_flow_1d(flux, sw, sg, _params())
    expected = fractional_flow_three_phase(sw[1], sg[1], _params())
    assert fw[1] == pytest.approx(expected[0])
    assert fo[1] == pytest.approx(expected[1])
    assert fg[1] == pytest.approx(expected[2])


def test_three_phase_flux_1d_zero_flux() -> None:
    flux, sw, sg = np.zeros(4), np.array([0.25, 0.35, 0.45]), np.array([0.10, 0.15, 0.20])
    water, oil, gas, _ = compute_three_phase_flux_1d(flux, sw, sg, _params())
    assert np.allclose(water, 0.0)
    assert np.allclose(oil, 0.0)
    assert np.allclose(gas, 0.0)


def test_three_phase_flux_3d_shapes() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    result = compute_three_phase_fluxes_3d(fx, fy, fz, sw, sg, _params())
    for actual, expected in zip(result[:9], [fx, fy, fz, fx, fy, fz, fx, fy, fz], strict=True):
        assert actual.shape == expected.shape


def test_three_phase_flux_3d_closure_x() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    water_x, _, _, oil_x, _, _, gas_x, _, _, _ = compute_three_phase_fluxes_3d(fx, fy, fz, sw, sg, _params())
    assert np.allclose(water_x + oil_x + gas_x, fx)


def test_three_phase_flux_3d_closure_y() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    _, water_y, _, _, oil_y, _, _, gas_y, _, _ = compute_three_phase_fluxes_3d(fx, fy, fz, sw, sg, _params())
    assert np.allclose(water_y + oil_y + gas_y, fy)


def test_three_phase_flux_3d_closure_z() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    _, _, water_z, _, _, oil_z, _, _, gas_z, _ = compute_three_phase_fluxes_3d(fx, fy, fz, sw, sg, _params())
    assert np.allclose(water_z + oil_z + gas_z, fz)


def test_three_phase_flux_3d_x_upwind_positive_negative() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    fx[:] = 0.0
    fx[0, 0, 1] = 1.0
    fx[0, 0, 2] = -1.0
    fw_x, fo_x, fg_x, *_ = compute_upwind_fractional_flow_3d(fx, fy, fz, sw, sg, _params())
    expected_left = fractional_flow_three_phase(sw[0, 0, 0], sg[0, 0, 0], _params())
    expected_right = fractional_flow_three_phase(sw[0, 0, 2], sg[0, 0, 2], _params())
    assert (fw_x[0, 0, 1], fo_x[0, 0, 1], fg_x[0, 0, 1]) == pytest.approx(expected_left)
    assert (fw_x[0, 0, 2], fo_x[0, 0, 2], fg_x[0, 0, 2]) == pytest.approx(expected_right)


def test_three_phase_flux_3d_y_upwind_positive_negative() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    fy[:] = 0.0
    fy[0, 1, 0] = 1.0
    fy[0, 2, 0] = -1.0
    *_, fw_y, fo_y, fg_y, _, _, _ = compute_upwind_fractional_flow_3d(fx, fy, fz, sw, sg, _params())
    expected_front = fractional_flow_three_phase(sw[0, 0, 0], sg[0, 0, 0], _params())
    expected_back = fractional_flow_three_phase(sw[0, 2, 0], sg[0, 2, 0], _params())
    assert (fw_y[0, 1, 0], fo_y[0, 1, 0], fg_y[0, 1, 0]) == pytest.approx(expected_front)
    assert (fw_y[0, 2, 0], fo_y[0, 2, 0], fg_y[0, 2, 0]) == pytest.approx(expected_back)


def test_three_phase_flux_3d_z_upwind_positive_negative() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    fz[:] = 0.0
    fz[1, 0, 0] = 1.0
    fz[2, 0, 0] = -1.0
    *_, fw_z, fo_z, fg_z = compute_upwind_fractional_flow_3d(fx, fy, fz, sw, sg, _params())
    expected_bottom = fractional_flow_three_phase(sw[0, 0, 0], sg[0, 0, 0], _params())
    expected_top = fractional_flow_three_phase(sw[2, 0, 0], sg[2, 0, 0], _params())
    assert (fw_z[1, 0, 0], fo_z[1, 0, 0], fg_z[1, 0, 0]) == pytest.approx(expected_bottom)
    assert (fw_z[2, 0, 0], fo_z[2, 0, 0], fg_z[2, 0, 0]) == pytest.approx(expected_top)


def test_three_phase_flux_no_nan_inf() -> None:
    result = compute_three_phase_fluxes_3d(*_case_3d(), _params())
    for values in result[:9]:
        assert np.isfinite(values).all()


def test_three_phase_flux_report_keys() -> None:
    report = compute_three_phase_fluxes_3d(*_case_3d(), _params())[-1]
    keys = {
        "max_total_flux",
        "max_water_flux",
        "max_oil_flux",
        "max_gas_flux",
        "min_water_flux",
        "min_oil_flux",
        "min_gas_flux",
        "phase_flux_closure_error_max",
        "has_nan",
        "has_inf",
        "flux_shape_x",
        "flux_shape_y",
        "flux_shape_z",
    }
    assert keys.issubset(report)


def test_three_phase_flux_report_closure_error_small() -> None:
    report = compute_three_phase_fluxes_3d(*_case_3d(), _params())[-1]
    assert report["phase_flux_closure_error_max"] <= 1.0e-14


def test_invalid_flux_shape_raises() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    with pytest.raises(FieldShapeError):
        validate_three_phase_flux_inputs(fx[:, :, :-1], fy, fz, sw, sg, _params())


def test_invalid_saturation_shape_raises() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    with pytest.raises(FieldShapeError):
        validate_three_phase_flux_inputs(fx, fy, fz, sw[:, :, :-1], sg, _params())


def test_invalid_saturation_state_raises() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    sw = sw.copy()
    sw[0, 0, 0] = 0.9
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_flux_inputs(fx, fy, fz, sw, sg, _params())


def test_invalid_three_phase_params_raises() -> None:
    fx, fy, fz, sw, sg = _case_3d()
    params = _params(swi=0.5, sor=0.4, sgc=0.1)
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_flux_inputs(fx, fy, fz, sw, sg, params)


def test_flux_inputs_not_modified() -> None:
    arrays = _case_3d()
    originals = tuple(array.copy() for array in arrays)
    compute_three_phase_fluxes_3d(*arrays, _params())
    for array, original in zip(arrays, originals, strict=True):
        assert np.allclose(array, original)


def test_repeatability() -> None:
    first = compute_three_phase_fluxes_3d(*_case_3d(), _params())
    second = compute_three_phase_fluxes_3d(*_case_3d(), _params())
    for a, b in zip(first[:9], second[:9], strict=True):
        assert np.allclose(a, b)
    assert first[-1] == second[-1]


def test_three_phase_transport_still_not_claimed() -> None:
    text = Path("STATUS.md").read_text(encoding="utf-8")
    assert "三相" in text

def _params(**overrides: float) -> dict[str, float]:
    params = {
        "swi": 0.2,
        "sor": 0.2,
        "sgc": 0.05,
        "krw0": 0.3,
        "kro0": 0.8,
        "krg0": 0.6,
        "nw": 2.0,
        "no": 2.0,
        "ng": 2.0,
        "mu_w": 1.0e-3,
        "mu_o": 5.0e-3,
        "mu_g": 1.0e-5,
    }
    params.update(overrides)
    return params


def _case_1d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flux = np.array([1.0, 2.0, -1.5, 0.5])
    sw = np.array([0.25, 0.35, 0.45])
    sg = np.array([0.10, 0.15, 0.20])
    return flux, sw, sg


def _case_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = 3, 3, 4
    k, j, i = np.indices((nz, ny, nx), dtype=float)
    sw = 0.25 + 0.03 * i + 0.02 * j
    sg = 0.10 + 0.02 * k + 0.01 * i
    fx = np.full((nz, ny, nx + 1), 1.0)
    fy = np.full((nz, ny + 1, nx), -0.5)
    fz = np.full((nz + 1, ny, nx), 0.25)
    fx[:, :, 0] = 0.0
    fx[:, :, -1] = 0.0
    fy[:, 0, :] = 0.0
    fy[:, -1, :] = 0.0
    fz[0, :, :] = 0.0
    fz[-1, :, :] = 0.0
    return fx, fy, fz, sw, sg
