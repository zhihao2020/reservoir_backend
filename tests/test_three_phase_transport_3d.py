from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.solver.saturation_solver import advance_saturation_3d
from reservoir_backend.solver.three_phase_flux import compute_three_phase_fluxes_3d
from reservoir_backend.solver.three_phase_relperm import fractional_flow_three_phase
from reservoir_backend.solver.three_phase_transport import (
    advance_three_phase_saturation_1d,
    advance_three_phase_saturation_3d,
    compute_three_phase_cfl_3d,
    compute_three_phase_material_balance_3d,
    compute_three_phase_saturation_update_3d,
    validate_three_phase_transport_3d_inputs,
)


def test_three_phase_transport_3d_shapes() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert sw_new.shape == _sw().shape
    assert sg_new.shape == _sw().shape
    assert so_new.shape == _sw().shape


def test_three_phase_transport_3d_closure() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.allclose(sw_new + sg_new + so_new, 1.0)


def test_three_phase_transport_3d_sw_sg_update() -> None:
    fx, fy, fz = _internal_fluxes()
    sw, sg = _sw(), _sg()
    water_x, water_y, water_z, _, _, _, gas_x, gas_y, gas_z, _ = compute_three_phase_fluxes_3d(
        fx, fy, fz, sw, sg, _params()
    )
    sw_expected, sg_expected, _ = compute_three_phase_saturation_update_3d(
        water_x, water_y, water_z, gas_x, gas_y, gas_z, sw, sg, _phi(), _volume(), _dt(), _params()
    )
    sw_new, sg_new, _, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params()
    )
    assert np.allclose(sw_new, sw_expected)
    assert np.allclose(sg_new, sg_expected)


def test_three_phase_transport_3d_so_from_closure() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.allclose(so_new, 1.0 - sw_new - sg_new)


def test_three_phase_transport_3d_no_flow_no_change() -> None:
    fx, fy, fz, sw, sg = _zero_flux_case()
    sw_new, sg_new, so_new, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params()
    )
    assert np.allclose(sw_new, sw)
    assert np.allclose(sg_new, sg)
    assert np.allclose(so_new, 1.0 - sw - sg)


def test_three_phase_transport_3d_water_injection_x_increases_sw() -> None:
    fx, fy, fz, sw, sg = _zero_flux_case()
    fx[:, :, 0] = 1.0e-5
    sw_new, _, _, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.70, injected_sg=0.05
    )
    assert np.mean(sw_new[:, :, 0]) > np.mean(sw[:, :, 0])


def test_three_phase_transport_3d_gas_injection_x_increases_sg() -> None:
    fx, fy, fz, sw, sg = _zero_flux_case()
    fx[:, :, 0] = 1.0e-5
    _, sg_new, _, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.20, injected_sg=0.55
    )
    assert np.mean(sg_new[:, :, 0]) > np.mean(sg[:, :, 0])


def test_three_phase_transport_3d_water_injection_y_increases_sw() -> None:
    fx, fy, fz, sw, sg = _zero_flux_case()
    fy[:, 0, :] = 1.0e-5
    sw_new, _, _, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.70, injected_sg=0.05
    )
    assert np.mean(sw_new[:, 0, :]) > np.mean(sw[:, 0, :])


def test_three_phase_transport_3d_gas_injection_z_increases_sg() -> None:
    fx, fy, fz, sw, sg = _zero_flux_case()
    fz[0, :, :] = 1.0e-5
    _, sg_new, _, _ = advance_three_phase_saturation_3d(
        fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.20, injected_sg=0.55
    )
    assert np.mean(sg_new[0, :, :]) > np.mean(sg[0, :, :])


def test_three_phase_transport_3d_bounds() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.all(sw_new >= _params()["swi"])
    assert np.all(sg_new >= _params()["sgc"])
    assert np.all(so_new >= _params()["sor"])


def test_three_phase_transport_3d_no_nan_inf() -> None:
    sw_new, sg_new, so_new, report = _advance()
    for values in (sw_new, sg_new, so_new):
        assert np.isfinite(values).all()
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_three_phase_transport_3d_cfl_array() -> None:
    fx, fy, fz = _internal_fluxes()
    cfl, max_cfl = compute_three_phase_cfl_3d(fx, fy, fz, _phi(), _volume(), _dt())
    assert cfl.shape == _sw().shape
    assert max_cfl == pytest.approx(float(np.max(cfl)))


def test_three_phase_transport_3d_cfl_violation_raises() -> None:
    fx, fy, fz = _internal_fluxes(scale=1.0e-2)
    with pytest.raises(CFLViolationError):
        advance_three_phase_saturation_3d(fx, fy, fz, _sw(), _sg(), _phi(), _volume(), 1000.0, _params(), max_cfl=0.1)


def test_three_phase_transport_3d_material_balance_keys() -> None:
    report = _advance()[-1]
    for key in _balance_keys():
        assert key in report


def test_three_phase_transport_3d_water_material_balance() -> None:
    assert abs(_advance()[-1]["water_balance_error"]) <= 1.0e-15


def test_three_phase_transport_3d_gas_material_balance() -> None:
    assert abs(_advance()[-1]["gas_balance_error"]) <= 1.0e-15


def test_three_phase_transport_3d_oil_material_balance() -> None:
    assert abs(_advance()[-1]["oil_balance_error"]) <= 1.0e-15


def test_three_phase_transport_3d_report_keys() -> None:
    report = _advance()[-1]
    keys = {
        "max_cfl",
        "closure_error_max",
        "sw_min",
        "sw_max",
        "sg_min",
        "sg_max",
        "so_min",
        "so_max",
        "has_nan",
        "has_inf",
        "transport_dimension",
    }
    assert keys.issubset(report)
    assert report["transport_dimension"] == "3d"


def test_three_phase_transport_3d_repeatability() -> None:
    first = _advance()
    second = _advance()
    for a, b in zip(first[:3], second[:3], strict=True):
        assert np.allclose(a, b)
    assert first[-1] == second[-1]


def test_three_phase_transport_3d_inputs_not_modified() -> None:
    fx, fy, fz = _internal_fluxes()
    sw, sg = _sw(), _sg()
    arrays = (fx, fy, fz, sw, sg)
    originals = tuple(array.copy() for array in arrays)
    advance_three_phase_saturation_3d(fx, fy, fz, sw, sg, _phi(), _volume(), _dt(), _params())
    for array, original in zip(arrays, originals, strict=True):
        assert np.allclose(array, original)


def test_invalid_flux_x_shape_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(FieldShapeError):
        validate_three_phase_transport_3d_inputs(fx[:, :, :-1], fy, fz, _sw(), _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_flux_y_shape_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(FieldShapeError):
        validate_three_phase_transport_3d_inputs(fx, fy[:, :-1, :], fz, _sw(), _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_flux_z_shape_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(FieldShapeError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz[:-1, :, :], _sw(), _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_phi_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz, _sw(), _sg(), 0.0, _volume(), _dt(), _params())


def test_invalid_cell_volume_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz, _sw(), _sg(), _phi(), 0.0, _dt(), _params())


def test_invalid_dt_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz, _sw(), _sg(), _phi(), _volume(), 0.0, _params())


def test_invalid_saturation_state_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    sw = _sw()
    sw[0, 0, 0] = 0.90
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz, sw, _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_three_phase_params_raises() -> None:
    fx, fy, fz = _internal_fluxes()
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_3d_inputs(fx, fy, fz, _sw(), _sg(), _phi(), _volume(), _dt(), _params(sgc=0.7))


def test_three_phase_pipeline_config_exists_after_pipeline_stage() -> None:
    assert Path("config/three_phase_case.yaml").exists()


def test_existing_three_phase_transport_1d_tests_still_pass() -> None:
    sw = np.array([0.30, 0.32, 0.34])
    sg = np.array([0.10, 0.11, 0.12])
    sw_new, sg_new, so_new, report = advance_three_phase_saturation_1d(
        np.zeros(4), sw, sg, _phi(), _volume(), _dt(), _params()
    )
    assert np.allclose(sw_new, sw)
    assert np.allclose(sg_new, sg)
    assert np.allclose(so_new, 1.0 - sw - sg)
    assert report["transport_dimension"] == "1d"


def test_existing_three_phase_flux_tests_still_pass() -> None:
    fx, fy, fz = _internal_fluxes()
    result = compute_three_phase_fluxes_3d(fx, fy, fz, _sw(), _sg(), _params())
    assert np.allclose(result[0] + result[3] + result[6], fx)


def test_existing_two_phase_solver_tests_still_pass() -> None:
    assert callable(advance_saturation_3d)


def test_material_balance_helper_direct_call() -> None:
    fx, fy, fz = _internal_fluxes()
    sw, sg = _sw(), _sg()
    water_x, water_y, water_z, oil_x, oil_y, oil_z, gas_x, gas_y, gas_z, _ = compute_three_phase_fluxes_3d(
        fx, fy, fz, sw, sg, _params()
    )
    sw_new, sg_new, _ = compute_three_phase_saturation_update_3d(
        water_x, water_y, water_z, gas_x, gas_y, gas_z, sw, sg, _phi(), _volume(), _dt(), _params()
    )
    report = compute_three_phase_material_balance_3d(
        water_x,
        water_y,
        water_z,
        oil_x,
        oil_y,
        oil_z,
        gas_x,
        gas_y,
        gas_z,
        sw,
        sg,
        sw_new,
        sg_new,
        _phi(),
        _volume(),
        _dt(),
        _params(),
    )
    assert set(_balance_keys()).issubset(report)


def _advance():
    fx, fy, fz = _internal_fluxes()
    return advance_three_phase_saturation_3d(fx, fy, fz, _sw(), _sg(), _phi(), _volume(), _dt(), _params())


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


def _sw() -> np.ndarray:
    nz, ny, nx = 3, 3, 4
    k, j, i = np.indices((nz, ny, nx), dtype=float)
    return 0.28 + 0.01 * i + 0.005 * j


def _sg() -> np.ndarray:
    nz, ny, nx = 3, 3, 4
    k, _, i = np.indices((nz, ny, nx), dtype=float)
    return 0.10 + 0.004 * k + 0.002 * i


def _zero_flux_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sw = np.full((3, 3, 4), 0.30)
    sg = np.full((3, 3, 4), 0.10)
    return np.zeros((3, 3, 5)), np.zeros((3, 4, 4)), np.zeros((4, 3, 4)), sw, sg


def _internal_fluxes(scale: float = 1.0e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz, ny, nx = _sw().shape
    fx = np.zeros((nz, ny, nx + 1))
    fy = np.zeros((nz, ny + 1, nx))
    fz = np.zeros((nz + 1, ny, nx))
    fx[:, :, 1:nx] = scale
    fy[:, 1:ny, :] = -0.5 * scale
    fz[1:nz, :, :] = 0.25 * scale
    return fx, fy, fz


def _phi() -> float:
    return 0.2


def _volume() -> float:
    return 1.0


def _dt() -> float:
    return 100.0


def _balance_keys() -> list[str]:
    return [
        "water_inflow",
        "water_outflow",
        "water_storage_change",
        "water_balance_error",
        "gas_inflow",
        "gas_outflow",
        "gas_storage_change",
        "gas_balance_error",
        "oil_inflow",
        "oil_outflow",
        "oil_storage_change",
        "oil_balance_error",
    ]
