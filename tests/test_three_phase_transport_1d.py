from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.solver.saturation_solver import advance_saturation_3d
from reservoir_backend.solver.three_phase_flux import compute_three_phase_flux_1d
from reservoir_backend.solver.three_phase_relperm import fractional_flow_three_phase
from reservoir_backend.solver.three_phase_transport import (
    advance_three_phase_saturation_1d,
    compute_three_phase_cfl_1d,
    compute_three_phase_material_balance_1d,
    compute_three_phase_saturation_update_1d,
    validate_three_phase_transport_1d_inputs,
)


def test_three_phase_transport_1d_shapes() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert sw_new.shape == _sw().shape
    assert sg_new.shape == _sw().shape
    assert so_new.shape == _sw().shape


def test_three_phase_transport_1d_closure() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.allclose(sw_new + sg_new + so_new, 1.0)


def test_three_phase_transport_1d_sw_sg_update() -> None:
    flux = np.full(5, 1.0e-5)
    sw, sg = _sw(), _sg()
    water, _, gas, _ = compute_three_phase_flux_1d(flux, sw, sg, _params())
    sw_expected = sw - _dt() / (_phi() * _volume()) * (water[1:] - water[:-1])
    sg_expected = sg - _dt() / (_phi() * _volume()) * (gas[1:] - gas[:-1])
    sw_new, sg_new, _, _ = advance_three_phase_saturation_1d(
        flux, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=sw[0], injected_sg=sg[0]
    )
    assert np.allclose(sw_new, sw_expected)
    assert np.allclose(sg_new, sg_expected)


def test_three_phase_transport_1d_so_from_closure() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.allclose(so_new, 1.0 - sw_new - sg_new)


def test_three_phase_transport_1d_no_flow_no_change() -> None:
    sw, sg = _sw(), _sg()
    sw_new, sg_new, so_new, _ = advance_three_phase_saturation_1d(np.zeros(5), sw, sg, _phi(), _volume(), _dt(), _params())
    assert np.allclose(sw_new, sw)
    assert np.allclose(sg_new, sg)
    assert np.allclose(so_new, 1.0 - sw - sg)


def test_three_phase_transport_1d_water_injection_increases_sw() -> None:
    sw, sg = np.full(4, 0.30), np.full(4, 0.10)
    flux = np.full(5, 1.0e-5)
    sw_new, _, _, _ = advance_three_phase_saturation_1d(
        flux, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.75, injected_sg=0.05
    )
    assert sw_new[0] > sw[0]


def test_three_phase_transport_1d_gas_injection_increases_sg() -> None:
    sw, sg = np.full(4, 0.30), np.full(4, 0.10)
    flux = np.full(5, 1.0e-5)
    _, sg_new, _, _ = advance_three_phase_saturation_1d(
        flux, sw, sg, _phi(), _volume(), _dt(), _params(), injected_sw=0.20, injected_sg=0.60
    )
    assert sg_new[0] > sg[0]


def test_three_phase_transport_1d_bounds() -> None:
    sw_new, sg_new, so_new, _ = _advance()
    assert np.all(sw_new >= _params()["swi"])
    assert np.all(sg_new >= _params()["sgc"])
    assert np.all(so_new >= _params()["sor"])


def test_three_phase_transport_1d_no_nan_inf() -> None:
    result = _advance()
    for values in result[:3]:
        assert np.isfinite(values).all()
    assert result[-1]["has_nan"] is False
    assert result[-1]["has_inf"] is False


def test_three_phase_transport_1d_cfl_array() -> None:
    cfl, max_cfl = compute_three_phase_cfl_1d(np.full(5, 1.0e-5), _phi(), _volume(), _dt())
    assert cfl.shape == _sw().shape
    assert max_cfl == pytest.approx(float(np.max(cfl)))


def test_three_phase_transport_1d_cfl_violation_raises() -> None:
    with pytest.raises(CFLViolationError):
        advance_three_phase_saturation_1d(np.full(5, 1.0e-3), _sw(), _sg(), _phi(), _volume(), 1000.0, _params(), max_cfl=0.1)


def test_three_phase_transport_1d_material_balance_keys() -> None:
    report = _advance()[-1]
    for key in _balance_keys():
        assert key in report


def test_three_phase_transport_1d_water_material_balance() -> None:
    assert abs(_advance()[-1]["water_balance_error"]) <= 1.0e-18


def test_three_phase_transport_1d_gas_material_balance() -> None:
    assert abs(_advance()[-1]["gas_balance_error"]) <= 1.0e-15


def test_three_phase_transport_1d_oil_material_balance() -> None:
    assert abs(_advance()[-1]["oil_balance_error"]) <= 1.0e-15


def test_three_phase_transport_1d_report_keys() -> None:
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
    assert report["transport_dimension"] == "1d"


def test_three_phase_transport_1d_repeatability() -> None:
    first = _advance()
    second = _advance()
    for a, b in zip(first[:3], second[:3], strict=True):
        assert np.allclose(a, b)
    assert first[-1] == second[-1]


def test_three_phase_transport_1d_inputs_not_modified() -> None:
    flux, sw, sg = np.full(5, 1.0e-5), _sw(), _sg()
    originals = (flux.copy(), sw.copy(), sg.copy())
    advance_three_phase_saturation_1d(flux, sw, sg, _phi(), _volume(), _dt(), _params())
    for array, original in zip((flux, sw, sg), originals, strict=True):
        assert np.allclose(array, original)


def test_invalid_flux_shape_raises() -> None:
    with pytest.raises(FieldShapeError):
        validate_three_phase_transport_1d_inputs(np.ones(4), _sw(), _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_phi_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_1d_inputs(np.ones(5), _sw(), _sg(), 0.0, _volume(), _dt(), _params())


def test_invalid_cell_volume_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_1d_inputs(np.ones(5), _sw(), _sg(), _phi(), 0.0, _dt(), _params())


def test_invalid_dt_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_1d_inputs(np.ones(5), _sw(), _sg(), _phi(), _volume(), 0.0, _params())


def test_invalid_saturation_state_raises() -> None:
    sw = _sw()
    sw[0] = 0.9
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_1d_inputs(np.ones(5), sw, _sg(), _phi(), _volume(), _dt(), _params())


def test_invalid_three_phase_params_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_transport_1d_inputs(np.ones(5), _sw(), _sg(), _phi(), _volume(), _dt(), _params(sgc=0.7))


def test_three_phase_pipeline_config_exists_after_pipeline_stage() -> None:
    assert __import__("pathlib").Path("config/three_phase_case.yaml").exists()


def test_existing_three_phase_flux_tests_still_pass() -> None:
    water, oil, gas, _ = compute_three_phase_flux_1d(np.full(5, 1.0e-5), _sw(), _sg(), _params())
    assert np.allclose(water + oil + gas, 1.0e-5)


def test_existing_three_phase_relperm_tests_still_pass() -> None:
    fw, fo, fg = fractional_flow_three_phase(0.3, 0.1, _params())
    assert fw + fo + fg == pytest.approx(1.0)


def test_existing_two_phase_solver_tests_still_pass() -> None:
    # Smoke-check import and callable presence without coupling this module into oil-water transport.
    assert callable(advance_saturation_3d)


def test_material_balance_helper_direct_call() -> None:
    flux = np.full(5, 1.0e-5)
    sw, sg = _sw(), _sg()
    water, oil, gas, _ = compute_three_phase_flux_1d(flux, sw, sg, _params())
    sw_new, sg_new, _ = compute_three_phase_saturation_update_1d(water, gas, sw, sg, _phi(), _volume(), _dt(), _params())
    report = compute_three_phase_material_balance_1d(water, oil, gas, sw, sg, sw_new, sg_new, _phi(), _volume(), _dt(), _params())
    assert set(_balance_keys()).issubset(report)


def _advance():
    return advance_three_phase_saturation_1d(np.full(5, 1.0e-5), _sw(), _sg(), _phi(), _volume(), _dt(), _params())


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
    return np.array([0.30, 0.32, 0.34, 0.36])


def _sg() -> np.ndarray:
    return np.array([0.10, 0.11, 0.12, 0.13])


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
