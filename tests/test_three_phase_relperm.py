from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.relperm import fractional_flow_water
from reservoir_backend.solver.three_phase_relperm import (
    build_three_phase_relperm_report,
    clip_three_phase_saturations,
    compute_oil_saturation,
    corey_three_phase_relative_permeability,
    effective_saturations_three_phase,
    fractional_flow_three_phase,
    three_phase_mobility,
    validate_three_phase_params,
    validate_three_phase_saturations,
)


def test_compute_oil_saturation_scalar() -> None:
    assert compute_oil_saturation(0.3, 0.1) == pytest.approx(0.6)


def test_compute_oil_saturation_array_shape() -> None:
    sw = np.array([0.2, 0.3, 0.4])
    sg = np.array([0.05, 0.1, 0.15])
    so = compute_oil_saturation(sw, sg)
    assert isinstance(so, np.ndarray)
    assert so.shape == sw.shape


def test_three_phase_closure() -> None:
    sw, sg = _sample_saturations()
    so = compute_oil_saturation(sw, sg)
    assert np.allclose(sw + so + sg, 1.0)


def test_validate_three_phase_params_valid() -> None:
    validate_three_phase_params(_params())


def test_validate_three_phase_params_invalid_residual_sum() -> None:
    params = _params(swi=0.4, sor=0.4, sgc=0.2)
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_params(params)


def test_validate_three_phase_saturations_valid() -> None:
    validate_three_phase_saturations(np.array([0.25, 0.35]), np.array([0.1, 0.15]), _params())


def test_validate_three_phase_saturations_negative_oil_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_three_phase_saturations(0.7, 0.2, _params())


def test_effective_saturations_bounds() -> None:
    sew, seo, seg = effective_saturations_three_phase(np.array([0.0, 0.4, 1.0]), np.array([0.0, 0.1, 1.0]), _params())
    for values in [sew, seo, seg]:
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_corey_three_phase_relperm_shapes() -> None:
    sw, sg = _sample_saturations()
    krw, kro, krg = corey_three_phase_relative_permeability(sw, sg, _params())
    assert krw.shape == sw.shape
    assert kro.shape == sw.shape
    assert krg.shape == sw.shape


def test_corey_three_phase_relperm_nonnegative() -> None:
    krw, kro, krg = corey_three_phase_relative_permeability(*_sample_saturations(), _params())
    assert np.all(krw >= 0.0)
    assert np.all(kro >= 0.0)
    assert np.all(krg >= 0.0)


def test_corey_three_phase_relperm_endpoints() -> None:
    params = _params()
    krw, _, _ = corey_three_phase_relative_permeability(0.2, 0.05, params)
    assert krw == pytest.approx(0.0)
    _, _, krg = corey_three_phase_relative_permeability(0.2, 0.05, params)
    assert krg == pytest.approx(0.0)
    _, kro, _ = corey_three_phase_relative_permeability(0.6, 0.2, params)
    assert kro == pytest.approx(0.0)


def test_three_phase_mobility_shapes() -> None:
    sw, sg = _sample_saturations()
    lambda_w, lambda_o, lambda_g, lambda_t = three_phase_mobility(sw, sg, _params())
    assert lambda_w.shape == sw.shape
    assert lambda_o.shape == sw.shape
    assert lambda_g.shape == sw.shape
    assert lambda_t.shape == sw.shape


def test_three_phase_mobility_nonnegative() -> None:
    for values in three_phase_mobility(*_sample_saturations(), _params()):
        assert np.all(values >= 0.0)


def test_fractional_flow_shapes() -> None:
    sw, sg = _sample_saturations()
    fw, fo, fg = fractional_flow_three_phase(sw, sg, _params())
    assert fw.shape == sw.shape
    assert fo.shape == sw.shape
    assert fg.shape == sw.shape


def test_fractional_flow_sum_to_one() -> None:
    fw, fo, fg = fractional_flow_three_phase(*_sample_saturations(), _params())
    assert np.allclose(fw + fo + fg, 1.0)


def test_fractional_flow_no_nan_inf() -> None:
    flows = fractional_flow_three_phase(*_sample_saturations(), _params())
    for values in flows:
        assert np.isfinite(values).all()


def test_invalid_viscosity_raises() -> None:
    for key in ["mu_w", "mu_o", "mu_g"]:
        params = _params(**{key: 0.0})
        with pytest.raises(InvalidPhysicalValueError):
            fractional_flow_three_phase(0.3, 0.1, params)


def test_clip_three_phase_saturations() -> None:
    sw, sg = clip_three_phase_saturations(np.array([0.1, 0.9]), np.array([0.0, 0.9]), _params())
    validate_three_phase_saturations(sw, sg, _params())
    so = compute_oil_saturation(sw, sg)
    assert np.allclose(sw + so + sg, 1.0)


def test_field3d_input_if_supported() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    sw = Field3D(grid, np.array([[[0.25, 0.35]]]), name="sw", unit="fraction")
    sg = Field3D(grid, np.array([[[0.10, 0.15]]]), name="sg", unit="fraction")
    fw, fo, fg = fractional_flow_three_phase(sw, sg, _params())
    assert isinstance(fw, np.ndarray)
    assert fw.shape == grid.shape
    assert fo.shape == grid.shape
    assert fg.shape == grid.shape


def test_report_keys() -> None:
    report = build_three_phase_relperm_report(*_sample_saturations(), _params())
    keys = {
        "sw_min",
        "sw_max",
        "sg_min",
        "sg_max",
        "so_min",
        "so_max",
        "krw_min",
        "krw_max",
        "kro_min",
        "kro_max",
        "krg_min",
        "krg_max",
        "fw_min",
        "fw_max",
        "fo_min",
        "fo_max",
        "fg_min",
        "fg_max",
        "has_nan",
        "has_inf",
        "closure_error_max",
    }
    assert keys.issubset(report)


def test_report_closure_error_small() -> None:
    report = build_three_phase_relperm_report(*_sample_saturations(), _params())
    assert report["closure_error_max"] <= 1.0e-14


def test_repeatability() -> None:
    first = fractional_flow_three_phase(*_sample_saturations(), _params())
    second = fractional_flow_three_phase(*_sample_saturations(), _params())
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a, b)


def test_three_phase_design_doc_still_exists() -> None:
    assert __import__("pathlib").Path("specs/12_three_phase_flow_design.md").exists()


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


def _sample_saturations() -> tuple[np.ndarray, np.ndarray]:
    sw = np.array([0.25, 0.35, 0.45])
    sg = np.array([0.10, 0.15, 0.20])
    return sw, sg
