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
from reservoir_backend.solver.capillary_pressure import (
    brooks_corey_pc,
    build_capillary_model_from_config,
    capillary_pressure,
    capillary_pressure_derivative_numeric,
    effective_saturation_for_pc,
    no_capillary_pressure,
    validate_capillary_params,
    van_genuchten_pc,
)


def test_no_capillary_pressure_scalar() -> None:
    assert no_capillary_pressure(0.4) == pytest.approx(0.0)


def test_no_capillary_pressure_array_shape() -> None:
    sw = np.full((2, 3), 0.4)
    pc = no_capillary_pressure(sw)
    assert isinstance(pc, np.ndarray)
    assert pc.shape == sw.shape
    assert np.allclose(pc, 0.0)


def test_effective_saturation_pc_bounds() -> None:
    se = effective_saturation_for_pc(np.array([0.0, 0.5, 1.0]), 0.2, 0.2, eps=1.0e-4)
    assert np.all(se >= 1.0e-4)
    assert np.all(se <= 1.0)


def test_brooks_corey_pc_positive() -> None:
    pc = brooks_corey_pc(np.linspace(0.2, 0.8, 8), 0.2, 0.2, entry_pressure=1000.0, lambda_pc=2.0)
    assert np.all(np.asarray(pc) >= 0.0)


def test_brooks_corey_pc_monotonic_decreasing() -> None:
    pc = np.asarray(brooks_corey_pc(np.linspace(0.21, 0.8, 20), 0.2, 0.2, 1000.0, 2.0))
    assert np.all(np.diff(pc) <= 0.0)


def test_brooks_corey_pc_endpoint_high_sw() -> None:
    pc = brooks_corey_pc(0.8, 0.2, 0.2, entry_pressure=1000.0, lambda_pc=2.0)
    assert pc == pytest.approx(1000.0)


def test_van_genuchten_pc_positive() -> None:
    pc = van_genuchten_pc(np.linspace(0.2, 0.8, 8), 0.2, 0.2, p0=1000.0, m=0.5, n=2.0)
    assert np.all(np.asarray(pc) >= 0.0)


def test_van_genuchten_pc_monotonic_decreasing() -> None:
    pc = np.asarray(van_genuchten_pc(np.linspace(0.21, 0.8, 20), 0.2, 0.2, 1000.0, 0.5, 2.0))
    assert np.all(np.diff(pc) <= 0.0)


def test_capillary_pressure_field3d_input() -> None:
    grid = Grid3D(nx=4, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    sw = Field3D.from_constant(grid, 0.5, name="sw", unit="fraction")
    pc = capillary_pressure(sw, _brooks_corey_params())
    assert isinstance(pc, Field3D)
    assert pc.grid == grid
    assert pc.values.shape == grid.shape
    assert pc.name == "capillary_pressure"
    assert pc.unit == "Pa"


def test_capillary_pressure_derivative_numeric() -> None:
    sw = np.linspace(0.25, 0.75, 16)
    derivative = np.asarray(capillary_pressure_derivative_numeric(sw, _brooks_corey_params()))
    assert np.isfinite(derivative).all()
    assert np.count_nonzero(derivative <= 0.0) >= 14


def test_invalid_saturation_params_raises() -> None:
    params = _brooks_corey_params()
    params["swi"] = 0.7
    params["sor"] = 0.4
    with pytest.raises(InvalidPhysicalValueError):
        validate_capillary_params(params)


def test_invalid_brooks_corey_params_raises() -> None:
    params = _brooks_corey_params()
    params["entry_pressure_pa"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_capillary_params(params)
    params = _brooks_corey_params()
    params["lambda_pc"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_capillary_params(params)


def test_invalid_van_genuchten_params_raises() -> None:
    params = _van_genuchten_params()
    params["p0_pa"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        validate_capillary_params(params)
    for key in ["m", "n"]:
        params = _van_genuchten_params()
        params[key] = 0.0
        with pytest.raises(InvalidPhysicalValueError):
            validate_capillary_params(params)


def test_capillary_pressure_no_nan_inf() -> None:
    pc = capillary_pressure(np.linspace(0.0, 1.0, 50), _van_genuchten_params())
    assert np.isfinite(pc).all()


def test_build_capillary_model_from_config_disabled() -> None:
    config = load_case_config("config/demo_case.yaml")
    params = build_capillary_model_from_config(config)
    assert params["model"] == "none"
    assert params["enabled"] is False


def test_config_loader_accepts_capillary_section() -> None:
    config = load_case_config("config/demo_case.yaml")
    assert "capillary_pressure" in config
    assert config["capillary_pressure"]["entry_pressure_pa"] == pytest.approx(1000.0)


def test_existing_full_pipeline_unchanged_when_disabled(tmp_path) -> None:
    result = run_demo(case_id="cap_disabled", results_root=tmp_path)
    assert result["summary"]["success"] is True
    assert "capillary_pressure.npy" not in {path.name for path in result["case_dir"].iterdir()}


def _brooks_corey_params() -> dict:
    return {
        "enabled": True,
        "model": "brooks_corey",
        "swi": 0.2,
        "sor": 0.2,
        "entry_pressure_pa": 1000.0,
        "lambda_pc": 2.0,
    }


def _van_genuchten_params() -> dict:
    return {
        "enabled": True,
        "model": "van_genuchten",
        "swi": 0.2,
        "sor": 0.2,
        "p0_pa": 1000.0,
        "m": 0.5,
        "n": 2.0,
    }
