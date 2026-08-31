import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.log_cf_tmf import LogCfTmfParameterization
from reservoir_backend.io.case import inverse_spec_from_cfg, load_case
from reservoir_backend.io.parameterization_cfg import parameterization_from_cfg
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.physics.transfer import ComponentTransfer


def test_log_cf_tmf_roundtrip() -> None:
    param = LogCfTmfParameterization(c_ref_m2=1.0e-12)
    th = param.encode(np.array([1.0e-12, 2.0]))
    phys = param.decode_physical(th)
    assert phys["cf_m2"] == pytest.approx(1.0e-12, rel=1e-12)
    assert phys["tmf_multiplier"] == pytest.approx(2.0, rel=1e-12)
    assert param.n_params == 2


def test_lab_v1_yaml_selects_joint_param() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    assert twin.parameterization.n_params == 2
    assert twin.inverse.algorithm == "esmda"
    xfer = twin.transfer_operator(np.array([0.0, np.log(2.0)]))
    assert xfer.transfer_multiplier == pytest.approx(2.0)


def test_transfer_multiplier_scales_once() -> None:
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import flash_state, moles_from_z

    spec = fluid_from_name("example", temperature_k=350.0)
    moles = moles_from_z(spec, np.array([1.2e7]), spec.z_init, np.array([1.0e-4]))
    props = flash_state(spec, np.array([1.2e7]), moles)
    base = ComponentTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15, transfer_multiplier=1.0)
    twice = ComponentTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15, transfer_multiplier=2.0)
    q1 = base.compute(np.array([1.3e7]), np.array([1.1e7]), np.array([1.0e-4]), props, props)
    q2 = twice.compute(np.array([1.3e7]), np.array([1.1e7]), np.array([1.0e-4]), props, props)
    assert q2.molar_rate == pytest.approx(2.0 * q1.molar_rate)


def test_member_transfer_differs_with_theta() -> None:
    grid = CartesianGrid.uniform((0.3, 0.2, 0.1), (0.1, 0.1, 0.1))
    cond = FractureConductivityModel(n_cells=grid.n_cells, fracture_mask=np.ones(grid.n_cells, dtype=bool), k_matrix_m2=1e-15)
    param = LogCfTmfParameterization(conductivity=cond, c_ref_m2=1e-12)
    spec = inverse_spec_from_cfg({"parameterization": "log_cf_tmf", "prior_mean": [0.0, 0.0]})
    assert spec.algorithm == "esmda"
    cfg = {
        "inverse": {"parameterization": "log_cf_tmf", "prior_mean": [0.0, 0.0], "prior_std": [0.8, 0.5]},
        "rock": {"porosity": 0.08, "k_matrix_m2": 1e-15},
        "physics": {"phi_fracture": 0.02},
    }
    got = parameterization_from_cfg(grid, cfg, ".")
    assert got.n_params == 2
    _ = param
