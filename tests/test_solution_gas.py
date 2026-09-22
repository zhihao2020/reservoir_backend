"""Solution-gas (CO2-in-oil) extension regression tests.

Pins the extended black-oil upgrade: Henry's-law Rs, the ``rs``/``co2`` output
fields, backward compatibility at ``rs_slope == 0``, and that a nonzero slope
keeps the inversion finite (the CO2-component mass balance replaces the gas
phase balance).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.core.lab_case import load_lab_case
from src.programs.pipeline import run_pipeline
from src.programs.rock import BlackOilParams, solution_gas_ratio

ROOT = Path(__file__).resolve().parents[1]
SHALE = ROOT / "example" / "case.yaml"


def _case(n_times: int = 3):
    case = load_lab_case(SHALE)
    case.times = case.times[:n_times]
    case.pressure = case.pressure[:n_times]
    case.sw = case.sw[:n_times]
    case.so = case.so[:n_times]
    case.sg = case.sg[:n_times]
    case.well_pw = case.well_pw[:n_times]
    case.well_q = case.well_q[:n_times]
    case.well_qw = case.well_qw[:n_times]
    case.well_qo = case.well_qo[:n_times]
    case.well_qg = case.well_qg[:n_times]
    return case


def test_solution_gas_ratio_linear():
    params = BlackOilParams(rs_slope=1.0e-5)
    p = np.array([0.0, 10.0e6, 20.0e6])
    rs = solution_gas_ratio(p, params)
    assert np.allclose(rs, [0.0, 100.0, 200.0])


def test_solution_gas_ratio_non_negative():
    # Negative pressure (should not occur, but the guard must not produce Rs < 0).
    rs = solution_gas_ratio(np.array([-1.0e6]), BlackOilParams(rs_slope=1.0e-5))
    assert float(rs[0]) >= 0.0


def test_zero_slope_is_backward_compatible():
    # rs_slope == 0 must reduce exactly to the dead-oil model: Rs = 0 and the
    # total CO2 component equals the free gas saturation.
    case = _case()
    case.black_oil = replace(case.black_oil, rs_slope=0.0)
    fields = run_pipeline(case)
    assert fields.rs is not None and fields.co2 is not None
    assert np.allclose(fields.rs, 0.0)
    assert np.allclose(fields.co2, fields.sg)
    assert np.isfinite(fields.k).all()
    assert np.isfinite(fields.phi).all()


def test_nonzero_slope_outputs_rs_and_co2():
    case = _case()
    case.k_homogeneous = False
    slope = 2.0e-6
    case.black_oil = replace(case.black_oil, rs_slope=slope)
    fields = run_pipeline(case)
    # Rs is derived from the reconstructed pressure field.
    assert np.allclose(fields.rs, slope * fields.p)
    # Total CO2 component = free gas + dissolved gas.
    assert np.allclose(fields.co2, fields.sg + fields.rs * fields.so)
    assert np.isfinite(fields.rs).all()
    assert np.isfinite(fields.co2).all()
    assert np.isfinite(fields.k).all()
    assert np.isfinite(fields.phi).all()
    # The CO2-component mass-balance residual must stay finite (not NaN/inf).
    assert np.isfinite(fields.diagnostics.get("mass_residual_rel", float("nan")))


def test_solution_gas_three_phase_inversion_finite():
    # Exercise the three-phase component-balance branch with solution gas and
    # joint rel-perm inversion together.
    case = _case()
    case.k_homogeneous = False
    case.black_oil = replace(case.black_oil, rs_slope=1.0e-6)
    case.relperm = ("nw", "no", "ng")
    fields = run_pipeline(case)
    assert np.isfinite(fields.k).all()
    assert np.isfinite(fields.phi).all()
    assert fields.diagnostics.get("relperm")
