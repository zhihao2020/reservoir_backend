"""Numerical-robustness regression guards.

Each test pins a precision/stability fix from the audit so it cannot silently
regress: kriging co-located-point singularity, the IDW power bound, zero-flow
``q_scale`` degeneracy, negative ``kv_kh`` rejection, and zero-field smoothing.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.core.lab_case import load_lab_case
from src.exceptions import InvalidObservation
from src.programs.interpolate import VariogramModel, inverse_distance, ordinary_kriging
from src.programs.pipeline import run_pipeline
from src.programs.saturation import smooth_fields

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "small" / "case.yaml"


def test_covariance_co_located_points_not_singular():
    # Two distinct points at the same location must yield a positive-definite
    # covariance block [[sill+n, sill],[sill, sill+n]] (min eigenvalue == nugget),
    # not the exactly-singular [[sill+n, sill+n],[sill+n, sill+n]] the old code
    # produced (nugget misapplied to off-diagonal co-located pairs).
    model = VariogramModel(kind="exponential", nugget=1.0e-6, sill=1.0, range_h=1.0, range_v=1.0)
    dh = np.zeros((2, 2))
    dv = np.zeros((2, 2))
    cov = model.covariance(dh, dv)
    assert np.linalg.eigvalsh(cov).min() > 0.0


def test_kriging_co_located_points_finite():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vals = np.array([1.0, 1.0, 3.0])
    tgt = np.array([[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
    out, var = ordinary_kriging(pts, vals, tgt)
    assert np.isfinite(out).all()
    assert np.isfinite(var).all()


def test_idw_power_upper_bound_rejected():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vals = np.array([1.0, 2.0])
    tgt = np.array([[0.5, 0.0, 0.0]])
    with pytest.raises(InvalidObservation):
        inverse_distance(pts, vals, tgt, power=100.0)


def test_zero_flow_inversion_stays_finite():
    # All well rates zero -> q_scale would collapse to ~1e-18 and blow the
    # mass-balance residual up into noise-fit territory. It must instead fall
    # back to a unit scale and still return finite k/phi.
    case = load_lab_case(SMALL)
    case.well_qw = np.zeros_like(case.well_qw)
    case.well_qo = np.zeros_like(case.well_qo)
    case.well_qg = np.zeros_like(case.well_qg)
    fields = run_pipeline(case)
    assert np.isfinite(fields.k).all()
    assert np.isfinite(fields.phi).all()


def test_negative_kv_kh_flagged():
    case = load_lab_case(SMALL)
    case.well = replace(case.well, kv_kh=-1.0)
    assert any("kv_kh" in issue for issue in case.static_issues())


def test_relperm_inversion_finite():
    # Exercise the rel-perm (n_rel>0) branch: central-difference Jacobian and
    # rel-perm parameter bounds. The small case leaves relperm empty by default,
    # so this path is otherwise untested.
    case = load_lab_case(SMALL)
    case.relperm = ("nw", "no", "ng")
    fields = run_pipeline(case)
    assert np.isfinite(fields.k).all()
    assert np.isfinite(fields.phi).all()
    relperm = fields.diagnostics.get("relperm", {})
    assert relperm  # branch actually ran
    assert all(np.isfinite(float(v)) for v in relperm.values())


def test_smooth_fields_zero_predecessor_smoothed():
    # A zero predecessor must not be treated as a jump: the first non-zero step
    # should still be smoothed (the old +1e-18 floor classified it as a jump).
    arr = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = smooth_fields(arr)
    assert np.all(out[1] < arr[1])
