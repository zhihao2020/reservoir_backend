"""FastPR vs reference near the envelope, extremes, and DPDP-sampled states."""

import numpy as np
import pytest

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.eos.flash_backend import FastPRBackend, ReferencePRBackend

pytestmark = pytest.mark.dpdp


def _states():
    eos = example_c1_nc10()
    t = 350.0
    rows = []
    for p in (1.0e6, 3.0e6, 8.0e6, 1.5e7, 3.0e7, 4.5e7):
        for z1 in (0.02, 0.08, 0.30, 0.55, 0.80, 0.95, 0.99):
            rows.append((p, np.array([z1, 1.0 - z1])))
    for p in (5.0e6, 2.0e7):
        for t2 in (280.0, 350.0, 420.0):
            rows.append((p, np.array([0.60, 0.40]), t2))
    return eos, t, rows


def test_fastpr_phase_class_and_properties_on_envelope() -> None:
    eos, t0, rows = _states()
    fast = FastPRBackend()
    n_phase_err = 0
    max_rel_v = 0.0
    max_xy = 0.0
    for row in rows:
        p, z = row[0], row[1]
        t = float(row[2]) if len(row) > 2 else t0
        ref = flash_tp(eos, p, t, z)
        got = fast.flash_tp(eos, p, t, z)
        if bool(ref.two_phase) != bool(got.two_phase):
            n_phase_err += 1
            continue
        denom = max(abs(ref.v_mix), 1.0e-12)
        max_rel_v = max(max_rel_v, abs(got.v_mix - ref.v_mix) / denom)
        max_xy = max(max_xy, float(np.max(np.abs(got.x - ref.x))), float(np.max(np.abs(got.y - ref.y))))
    assert n_phase_err == 0
    assert max_rel_v < 1.0e-6
    assert max_xy < 1.0e-6


def test_fastpr_matches_reference_backend_batch() -> None:
    eos = example_c1_nc10()
    p = np.array([1.2e6, 8.0e6, 2.8e7])
    z = np.array([[0.05, 0.95], [0.55, 0.45], [0.97, 0.03]])
    a = ReferencePRBackend().evaluate_batch(eos, p, 350.0, z)
    b = FastPRBackend().evaluate_batch(eos, p, 350.0, z)
    assert np.array_equal(a.two_phase, b.two_phase)
    np.testing.assert_allclose(a.v_mix, b.v_mix, rtol=1.0e-6, atol=1.0e-14)
    np.testing.assert_allclose(a.x, b.x, rtol=1.0e-6, atol=1.0e-12)
