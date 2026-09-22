"""Online session: bounded sliding window and non-blocking background inversion."""

import time
from pathlib import Path

import numpy as np

from src.core.lab_case import load_lab_case
from src.programs.session import InversionSession

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "small" / "case.yaml"
SHALE_OIL = ROOT / "examples" / "shale_oil" / "case.yaml"


def _empty_case(path):
    """Load the case but with empty observation/well-series history."""
    case = load_lab_case(path)
    n_p, n_w = len(case.probes), len(case.wells)
    case.times = np.zeros(0, dtype=float)
    case.pressure = np.zeros((0, n_p))
    case.sw = np.zeros((0, n_p))
    case.so = np.zeros((0, n_p))
    case.sg = np.zeros((0, n_p))
    case.well_pw = np.zeros((0, n_w))
    case.well_q = np.zeros((0, n_w))
    case.well_qw = np.zeros((0, n_w))
    case.well_qo = np.zeros((0, n_w))
    case.well_qg = np.zeros((0, n_w))
    return case


def _step(sess, full, t):
    return sess.step(
        float(full.times[t]),
        full.pressure[t],
        full.sw[t],
        full.so[t],
        full.sg[t],
        full.well_pw[t],
        full.well_q[t],
        full.well_qw[t],
        full.well_qo[t],
        full.well_qg[t],
    )


def test_window_is_bounded():
    empty = _empty_case(SMALL)
    full = load_lab_case(SMALL)
    sess = InversionSession.open(empty, window=2)

    fields = None
    for t in range(3):
        fields = _step(sess, full, t)

    n_cells = empty.nx * empty.ny * empty.nz
    assert empty.times.size == 2  # trimmed to window
    assert fields.times.size == 2
    assert fields.p.shape == (2, n_cells)
    assert fields.sw.shape == (2, n_cells)
    # the two retained times are the last two
    assert fields.times.tolist() == full.times[-2:].tolist()


def test_background_inversion_populates_k_phi():
    empty = _empty_case(SMALL)
    full = load_lab_case(SMALL)
    sess = InversionSession.open(empty, window=3)

    for t in range(3):
        _step(sess, full, t)

    deadline = time.monotonic() + 20.0
    while sess._phi is None and time.monotonic() < deadline:
        time.sleep(0.1)

    n_cells = empty.nx * empty.ny * empty.nz
    assert sess._phi is not None, "background inversion did not finish"
    assert sess._phi.shape == (n_cells,)
    assert sess._k.shape == (n_cells,)
    assert np.isfinite(sess._phi).all()
    assert np.isfinite(sess._k).all()
    # a later snapshot reflects the inverted k/phi (not just the prior)
    fields = sess.snapshot_fields()
    assert fields.phi.shape == (3, n_cells)
    assert np.allclose(fields.phi[0], sess._phi)


def test_homogeneous_rock_skips_inversion():
    """``k_homogeneous`` in the online session short-circuits the ill-posed k
    inversion to constant k0/phi0, matching the offline ``pipeline`` path."""
    empty = _empty_case(SHALE_OIL)
    full = load_lab_case(SHALE_OIL)
    assert empty.k_homogeneous
    sess = InversionSession.open(empty, window=2)

    for t in range(3):
        _step(sess, full, t)

    deadline = time.monotonic() + 20.0
    while sess._phi is None and time.monotonic() < deadline:
        time.sleep(0.1)

    n_cells = empty.nx * empty.ny * empty.nz
    assert sess._phi is not None, "background inversion did not finish"
    assert sess._k is not None
    # homogeneous rock -> constant prior values, no spatial variation
    assert np.allclose(sess._phi, empty.phi0)
    assert np.allclose(sess._k, empty.k0)
    assert sess._phi.shape == (n_cells,)
    assert sess._k.shape == (n_cells,)
    assert sess._diag.get("converged") is True
