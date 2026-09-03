"""Unit tests for local dense CellAD."""

from __future__ import annotations

import numpy as np

from reservoir_backend.solver.adnum import CellAD, clip, interp_with_slope, maximum, table_eval, where


def test_cellad_arithmetic_matches_fd() -> None:
    n = 5
    p = CellAD.independent(np.linspace(1.0e5, 2.0e5, n), 0)
    sw = CellAD.independent(np.linspace(0.2, 0.4, n), 1)
    # f = p * sw + 1/p
    f = p * sw + (1.0 / p)
    eps = 1.0e-4
    # ∂f/∂p via FD
    p_h = p.value * (1.0 + eps)
    p_l = p.value * (1.0 - eps)
    fh = p_h * sw.value + 1.0 / p_h
    fl = p_l * sw.value + 1.0 / p_l
    dfd_p = (fh - fl) / (p_h - p_l)
    assert np.allclose(f.derivs[:, 0], dfd_p, rtol=1.0e-5, atol=1.0e-8)
    # ∂f/∂sw = p
    assert np.allclose(f.derivs[:, 1], p.value, rtol=1.0e-12)


def test_where_min_max_clip() -> None:
    a = CellAD.independent(np.array([1.0, 3.0, -1.0]), 0)
    b = CellAD.constant(0.0, n=3)
    m = maximum(a, b)
    assert np.allclose(m.value, [1.0, 3.0, 0.0])
    assert np.allclose(m.derivs[:, 0], [1.0, 1.0, 0.0])
    c = clip(a, 0.0, 2.0)
    assert np.allclose(c.value, [1.0, 2.0, 0.0])


def test_table_eval_slope() -> None:
    xp = np.array([0.0, 1.0, 2.0])
    fp = np.array([0.0, 2.0, 3.0])
    x = CellAD.independent(np.array([0.5, 1.5]), 0)
    y = table_eval(x, xp, fp)
    assert np.allclose(y.value, [1.0, 2.5])
    assert np.allclose(y.derivs[:, 0], [2.0, 1.0])
    _v, s = interp_with_slope(np.array([0.5]), xp, fp)
    assert abs(float(s[0]) - 2.0) < 1.0e-12


def test_where_branches() -> None:
    a = CellAD.independent(np.array([1.0, 2.0]), 0)
    b = CellAD.independent(np.array([3.0, 4.0]), 1)
    w = where(np.array([True, False]), a, b)
    assert np.allclose(w.value, [1.0, 4.0])
    assert np.allclose(w.derivs[0], [1.0, 0.0, 0.0])
    assert np.allclose(w.derivs[1], [0.0, 1.0, 0.0])
