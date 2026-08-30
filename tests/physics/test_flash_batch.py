"""Vectorized PR kernels vs scalar truth."""

import numpy as np
import pytest

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp, rachford_rice
from reservoir_backend.eos.flash_batch import flash_batch, pr_z_factors_batch, wilson_k_batch
from reservoir_backend.eos.pr import pr_z_factors
from reservoir_backend.eos.stability import wilson_k


def test_pr_z_factors_batch_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    A = rng.uniform(0.01, 2.0, size=80)
    B = rng.uniform(0.001, 0.4, size=80)
    zl, zv = pr_z_factors_batch(A, B)
    for i in range(A.size):
        a, b = pr_z_factors(float(A[i]), float(B[i]))
        assert zl[i] == pytest.approx(a, rel=1.0e-10, abs=1.0e-12)
        assert zv[i] == pytest.approx(b, rel=1.0e-10, abs=1.0e-12)


def test_wilson_k_batch_matches_scalar() -> None:
    eos = example_c1_nc10()
    p = np.array([5.0e6, 1.2e7, 2.0e7])
    kb = wilson_k_batch(eos, p, 350.0)
    for i, pi in enumerate(p):
        ks = wilson_k(eos, float(pi), 350.0)
        np.testing.assert_allclose(kb[i], ks, rtol=1.0e-12)


def test_flash_batch_matches_scalar_random() -> None:
    eos = example_c1_nc10()
    rng = np.random.default_rng(4)
    p = rng.uniform(3.0e6, 2.5e7, size=64)
    z1 = rng.uniform(0.1, 0.9, size=64)
    z = np.stack((z1, 1.0 - z1), axis=1)
    arr = flash_batch(eos, p, 350.0, z)
    for i in range(p.size):
        fl = flash_tp(eos, float(p[i]), 350.0, z[i])
        assert bool(arr.two_phase[i]) == bool(fl.two_phase)
        assert arr.vapor_frac[i] == pytest.approx(fl.vapor_frac, rel=1.0e-7, abs=1.0e-8)
        np.testing.assert_allclose(arr.x[i], fl.x, rtol=1.0e-7, atol=1.0e-8)
        np.testing.assert_allclose(arr.v_mix[i], fl.v_mix, rtol=1.0e-7, atol=1.0e-14)


def test_rachford_rice_batch_binary() -> None:
    from reservoir_backend.eos.flash_batch import rachford_rice_batch

    k = np.array([[3.0, 0.2], [1.8, 0.4]])
    z = np.array([[0.6, 0.4], [0.3, 0.7]])
    vb = rachford_rice_batch(k, z)
    for i in range(2):
        assert vb[i] == pytest.approx(rachford_rice(k[i], z[i]), rel=1.0e-12)
