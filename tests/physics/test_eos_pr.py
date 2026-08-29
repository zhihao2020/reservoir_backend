"""Peng–Robinson cubic: published C1–nC10 EXAMPLE, not a GEM card."""

import numpy as np

from reservoir_backend.eos.example import EXAMPLE_NAMES, example_c1_nc10
from reservoir_backend.eos.pr import R_GAS, pr_z_factors


def test_example_criticals_are_published_c1_nc10() -> None:
    eos = example_c1_nc10()
    assert eos.names == EXAMPLE_NAMES
    assert eos.nc == 2
    np.testing.assert_allclose(eos.tc, [190.564, 617.70])
    np.testing.assert_allclose(eos.pc, [4.5992e6, 2.103e6])
    np.testing.assert_allclose(eos.omega, [0.01142, 0.490])
    np.testing.assert_allclose(eos.kij, [[0.0, 0.049], [0.049, 0.0]])


def test_pr_z_methane_near_ideal_at_low_p() -> None:
    eos = example_c1_nc10()
    z = np.array([1.0, 0.0])
    zl, zv = eos.z_roots(1.0e5, 300.0, z)
    assert 0.99 < zv < 1.01
    assert abs(zv - zl) < 0.02


def test_pr_z_decane_liquid_at_ambient() -> None:
    eos = example_c1_nc10()
    z = np.array([0.0, 1.0])
    zl, zv = eos.z_roots(1.0e5, 300.0, z)
    assert zl < 0.05
    v = eos.molar_volume(1.0e5, 300.0, z, vapor=False)
    assert v > 0.0
    assert abs(zl * R_GAS * 300.0 / 1.0e5 - v) / v < 1.0e-12


def test_fugacity_coeff_finite_and_positive() -> None:
    eos = example_c1_nc10()
    z = np.array([0.6, 0.4])
    ln_v = eos.ln_fugacity_coeff(8.0e6, 350.0, z, vapor=True)
    ln_l = eos.ln_fugacity_coeff(8.0e6, 350.0, z, vapor=False)
    assert np.all(np.isfinite(ln_v)) and np.all(np.isfinite(ln_l))
    f = eos.fugacity(8.0e6, 350.0, z, vapor=True)
    assert np.all(f > 0.0)


def test_pr_z_factors_respect_b() -> None:
    zl, zv = pr_z_factors(0.1, 0.01)
    assert zv >= zl
    assert zl > 0.01


def test_pr_z_cardano_matches_numpy_roots() -> None:
    for A, B in ((0.1, 0.01), (0.8, 0.05), (1.5, 0.12), (0.02, 0.002)):
        zl, zv = pr_z_factors(A, B)
        coeffs = [1.0, -(1.0 - B), A - 3.0 * B * B - 2.0 * B, -(A * B - B * B - B * B * B)]
        roots = np.roots(coeffs)
        real = np.sort(np.real(roots[np.abs(np.imag(roots)) < 1.0e-8]))
        real = real[real > B + 1.0e-12]
        assert abs(zl - float(real[0])) < 1.0e-8
        assert abs(zv - float(real[-1])) < 1.0e-8
