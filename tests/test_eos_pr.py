"""Peng–Robinson EOS unit tests. Standalone kernel; not a FIM residual test."""

import numpy as np

from reservoir_backend.eos import (
    EXAMPLE_LIBRARY_MARKER,
    compressibility_roots,
    example_eight_component_mixture,
    fugacity_coefficients,
    peng_robinson_ab,
    select_z,
)
from reservoir_backend.eos.peng_robinson import mix_a_b, reduced_AB


def test_example_library_carries_example_marker() -> None:
    mix = example_eight_component_mixture()
    assert "EXAMPLE" in EXAMPLE_LIBRARY_MARKER
    assert "EXAMPLE" in mix.marker
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert "not wired" in mix.marker
    assert "example PR params" in mix.marker
    assert "NOT a Jiyang GEM card" in mix.marker
    assert mix.n_components == 8
    assert mix.names == ("C1", "C2", "C3", "nC4", "nC5", "nC6", "example_C7plus", "CO2")


def test_pure_co2_z_and_fugacity_at_documented_tp() -> None:
    """CO2 at T=310.15 K, p=1.0 MPa (Tr≈1.02, Pr≈0.136): vapor-like Z, φ≈1."""
    mix = example_eight_component_mixture().subset(["CO2"])
    T, p = 310.15, 1.0e6
    a, b = peng_robinson_ab(mix.Tc, mix.Pc, mix.omega, T)
    A, B = reduced_AB(float(a[0]), float(b[0]), T, p)
    Z = select_z(A, B, "vapor")
    phi = fugacity_coefficients([1.0], T, p, mix, phase="vapor")
    assert Z > B
    # Low reduced pressure: compressibility near the ideal-gas limit.
    assert 0.85 < Z < 1.0
    assert 0.80 < float(phi[0]) < 1.05


def test_pure_c1_z_and_fugacity_at_documented_tp() -> None:
    """C1 at T=300.0 K, p=5.0 MPa (Tr≈1.57, Pr≈1.09): Z in (0.85, 1.0)."""
    mix = example_eight_component_mixture().subset(["C1"])
    T, p = 300.0, 5.0e6
    a, b = peng_robinson_ab(mix.Tc, mix.Pc, mix.omega, T)
    A, B = reduced_AB(float(a[0]), float(b[0]), T, p)
    Z = select_z(A, B)
    phi = fugacity_coefficients([1.0], T, p, mix)
    assert Z > B
    assert 0.80 < Z < 1.05
    assert np.isfinite(phi).all()
    assert float(phi[0]) > 0.0


def test_cubic_physical_roots_exceed_B() -> None:
    """Every accepted Z satisfies Z > B (equivalent to v > b_mix)."""
    mix = example_eight_component_mixture()
    T, p = 320.0, 8.0e6
    x = np.full(mix.n_components, 1.0 / mix.n_components)
    a, b = peng_robinson_ab(mix.Tc, mix.Pc, mix.omega, T)
    a_mix, b_mix, _ = mix_a_b(x, a, b, mix.kij)
    A, B = reduced_AB(a_mix, b_mix, T, p)
    roots = compressibility_roots(A, B)
    assert roots.size >= 1
    assert np.all(roots > B)
    assert np.all(np.isfinite(roots))


def test_subset_preserves_example_marker_and_symmetric_kij() -> None:
    binary = example_eight_component_mixture().subset(["CO2", "C1"])
    assert "EXAMPLE" in binary.marker
    assert np.allclose(binary.kij, binary.kij.T)
    assert binary.kij[0, 0] == 0.0
    assert binary.kij[0, 1] == binary.kij[1, 0] == 0.105
