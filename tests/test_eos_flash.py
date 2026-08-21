"""Two-phase TP-flash tests. Standalone; does not touch FIM / black-oil PVT."""

import numpy as np

from reservoir_backend.eos import (
    EXAMPLE_LIBRARY_MARKER,
    example_eight_component_mixture,
    example_feed_z,
    flash_tp,
    fugacity_coefficients,
    solve_rachford_rice,
    wilson_k,
)


def test_binary_co2_c1_two_phase_fugacity_and_balance() -> None:
    """CO2–C1 VLE at 250 K, 5 MPa, z_CO2=0.60.

    Interior point of the published CH4–CO2 envelope on the 250.00 K
    isotherm (Davalos et al., J. Chem. Eng. Data 1976; Donnelly & Katz,
    Ind. Eng. Chem. 1954). A 40 mol% CO2 feed at this (T, p) sits on the
    vapor side of the dew curve. Not a GEM / Jiyang match.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    result = flash_tp(z, T, p, mix)
    assert result.converged
    assert result.phase_state == "two-phase"
    assert 0.0 < result.V < 1.0
    assert np.isclose(result.x.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.y.sum(), 1.0, atol=1e-12)
    assert np.allclose(result.z, (1.0 - result.V) * result.x + result.V * result.y, atol=1e-10)
    ln_f_l = np.log(fugacity_coefficients(result.x, T, p, mix, phase="liquid")) + np.log(result.x)
    ln_f_v = np.log(fugacity_coefficients(result.y, T, p, mix, phase="vapor")) + np.log(result.y)
    assert np.max(np.abs(ln_f_l - ln_f_v)) < 1.0e-6
    # Methane prefers the vapor; CO2 prefers the liquid on this isotherm.
    assert result.y[0] > result.x[0]
    assert result.x[1] > result.y[1]


def test_eight_component_example_oil_plus_co2_flash() -> None:
    """EXAMPLE 8-component oil + CO2 feed. Not a Jiyang / GEM composition."""
    mix = example_eight_component_mixture()
    z = example_feed_z()
    assert z.size == 8
    assert np.isclose(z.sum(), 1.0)
    result = flash_tp(z, 350.0, 8.0e6, mix)
    assert result.converged
    assert result.phase_state == "two-phase"
    assert 0.0 < result.V < 1.0
    assert np.isclose(result.x.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.y.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.z.sum(), 1.0, atol=1e-12)
    reconstructed = (1.0 - result.V) * result.x + result.V * result.y
    assert np.allclose(result.z, reconstructed, atol=1e-9)
    assert np.allclose(z, reconstructed, atol=1e-9)


def test_single_phase_almost_pure_c1_high_t() -> None:
    """Almost-pure C1 at 400 K, 5 MPa: single-phase vapor, no crash."""
    mix = example_eight_component_mixture()
    z = np.full(mix.n_components, 1.0e-4)
    z[mix.names.index("C1")] = 1.0 - 7.0e-4
    z = z / z.sum()
    result = flash_tp(z, 400.0, 5.0e6, mix)
    assert result.converged
    assert result.phase_state in ("vapor", "liquid")
    assert result.phase_state != "two-phase"
    assert result.V in (0.0, 1.0)
    assert result.V >= 0.0 and result.V <= 1.0
    assert np.allclose(result.z, (1.0 - result.V) * result.x + result.V * result.y, atol=1e-12)


def test_negative_flash_not_returned_as_two_phase() -> None:
    """RR roots outside (0, 1) are single-phase; V is never <0 or >1."""
    mix = example_eight_component_mixture().subset(["C1", "example_C7plus"])
    z = np.array([0.20, 0.80])
    # All K ≪ 1 → unbounded RR would go negative; public solver returns liquid.
    V, state = solve_rachford_rice(z, np.array([0.05, 0.02]))
    assert state == "liquid"
    assert V == 0.0
    # All K ≫ 1 → unbounded RR would exceed 1; public solver returns vapor.
    V, state = solve_rachford_rice(z, np.array([20.0, 8.0]))
    assert state == "vapor"
    assert V == 1.0

    # Heavy, cold, high-p feed: flash must not emit V ∉ [0, 1] as two-phase.
    heavy = example_eight_component_mixture()
    z_heavy = np.zeros(heavy.n_components)
    z_heavy[heavy.names.index("example_C7plus")] = 0.92
    z_heavy[heavy.names.index("nC6")] = 0.08
    result = flash_tp(z_heavy, 280.0, 25.0e6, heavy)
    assert 0.0 <= result.V <= 1.0
    if result.V == 0.0 or result.V == 1.0:
        assert result.phase_state != "two-phase"
    if result.phase_state == "two-phase":
        assert 0.0 < result.V < 1.0


def test_two_phase_equilibrium_fugacity_balance_and_k() -> None:
    """Converged two-phase VLE for the EXAMPLE CO2–C1 pair.

    Same documented interior point as the binary flash test (250 K, 5 MPa,
    z_CO2=0.60; Davalos et al. 1976; Donnelly & Katz 1954). Checks fugacity
    equality, material balance, 0<V<1, mole-fraction sums, and K_i = y_i/x_i.
    EXAMPLE public literature; not a GEM / Jiyang match.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert "EXAMPLE" in mix.marker
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    result = flash_tp(z, T, p, mix)
    assert result.converged
    assert result.phase_state == "two-phase"
    assert 0.0 < result.V < 1.0
    assert np.isclose(result.x.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.y.sum(), 1.0, atol=1e-12)
    assert np.allclose(result.material_balance_residual(), 0.0, atol=1e-10)
    assert np.allclose(result.z, (1.0 - result.V) * result.x + result.V * result.y, atol=1e-10)
    phi_l = fugacity_coefficients(result.x, T, p, mix, phase="liquid")
    phi_v = fugacity_coefficients(result.y, T, p, mix, phase="vapor")
    ln_f_gap = np.log(phi_l * result.x) - np.log(phi_v * result.y)
    assert np.max(np.abs(ln_f_gap)) < 1.0e-6
    assert np.allclose(result.K, result.y / result.x, rtol=1e-10, atol=1e-12)
    assert np.all(result.K > 0.0)


def test_example_co2_nc10_liquid_mole_fraction_order() -> None:
    """EXAMPLE CO2–nC10 liquid CO2 mole fraction, published order of magnitude.

    Binary flash of library CO2 + example_C7plus (published n-decane criticals)
    at T=344.15 K, p=8.0 MPa, z_CO2=0.70 — interior to the 344.3 K isotherm.

    Open literature (not GEM, not Jiyang / 济阳, not site-calibrated):
    Reamer & Sage, J. Chem. Eng. Data 8, 508 (1963); Nagarajan & Robinson,
    J. Chem. Eng. Data 31, 168 (1986). On the ~344 K isotherm between about
    6–10 MPa, measured liquid x_CO2 is typically several tenths (order
    0.4–0.8). This test only asserts the loose band 0.2–0.9 (factor-of-few /
    order of magnitude), not a tight regression.

    Standalone example flash, not wired, example PR params.
    """
    mix = example_eight_component_mixture().subset(["CO2", "example_C7plus"])
    assert "EXAMPLE" in mix.marker
    assert "NOT a Jiyang GEM card" in mix.marker
    T, p = 344.15, 8.0e6
    z = np.array([0.70, 0.30])
    result = flash_tp(z, T, p, mix)
    assert result.converged
    assert result.phase_state == "two-phase"
    assert 0.0 < result.V < 1.0
    i_co2 = mix.names.index("CO2")
    x_co2 = float(result.x[i_co2])
    # Loose published-order band; not a field PVT or GEM-card match.
    assert 0.2 < x_co2 < 0.9
    assert np.isclose(result.x.sum(), 1.0, atol=1e-12)
    assert np.isclose(result.y.sum(), 1.0, atol=1e-12)
    assert np.allclose(result.z, (1.0 - result.V) * result.x + result.V * result.y, atol=1e-10)


def test_wilson_k_and_example_marker_on_flash_mixture() -> None:
    mix = example_eight_component_mixture()
    assert "EXAMPLE" in mix.marker
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    K = wilson_k(mix, 350.0, 8.0e6)
    assert K.shape == (8,)
    assert np.all(K > 0.0)
    # Light ends more volatile than the C7+ example pseudo.
    assert K[mix.names.index("C1")] > K[mix.names.index("example_C7plus")]
