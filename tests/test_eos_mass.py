"""PR density and standalone component-mass inventory tests. Not a FIM term."""

import numpy as np

from reservoir_backend.eos import (
    EXAMPLE_LIBRARY_MARKER,
    component_inventory,
    example_eight_component_mixture,
    flash_tp,
    mass_density,
    molar_volume,
)


def test_example_mw_are_public_and_labeled() -> None:
    mix = example_eight_component_mixture()
    assert mix.Mw is not None
    assert mix.Mw.size == 8
    assert np.all(mix.Mw > 0.0)
    assert "EXAMPLE" in mix.marker
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    # NIST / IUPAC methane and CO2, stored as kg/mol.
    assert abs(mix.Mw[mix.names.index("C1")] - 0.0160425) < 1e-8
    assert abs(mix.Mw[mix.names.index("CO2")] - 0.0440095) < 1e-8
    assert abs(mix.Mw[mix.names.index("example_C7plus")] - 0.1422817) < 1e-8


def test_two_phase_liquid_denser_than_vapor() -> None:
    """PR ``ρ = M / (ZRT/p)``: liquid denser than vapor at EXAMPLE CO2–C1 VLE."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    result = flash_tp(z, T, p, mix)
    assert result.phase_state == "two-phase"
    rho_l = mass_density(result.x, T, p, mix, phase="liquid")
    rho_v = mass_density(result.y, T, p, mix, phase="vapor")
    assert np.isfinite(rho_l) and np.isfinite(rho_v)
    assert rho_l > 0.0 and rho_v > 0.0
    assert rho_l > rho_v
    assert result.rho_liquid is not None and result.rho_vapor is not None
    assert result.rho_liquid > result.rho_vapor > 0.0
    v_l = molar_volume(result.x, T, p, mix, phase="liquid")
    v_v = molar_volume(result.y, T, p, mix, phase="vapor")
    assert v_l > 0.0 and v_v > 0.0
    assert result.v_liquid is not None and result.Z_liquid is not None
    assert np.isclose(result.v_liquid, result.Z_liquid * 8.314462618 * T / p, rtol=1e-12)


def test_component_inventory_mole_and_mass_balance() -> None:
    """1 mol and 1 kg bases: phase component moles sum to the feed; masses ≥ 0."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert "EXAMPLE" in mix.marker
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    for basis in ("mol", "kg"):
        inv = component_inventory(z, T, p, mix, basis=basis)
        assert inv.basis == basis
        assert inv.phase_state == "two-phase"
        assert np.allclose(inv.n_liquid + inv.n_vapor, inv.n_feed, atol=1e-12)
        assert np.allclose(inv.mole_balance_residual(), 0.0, atol=1e-12)
        assert np.all(inv.n_liquid >= -1e-15)
        assert np.all(inv.n_vapor >= -1e-15)
        assert inv.mass_liquid >= 0.0
        assert inv.mass_vapor >= 0.0
        assert np.isclose(inv.mass_liquid + inv.mass_vapor, inv.feed_mass, atol=1e-12)
        if basis == "mol":
            assert np.isclose(inv.feed_moles, 1.0)
        else:
            assert np.isclose(inv.feed_mass, 1.0)
