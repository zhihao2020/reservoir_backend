"""Michelsen TPD stability tests. Standalone; not a FIM residual test."""

import numpy as np

from reservoir_backend.eos import (
    example_eight_component_mixture,
    flash_tp,
    michelsen_stability,
)


def test_two_phase_co2_c1_is_tpd_unstable() -> None:
    """EXAMPLE CO2–C1 at 250 K, 5 MPa, z_CO2=0.60 is TPD-unstable.

    Same documented interior point as the binary flash tests (Davalos 1976;
    Donnelly & Katz 1954). Single-phase feed has TPD < 0; flash stays
    two-phase. EXAMPLE, not Jiyang / GEM.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert "EXAMPLE" in mix.marker
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    stab = michelsen_stability(z, T, p, mix)
    assert not stab.stable
    assert stab.tpd_min < 0.0
    result = flash_tp(z, T, p, mix)
    assert result.converged
    assert result.phase_state == "two-phase"
    assert 0.0 < result.V < 1.0
    assert result.tpd_min is not None and result.tpd_min < 0.0


def test_almost_pure_c1_is_tpd_stable() -> None:
    """Almost-pure C1 at 400 K, 5 MPa is TPD-stable and stays single-phase."""
    mix = example_eight_component_mixture()
    z = np.full(mix.n_components, 1.0e-4)
    z[mix.names.index("C1")] = 1.0 - 7.0e-4
    z = z / z.sum()
    T, p = 400.0, 5.0e6
    stab = michelsen_stability(z, T, p, mix)
    result = flash_tp(z, T, p, mix)
    assert stab.stable
    assert stab.tpd_min >= 0.0
    assert result.phase_state != "two-phase"
    assert result.V in (0.0, 1.0)
    assert result.tpd_min is not None and result.tpd_min >= 0.0
