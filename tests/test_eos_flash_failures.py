"""Flash failure paths: negative RR and failed-SSI fallback. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.eos import (
    EXAMPLE_LIBRARY_MARKER,
    example_eight_component_mixture,
    flash_tp,
    michelsen_stability,
    solve_rachford_rice,
)


def _assert_finite_single_phase(result) -> None:
    assert result.phase_state in ("liquid", "vapor")
    assert result.phase_state != "two-phase"
    assert result.V in (0.0, 1.0)
    assert np.all(np.isfinite(result.z))
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.y))
    assert np.all(np.isfinite(result.K))
    assert np.isfinite(result.V)
    assert np.allclose(result.x, result.z)
    assert np.allclose(result.y, result.z)


def test_negative_flash_mixed_k_is_single_phase() -> None:
    """RR with mixed K that has no root in (0, 1) is liquid or vapor, not two-phase."""
    # f(0) < 0: would be a negative-flash V < 0.
    V, state = solve_rachford_rice(np.array([0.20, 0.80]), np.array([1.2, 0.5]))
    assert state == "liquid"
    assert V == 0.0
    # f(1) > 0: would be a negative-flash V > 1.
    V, state = solve_rachford_rice(np.array([0.80, 0.20]), np.array([10.0, 0.8]))
    assert state == "vapor"
    assert V == 1.0


def test_heavy_example_feed_negative_flash_via_flash_tp() -> None:
    """Cold heavy EXAMPLE C7+ / nC6 at 280 K, 25 MPa: single-phase, finite, no NaN.

    Documented EXAMPLE library species (published nC10 as example_C7plus).
    Not a GEM / Jiyang card.
    """
    mix = example_eight_component_mixture()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    z = np.zeros(mix.n_components)
    z[mix.names.index("example_C7plus")] = 0.92
    z[mix.names.index("nC6")] = 0.08
    result = flash_tp(z, 280.0, 25.0e6, mix)
    _assert_finite_single_phase(result)
    assert result.converged
    assert 0.0 <= result.V <= 1.0


def test_tpd_stable_feed_is_single_phase_fallback() -> None:
    """Almost-pure C1 at 400 K, 5 MPa: TPD-stable → single-phase, converged=True."""
    mix = example_eight_component_mixture()
    z = np.full(mix.n_components, 1.0e-4)
    z[mix.names.index("C1")] = 1.0 - 7.0e-4
    z = z / z.sum()
    T, p = 400.0, 5.0e6
    stab = michelsen_stability(z, T, p, mix)
    result = flash_tp(z, T, p, mix)
    assert stab.stable
    assert result.converged is True
    _assert_finite_single_phase(result)


def test_failed_ssi_falls_back_to_single_phase() -> None:
    """Documented CO2–C1 two-phase point (250 K, 5 MPa, z_CO2=0.60) with max_iter=1.

    SSI cannot meet tol in one iteration. Fallback is single-phase by Gibbs,
    converged=False, compositions finite, V in {0, 1}. Davalos 1976 /
    Donnelly & Katz 1954 EXAMPLE window; not a GEM card.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = np.array([0.40, 0.60])
    T, p = 250.0, 5.0e6
    full = flash_tp(z, T, p, mix)
    assert full.converged and full.phase_state == "two-phase"
    result = flash_tp(z, T, p, mix, max_iter=1)
    assert result.converged is False
    _assert_finite_single_phase(result)
    assert result.n_iter == 1
