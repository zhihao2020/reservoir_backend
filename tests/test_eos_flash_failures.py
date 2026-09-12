"""Flash failure paths: negative RR and failed-SSI fallback. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.eos import (
    EXAMPLE_LIBRARY_MARKER,
    example_eight_component_mixture,
    flash_tp,
    michelsen_stability,
    solve_rachford_rice,
)
from reservoir_backend.eos.flash import _damped_k_update


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


def test_max_iter_zero_falls_back_to_single_phase() -> None:
    """Unstable EXAMPLE CO2–C1 window with max_iter=0: no SSI, usable single-phase."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    result = flash_tp(np.array([0.40, 0.60]), 250.0, 5.0e6, mix, max_iter=0)
    assert result.converged is False
    assert result.n_iter == 0
    _assert_finite_single_phase(result)


def test_allow_negative_rr_still_clips_to_single_phase() -> None:
    """``allow_negative`` is accepted and still returns V in {0, 1}, not two-phase."""
    V, state = solve_rachford_rice(
        np.array([0.20, 0.80]), np.array([1.2, 0.5]), allow_negative=True
    )
    assert state == "liquid"
    assert V == 0.0
    V, state = solve_rachford_rice(
        np.array([0.80, 0.20]), np.array([10.0, 0.8]), allow_negative=True
    )
    assert state == "vapor"
    assert V == 1.0


def test_rr_extreme_k_is_finite_single_phase() -> None:
    """Singular / clipped K cannot emit NaN or two-phase V outside (0, 1)."""
    z = np.array([0.50, 0.50])
    V, state = solve_rachford_rice(z, np.array([1.0e-20, 1.0e20]))
    assert state in ("liquid", "vapor", "two-phase")
    assert np.isfinite(V)
    assert 0.0 <= V <= 1.0
    if state != "two-phase":
        assert V in (0.0, 1.0)
    V0, s0 = solve_rachford_rice(z, np.array([1.0, 1.0]))
    assert s0 in ("liquid", "vapor")
    assert V0 in (0.0, 1.0)


def test_ssi_damping_update_stays_finite() -> None:
    """Damped SSI ln-K step stays in the published K clip; not a GEM card."""
    k0 = np.array([2.0, 0.4])
    k_ss = np.array([20.0, 0.04])
    k_half = _damped_k_update(k0, k_ss, 0.5)
    k_full = _damped_k_update(k0, k_ss, 1.0)
    assert np.all(np.isfinite(k_half))
    assert np.all(np.isfinite(k_full))
    assert np.all(k_half > 0.0) and np.all(k_full > 0.0)
    # Half-step is between the current K and the undamped SSI update.
    assert np.all((k_half - k0) * (k_ss - k0) > 0.0)
    assert np.max(np.abs(np.log(k_half) - np.log(k0))) < np.max(np.abs(np.log(k_full) - np.log(k0)))
