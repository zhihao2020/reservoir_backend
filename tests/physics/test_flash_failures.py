"""Flash failure paths on the EXAMPLE C1–nC10 card. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import (
    flash_tp,
    negative_flash_vapor_frac,
    rachford_rice,
)
from reservoir_backend.eos.flash_backend import FastPRBackend


def _assert_finite_single_phase(result) -> None:
    assert result.two_phase is False
    assert result.vapor_frac in (0.0, 1.0)
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(result.y))
    assert np.isfinite(result.vapor_frac)
    assert np.isfinite(result.v_mix)
    np.testing.assert_allclose(result.x, result.y)
    assert 0.0 <= result.vapor_frac <= 1.0


def test_negative_flash_mixed_k_is_single_phase() -> None:
    """RR with mixed K that has no root in (0, 1) is liquid or vapor."""
    assert negative_flash_vapor_frac(np.array([1.2, 0.5]), np.array([0.20, 0.80])) == 0.0
    assert rachford_rice(np.array([1.2, 0.5]), np.array([0.20, 0.80])) == 0.0
    assert negative_flash_vapor_frac(np.array([10.0, 0.8]), np.array([0.80, 0.20])) == 1.0
    assert rachford_rice(np.array([10.0, 0.8]), np.array([0.80, 0.20])) == 1.0


def test_heavy_example_feed_negative_flash_via_flash_tp() -> None:
    """Cold heavy EXAMPLE nC10 at 280 K, 25 MPa: single-phase, finite, no NaN."""
    eos = example_c1_nc10()
    z = np.array([0.08, 0.92])
    result = flash_tp(eos, 25.0e6, 280.0, z)
    _assert_finite_single_phase(result)
    assert result.converged
    assert result.vapor_frac == 0.0


def test_almost_pure_c1_high_t_is_single_phase() -> None:
    """Almost-pure C1 at 400 K, 5 MPa: TPD-stable → single-phase vapor."""
    eos = example_c1_nc10()
    z = np.array([0.9993, 0.0007])
    result = flash_tp(eos, 5.0e6, 400.0, z)
    assert result.converged is True
    _assert_finite_single_phase(result)
    assert result.vapor_frac == 1.0


def test_failed_ssi_falls_back_to_single_phase() -> None:
    """Interior C1–nC10 two-phase point with max_iter=1: SSI cannot meet tol.

    Fallback is single-phase by Gibbs, converged=False, compositions finite.
    EXAMPLE C1–nC10 (Reid / Katz–Firoozabadi); not a GEM card.
    """
    eos = example_c1_nc10()
    z = np.array([0.70, 0.30])
    full = flash_tp(eos, 8.0e6, 350.0, z)
    assert full.converged and full.two_phase
    result = flash_tp(eos, 8.0e6, 350.0, z, max_iter=1)
    assert result.converged is False
    assert result.fallback_used is True
    _assert_finite_single_phase(result)
    assert result.iterations == 1


def test_fastpr_failed_ssi_matches_reference_fallback() -> None:
    eos = example_c1_nc10()
    z = np.array([0.70, 0.30])
    ref = flash_tp(eos, 8.0e6, 350.0, z, max_iter=1)
    got = FastPRBackend().flash_tp(eos, 8.0e6, 350.0, z, max_iter=1)
    assert ref.two_phase is False
    assert got.two_phase is False
    assert got.converged is False
    assert got.fallback_used is True
    assert got.vapor_frac == ref.vapor_frac
