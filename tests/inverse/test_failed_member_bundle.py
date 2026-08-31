import numpy as np
import pytest

from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
from reservoir_backend.exceptions import AssimilationError
from reservoir_backend.inverse.ensemble import replace_failed_member_bundle


def _dual(p: float) -> DualCompositionalState:
    return DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([p]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([p + 1.0e6]), np.ones((1, 2)) * 4.0e-4),
        time_s=1.0,
    )


def test_bundle_copies_donor_dual_with_theta() -> None:
    rng = np.random.default_rng(0)
    members = np.array([[0.0, 1.0, 2.0], [0.1, 0.2, 0.3]])
    duals = [_dual(1.0e7), _dual(1.1e7), _dual(1.2e7)]
    failed = np.array([False, True, False])
    x, out_duals, caches = replace_failed_member_bundle(members, failed, rng, 0.1, dual_states=duals)
    assert caches is None
    assert not np.allclose(x[:, 1], members[:, 1])
    # donor is 0 or 2; pressure matches that donor
    donor_p = float(out_duals[1].fracture.pressure[0])
    assert donor_p in {1.0e7, 1.2e7}
    assert donor_p != 1.1e7


def test_bundle_all_failed_raises() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(AssimilationError):
        replace_failed_member_bundle(np.ones((1, 2)), np.array([True, True]), rng, 0.1)
