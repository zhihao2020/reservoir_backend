import numpy as np
import pytest

from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState


def test_dual_compositional_moles_shape() -> None:
    n_cells, nc = 3, 2
    f = CompositionalContinuumState(np.full(n_cells, 1.0e7), np.ones((n_cells, nc)))
    m = CompositionalContinuumState(np.full(n_cells, 1.2e7), 2.0 * np.ones((n_cells, nc)))
    state = DualCompositionalState(fracture=f, matrix=m, time_s=1.0)
    assert state.total_moles().shape == (nc,)
    assert state.total_moles()[0] == pytest.approx(9.0)
    copied = state.copy()
    copied.fracture.pressure[0] = 0.0
    assert state.fracture.pressure[0] == pytest.approx(1.0e7)


def test_mismatched_comp_errors() -> None:
    f = CompositionalContinuumState(np.ones(2), np.ones((2, 2)))
    m = CompositionalContinuumState(np.ones(2), np.ones((2, 3)))
    with pytest.raises(ValueError, match="n_comp"):
        DualCompositionalState(fracture=f, matrix=m)
