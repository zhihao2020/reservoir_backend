import numpy as np
import pytest

from reservoir_backend.domain.state import ContinuumState, DualContinuumState


def test_dual_continuum_state_separates_fields() -> None:
    n = 4
    nc = 2
    m = ContinuumState(
        pressure=np.full(n, 1.5e5),
        saturation=np.full(n, 0.2),
        composition=np.tile(np.array([0.6, 0.4]), (n, 1)),
    )
    f = ContinuumState(
        pressure=np.full(n, 1.2e5),
        saturation=np.full(n, 0.8),
        composition=np.tile(np.array([0.9, 0.1]), (n, 1)),
    )
    state = DualContinuumState(matrix=m, fracture=f, time_s=10.0)
    assert state.matrix.pressure[0] != state.fracture.pressure[0]
    assert state.matrix.saturation[0] == pytest.approx(0.2)
    assert state.fracture.saturation[0] == pytest.approx(0.8)
    assert state.matrix.composition.shape == (n, nc)
    copied = state.copy()
    copied.matrix.pressure[0] = 0.0
    assert state.matrix.pressure[0] == pytest.approx(1.5e5)


def test_mismatched_n_cells_errors() -> None:
    m = ContinuumState(pressure=np.ones(3), saturation=np.ones(3))
    f = ContinuumState(pressure=np.ones(2), saturation=np.ones(2))
    with pytest.raises(ValueError, match="n_cells"):
        DualContinuumState(matrix=m, fracture=f)
