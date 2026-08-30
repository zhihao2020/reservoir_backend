import numpy as np
import pytest

from reservoir_backend.physics.transfer import WarrenRootTransfer


def test_warren_root_sign_and_scale() -> None:
    tr = WarrenRootTransfer(shape_factor=4.0, k_matrix_m2=1.0e-15)
    q = tr.compute_transfer(p_matrix=2.0e5, p_fracture=1.0e5, cell_volume=0.001)
    assert float(q) == pytest.approx(4.0 * 1.0e-15 * 0.001 * 1.0e5)
    q_rev = tr.compute_transfer(p_matrix=1.0e5, p_fracture=2.0e5, cell_volume=0.001)
    assert float(q_rev) == pytest.approx(-float(q))


def test_transfer_vectorized() -> None:
    tr = WarrenRootTransfer(shape_factor=1.0, k_matrix_m2=1.0e-14)
    pm = np.array([2.0, 3.0])
    pf = np.array([1.0, 1.0])
    vol = np.array([1.0, 2.0])
    q = tr.compute_transfer(pm, pf, vol)
    assert q.shape == (2,)
    assert q[1] == pytest.approx(2.0 * q[0])
