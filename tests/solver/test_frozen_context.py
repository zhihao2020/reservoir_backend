"""Gate 8: frozen-pressure LU reuse matches a fresh factorisation."""

import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.frozen_pressure import FrozenPressureContext, step_frozen_pressure


def test_frozen_lu_reuse_matches_fresh_solve() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.15e7, 1.00e7])
    pm = np.array([1.20e7, 1.18e7])
    lam = np.array([1.0e-3, 1.0e-3])
    factor = FrozenPressureContext()
    a1, b1 = step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, dt=1.0, factor=factor)
    assert factor.n_factor == 1
    a2, b2 = step_frozen_pressure(grid, ctx, dual, tr, a1, b1, lam, lam, dt=1.0, factor=factor)
    assert factor.n_reuse == 1
    c2, d2 = step_frozen_pressure(grid, ctx, dual, tr, a1, b1, lam, lam, dt=1.0)
    np.testing.assert_allclose(a2, c2, rtol=1.0e-10, atol=1.0)
    np.testing.assert_allclose(b2, d2, rtol=1.0e-10, atol=1.0)
    _ = pytest
