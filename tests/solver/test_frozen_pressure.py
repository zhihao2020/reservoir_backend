import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.frozen_pressure import step_frozen_pressure


def test_frozen_step_reduces_matrix_fracture_gap() -> None:
    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    dual = DualRock.from_cf(1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.0e7])
    pm = np.array([1.2e7])
    lam = np.array([1.0e-3])
    pf2, pm2 = step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, dt=10.0)
    assert abs(float(pm2[0] - pf2[0])) < abs(float(pm[0] - pf[0]))
    assert np.all(np.isfinite(pf2)) and np.all(np.isfinite(pm2))


def test_frozen_step_finite_on_two_cells() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.15e7, 1.00e7])
    pm = np.array([1.20e7, 1.18e7])
    lam = np.array([1.0e-3, 1.0e-3])
    pf2, pm2 = step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, dt=1.0)
    assert pf2.shape == (2,)
    assert np.all(np.isfinite(pf2))
    assert np.all(np.isfinite(pm2))
    _ = pytest
