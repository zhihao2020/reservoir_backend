"""Gate 4 Case D0: one cell, transfer only, component moles conserved."""

import numpy as np
import pytest

pytestmark = pytest.mark.dpdp

from reservoir_backend.comp.dual_residual import dual_residual
from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import moles_from_z
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.fi_comp_dual import solve_dual_comp_step


def _d0_state():
    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(
        1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02
    )
    vol = grid.cell_volumes()
    p_f = np.array([1.05e7])
    p_m = np.array([1.25e7])
    n_f = moles_from_z(spec, p_f, spec.z_init, dual.fracture.porosity * vol)
    n_m = moles_from_z(spec, p_m, spec.z_init, dual.matrix.porosity * vol)
    state = DualCompositionalState(
        fracture=CompositionalContinuumState(p_f, n_f),
        matrix=CompositionalContinuumState(p_m, n_m),
        time_s=0.0,
    )
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    return grid, spec, dual, state, transfer


def test_d0_transfer_antisymmetry_in_residual() -> None:
    grid, spec, dual, state, transfer = _d0_state()
    res, _, _, rates = dual_residual(grid, dual, spec, state, state, dt=1.0, transfer=transfer)
    nc = spec.nc
    rf = res[: nc + 1]
    rm = res[nc + 1 :]
    assert float(np.sum(rates.molar_rate)) > 0.0
    assert rf[:nc] == pytest.approx(-rates.molar_rate[0], rel=1e-8, abs=1e-12)
    assert rm[:nc] == pytest.approx(rates.molar_rate[0], rel=1e-8, abs=1e-12)
    assert rf[:nc] + rm[:nc] == pytest.approx(0.0, abs=1e-12)


def test_d0_closed_transfer_zero_when_sigma_zero() -> None:
    grid, spec, dual, state, _ = _d0_state()
    dead = ComponentTransfer(shape_factor=0.0, k_matrix_m2=1.0e-15)
    res0, _, _, rates = dual_residual(grid, dual, spec, state, state, dt=10.0, transfer=dead)
    assert rates.molar_rate == pytest.approx(0.0, abs=1e-18)
    nc = spec.nc
    assert res0[:nc] == pytest.approx(0.0, abs=1e-9)
    assert res0[nc + 1 : nc + 1 + nc] == pytest.approx(0.0, abs=1e-9)


def test_d1_has_fracture_matrix_and_transfer_flux() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    assert grid.n_cells == 2
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(
        2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02
    )
    vol = grid.cell_volumes()
    p_f = np.array([1.15e7, 1.00e7])
    p_m = np.array([1.30e7, 1.20e7])
    n_f = moles_from_z(spec, p_f, spec.z_init, dual.fracture.porosity * vol)
    n_m = moles_from_z(spec, p_m, spec.z_init, dual.matrix.porosity * vol)
    state = DualCompositionalState(
        fracture=CompositionalContinuumState(p_f, n_f),
        matrix=CompositionalContinuumState(p_m, n_m),
        time_s=0.0,
    )
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    dead = ComponentTransfer(shape_factor=0.0, k_matrix_m2=1.0e-15)
    res, _, _, rates = dual_residual(grid, dual, spec, state, state, dt=1.0, transfer=transfer)
    res0, _, _, _ = dual_residual(grid, dual, spec, state, state, dt=1.0, transfer=dead)
    nc = spec.nc
    half = 2 * (nc + 1)
    rf0 = res0[:half].reshape(2, nc + 1)[:, :nc]
    rm0 = res0[half:].reshape(2, nc + 1)[:, :nc]
    assert float(np.max(np.abs(rates.molar_rate))) > 0.0
    assert float(np.max(np.abs(rf0))) > 0.0
    assert float(np.max(np.abs(rm0))) > 0.0
    assert float(np.max(np.abs(res[:half] - res0[:half]))) > 0.0


def test_d0_newton_conserves_moles_and_relaxes_pressure() -> None:
    grid, spec, dual, state, transfer = _d0_state()
    n0 = state.total_moles()
    dp0 = abs(float(state.matrix.pressure[0] - state.fracture.pressure[0]))
    for _ in range(6):
        out = solve_dual_comp_step(grid, dual, spec, state, dt=5.0, transfer=transfer, tol=1.0e-7)
        state = out.state
    n1 = state.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    assert rel < 1.0e-4
    dp1 = abs(float(state.matrix.pressure[0] - state.fracture.pressure[0]))
    assert dp1 < dp0
    assert n1 == pytest.approx(n0, rel=1e-4)
