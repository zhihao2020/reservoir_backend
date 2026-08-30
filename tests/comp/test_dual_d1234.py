"""Gate 4 Cases D1–D4: coupled fluxes, σ=0, k_m^intercell→0, small 3D wells."""

import numpy as np
import pytest

from reservoir_backend.comp.dual_residual import dual_residual
from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import moles_from_z
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi_comp import simulate_comp, initialize_state
from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp


def _spec():
    return fluid_from_name("example", temperature_k=350.0)


def _two_cell():
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = _spec()
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
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
    return grid, spec, dual, state


def test_d1_newton_three_fluxes_and_conservation() -> None:
    grid, spec, dual, state = _two_cell()
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    n0 = state.total_moles()
    dp_f0 = abs(float(state.fracture.pressure[0] - state.fracture.pressure[1]))
    dp_mf0 = abs(float(state.matrix.pressure[0] - state.fracture.pressure[0]))
    _, state = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], state, t_end=8.0, dt_init=2.0, dt_max=4.0, max_steps=80
    )
    n1 = state.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    assert rel < 1.0e-4
    assert abs(float(state.fracture.pressure[0] - state.fracture.pressure[1])) < dp_f0
    assert abs(float(state.matrix.pressure[0] - state.fracture.pressure[0])) < dp_mf0


def test_d2_zero_sigma_matches_uncoupled_simulators() -> None:
    grid, spec, dual, state = _two_cell()
    dead = ComponentTransfer(shape_factor=0.0, k_matrix_m2=1.0e-15)
    traj, dual_end = simulate_dual_comp(
        grid, dual, spec, dead, [], [], state, t_end=20.0, dt_init=5.0, dt_max=10.0
    )
    assert traj.reports
    st_f = initialize_state(grid, dual.fracture, spec, 1.0e7)
    st_f.pressure = state.fracture.pressure.copy()
    st_f.moles = state.fracture.moles.copy()
    st_m = initialize_state(grid, dual.matrix, spec, 1.0e7)
    st_m.pressure = state.matrix.pressure.copy()
    st_m.moles = state.matrix.moles.copy()
    traj_f = simulate_comp(grid, dual.fracture, spec, [], [], st_f, t_end=20.0, dt_init=5.0, dt_max=10.0)
    traj_m = simulate_comp(grid, dual.matrix, spec, [], [], st_m, t_end=20.0, dt_init=5.0, dt_max=10.0)
    assert dual_end.fracture.pressure == pytest.approx(traj_f.states[-1].pressure, rel=2e-3, abs=1.0e3)
    assert dual_end.matrix.pressure == pytest.approx(traj_m.states[-1].pressure, rel=2e-3, abs=1.0e3)
    assert dual_end.fracture.moles == pytest.approx(traj_f.states[-1].moles, rel=2e-3, abs=1.0e-8)
    assert dual_end.matrix.moles == pytest.approx(traj_m.states[-1].moles, rel=2e-3, abs=1.0e-8)


def test_d3_matrix_intercell_off_is_dual_porosity() -> None:
    grid, spec, dual, state = _two_cell()
    closed = ComponentTransfer(shape_factor=0.0, k_matrix_m2=1.0e-15)
    open_t = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    n_m1_0 = state.matrix.moles[1].copy()
    _, dead_end = simulate_dual_comp(
        grid, dual, spec, closed, [], [], state, t_end=15.0, dt_init=5.0, dt_max=8.0, matrix_intercell=False
    )
    _, live_end = simulate_dual_comp(
        grid, dual, spec, open_t, [], [], state, t_end=15.0, dt_init=5.0, dt_max=8.0, matrix_intercell=False
    )
    dead_change = float(np.max(np.abs(dead_end.matrix.moles[1] - n_m1_0)))
    live_change = float(np.max(np.abs(live_end.matrix.moles[1] - n_m1_0)))
    assert dead_change < 1.0e-6 * max(float(np.max(np.abs(n_m1_0))), 1.0)
    assert live_change > 10.0 * max(dead_change, 1.0e-18)
    _, _, _, rates = dual_residual(grid, dual, spec, state, state, dt=1.0, transfer=closed)
    assert rates.molar_rate == pytest.approx(0.0, abs=1e-18)


def test_d4_small_3d_wells_mass_balance() -> None:
    grid = CartesianGrid.uniform((0.4, 0.3, 0.2), (0.1, 0.1, 0.1))
    assert grid.n_cells == 24
    spec = _spec()
    dual = DualRock.from_cf(24, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7)
    inj = FlowPort.column(grid, "INJ", "injector", "rate", float(grid.dx[0] * 0.5), 0.15)
    prod = FlowPort.column(grid, "PROD", "producer", "pressure", 0.4 - float(grid.dx[-1] * 0.5), 0.15)
    times = np.array([0.0, 1.5])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([2.0e-4, 2.0e-4])),
        ControlSeries("INJ", "composition", times, np.array([0.95, 0.95])),
        ControlSeries("PROD", "pressure", times, np.array([1.18e7, 1.18e7])),
    ]
    traj, dual_end = simulate_dual_comp(
        grid,
        dual,
        spec,
        transfer,
        [inj, prod],
        controls,
        state,
        t_end=1.5,
        dt_init=0.5,
        dt_max=1.5,
        max_steps=20,
    )
    assert traj.reports
    rel = float(traj.reports[-1].mass.relative_balance_error)
    assert rel < 1.0e-4
    assert np.all(np.isfinite(dual_end.fracture.pressure))
    assert np.all(np.isfinite(dual_end.matrix.pressure))
    props_ok = np.all(dual_end.fracture.moles > 0.0) and np.all(dual_end.matrix.moles > 0.0)
    assert props_ok
    n0 = state.total_moles()
    n1 = dual_end.total_moles()
    assert float(np.sum(np.abs(n1 - n0))) > 0.0
