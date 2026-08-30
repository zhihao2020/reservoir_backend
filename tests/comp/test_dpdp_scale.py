"""Scalability smoke: 5³ one forward. Larger grids live in scripts/dpdp_scale_bench.py."""

import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp


def test_five_cubed_closed_transfer_conserves() -> None:
    n = 5
    dx = 0.02
    grid = CartesianGrid.uniform((n * dx, n * dx, n * dx), dx)
    assert grid.n_cells == 125
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(125, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.25e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    n0 = state.total_moles()
    traj, end = simulate_dual_comp(
        grid,
        dual,
        spec,
        transfer,
        [],
        [],
        state,
        t_end=1.0,
        dt_init=0.5,
        dt_max=1.0,
        max_steps=8,
        context=ctx,
    )
    assert traj.reports
    n1 = end.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    assert rel < 1.0e-4
    assert ctx.pattern.nnz < ctx.pattern.n_u ** 2


def test_ten_cubed_closed_transfer_conserves() -> None:
    n = 10
    dx = 0.01
    grid = CartesianGrid.uniform((n * dx, n * dx, n * dx), dx)
    assert grid.n_cells == 1000
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(1000, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    n0 = state.total_moles()
    traj, end = simulate_dual_comp(
        grid,
        dual,
        spec,
        transfer,
        [],
        [],
        state,
        t_end=0.5,
        dt_init=0.5,
        dt_max=0.5,
        max_steps=4,
        context=ctx,
    )
    assert traj.reports
    n1 = end.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    assert rel < 1.0e-4
    notes = " ".join(traj.reports[-1].notes)
    assert "sum_jac_s=" in notes
    assert "sum_solve_s=" in notes
