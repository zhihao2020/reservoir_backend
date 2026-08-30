"""DPDP restart must restore p_f, n_f, p_m, n_m, not only time_s."""

import numpy as np
import pytest

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp_dual import (
    dual_from_visual_state,
    dual_to_state,
    initialize_dual_state,
    simulate_dual_comp,
)

pytestmark = pytest.mark.dpdp


def _closed():
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.25e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    return grid, spec, dual, transfer, state, ctx


def test_visual_state_roundtrip_keeps_matrix_moles() -> None:
    grid, spec, dual, transfer, state, ctx = _closed()
    traj, end = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], state, t_end=2.0, dt_init=1.0, dt_max=2.0, max_steps=8, context=ctx
    )
    vis = dual_to_state(spec, end, dual)
    assert vis.moles_matrix is not None
    restored = dual_from_visual_state(grid, dual, spec, vis)
    assert restored.fracture.pressure == pytest.approx(end.fracture.pressure)
    assert restored.matrix.pressure == pytest.approx(end.matrix.pressure)
    assert restored.fracture.moles == pytest.approx(end.fracture.moles)
    assert restored.matrix.moles == pytest.approx(end.matrix.moles)
    assert restored.time_s == pytest.approx(end.time_s)
    _ = traj


def test_restart_from_dual_matches_restart_from_visual() -> None:
    grid, spec, dual, transfer, state, ctx = _closed()
    _, mid = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], state, t_end=2.0, dt_init=1.0, dt_max=2.0, max_steps=8, context=ctx
    )
    vis = dual_to_state(spec, mid, dual)
    _, a = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], mid, t_end=3.0, dt_init=1.0, dt_max=2.0, max_steps=8, context=ctx
    )
    _, b = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], vis, t_end=3.0, dt_init=1.0, dt_max=2.0, max_steps=8, context=ctx
    )
    assert a.matrix.moles == pytest.approx(b.matrix.moles, rel=1e-8, abs=1e-12)
    fresh = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.25e7)
    assert np.max(np.abs(mid.matrix.moles - fresh.matrix.moles)) > 1.0e-12


def test_restart_without_matrix_moles_is_rejected() -> None:
    grid, spec, dual, transfer, state, ctx = _closed()
    _, mid = simulate_dual_comp(
        grid, dual, spec, transfer, [], [], state, t_end=2.0, dt_init=1.0, dt_max=2.0, max_steps=8, context=ctx
    )
    vis = dual_to_state(spec, mid, dual)
    vis.moles_matrix = None
    with pytest.raises(ValueError, match="moles_matrix"):
        dual_from_visual_state(grid, dual, spec, vis)
