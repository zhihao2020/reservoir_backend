"""Colored FD Jacobian vs brute-force one-column FD on a tiny DPDP grid."""

import numpy as np
import pytest

from reservoir_backend.comp.dual_residual import pack_dual
from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext, cartesian_cell_colors, verify_coloring_no_row_collision
from reservoir_backend.solver.fi_comp import _neighbor_cells
from reservoir_backend.comp.dual_residual import unpack_dual
from reservoir_backend.solver.dpdp_blocks import assemble_block_jacobian
from reservoir_backend.solver.fi_comp_dual import _coloring_jacobian, _residual, initialize_dual_state

pytestmark = pytest.mark.dpdp


def test_seven_color_has_no_row_collision() -> None:
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    colors = cartesian_cell_colors(grid)
    assert int(np.max(colors)) + 1 <= 7
    neighbors = [_neighbor_cells(grid, c) for c in range(grid.n_cells)]
    verify_coloring_no_row_collision(neighbors, colors)


def test_colored_jacobian_matches_brute_force() -> None:
    grid = CartesianGrid.uniform((0.3, 0.2, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(grid.n_cells, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    assert int(np.max(ctx.colors)) + 1 <= 7
    dt = 1.0
    t_f, t_m = ctx.transmissibilities(dual)
    u = pack_dual(state)
    t1 = float(state.time_s) + dt
    res0, props_f, props_m, *_rest = _residual(
        grid, dual, spec, state, state, dt, transfer, t_f, t_m, [], {}, t1, reflash_all=True
    )
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)
    jac, _fls = _coloring_jacobian(
        ctx, spec, dual, state, dt, transfer, t_f, t_m, [], {}, t1, u, res0, props_f, props_m, n_scale, p_scale
    )
    n_cells = grid.n_cells
    nc = spec.nc
    nu = nc + 1
    half = n_cells * nu
    js = np.asarray(jac.todense())
    jd = np.zeros_like(js)
    nf0, pf0, nm0, pm0 = unpack_dual(u, n_cells, nc)
    from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState

    for col in range(u.size):
        cont = 0 if col < half else 1
        local = col if cont == 0 else col - half
        c = local // nu
        slot = local % nu
        eps = 1.0e-8 * n_scale if slot < nc else 1.0e-8 * p_scale
        nf, pf, nm, pm = nf0.copy(), pf0.copy(), nm0.copy(), pm0.copy()
        if cont == 0:
            if slot < nc:
                nf[c, slot] = nf[c, slot] + eps
            else:
                pf[c] = pf[c] + eps
            re_f, re_m = np.array([c], dtype=np.int64), None
        else:
            if slot < nc:
                nm[c, slot] = nm[c, slot] + eps
            else:
                pm[c] = pm[c] + eps
            re_f, re_m = None, np.array([c], dtype=np.int64)
        trial = DualCompositionalState(
            fracture=CompositionalContinuumState(pf, nf),
            matrix=CompositionalContinuumState(pm, nm),
            time_s=t1,
        )
        r2, *_ = _residual(
            grid,
            dual,
            spec,
            trial,
            state,
            dt,
            transfer,
            t_f,
            t_m,
            [],
            {},
            t1,
            props_f=props_f.copy(),
            props_m=props_m.copy(),
            reflash_f=re_f,
            reflash_m=re_m,
        )
        jd[:, col] = (r2 - res0) / eps
    denom = max(float(np.linalg.norm(jd)), 1.0e-30)
    rel = float(np.linalg.norm(js - jd) / denom)
    assert rel < 1.0e-4, f"colored vs brute Jacobian rel {rel}"


def test_block_jacobian_matches_coloring() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(grid.n_cells, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    dt = 1.0
    t_f, t_m = ctx.transmissibilities(dual)
    t1 = float(state.time_s) + dt
    u = pack_dual(state)
    res0, props_f, props_m, *_ = _residual(
        grid, dual, spec, state, state, dt, transfer, t_f, t_m, [], {}, t1, reflash_all=True
    )
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)
    jc, _ = _coloring_jacobian(
        ctx, spec, dual, state, dt, transfer, t_f, t_m, [], {}, t1, u, res0, props_f, props_m, n_scale, p_scale
    )
    jb, _ = assemble_block_jacobian(
        grid, spec, dual, state, dt, transfer, t_f, t_m, props_f, props_m, n_scale, p_scale
    )
    ac = np.asarray(jc.todense())
    ab = np.asarray(jb.todense())
    rel = float(np.linalg.norm(ab - ac) / max(float(np.linalg.norm(ac)), 1.0e-30))
    assert rel < 1.0e-3, f"block vs coloring rel {rel}"


def test_block_jacobian_is_finite_and_sparse() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(grid.n_cells, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    dt = 1.0
    t_f, t_m = ctx.transmissibilities(dual)
    t1 = float(state.time_s) + dt
    _, props_f, props_m, *_ = _residual(
        grid, dual, spec, state, state, dt, transfer, t_f, t_m, [], {}, t1, reflash_all=True
    )
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)
    jac, fls = assemble_block_jacobian(
        grid, spec, dual, state, dt, transfer, t_f, t_m, props_f, props_m, n_scale, p_scale
    )
    n_u = pack_dual(state).size
    assert jac.shape == (n_u, n_u)
    assert np.all(np.isfinite(jac.data))
    assert jac.nnz < n_u * n_u
    assert fls >= 0.0
