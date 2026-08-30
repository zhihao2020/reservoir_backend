import numpy as np

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.dpdp_jacobian import build_sparsity_pattern
from reservoir_backend.solver.fi_comp import _neighbor_cells


def test_sparsity_has_transfer_and_tpfa_blocks() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    n = grid.n_cells
    nu = 3
    neighbors = [_neighbor_cells(grid, c) for c in range(n)]
    pattern = build_sparsity_pattern(n, nu, neighbors)
    half = n * nu
    # same-cell transfer: fracture unknown 0 couples to matrix unknown half
    rows0 = set(pattern.indices[pattern.indptr[0] : pattern.indptr[1]])
    assert half in rows0
    # 7-point: cell 0 neighbors cell 1 on this 2-cell grid
    assert nu in rows0
    assert pattern.n_u == 2 * half
    assert pattern.nnz < pattern.n_u * pattern.n_u
    assert pattern.nnz > 2 * n * nu


def test_context_scales_tf_with_cf() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    ctx = DPDPModelContext.build(grid, n_comp=2)
    from reservoir_backend.physics.dual_rock import DualRock

    a = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    b = a.with_cf(2.0e-12)
    tf_a, tm_a = ctx.transmissibilities(a)
    tf_b, tm_b = ctx.transmissibilities(b)
    if tf_a[0].size:
        assert tf_b[0] == np.asarray(tf_a[0]) * 2.0
    assert tm_a[0].shape == tm_b[0].shape


def test_matrix_intercell_off_zeros_tm() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    ctx = DPDPModelContext.build(grid, n_comp=2, matrix_intercell=False)
    from reservoir_backend.physics.dual_rock import DualRock

    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    _, t_m = ctx.transmissibilities(dual)
    assert float(np.max(np.abs(t_m[0]))) == 0.0
