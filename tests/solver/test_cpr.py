import numpy as np
from scipy import sparse

from reservoir_backend.solver.linear import CPRLikeSolver, SparseDirectSolver, _pressure_dofs


def test_pressure_dofs_binary() -> None:
    # 2 cells, nc=2, nu=3, 2 continua → 12 unknowns. p slots 2,5,8,11
    idx = _pressure_dofs(12, 2)
    assert list(idx) == [2, 5, 8, 11]


def test_cpr_matches_direct_on_diagonal_dominant() -> None:
    rng = np.random.default_rng(0)
    n_cells = 4
    nc = 2
    nu = nc + 1
    n = 2 * n_cells * nu
    a = rng.normal(size=(n, n))
    a = a @ a.T + 3.0 * np.eye(n)
    j = sparse.csc_matrix(a)
    rhs = rng.normal(size=n)
    xd = SparseDirectSolver().solve(j, rhs).x
    xc = CPRLikeSolver(n_comp=nc).solve(j, rhs).x
    np.testing.assert_allclose(xc, xd, rtol=2.0e-6, atol=1.0e-8)
