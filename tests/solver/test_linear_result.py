import numpy as np
from scipy import sparse

from reservoir_backend.solver.linear import (
    CPRLikeSolver,
    GMRESILUSolver,
    LinearSolveResult,
    SparseDirectSolver,
    _schur_pressure,
    _pressure_dofs,
)


def test_linear_solve_result_fields() -> None:
    rng = np.random.default_rng(0)
    n = 12
    a = rng.normal(size=(n, n))
    a = a @ a.T + 3.0 * np.eye(n)
    j = sparse.csc_matrix(a)
    rhs = rng.normal(size=n)
    for solver in (SparseDirectSolver(), GMRESILUSolver()):
        res = solver.solve(j, rhs)
        assert isinstance(res, LinearSolveResult)
        assert res.x.size == n
        assert res.iterations >= 0
        assert np.isfinite(res.final_residual)
        assert res.setup_s >= 0.0
        assert res.solve_s >= 0.0
        assert isinstance(res.preconditioner, str)
        assert res.fallback_used is False


def test_cpr_records_setup_and_schur_operator() -> None:
    rng = np.random.default_rng(1)
    n_cells = 4
    nc = 2
    nu = nc + 1
    n = 2 * n_cells * nu
    a = rng.normal(size=(n, n))
    a = a @ a.T + 4.0 * np.eye(n)
    j = sparse.csc_matrix(a)
    rhs = rng.normal(size=n)
    res = CPRLikeSolver(n_comp=nc).solve(j, rhs)
    assert res.method in {"cpr_gmres", "gmres_ilu"}
    assert res.preconditioner
    assert res.setup_s >= 0.0
    pdofs = _pressure_dofs(n, nc)
    schur, _ = _schur_pressure(j, pdofs)
    assert schur.shape[0] == pdofs.size
