"""Sparse linear solvers for DPDP Newton. fi_comp_dual does not pick scipy details."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import gmres, spilu, spsolve
from scipy.sparse.linalg import LinearOperator

from reservoir_backend.exceptions import PhysicsConvergenceError

_DIRECT_MAX = 2000


@dataclass
class LinearSolveResult:
    x: NDArray[np.float64]
    method: str


class LinearSolver:
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        raise NotImplementedError


class SparseDirectSolver(LinearSolver):
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        j = jacobian.tocsc()
        x = np.asarray(spsolve(j, np.asarray(rhs, dtype=float)), dtype=float).ravel()
        if x.size != int(rhs.size) or not np.all(np.isfinite(x)):
            raise PhysicsConvergenceError("DPDP sparse direct solve failed")
        return LinearSolveResult(x=x, method="spsolve")


class GMRESILUSolver(LinearSolver):
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        j = jacobian.tocsc()
        rhs = np.asarray(rhs, dtype=float).ravel()
        try:
            ilu = spilu(j, drop_tol=1.0e-4, fill_factor=10)
            prec = LinearOperator(j.shape, matvec=ilu.solve)
        except Exception:
            prec = None
        try:
            x, info = gmres(j, rhs, M=prec, rtol=1.0e-8, atol=0.0, restart=40, maxiter=400)
        except TypeError:
            x, info = gmres(j, rhs, M=prec, tol=1.0e-8, restart=40, maxiter=400)
        x = np.asarray(x, dtype=float).ravel()
        if int(info) != 0 or x.size != rhs.size or not np.all(np.isfinite(x)):
            raise PhysicsConvergenceError(f"DPDP GMRES failed, info={info}")
        return LinearSolveResult(x=x, method="gmres_ilu")


def solve_newton_system(jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
    n = int(np.asarray(rhs).size)
    solver: LinearSolver = SparseDirectSolver() if n <= _DIRECT_MAX else GMRESILUSolver()
    try:
        return solver.solve(jacobian, rhs)
    except Exception:
        if n <= _DIRECT_MAX:
            return GMRESILUSolver().solve(jacobian, rhs)
        raise
