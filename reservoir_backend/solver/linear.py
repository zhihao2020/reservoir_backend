"""Sparse linear solvers for DPDP Newton. fi_comp_dual does not pick scipy details."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import gmres, spilu, spsolve
from scipy.sparse.linalg import LinearOperator

from reservoir_backend.exceptions import PhysicsConvergenceError

_DIRECT_MAX = 2000
_PREC_CACHE: dict = {"mat": None, "kind": None, "prec": None}


def _cached_prec(mat, kind: str, factory):
    if _PREC_CACHE["mat"] is mat and _PREC_CACHE["kind"] == kind and _PREC_CACHE["prec"] is not None:
        return _PREC_CACHE["prec"]
    prec = factory()
    _PREC_CACHE["mat"] = mat
    _PREC_CACHE["kind"] = kind
    _PREC_CACHE["prec"] = prec
    return prec


@dataclass
class LinearSolveResult:
    x: NDArray[np.float64]
    method: str
    iterations: int = 0
    final_residual: float = 0.0
    setup_s: float = 0.0
    solve_s: float = 0.0
    preconditioner: str = ""
    fallback_used: bool = False


class LinearSolver:
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        raise NotImplementedError


class SparseDirectSolver(LinearSolver):
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        t0 = time.perf_counter()
        j = jacobian.tocsc()
        x = np.asarray(spsolve(j, np.asarray(rhs, dtype=float)), dtype=float).ravel()
        dt = time.perf_counter() - t0
        if x.size != int(rhs.size) or not np.all(np.isfinite(x)):
            raise PhysicsConvergenceError("DPDP sparse direct solve failed")
        r = float(np.linalg.norm(j.dot(x) - np.asarray(rhs, dtype=float).ravel()))
        return LinearSolveResult(x=x, method="spsolve", iterations=1, final_residual=r, setup_s=0.0, solve_s=dt, preconditioner="none")


class GMRESILUSolver(LinearSolver):
    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        j = jacobian.tocsc()
        rhs = np.asarray(rhs, dtype=float).ravel()
        n = int(rhs.size)
        drop = 1.0e-3 if n > 20000 else 1.0e-4
        fill = 3 if n > 20000 else 10
        t_setup0 = time.perf_counter()
        try:
            def _fact():
                ilu = spilu(j, drop_tol=drop, fill_factor=fill)
                return LinearOperator(j.shape, matvec=ilu.solve)

            prec = _cached_prec(jacobian, f"ilu-{drop}-{fill}", _fact)
        except Exception:
            prec = None
        setup_s = time.perf_counter() - t_setup0
        niter = [0]

        def _cb(_r):
            niter[0] += 1

        t1 = time.perf_counter()
        try:
            x, info = gmres(j, rhs, M=prec, rtol=1.0e-8, atol=0.0, restart=40, maxiter=400, callback=_cb)
        except TypeError:
            x, info = gmres(j, rhs, M=prec, tol=1.0e-8, restart=40, maxiter=400, callback=_cb)
        solve_s = time.perf_counter() - t1
        x = np.asarray(x, dtype=float).ravel()
        if int(info) != 0 or x.size != rhs.size or not np.all(np.isfinite(x)):
            raise PhysicsConvergenceError(f"DPDP GMRES failed, info={info}")
        rfin = float(np.linalg.norm(j.dot(x) - rhs))
        return LinearSolveResult(
            x=x,
            method="gmres_ilu",
            iterations=niter[0],
            final_residual=rfin,
            setup_s=setup_s,
            solve_s=solve_s,
            preconditioner="ilu",
        )


def _pressure_dofs(n_unknowns: int, n_comp: int) -> NDArray[np.int64]:
    nu = int(n_comp) + 1
    if n_unknowns % (2 * nu) != 0:
        raise ValueError("unknown count is not 2 n_cells (nc+1)")
    n_cells = n_unknowns // (2 * nu)
    f = np.arange(n_cells, dtype=np.int64) * nu + n_comp
    m = n_cells * nu + np.arange(n_cells, dtype=np.int64) * nu + n_comp
    return np.concatenate([f, m])


class CPRLikeSolver(LinearSolver):
    """Pressure-block ILU + Jacobi global correction. No full-system ILU."""

    def __init__(self, n_comp: int = 2):
        self.n_comp = int(n_comp)

    def solve(self, jacobian: sparse.spmatrix, rhs: NDArray[np.float64]) -> LinearSolveResult:
        rhs = np.asarray(rhs, dtype=float).ravel()
        n = int(rhs.size)
        j = jacobian.tocsr()
        jcsc = j.tocsc()
        t0 = time.perf_counter()
        try:
            def _fact():
                pdofs = _pressure_dofs(n, self.n_comp)
                jpp, _ = _schur_pressure(j, pdofs)
                diag = np.asarray(j.diagonal(), dtype=float)
                diag = np.where(np.abs(diag) < 1.0e-30, 1.0, diag)
                invd = 1.0 / diag
                ilu_p = spilu(jpp, drop_tol=1.0e-3, fill_factor=5)

                def _prec(v, _ilu=ilu_p, _invd=invd, _j=j, _pdofs=pdofs):
                    v = np.asarray(v, dtype=float).ravel()
                    y = np.zeros_like(v)
                    y[_pdofs] = _ilu.solve(v[_pdofs])
                    r = v - _j.dot(y)
                    return y + _invd * r

                return LinearOperator(j.shape, matvec=_prec)

            prec = _cached_prec(jacobian, "cpr-jacobi", _fact)
        except Exception:
            fb = GMRESILUSolver().solve(jacobian, rhs)
            fb.fallback_used = True
            return fb
        setup_s = time.perf_counter() - t0
        rtol = 1.0e-5 if n > 20000 else 1.0e-8
        maxiter = 80 if n > 20000 else 400
        niter = [0]

        def _cb(_r):
            niter[0] += 1

        t1 = time.perf_counter()
        try:
            x, info = gmres(jcsc, rhs, M=prec, rtol=rtol, atol=0.0, restart=30, maxiter=maxiter, callback=_cb)
        except TypeError:
            x, info = gmres(jcsc, rhs, M=prec, tol=rtol, restart=30, maxiter=maxiter, callback=_cb)
        solve_s = time.perf_counter() - t1
        x = np.asarray(x, dtype=float).ravel()
        if int(info) != 0 or x.size != rhs.size or not np.all(np.isfinite(x)):
            fb = GMRESILUSolver().solve(jacobian, rhs)
            fb.fallback_used = True
            return fb
        rfin = float(np.linalg.norm(j.dot(x) - rhs))
        return LinearSolveResult(
            x=x,
            method="cpr_gmres",
            iterations=niter[0],
            final_residual=rfin,
            setup_s=setup_s,
            solve_s=solve_s,
            preconditioner="schur_ilu_jacobi",
        )


def _schur_pressure(j: sparse.spmatrix, pdofs: NDArray[np.int64]):
    """S_p ≈ J_pp - J_pn diag(J_nn)^{-1} J_np. Does not invert J_nn."""
    n = int(j.shape[0])
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(pdofs, dtype=np.int64)] = False
    ndofs = np.flatnonzero(mask)
    jsr = j.tocsr()
    jpp = jsr[pdofs, :][:, pdofs]
    if ndofs.size == 0:
        return jpp.tocsc(), pdofs
    jpn = jsr[pdofs, :][:, ndofs]
    jnp = jsr[ndofs, :][:, pdofs]
    dnn = np.asarray(jsr.diagonal(), dtype=float)[ndofs]
    dnn = np.where(np.abs(dnn) < 1.0e-30, 1.0, dnn)
    schur = jpp - (jpn @ sparse.diags(1.0 / dnn) @ jnp)
    return schur.tocsc(), pdofs


def solve_newton_system(
    jacobian: sparse.spmatrix,
    rhs: NDArray[np.float64],
    *,
    n_comp: int | None = None,
    backend: str | None = None,
) -> LinearSolveResult:
    import os

    n = int(np.asarray(rhs).size)
    name = (backend or os.environ.get("RESERVOIR_LINEAR") or "").strip().lower()
    if name in {"cpr", "cprlike"} and n_comp is not None:
        solver: LinearSolver = CPRLikeSolver(n_comp=n_comp)
    elif name in {"gmres", "gmres_ilu"}:
        solver = GMRESILUSolver()
    elif name in {"direct", "spsolve"} or n <= _DIRECT_MAX:
        solver = SparseDirectSolver()
    elif n_comp is not None and n > 20000:
        solver = CPRLikeSolver(n_comp=n_comp)
    else:
        solver = GMRESILUSolver()
    try:
        return solver.solve(jacobian, rhs)
    except Exception:
        if n <= _DIRECT_MAX:
            return GMRESILUSolver().solve(jacobian, rhs)
        raise
