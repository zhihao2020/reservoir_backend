"""Optional linear-solver backend wrapper with JSON-serializable stats."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_linear_system(
    matrix: ArrayLike,
    rhs: ArrayLike,
    backend: str = "direct",
    tolerance: float = 1.0e-10,
    max_iterations: int = 1000,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Solve a linear system and return `(solution, solver_stats)`.

    Backends are best-effort. `direct` is the baseline. Iterative, ILU, and AMG
    requests fall back to direct solve when the optional implementation is not
    available or fails.
    """
    requested = backend.lower()
    warnings: list[str] = []
    fallback_used = False
    actual_backend = requested
    num_iterations: int | None = None
    a_dense = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    _validate_system(a_dense, b)

    try:
        if requested == "direct":
            solution = _solve_direct(a_dense, b)
        elif requested in {"cg", "gmres"}:
            solution, num_iterations = _solve_iterative(a_dense, b, requested, tolerance, max_iterations)
        elif requested == "ilu":
            solution, num_iterations = _solve_ilu(a_dense, b, tolerance, max_iterations)
        elif requested == "amg":
            solution, num_iterations = _solve_amg(a_dense, b, tolerance, max_iterations)
        else:
            raise ValueError(f"unsupported linear solver backend: {backend}")
    except Exception as exc:  # pragma: no cover - exercised by optional backend availability
        if requested == "direct":
            raise
        warnings.append(f"{requested} backend unavailable or failed; used direct fallback: {exc}")
        solution = _solve_direct(a_dense, b)
        fallback_used = True
        actual_backend = "direct"

    residual = a_dense @ solution - b
    residual_norm = float(np.linalg.norm(residual))
    stats = {
        "backend": actual_backend,
        "requested_backend": requested,
        "success": bool(np.isfinite(solution).all() and np.isfinite(residual_norm)),
        "num_iterations": num_iterations,
        "residual_norm": residual_norm,
        "mass_balance_error": float(abs(np.sum(residual))),
        "flux_conservation_error": residual_norm,
        "warnings": warnings,
        "fallback_used": fallback_used,
    }
    return solution, stats


def compute_solver_stats(
    matrix: ArrayLike,
    solution: ArrayLike,
    rhs: ArrayLike,
    backend: str = "external",
    warnings: list[str] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """Compute solver stats for a solution produced elsewhere."""
    a = np.asarray(matrix, dtype=float)
    x = np.asarray(solution, dtype=float)
    b = np.asarray(rhs, dtype=float)
    residual = a @ x - b
    residual_norm = float(np.linalg.norm(residual))
    return {
        "backend": backend,
        "success": bool(np.isfinite(x).all() and np.isfinite(residual_norm)),
        "num_iterations": None,
        "residual_norm": residual_norm,
        "mass_balance_error": float(abs(np.sum(residual))),
        "flux_conservation_error": residual_norm,
        "warnings": [] if warnings is None else list(warnings),
        "fallback_used": bool(fallback_used),
    }


def _validate_system(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("rhs shape must match matrix rows")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("matrix and rhs must be finite")


def _solve_direct(matrix: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.linalg.solve(matrix, rhs), dtype=float)


def _solve_iterative(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    method: str,
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int | None]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg, gmres

    iterations = 0

    def _callback(_: Any) -> None:
        nonlocal iterations
        iterations += 1

    solver: Callable[..., Any] = cg if method == "cg" else gmres
    kwargs = {"rtol": tolerance, "atol": 0.0, "maxiter": max_iterations, "callback": _callback}
    if method == "gmres":
        kwargs["callback_type"] = "legacy"
    solution, info = solver(csr_matrix(matrix), rhs, **kwargs)
    if info != 0:
        raise RuntimeError(f"{method} did not converge, info={info}")
    return np.asarray(solution, dtype=float), iterations


def _solve_ilu(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int | None]:
    from scipy.sparse import csc_matrix, csr_matrix
    from scipy.sparse.linalg import LinearOperator, gmres, spilu

    sparse = csc_matrix(matrix)
    ilu = spilu(sparse)
    operator = LinearOperator(sparse.shape, ilu.solve)
    iterations = 0

    def _callback(_: Any) -> None:
        nonlocal iterations
        iterations += 1

    solution, info = gmres(
        csr_matrix(matrix),
        rhs,
        M=operator,
        rtol=tolerance,
        atol=0.0,
        maxiter=max_iterations,
        callback=_callback,
        callback_type="legacy",
    )
    if info != 0:
        raise RuntimeError(f"ilu-preconditioned gmres did not converge, info={info}")
    return np.asarray(solution, dtype=float), iterations


def _solve_amg(
    matrix: NDArray[np.float64],
    rhs: NDArray[np.float64],
    tolerance: float,
    max_iterations: int,
) -> tuple[NDArray[np.float64], int | None]:
    try:
        import pyamg  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("pyamg is not installed") from exc

    from scipy.sparse import csr_matrix

    ml = pyamg.smoothed_aggregation_solver(csr_matrix(matrix))
    solution = ml.solve(rhs, tol=tolerance, maxiter=max_iterations)
    return np.asarray(solution, dtype=float), None
