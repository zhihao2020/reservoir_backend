"""Limit BLAS/Numba threads so ensemble workers do not oversubscribe."""

from __future__ import annotations

import os


def cap_flash_threads(n: int) -> None:
    """Inner flash/linear algebra threads. Ensemble workers should pass 1 or 2."""
    n = max(int(n), 1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass


def ensemble_flash_threads() -> int:
    raw = os.environ.get("RESERVOIR_FLASH_THREADS")
    if raw:
        return max(int(raw), 1)
    return 1


def production_flash_threads() -> int:
    raw = os.environ.get("RESERVOIR_FLASH_THREADS")
    if raw:
        return max(int(raw), 1)
    return max(int(os.cpu_count() or 1), 1)


def jacobian_threads(n_slots: int | None = None) -> int:
    """Worker count for independent flash-FD slots. ``RESERVOIR_JAC_THREADS`` overrides."""
    raw = os.environ.get("RESERVOIR_JAC_THREADS")
    ncpu = max(int(os.cpu_count() or 1), 1)
    n = max(int(raw), 1) if raw else min(ncpu, 8)
    if n_slots is not None:
        n = min(n, max(int(n_slots), 1))
    return max(n, 1)


def configure_forward_threads(*, n_slots: int | None = None) -> int:
    """One BLAS thread per Jacobian worker so slot flashes do not oversubscribe."""
    workers = jacobian_threads(n_slots)
    if workers > 1:
        cap_flash_threads(1)
    else:
        cap_flash_threads(production_flash_threads())
    return workers
