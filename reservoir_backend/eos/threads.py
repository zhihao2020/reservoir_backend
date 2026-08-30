"""Limit BLAS/Numba threads so ensemble workers do not oversubscribe."""

from __future__ import annotations

import os


def cap_flash_threads(n: int) -> None:
    """Inner flash/linear algebra threads. Ensemble workers should pass 1 or 2."""
    n = max(int(n), 1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(n)


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
