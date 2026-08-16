"""Parallel ensemble forwards.

Members are independent. Default is a thread pool: IMPES spends most time in
``scipy.sparse.linalg`` (GIL released), and the forward is a closure over the
twin — Windows ``spawn`` cannot pickle that for a process pool.

Set ``n_workers=1`` to force serial. ``None`` means auto (capped).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def resolve_n_workers(n_workers: int | None, n_tasks: int) -> int:
    n_tasks = max(int(n_tasks), 0)
    if n_tasks <= 1:
        return 1
    if n_workers is not None:
        return max(1, min(int(n_workers), n_tasks))
    cpus = int(os.cpu_count() or 1)
    if n_tasks < 4:
        return 1
    return max(1, min(n_tasks, cpus, 8))


def _one_blas_thread():
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except Exception:
        return nullcontext()


def map_members(fn: Callable[[T], R], items: Sequence[T] | Iterable[T], n_workers: int | None) -> list[R]:
    batch = list(items)
    workers = resolve_n_workers(n_workers, len(batch))
    if workers <= 1 or len(batch) <= 1:
        return [fn(item) for item in batch]
    with _one_blas_thread():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, batch))
