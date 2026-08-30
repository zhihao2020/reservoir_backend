"""Offline ES-MDA ensemble-size scan for scalar C_f (n_theta = 1)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

CANDIDATE_NE = (8, 12, 16, 24, 32)


def candidate_sizes() -> tuple[int, ...]:
    return CANDIDATE_NE


def posterior_spread(theta_members: NDArray[np.float64]) -> dict[str, float]:
    x = np.asarray(theta_members, dtype=float)
    col = x.ravel() if x.ndim == 1 else x[:, 0]
    q = np.quantile(col, [0.05, 0.50, 0.95])
    return {
        "mean": float(np.mean(col)),
        "std": float(np.std(col, ddof=1)) if col.size > 1 else 0.0,
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
    }


def recommend_ne(rows: Iterable[dict], *, rtol: float = 0.15) -> int:
    """Smallest Ne whose mean/std match the largest Ne within ``rtol``."""
    items = list(rows)
    if not items:
        return CANDIDATE_NE[0]
    ref = items[-1]
    pick = int(ref["ne"])
    for row in items:
        mean_ok = abs(row["mean"] - ref["mean"]) <= rtol * max(abs(ref["mean"]), 1.0e-8)
        std_ok = abs(row["std"] - ref["std"]) <= rtol * max(abs(ref["std"]), 1.0e-8)
        if mean_ok and std_ok:
            pick = int(row["ne"])
            break
    return pick
