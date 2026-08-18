"""Instrument noise vs forward residual. Do not dump F-mismatch into K twice."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def inflate_sigma(
    pred: NDArray[np.float64] | list[float],
    clean: NDArray[np.float64] | list[float],
    instrument: float,
    *,
    extra_cap: float | None = None,
) -> tuple[float, float]:
    """Return ``(extra, sigma)`` with ``sigma = sqrt(σ_inst² + extra²)``.

    ``extra`` is RMSE(F, clean gauges). Passing noisy values as ``clean``
    folds the instrument draw into R and over-damps the update.
    ``extra_cap`` clips the model-error term so one bad channel cannot
    bury the update.
    """
    p = np.asarray(pred, dtype=float).ravel()
    c = np.asarray(clean, dtype=float).ravel()
    if p.size != c.size or p.size == 0:
        raise ValueError("pred and clean must be non-empty and aligned")
    inst = float(instrument)
    if not np.isfinite(inst) or inst < 0.0:
        raise ValueError("instrument sigma must be finite and >= 0")
    extra = float(np.sqrt(np.mean((p - c) ** 2)))
    if not np.isfinite(extra):
        extra = 0.0
    if extra_cap is not None:
        extra = min(extra, max(float(extra_cap), 0.0))
    return extra, float(np.sqrt(inst * inst + extra * extra))
