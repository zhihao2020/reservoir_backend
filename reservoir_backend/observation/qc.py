"""Observation QC before ES-MDA. The smoother only sees ACTIVE data."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ObservationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    MISSING_RESPONSE = "MISSING_RESPONSE"
    LOW_ENSEMBLE_SPREAD = "LOW_ENSEMBLE_SPREAD"
    OUTLIER = "OUTLIER"


def classify_observations(
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    sigma: NDArray[np.float64],
    *,
    min_spread: float = 1.0e-12,
    outlier_nsigma: float = 8.0,
) -> NDArray[np.str_]:
    """Return one status per observation (rows of ``predicted``)."""
    y = np.asarray(predicted, dtype=float)
    d = np.asarray(observations, dtype=float).ravel()
    sig = np.asarray(sigma, dtype=float).ravel()
    n_obs = y.shape[0]
    out = np.full(n_obs, ObservationStatus.ACTIVE.value, dtype=object)
    for i in range(n_obs):
        row = y[i]
        if not np.any(np.isfinite(row)):
            out[i] = ObservationStatus.MISSING_RESPONSE.value
            continue
        finite = row[np.isfinite(row)]
        if float(np.std(finite)) < float(min_spread) * max(abs(float(d[i])), 1.0):
            out[i] = ObservationStatus.LOW_ENSEMBLE_SPREAD.value
            continue
        y_mean = float(np.mean(finite))
        if abs(y_mean - float(d[i])) > float(outlier_nsigma) * float(sig[i]):
            out[i] = ObservationStatus.OUTLIER.value
    return out.astype(str)
