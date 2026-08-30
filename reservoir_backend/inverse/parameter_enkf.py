"""Online parameter filter. Updates log C_f only; caller reruns F(m).

Forecast (random walk) and analysis are separate. Predicted observations
must come from F of the forecast parameters, not from the previous analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.inverse.esmda import esmda_update


def forecast_parameters(
    members: NDArray[np.float64],
    q_std: float | NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Parameter random walk in log space. ``q_std`` is small by default."""
    x = np.asarray(members, dtype=float)
    sd = np.broadcast_to(np.asarray(q_std, dtype=float).ravel(), (x.shape[0],))
    return x + sd[:, None] * rng.standard_normal(x.shape)


def analysis_parameters(
    members: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """ES update with α = 1. ``predicted`` must be H(F(members))."""
    return esmda_update(members, predicted, observations, sigma, alpha=1.0, rng=rng)


def parameter_random_walk(
    members: NDArray[np.float64],
    q_std: float | NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    return forecast_parameters(members, q_std, rng)


def parameter_enkf_update(
    members: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Analysis only. Forecast first, then recompute predicted, then call this."""
    return analysis_parameters(members, predicted, observations, sigma, rng)
