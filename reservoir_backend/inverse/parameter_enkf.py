"""Online parameter ensemble Kalman filter. Updates log C_f only, then rerun F.

V1: one ES step with α = 1 plus a small parameter random walk.
Does not overwrite pressure or saturation fields.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.inverse.esmda import esmda_update


def parameter_random_walk(
    members: NDArray[np.float64],
    q_std: float | NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """C_f(t+1) = C_f(t) + η in log space. ``q_std`` is small by default."""
    x = np.asarray(members, dtype=float)
    sd = np.broadcast_to(np.asarray(q_std, dtype=float).ravel(), (x.shape[0],))
    return x + sd[:, None] * rng.standard_normal(x.shape)


def parameter_enkf_update(
    members: NDArray[np.float64],
    predicted: NDArray[np.float64],
    observations: NDArray[np.float64],
    sigma: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    q_std: float = 0.02,
) -> NDArray[np.float64]:
    """Forecast with random walk, then assimilate (α = 1)."""
    xf = parameter_random_walk(members, q_std, rng)
    return esmda_update(xf, predicted, observations, sigma, alpha=1.0, rng=rng)
