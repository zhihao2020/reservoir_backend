"""Lab ↔ field comparison. Peripheral to the twin core; not used by invert."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def field_nrmse(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Normalised RMSE between two reconstructed fields of the same shape."""
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"field size {a.size} != {b.size}")
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0e-18)
    return float(np.linalg.norm(a - b) / denom)
