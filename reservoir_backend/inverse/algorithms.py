"""Assimilator variants extracted from references/methods — rewritten, not imported.

- ``es``: dass / Evensen single-step ensemble smoother (α = 1)
- ``esmda``: Emerick & Reynolds 2013, equal α with Σ 1/α_i = 1
- ``esmda_geo``: same budget, geometric α (large → small)
- ``esmda_rs``: pyesmda / Le 2016 restricted-step — next α from current misfit
- ``ies``: Chen–Oliver style damped iterative ES (α = 1+λ, not MDA-normalized)

Localization (Equinor Adaptive/Localized ESMDA) stays off until n_θ is large.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.inverse.ensemble import normalize_alpha_weights

ALGORITHMS = ("es", "esmda", "esmda_geo", "esmda_rs", "ies")


def next_rs_alpha(nrmse: float, remaining_inv: float) -> float:
    """Le / pyesmda: α ≈ max(0.25 S, 1), then spend leftover 1/α so the sum is 1."""
    leftover = max(float(remaining_inv), 1.0e-12)
    s = max(float(nrmse) * float(nrmse), 1.0e-6)
    alpha = max(0.25 * s, 1.0)
    if 1.0 / alpha >= leftover - 1.0e-12:
        return 1.0 / leftover
    return float(alpha)


def geometric_alphas(n_assimilations: int) -> NDArray[np.float64]:
    n = max(int(n_assimilations), 1)
    raw = 2.0 ** np.arange(n, 0, -1)
    return normalize_alpha_weights(raw)


def plan_alphas(algorithm: str, n_assimilations: int) -> NDArray[np.float64] | None:
    """Fixed schedule, or None if the algorithm chooses α after each forecast."""
    name = str(algorithm).strip().lower()
    if name not in ALGORITHMS:
        raise ValueError(f"unknown assimilator {algorithm!r}; choose from {ALGORITHMS}")
    if name == "es":
        return np.array([1.0], dtype=float)
    if name == "esmda":
        return normalize_alpha_weights(int(n_assimilations))
    if name == "esmda_geo":
        return geometric_alphas(int(n_assimilations))
    if name == "ies":
        # Damped Gauss–Newton in ensemble space (Chen & Oliver LM-EnRML idea).
        return np.full(max(int(n_assimilations), 1), 2.0, dtype=float)
    return None
