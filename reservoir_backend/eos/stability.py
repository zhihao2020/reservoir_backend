"""Michelsen tangent-plane distance (TPD) stability test.

Michelsen, Fluid Phase Equilibria 9, 1–19 (1982). A trial composition ``w``
is an unstable split of feed ``z`` when

    tpd(w) = Σ_i w_i [ln w_i + ln φ_i(w) − ln z_i − ln φ_i(z)]  <  0.

Vapor-like and liquid-like Wilson trials are iterated to stationarity.
Trivial solutions (w → z) are ignored. Standalone; not a FIM residual.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.peng_robinson import (
    EosMixture,
    _normalize_composition,
    fugacity_coefficients,
)

_TPD_UNSTABLE = -1.0e-8
_TRIVIAL_W = 1.0e-5


@dataclass(frozen=True)
class StabilityResult:
    """Outcome of a two-sided Michelsen TPD test on an EXAMPLE feed."""

    stable: bool
    tpd_min: float
    tpd_vapor_trial: float
    tpd_liquid_trial: float
    w_vapor: NDArray[np.float64]
    w_liquid: NDArray[np.float64]
    trivial_vapor: bool
    trivial_liquid: bool


def tangent_plane_distance(
    z: NDArray[np.float64] | float,
    w: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase_z: str | None = None,
    phase_w: str | None = None,
) -> float:
    """TPD of trial ``w`` against feed ``z`` at ``T`` [K], ``p`` [Pa]."""
    z_arr = _normalize_composition(z, mixture.n_components)
    w_arr = _normalize_composition(w, mixture.n_components)
    phi_z = fugacity_coefficients(z_arr, T, p, mixture, phase=phase_z)
    phi_w = fugacity_coefficients(w_arr, T, p, mixture, phase=phase_w)
    d = np.log(np.clip(z_arr, 1.0e-16, None)) + np.log(phi_z)
    return float(np.dot(w_arr, np.log(np.clip(w_arr, 1.0e-16, None)) + np.log(phi_w) - d))


def _stationary_trial(
    z: NDArray[np.float64],
    d: NDArray[np.float64],
    Y0: NDArray[np.float64],
    T: float,
    p: float,
    mixture: EosMixture,
    phase: str,
    *,
    max_iter: int = 80,
) -> tuple[NDArray[np.float64], float, bool]:
    Y = np.clip(np.asarray(Y0, dtype=float), 1.0e-16, None)
    for _ in range(max_iter):
        w = Y / float(Y.sum())
        phi = fugacity_coefficients(w, T, p, mixture, phase=phase)
        Y_new = np.exp(np.clip(d - np.log(np.clip(phi, 1.0e-30, None)), -40.0, 40.0))
        if float(np.max(np.abs(np.log(Y_new) - np.log(Y)))) < 1.0e-10:
            Y = Y_new
            break
        Y = Y_new
    w = Y / float(Y.sum())
    phi = fugacity_coefficients(w, T, p, mixture, phase=phase)
    tpd = float(np.dot(w, np.log(np.clip(w, 1.0e-16, None)) + np.log(phi) - d))
    trivial = float(np.max(np.abs(w - z))) < _TRIVIAL_W
    return w, tpd, trivial


def michelsen_stability(
    z: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
) -> StabilityResult:
    """Two-sided Michelsen TPD test. ``stable`` if no non-trivial TPD < 0."""
    from reservoir_backend.eos.flash import wilson_k

    z_arr = _normalize_composition(z, mixture.n_components)
    phi_z = fugacity_coefficients(z_arr, T, p, mixture, phase=None)
    d = np.log(np.clip(z_arr, 1.0e-16, None)) + np.log(phi_z)
    K = np.clip(wilson_k(mixture, T, p), 1.0e-8, 1.0e8)
    w_v, tpd_v, triv_v = _stationary_trial(z_arr, d, z_arr * K, T, p, mixture, "vapor")
    w_l, tpd_l, triv_l = _stationary_trial(z_arr, d, z_arr / K, T, p, mixture, "liquid")

    candidates: list[float] = []
    if not triv_v:
        candidates.append(tpd_v)
    if not triv_l:
        candidates.append(tpd_l)
    tpd_min = min(candidates) if candidates else 0.0
    stable = tpd_min >= _TPD_UNSTABLE
    if stable and not candidates:
        tpd_min = 0.0
    return StabilityResult(
        stable=stable,
        tpd_min=float(tpd_min),
        tpd_vapor_trial=float(tpd_v),
        tpd_liquid_trial=float(tpd_l),
        w_vapor=w_v,
        w_liquid=w_l,
        trivial_vapor=triv_v,
        trivial_liquid=triv_l,
    )
