"""Simple two-phase fractional flow (Corey-like) for water–oil proxy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def water_fractional_flow(
    sw: float | NDArray[np.float64],
    *,
    mu_w: float = 1.0e-3,
    mu_o: float = 5.0e-3,
    swc: float = 0.2,
    sor: float = 0.2,
    nw: float = 2.0,
    no: float = 2.0,
) -> float | NDArray[np.float64]:
    """Brooks–Corey style ``f_w(S_w)`` in [0, 1].

    ``krw = S^nw``, ``kro = (1-S)^no`` with normalized mobile saturation
    ``S = (sw - swc) / (1 - swc - sor)``.
    """
    sw_arr = np.asarray(sw, dtype=float)
    denom = max(1.0 - float(swc) - float(sor), 1.0e-6)
    s = np.clip((sw_arr - float(swc)) / denom, 0.0, 1.0)
    krw = np.power(s, float(nw))
    kro = np.power(1.0 - s, float(no))
    lw = krw / max(float(mu_w), 1.0e-30)
    lo = kro / max(float(mu_o), 1.0e-30)
    fw = lw / np.maximum(lw + lo, 1.0e-30)
    fw = np.clip(fw, 0.0, 1.0)
    if np.isscalar(sw) or (isinstance(sw, float)):
        return float(np.asarray(fw).reshape(-1)[0])
    return fw


def two_phase_relperms(
    sw: float | NDArray[np.float64],
    *,
    swc: float = 0.2,
    sor: float = 0.2,
    nw: float = 2.0,
    no: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Corey ``krw``, ``kro`` on the same saturation normalization as ``f_w``."""
    sw_arr = np.asarray(sw, dtype=float)
    denom = max(1.0 - float(swc) - float(sor), 1.0e-6)
    s = np.clip((sw_arr - float(swc)) / denom, 0.0, 1.0)
    krw = np.power(s, float(nw))
    kro = np.power(1.0 - s, float(no))
    return krw, kro


def total_mobility(
    sw: float | NDArray[np.float64],
    *,
    mu_w: float = 1.0e-3,
    mu_o: float = 5.0e-3,
    swc: float = 0.2,
    sor: float = 0.2,
    nw: float = 2.0,
    no: float = 2.0,
) -> NDArray[np.float64]:
    """Total mobility ``λ_t = krw/μw + kro/μo`` (1 / Pa.s)."""
    krw, kro = two_phase_relperms(sw, swc=swc, sor=sor, nw=nw, no=no)
    lw = krw / max(float(mu_w), 1.0e-30)
    lo = kro / max(float(mu_o), 1.0e-30)
    return np.asarray(lw + lo, dtype=float)
