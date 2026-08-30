"""Michelsen TPD stability test for two-phase PT flash."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.pr import PengRobinson


def wilson_k(eos: PengRobinson, pressure: float, temperature: float) -> NDArray[np.float64]:
    """Wilson K-value estimate."""
    p = max(float(pressure), 1.0)
    pack = eos._t_pack(temperature)
    return np.asarray(pack[4], dtype=float) / p


def tpd_trial(
    eos: PengRobinson,
    pressure: float,
    temperature: float,
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    ln_phi_z: NDArray[np.float64] | None = None,
) -> float:
    """Tangent-plane distance of trial composition ``y`` vs feed ``z``."""
    z = np.asarray(z, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    y = np.maximum(y, 1.0e-16)
    y = y / float(np.sum(y))
    if ln_phi_z is None:
        ln_phi_z = eos.ln_fugacity_coeff(pressure, temperature, z, vapor=True)
    ln_phi_y = eos.ln_fugacity_coeff(pressure, temperature, y, vapor=True)
    # Prefer the more liquid-like root for a liquid-looking trial.
    zl, zv = eos.z_roots(pressure, temperature, y)
    if zv - zl > 1.0e-6 and float(np.sum(y * eos.tc)) > 0.6 * float(np.sum(z * eos.tc)):
        ln_phi_y = eos.ln_fugacity_coeff(pressure, temperature, y, vapor=False)
    di = np.log(np.maximum(z, 1.0e-16)) + ln_phi_z
    return float(np.sum(y * (np.log(y) + ln_phi_y - di)))


def is_unstable(eos: PengRobinson, pressure: float, temperature: float, z: NDArray[np.float64]) -> bool:
    """True if a two-phase split is indicated."""
    z = np.asarray(z, dtype=float).ravel()
    k = np.clip(wilson_k(eos, pressure, temperature), 1.0e-8, 1.0e8)
    ln_phi_z = eos.ln_fugacity_coeff(pressure, temperature, z, vapor=True)
    trials = [z * k, z / k]
    for w in trials:
        y = np.maximum(w, 1.0e-16)
        y = y / float(np.sum(y))
        for _ in range(4):
            ln_phi_y = eos.ln_fugacity_coeff(pressure, temperature, y, vapor=True)
            y_new = z * np.exp(np.clip(ln_phi_z - ln_phi_y, -20.0, 20.0))
            y_new = np.maximum(y_new, 1.0e-16)
            y_new = y_new / float(np.sum(y_new))
            if float(np.max(np.abs(y_new - y))) < 1.0e-8:
                y = y_new
                break
            y = y_new
        if tpd_trial(eos, pressure, temperature, z, y, ln_phi_z=ln_phi_z) < -1.0e-8:
            return True
    return False
