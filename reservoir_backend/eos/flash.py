"""Isothermal PT flash: stability + Rachford–Rice + successive substitution.

Rachford–Rice bisection follows the same bracket as open-DARTS RR2
(ideas only; no import of ``references/``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.pr import R_GAS, PengRobinson, _frac
from reservoir_backend.eos.stability import is_unstable, wilson_k

_RR_EPS = 1.0e-11


@dataclass
class FlashResult:
    vapor_frac: float
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z_liq: float
    z_vap: float
    v_liq: float
    v_vap: float
    two_phase: bool
    k: NDArray[np.float64] | None = None
    converged: bool = True
    iterations: int = 0
    fugacity_error: float = 0.0
    stability_checked: bool = False
    fallback_used: bool = False
    stability_margin: float = 0.0

    @property
    def v_mix(self) -> float:
        v = float(self.vapor_frac)
        return v * self.v_vap + (1.0 - v) * self.v_liq

    @property
    def sl(self) -> float:
        vm = self.v_mix
        if vm <= 0.0:
            return 1.0
        return (1.0 - float(self.vapor_frac)) * self.v_liq / vm

    @property
    def sv(self) -> float:
        return 1.0 - self.sl


def rachford_rice(k: NDArray[np.float64], z: NDArray[np.float64], eps: float = _RR_EPS) -> float:
    """Vapor mole fraction in (1/(1-Kmax), 1/(1-Kmin)). Binary is closed-form."""
    k = np.asarray(k, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    k1 = k - 1.0
    a = 1.0 / (1.0 - float(np.max(k))) + eps
    b = 1.0 / (1.0 - float(np.min(k))) - eps
    if not np.isfinite(a) or not np.isfinite(b):
        return 0.5
    if a > b:
        a, b = b, a
    lo, hi = float(a), float(b)
    if k.size == 2 and abs(float(np.sum(z)) - 1.0) < 1.0e-12:
        den = float(k1[0] * k1[1])
        if abs(den) > 1.0e-18:
            v_bin = -float(z[0] * k1[0] + z[1] * k1[1]) / den
            return float(np.clip(v_bin, lo, hi))
    v = 0.5 * (lo + hi)
    for _ in range(12):
        denom = v * k1 + 1.0
        r = float(np.sum(z * k1 / denom))
        dr = float(-np.sum(z * k1 * k1 / (denom * denom)))
        if abs(r) < 1.0e-12:
            return float(np.clip(v, 0.0, 1.0))
        if abs(dr) < 1.0e-18:
            break
        v_n = v - r / dr
        if v_n <= lo or v_n >= hi:
            break
        v = v_n
    for _ in range(40):
        v = 0.5 * (lo + hi)
        r = float(np.sum(z * k1 / (v * k1 + 1.0)))
        if abs(r) < 1.0e-12:
            break
        if r > 0.0:
            lo = v
        else:
            hi = v
    return float(np.clip(v, 0.0, 1.0))


def _single(eos: PengRobinson, pressure: float, temperature: float, z: NDArray[np.float64], *, vapor: bool) -> FlashResult:
    z = _frac(z, eos.nc)
    zl, zv = eos.z_roots(pressure, temperature, z)
    zz = zv if vapor else zl
    vol = zz * R_GAS * float(temperature) / max(float(pressure), 1.0e-12)
    vfrac = 1.0 if vapor else 0.0
    return FlashResult(
        vapor_frac=vfrac,
        x=z.copy(),
        y=z.copy(),
        z_liq=zl,
        z_vap=zv,
        v_liq=vol if not vapor else eos.molar_volume(pressure, temperature, z, vapor=False),
        v_vap=vol if vapor else eos.molar_volume(pressure, temperature, z, vapor=True),
        two_phase=False,
    )


def flash_tp(
    eos: PengRobinson,
    pressure: float,
    temperature: float,
    z: NDArray[np.float64],
    *,
    max_iter: int = 20,
    tol: float = 1.0e-8,
    k_guess: NDArray[np.float64] | None = None,
    skip_stability: bool = False,
    single_vapor: bool | None = None,
) -> FlashResult:
    """Two-phase PT flash. Reuse ``k_guess``; skip stability when clearly single-phase."""
    z = _frac(z, eos.nc)
    p = float(pressure)
    t = float(temperature)
    if skip_stability and single_vapor is not None:
        fl = _single(eos, p, t, z, vapor=bool(single_vapor))
        fl.k = np.clip(wilson_k(eos, p, t), 1.0e-8, 1.0e8)
        return fl
    if k_guess is not None:
        k = np.clip(np.asarray(k_guess, dtype=float).ravel(), 1.0e-8, 1.0e8)
    else:
        k = np.clip(wilson_k(eos, p, t), 1.0e-8, 1.0e8)
    two_phase_guess = float(np.max(k)) > 1.0 + 1.0e-8 and float(np.min(k)) < 1.0 - 1.0e-8
    v_try = 0.5
    if two_phase_guess:
        v_try = rachford_rice(k, z)
        two_phase_guess = 1.0e-6 < v_try < 1.0 - 1.0e-6
    if not two_phase_guess:
        if skip_stability or not is_unstable(eos, p, t, z):
            vapor = float(np.sum(z * k)) > 1.0 if single_vapor is None else bool(single_vapor)
            fl = _single(eos, p, t, z, vapor=vapor)
            fl.k = k
            return fl

    v = v_try
    x = z.copy()
    y = z.copy()
    err = float("inf")
    n_it = 0
    for n_it in range(1, int(max_iter) + 1):
        if float(np.max(k)) < 1.0 + 1.0e-10 and float(np.min(k)) > 1.0 - 1.0e-10:
            break
        try:
            v = rachford_rice(k, z)
        except Exception:
            break
        x = z / (1.0 + v * (k - 1.0))
        x = _frac(np.maximum(x, 1.0e-16), eos.nc)
        y = k * x
        y = _frac(np.maximum(y, 1.0e-16), eos.nc)
        ln_l = eos.ln_fugacity_coeff(p, t, x, vapor=False)
        ln_v = eos.ln_fugacity_coeff(p, t, y, vapor=True)
        k_new = np.exp(np.clip(ln_l - ln_v, -20.0, 20.0))
        k_new = np.clip(k_new, 1.0e-8, 1.0e8)
        err = float(np.max(np.abs(np.log(np.maximum(k_new, 1.0e-30) / np.maximum(k, 1.0e-30)))))
        if err < float(tol):
            k = k_new
            break
        k = 0.6 * k_new + 0.4 * k

    v = float(np.clip(v, 0.0, 1.0))
    if v <= 1.0e-8:
        fl = _single(eos, p, t, z, vapor=False)
        fl.k = k
        fl.iterations = n_it
        fl.fugacity_error = float(err) if np.isfinite(err) else 0.0
        fl.converged = True
        fl.stability_checked = not skip_stability
        return fl
    if v >= 1.0 - 1.0e-8:
        fl = _single(eos, p, t, z, vapor=True)
        fl.k = k
        fl.iterations = n_it
        fl.fugacity_error = float(err) if np.isfinite(err) else 0.0
        fl.converged = True
        fl.stability_checked = not skip_stability
        return fl
    zl, _ = eos.z_roots(p, t, x)
    _, zv = eos.z_roots(p, t, y)
    v_liq = zl * R_GAS * t / max(p, 1.0e-12)
    v_vap = zv * R_GAS * t / max(p, 1.0e-12)
    ok = bool(np.isfinite(err) and err < float(tol))
    return FlashResult(
        vapor_frac=v,
        x=x,
        y=y,
        z_liq=zl,
        z_vap=zv,
        v_liq=v_liq,
        v_vap=v_vap,
        two_phase=True,
        k=k,
        converged=ok,
        iterations=n_it,
        fugacity_error=float(err) if np.isfinite(err) else 1.0,
        stability_checked=not skip_stability,
    )
