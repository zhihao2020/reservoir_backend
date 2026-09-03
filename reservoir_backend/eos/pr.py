"""Peng–Robinson cubic EOS (1976) with PR78 acentric correction.

Structure follows the usual industrial cubic (mixing, Z roots, fugacity).
Product code does not import ``references/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

R_GAS = 8.314462618  # J / (mol K), CODATA
# GEM *OMEGA/*OMEGB defaults (Peng–Robinson). GEOS CubicEOSPhaseModel.hpp
# uses 0.457235529 / 0.077796074; copied here, not imported from references/.
_OMEGA_A = 0.457235530
_OMEGA_B = 0.077796074
_SQRT2 = float(np.sqrt(2.0))


def _kappa(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    w = np.asarray(omega, dtype=float).ravel()
    k = 0.37464 + 1.54226 * w - 0.26992 * w * w
    heavy = w >= 0.49
    if np.any(heavy):
        wh = w[heavy]
        k = k.copy()
        k[heavy] = 0.379642 + 1.48503 * wh - 0.164423 * wh * wh + 0.016666 * wh**3
    return k


def _cbrt(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        return x ** (1.0 / 3.0)
    return -((-x) ** (1.0 / 3.0))


def pr_z_factors(A: float, B: float) -> tuple[float, float]:
    """Return (Z_liquid, Z_vapor). Both > B. Identical when only one real root."""
    A = float(A)
    B = float(B)
    a = -(1.0 - B)
    b = A - 3.0 * B * B - 2.0 * B
    c = -(A * B - B * B - B * B * B)
    aa = a * a
    p = b - aa / 3.0
    q = c + (2.0 * a * aa - 9.0 * a * b) / 27.0
    disc = (q * 0.5) * (q * 0.5) + (p / 3.0) ** 3
    shift = a / 3.0
    roots: list[float] = []
    if disc > 1.0e-16:
        sd = float(np.sqrt(disc))
        roots.append(_cbrt(-0.5 * q + sd) + _cbrt(-0.5 * q - sd) - shift)
    elif disc >= -1.0e-16:
        u = _cbrt(-0.5 * q)
        roots.extend((2.0 * u - shift, -u - shift))
    else:
        amp = float(np.sqrt(max(-p / 3.0, 0.0)))
        if amp <= 0.0:
            roots.append(-shift)
        else:
            arg = float(np.clip((-0.5 * q) / (amp * amp * amp), -1.0, 1.0))
            phi = float(np.arccos(arg))
            for k in range(3):
                roots.append(2.0 * amp * float(np.cos((phi - 2.0 * np.pi * k) / 3.0)) - shift)
    real = np.sort(np.asarray(roots, dtype=float))
    real = real[real > B + 1.0e-12]
    if real.size == 0:
        z = max(B + 1.0e-8, 1.0)
        return z, z
    return float(real[0]), float(real[-1])


@dataclass(frozen=True)
class PengRobinson:
    """Isothermal cubic for ``nc`` components. Criticals are SI (K, Pa)."""

    tc: NDArray[np.float64]
    pc: NDArray[np.float64]
    omega: NDArray[np.float64]
    mw: NDArray[np.float64]
    kij: NDArray[np.float64]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        tc = np.asarray(self.tc, dtype=float).ravel()
        pc = np.asarray(self.pc, dtype=float).ravel()
        omega = np.asarray(self.omega, dtype=float).ravel()
        mw = np.asarray(self.mw, dtype=float).ravel()
        n = tc.size
        if min(n, pc.size, omega.size, mw.size) != n or n < 1:
            raise ValueError("PengRobinson criticals must align")
        if np.any(tc <= 0.0) or np.any(pc <= 0.0) or np.any(mw <= 0.0):
            raise ValueError("Tc, Pc, Mw must be positive")
        kij = np.asarray(self.kij, dtype=float)
        if kij.shape != (n, n):
            raise ValueError("kij must be (nc, nc)")
        object.__setattr__(self, "tc", tc)
        object.__setattr__(self, "pc", pc)
        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "mw", mw)
        object.__setattr__(self, "kij", kij)

    @property
    def nc(self) -> int:
        return int(self.tc.size)

    def _t_pack(self, temperature: float):
        """Isothermal pack: a_i, b_i, a_ij, Wilson T-factor. Independent of p and z."""
        t = float(temperature)
        key = round(t, 6)
        cached = getattr(self, "_ab_cache", None)
        if cached is not None and cached[0] == key:
            return cached
        tr = t / self.tc
        kap = _kappa(self.omega)
        alpha = np.square(1.0 + kap * (1.0 - np.sqrt(np.maximum(tr, 1.0e-12))))
        a = _OMEGA_A * R_GAS**2 * self.tc**2 / self.pc * alpha
        b = _OMEGA_B * R_GAS * self.tc / self.pc
        aij = (1.0 - self.kij) * np.sqrt(np.outer(a, a))
        wilson_t = self.pc * np.exp(5.373 * (1.0 + self.omega) * (1.0 - self.tc / t))
        pack = (key, a, b, aij, wilson_t)
        object.__setattr__(self, "_ab_cache", pack)
        return pack

    def _ab_pure(self, temperature: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pack = self._t_pack(temperature)
        return pack[1], pack[2]

    def mix_ab(self, temperature: float, z: NDArray[np.float64]) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        z = _frac(z, self.nc)
        pack = self._t_pack(temperature)
        a_i, b_i, aij = pack[1], pack[2], pack[3]
        a = float(z @ aij @ z)
        b = float(z @ b_i)
        return a, b, a_i, b_i

    def reduced_ab(self, pressure: float, temperature: float, z: NDArray[np.float64]) -> tuple[float, float, float, float, NDArray[np.float64], NDArray[np.float64]]:
        a, b, a_i, b_i = self.mix_ab(temperature, z)
        p = float(pressure)
        t = float(temperature)
        A = a * p / (R_GAS * R_GAS * t * t)
        B = b * p / (R_GAS * t)
        return A, B, a, b, a_i, b_i

    def z_roots(self, pressure: float, temperature: float, z: NDArray[np.float64]) -> tuple[float, float]:
        A, B, *_ = self.reduced_ab(pressure, temperature, z)
        return pr_z_factors(A, B)

    def molar_volume(self, pressure: float, temperature: float, z: NDArray[np.float64], *, vapor: bool) -> float:
        zl, zv = self.z_roots(pressure, temperature, z)
        zz = zv if vapor else zl
        return zz * R_GAS * float(temperature) / max(float(pressure), 1.0e-12)

    def ln_fugacity_coeff(
        self,
        pressure: float,
        temperature: float,
        z: NDArray[np.float64],
        *,
        vapor: bool,
    ) -> NDArray[np.float64]:
        z = _frac(z, self.nc)
        A, B, a, b, a_i, b_i = self.reduced_ab(pressure, temperature, z)
        zl, zv = pr_z_factors(A, B)
        zz = zv if vapor else zl
        aij = self._t_pack(temperature)[3]
        sum_a = aij @ z
        b = max(b, 1.0e-18)
        B = max(B, 1.0e-18)
        bi_b = b_i / b
        a_term = 2.0 * sum_a / max(a, 1.0e-30) - bi_b
        log_arg = (zz + (1.0 + _SQRT2) * B) / max(zz + (1.0 - _SQRT2) * B, 1.0e-18)
        ln_phi = (
            bi_b * (zz - 1.0)
            - np.log(max(zz - B, 1.0e-18))
            - (A / (2.0 * _SQRT2 * B)) * a_term * np.log(max(log_arg, 1.0e-18))
        )
        return ln_phi

    def fugacity(
        self,
        pressure: float,
        temperature: float,
        z: NDArray[np.float64],
        *,
        vapor: bool,
    ) -> NDArray[np.float64]:
        z = _frac(z, self.nc)
        ln_phi = self.ln_fugacity_coeff(pressure, temperature, z, vapor=vapor)
        return z * float(pressure) * np.exp(np.clip(ln_phi, -50.0, 50.0))


def _frac(z: NDArray[np.float64] | float, nc: int) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=float).ravel()
    if z.size != nc:
        raise ValueError(f"composition size {z.size} != nc {nc}")
    z = np.maximum(z, 0.0)
    s = float(np.sum(z))
    if s <= 0.0:
        raise ValueError("composition sums to 0")
    return z / s
