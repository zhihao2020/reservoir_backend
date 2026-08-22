"""Peng–Robinson (1976) cubic EOS for a multicomponent mixture.

Units (SI, matching the rest of the product):
    T  kelvin
    p  pascal
    Tc kelvin
    Pc pascal
    R  8.314462618 Pa·m³/(mol·K)
    x, y, z  mole fractions

Standalone kernel: not imported by the reservoir residual / FIM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# CODATA 2018 molar gas constant.
GAS_CONSTANT = 8.314462618

# Peng & Robinson, Ind. Eng. Chem. Fundam. 15(1):59–64 (1976).
_OMEGA_A = 0.45724
_OMEGA_B = 0.07780
_KAPPA_0 = 0.37464
_KAPPA_1 = 1.54226
_KAPPA_2 = 0.26992
_SQRT2 = np.sqrt(2.0)


@dataclass(frozen=True)
class EosMixture:
    """Multicomponent PR mixture. ``kij`` is symmetric van der Waals BIP.

    ``marker`` must identify EXAMPLE / public-literature parameters when
    the mixture comes from :mod:`reservoir_backend.eos.example_library`.
    """

    names: tuple[str, ...]
    Tc: NDArray[np.float64]
    Pc: NDArray[np.float64]
    omega: NDArray[np.float64]
    kij: NDArray[np.float64]
    Mw: NDArray[np.float64] | None = None  # kg/mol, EXAMPLE public values when set
    marker: str = ""

    def __post_init__(self) -> None:
        nc = len(self.names)
        object.__setattr__(self, "Tc", np.asarray(self.Tc, dtype=float).ravel())
        object.__setattr__(self, "Pc", np.asarray(self.Pc, dtype=float).ravel())
        object.__setattr__(self, "omega", np.asarray(self.omega, dtype=float).ravel())
        object.__setattr__(self, "kij", np.asarray(self.kij, dtype=float))
        if self.Mw is not None:
            object.__setattr__(self, "Mw", np.asarray(self.Mw, dtype=float).ravel())
        if self.Tc.size != nc or self.Pc.size != nc or self.omega.size != nc:
            raise ValueError("Tc, Pc, omega must match the component list")
        if self.kij.shape != (nc, nc):
            raise ValueError(f"kij must be ({nc}, {nc}), got {self.kij.shape}")
        if self.Mw is not None and self.Mw.size != nc:
            raise ValueError("Mw must match the component list")

    @property
    def n_components(self) -> int:
        return len(self.names)

    def subset(self, names: tuple[str, ...] | list[str]) -> EosMixture:
        idx = [self.names.index(n) for n in names]
        mw = None if self.Mw is None else self.Mw[idx].copy()
        return EosMixture(
            names=tuple(names),
            Tc=self.Tc[idx].copy(),
            Pc=self.Pc[idx].copy(),
            omega=self.omega[idx].copy(),
            kij=self.kij[np.ix_(idx, idx)].copy(),
            Mw=mw,
            marker=self.marker,
        )


def peng_robinson_ab(
    Tc: NDArray[np.float64] | float,
    Pc: NDArray[np.float64] | float,
    omega: NDArray[np.float64] | float,
    T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pure-component ``a(T)`` and ``b``. ``T`` in K, ``Tc`` in K, ``Pc`` in Pa."""
    if T <= 0.0:
        raise ValueError("temperature must be positive (K)")
    tc = np.asarray(Tc, dtype=float)
    pc = np.asarray(Pc, dtype=float)
    w = np.asarray(omega, dtype=float)
    if np.any(tc <= 0.0) or np.any(pc <= 0.0):
        raise ValueError("Tc and Pc must be positive")
    kappa = _KAPPA_0 + _KAPPA_1 * w - _KAPPA_2 * w**2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(T / tc))) ** 2
    a = _OMEGA_A * GAS_CONSTANT**2 * tc**2 / pc * alpha
    b = _OMEGA_B * GAS_CONSTANT * tc / pc
    return a, b


def mix_a_b(
    x: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    kij: NDArray[np.float64],
) -> tuple[float, float, NDArray[np.float64]]:
    """Classic van der Waals mixing: ``a_mix``, ``b_mix``, ``a_ij``."""
    x = np.asarray(x, dtype=float).ravel()
    a_ij = np.sqrt(np.outer(a, a)) * (1.0 - kij)
    a_mix = float(x @ a_ij @ x)
    b_mix = float(x @ b)
    return a_mix, b_mix, a_ij


def reduced_AB(a_mix: float, b_mix: float, T: float, p: float) -> tuple[float, float]:
    """Dimensionless ``A = a p / (R² T²)``, ``B = b p / (R T)``."""
    if T <= 0.0 or p <= 0.0:
        raise ValueError("T (K) and p (Pa) must be positive")
    rt = GAS_CONSTANT * T
    return a_mix * p / (rt * rt), b_mix * p / rt


def compressibility_roots(A: float, B: float) -> NDArray[np.float64]:
    """Physical PR roots: real ``Z > B``, sorted ascending.

    Cubic: ``Z³ − (1−B) Z² + (A − 2B − 3B²) Z − (AB − B² − B³) = 0``.
    """
    c2 = B - 1.0
    c1 = A - 2.0 * B - 3.0 * B**2
    c0 = -(A * B - B**2 - B**3)
    roots = np.roots(np.array([1.0, c2, c1, c0], dtype=float))
    physical: list[float] = []
    for r in roots:
        if abs(float(r.imag)) > 1.0e-10:
            continue
        z = float(r.real)
        if z > B:
            physical.append(z)
    if not physical:
        # Degenerate / round-off: keep a slightly expanded volume.
        physical.append(max(B + 1.0e-12, 1.0e-8))
    return np.array(sorted(physical), dtype=float)


def _g_res_over_rt(Z: float, A: float, B: float) -> float:
    """Residual Gibbs energy over RT (mixture fugacity coefficient, ln φ)."""
    gap = Z - B
    if gap <= 0.0:
        return np.inf
    if B <= 1.0e-14:
        return (Z - 1.0) - np.log(max(Z, 1.0e-30)) - A / max(Z, 1.0e-30)
    arg = (Z + (1.0 + _SQRT2) * B) / (Z + (1.0 - _SQRT2) * B)
    if arg <= 0.0:
        return np.inf
    return (Z - 1.0) - np.log(gap) - (A / (2.0 * _SQRT2 * B)) * np.log(arg)


def select_z(A: float, B: float, phase: str | None = None) -> float:
    """Pick a physical Z. ``phase`` is ``'liquid'``, ``'vapor'``, or Gibbs-min."""
    roots = compressibility_roots(A, B)
    if roots.size == 1:
        return float(roots[0])
    if phase == "liquid":
        return float(roots[0])
    if phase == "vapor":
        return float(roots[-1])
    energies = np.array([_g_res_over_rt(float(z), A, B) for z in roots])
    return float(roots[int(np.argmin(energies))])


def _normalize_composition(x: NDArray[np.float64] | float, nc: int) -> NDArray[np.float64]:
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.size != nc:
        raise ValueError(f"composition size {x_arr.size} != n_components {nc}")
    if np.any(x_arr < -1.0e-15):
        raise ValueError("negative mole fraction")
    x_arr = np.clip(x_arr, 0.0, None)
    total = float(x_arr.sum())
    if total <= 0.0:
        raise ValueError("composition sums to zero")
    return x_arr / total


def mixture_AB(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
    """Return ``A, B, a, b, a_ij, a_mix, b_mix`` for composition ``x``."""
    x_arr = _normalize_composition(x, mixture.n_components)
    a, b = peng_robinson_ab(mixture.Tc, mixture.Pc, mixture.omega, T)
    a_mix, b_mix, a_ij = mix_a_b(x_arr, a, b, mixture.kij)
    A, B = reduced_AB(a_mix, b_mix, T, p)
    return A, B, a, b, a_ij, a_mix, b_mix


def fugacity_coefficients(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
    Z: float | None = None,
) -> NDArray[np.float64]:
    """Component fugacity coefficients ``φ_i`` (dimensionless).

    ``phase`` selects the liquid (smallest) or vapor (largest) root when three
    real roots exist. ``None`` uses the Gibbs-energy minimum. ``Z`` overrides.
    """
    x_arr = _normalize_composition(x, mixture.n_components)
    A, B, _a, b, a_ij, a_mix, b_mix = mixture_AB(x_arr, T, p, mixture)
    z_val = float(Z) if Z is not None else select_z(A, B, phase)
    if z_val <= B:
        z_val = B + 1.0e-12

    if B <= 1.0e-14 or a_mix <= 0.0 or b_mix <= 0.0:
        # Ideal-gas / vanishing-attraction limit: φ_i → 1.
        return np.ones(mixture.n_components, dtype=float)

    sum_aij = a_ij @ x_arr
    bi_over_b = b / b_mix
    log_term = np.log((z_val + (1.0 + _SQRT2) * B) / (z_val + (1.0 - _SQRT2) * B))
    alpha_i = 2.0 * sum_aij / a_mix - bi_over_b
    ln_phi = (
        bi_over_b * (z_val - 1.0)
        - np.log(z_val - B)
        - (A / (2.0 * _SQRT2 * B)) * alpha_i * log_term
    )
    return np.exp(np.clip(ln_phi, -50.0, 50.0))


def ln_fugacity(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
) -> NDArray[np.float64]:
    """``ln f_i = ln φ_i + ln x_i + ln p`` with ``p`` in Pa."""
    x_arr = _normalize_composition(x, mixture.n_components)
    phi = fugacity_coefficients(x_arr, T, p, mixture, phase=phase)
    return np.log(phi) + np.log(np.clip(x_arr, 1.0e-16, None)) + np.log(p)


def compressibility_factor(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
) -> float:
    """PR compressibility ``Z`` for composition ``x`` at ``T`` [K], ``p`` [Pa]."""
    A, B, *_ = mixture_AB(x, T, p, mixture)
    return select_z(A, B, phase)


def molar_volume(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
) -> float:
    """Molar volume ``v = Z R T / p`` in m³/mol."""
    z_val = compressibility_factor(x, T, p, mixture, phase=phase)
    return z_val * GAS_CONSTANT * T / p


def molar_mass(x: NDArray[np.float64] | float, mixture: EosMixture) -> float:
    """Mixture molar mass in kg/mol. Requires EXAMPLE / supplied ``mixture.Mw``."""
    if mixture.Mw is None:
        raise ValueError("mixture.Mw (kg/mol) is required for mass properties")
    x_arr = _normalize_composition(x, mixture.n_components)
    return float(x_arr @ mixture.Mw)


def mass_density(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
) -> float:
    """Mass density ``ρ = M / v`` in kg/m³."""
    vol = molar_volume(x, T, p, mixture, phase=phase)
    if vol <= 0.0 or not np.isfinite(vol):
        raise ValueError("molar volume must be positive and finite")
    return molar_mass(x, mixture) / vol


def phase_zv_rho(
    x: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    phase: str | None = None,
) -> tuple[float, float, float]:
    """Return ``(Z, v [m³/mol], ρ [kg/m³])`` for one phase composition."""
    z_val = compressibility_factor(x, T, p, mixture, phase=phase)
    vol = z_val * GAS_CONSTANT * T / p
    rho = molar_mass(x, mixture) / vol
    return z_val, vol, rho
