"""Peng-Robinson EOS for the shale-oil CO2 solubility (Rs_sat).

The GEM deck ``examples/shailoil.dat`` is a full compositional model
(``*MODEL *PR``, 14 components, ``*BIN``). Its CO2 solubility ``Rs_sat(p)`` is the
bubble-point dissolved-gas ratio computed by the EOS, *not* a fitted Henry's-law
slope. This module reproduces that solubility so the forward model's phase split
uses the thermodynamic bound instead of ``rs_slope * p``.

The 14 components and their critical properties are read from the GEM deck's
``*PCRIT`` / ``*TCRIT`` / ``*AC`` / ``*MW``; the binary-interaction coefficients
``*BIN`` have 0.15 for the CO2--C7+ pairs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

_R = 8.314  # J/(mol K)

# GEM deck *COMPNAME order.  (MW g/mol, Tc K, Pc Pa, acentric factor)
_NAMES = ("C1", "N2", "C2", "C3", "CO2", "IC4", "NC4", "IC5", "NC5", "NC6",
          "C7-10", "C11-14", "C15-19", "C20+")
_MW = np.array([16.043, 28.013, 30.07, 44.097, 44.01, 58.124, 58.124, 72.151,
                72.151, 86.178, 114.43, 177.78, 253.63, 350.0])
_TC = np.array([190.56, 126.2, 305.32, 369.83, 304.2, 407.8, 425.2, 460.4, 469.7,
                507.6, 573.45, 685.75, 748.331, 900.0])
_PC = np.array([45.99, 33.94, 48.72, 42.48, 72.8, 36.48, 37.96, 33.8, 33.7, 30.25,
                26.253, 19.987, 12.5544, 8.5]) * 101325.0
_ACENTRIC = np.array([0.011, 0.04, 0.099, 0.152, 0.225, 0.184, 0.201, 0.227, 0.252,
                      0.301, 0.361, 0.534, 0.724, 0.95])
# *ZGLOBALC initial oil composition (mole fractions).
_Z_OIL = np.array([0.35, 0.01, 0.08, 0.06, 0.03, 0.02, 0.04, 0.02, 0.02, 0.04,
                   0.12, 0.10, 0.07, 0.04])
_CO2_IDX = 4

# CO2-free (dead-oil) base composition: the reservoir oil before any injected CO2
# dissolves. The native _Z_OIL carries 0.03 CO2, so a two-component flash built on
# _Z_OIL would flash a 0.03-CO2 mixture even at z=0. The dead-oil base makes the
# injected-CO2 mole fraction zz the true total CO2 fraction of the flashed mixture.
_Z_OIL_DEAD = _Z_OIL.copy()
_Z_OIL_DEAD[_CO2_IDX] = 0.0
_Z_OIL_DEAD = _Z_OIL_DEAD / _Z_OIL_DEAD.sum()

# *BIN: CO2 (idx 4) with the C7+ pseudo-components (idx >= 10) = 0.15.
_BIN = np.zeros((14, 14))
for _i in range(14):
    for _j in range(14):
        if (_i == _CO2_IDX and _j >= 10) or (_j == _CO2_IDX and _i >= 10):
            _BIN[_i, _j] = 0.15

# Standard-condition molar volumes (m3/mol). CO2 at 1 atm / 15.6 C is an ideal
# gas; the oil is the liquid phase, so its molar volume is MW_oil / rho_oil.
# rho_oil_std ~ 800 kg/m3 for a light oil. Rs = (molCO2/molOil) * V_CO2 / V_oil.
_V_CO2_STD = 0.0237
_RHO_OIL_STD = 800.0  # kg/m3
_MW_OIL_DEAD = float(np.sum(_Z_OIL_DEAD * _MW))  # dead-oil MW (~86.6 g/mol, no CO2)

# The *surface* oil molar volume uses the dead-oil MW everywhere: the non-CO2
# moles of any liquid built from _Z_OIL or _Z_OIL_DEAD are dead oil. Reservoir
# phase molar masses and densities come from the flashed compositions
# (see phase_mass_densities).

# Peneloux volume-shift parameters (m3/mol). The PR EOS without a volume
# translation over-predicts the supercritical CO2 molar volume: pure CO2 at
# 20 MPa / 120 C gives Z=0.69 -> 390 kg/m3, while NIST gives ~590 kg/m3 (Z~0.45).
# The shift corrects v = Z*R*T/p - sum(x_i s_i); s_CO2 is set so the CO2-rich gas
# density hits NIST (~590 kg/m3), which the raw PR Z gets backwards (390 < oil
# 535 kg/m3, i.e. the raw EOS says CO2 floats when it actually sinks).
_VOL_SHIFT = np.zeros(14)
_VOL_SHIFT[_CO2_IDX] = 3.818e-5  # m3/mol, calibrated to NIST rho_CO2=590 @ 20MPa/120C

def _ab(T: float):
    """PR ``a`` (Pa m6/mol2) and ``b`` (m3/mol) per component at temperature T."""
    Tr = T / _TC
    kappa = 0.37464 + 1.54226 * _ACENTRIC - 0.26992 * _ACENTRIC ** 2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(Tr))) ** 2
    a = 0.45724 * _R ** 2 * _TC ** 2 / _PC * alpha
    b = 0.07780 * _R * _TC / _PC
    aij = np.sqrt(np.outer(a, a)) * (1.0 - _BIN)
    return aij, b

def _cubic_roots(a0: float, a1: float, a2: float) -> np.ndarray:
    """Real roots of ``Z³ + a2·Z² + a1·Z + a0 = 0`` (Cardano's formula).

    The PR-EOS compressibility cubic has real coefficients and (below the critical
    point) three real roots, so the analytic formula avoids the general
    ``np.roots``/``eigvals`` polynomial solver (the hot spot of the flash).
    """
    p = a1 - a2 ** 2 / 3.0
    q = 2.0 * a2 ** 3 / 27.0 - a2 * a1 / 3.0 + a0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if disc > 0.0:  # one real root
        s = np.sqrt(disc)
        u = np.cbrt(-q / 2.0 + s)
        v = np.cbrt(-q / 2.0 - s)
        return np.array([u + v - a2 / 3.0])
    if disc < 0.0:  # three real roots
        r = 2.0 * np.sqrt(-p / 3.0)
        theta = np.arccos(np.clip(3.0 * q / (2.0 * p) * np.sqrt(-3.0 / p), -1.0, 1.0)) / 3.0
        return np.array([r * np.cos(theta + 2.0 * np.pi * k / 3.0) - a2 / 3.0 for k in range(3)])
    u = np.cbrt(-q / 2.0)  # repeated roots
    return np.array([2.0 * u - a2 / 3.0, -u - a2 / 3.0])

def _z_factor(x, aij, b, P, T, phase):
    """Compressibility factor for composition ``x`` (liq = smallest root)."""
    amix = float(x @ aij @ x)
    bmix = float(x @ b)
    A = amix * P / (_R * T) ** 2
    B = bmix * P / (_R * T)
    roots = _cubic_roots(-(A * B - B ** 2 - B ** 3), A - 3.0 * B ** 2 - 2.0 * B, -(1.0 - B))
    Z = float(np.min(roots) if phase == "liq" else np.max(roots))
    return Z, A, B, amix, bmix

def _fugacity(x, aij, b, P, T, phase):
    """Log-fugacity coefficients ``ln(phi_i)`` for composition ``x``."""
    Z, A, B, amix, bmix = _z_factor(x, aij, b, P, T, phase)
    s = (2.0 * aij @ x) / amix - b / bmix
    return b / bmix * (Z - 1.0) - np.log(Z - B) - A / (2.0 * np.sqrt(2.0) * B) * s * np.log(
        (Z + (1.0 + np.sqrt(2.0)) * B) / (Z + (1.0 - np.sqrt(2.0)) * B)
    )

def _flash(z, P, T, aij=None, b=None, tol: float = 1.0e-5):
    """Rachford-Rice two-phase flash; returns (V, x, y).

    ``aij``/``b`` (the PR EOS parameters, functions of ``T`` only) are optional:
    pass them in to avoid recomputing them for every pressure point of the
    solubility table. ``tol`` is the K-value convergence tolerance; use a tighter
    value (e.g. 1e-8) when the flash feeds a finite-difference Jacobian so the
    derivative noise stays below the perturbation step.
    """
    if aij is None:
        aij, b = _ab(T)
    K = (_PC / P) * np.exp(5.373 * (1.0 + _ACENTRIC) * (1.0 - _TC / T))
    for _ in range(60):
        # solve sum z(K-1)/(1+V(K-1)) = 0 by Newton on V in (0,1)
        V = 0.5
        for _ in range(40):
            f = float(np.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0))))
            if abs(f) < 1.0e-10:
                break  # Rachford-Rice converged
            df = float(np.sum(-z * (K - 1.0) ** 2 / (1.0 + V * (K - 1.0)) ** 2))
            if abs(df) > 1.0e-12:
                V -= f / df
            else:
                V += 0.01 * (1.0 if f > 0.0 else -1.0)
            V = float(np.clip(V, 0.0, 1.0))
        x = z / (1.0 + V * (K - 1.0))
        y = K * x
        Knew = np.exp(_fugacity(x, aij, b, P, T, "liq") - _fugacity(y, aij, b, P, T, "vap"))
        if np.max(np.abs(Knew - K)) < tol:
            break
        K = Knew
    return V, x, y

def _eos_fingerprint() -> str:
    """Checksum of the EOS inputs, so the disk cache is invalidated whenever a
    critical property, composition, binary-interaction coefficient, or standard
    density/volume changes."""
    h = hashlib.sha256()
    for arr in (_MW, _TC, _PC, _ACENTRIC, _Z_OIL, _BIN):
        h.update(np.asarray(arr, dtype=float).tobytes())
    h.update(np.array([_R, _RHO_OIL_STD, _V_CO2_STD], dtype=float).tobytes())
    return h.hexdigest()

# --- two-phase flash table (vapor fraction V, liquid/gas CO2 mole fractions) ---
#
# The compositional phase split (GEM-style) uses the full Peng-Robinson flash
# ``V(z), x_CO2(z), y_CO2(z)`` rather than the black-oil bubble-point solubility
# ``Rs(p)``. At the injector the flash keeps most injected CO2 as a *gas phase*
# (V ≈ 0.86 at the accumulated state z ≈ 0.81), whereas the black-oil Rs dissolves
# it. The flash is per-cell and expensive, so it is tabulated over (p, z) and
# interpolated. ``x_CO2`` is the liquid (oil) phase CO2 mole fraction, ``y_CO2``
# the gas (vapor) phase CO2 mole fraction.
_FLASH_CACHE: dict[bool, tuple[np.ndarray, ...]] = {}
_FLASH_CACHE_FILE = Path(__file__).parent / ".flash_table_cache.npz"
_FLASH_DEAD_CACHE_FILE = Path(__file__).parent / ".flash_table_dead_cache.npz"

def flash_table(
    pmin: float = 15.0e6,
    pmax: float = 22.0e6,
    np_: int = 36,
    zmin: float = 0.0,
    zmax: float = 0.999,
    nz: int = 50,
    T: float = 393.0,
    *,
    dead_oil: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the two-phase flash over a ``(pressure, CO2 mole fraction)`` grid.

    Returns ``(P, Z, V, X, Y)`` where ``V[i, j]`` is the vapor (gas) mole fraction,
    ``X[i, j]`` the liquid (oil) CO2 mole fraction and ``Y[i, j]`` the gas (vapor)
    CO2 mole fraction at ``(P[i], Z[j])``. ``X``/``Y`` are the *overall* CO2 mole
    fraction ``Z`` in the single-phase regions (X = Z for V = 0, Y = Z for V = 1)
    so the composition fields stay continuous across the phase boundary.

    ``dead_oil`` flashes ``Z`` CO2 on the CO2-free ``_Z_OIL_DEAD`` base (the base of
    :func:`flash_direct_volumes`) instead of the native ``_Z_OIL`` (0.03 CO2); the
    two bases put the bubble point at different ``Z``.
    """
    base = _Z_OIL_DEAD if dead_oil else _Z_OIL
    P = np.linspace(pmin, pmax, np_)
    Z = np.linspace(zmin, zmax, nz)
    aij, b = _ab(T)
    V = np.zeros((np_, nz))
    X = np.zeros((np_, nz))
    Y = np.zeros((np_, nz))
    for i, p in enumerate(P):
        for j, zc in enumerate(Z):
            z = (1.0 - zc) * base
            z[_CO2_IDX] += zc
            v, x, y = _flash(z, float(p), T, aij, b)
            V[i, j] = v
            X[i, j] = x[_CO2_IDX] if v > 1.0e-6 else zc
            Y[i, j] = y[_CO2_IDX] if v < 1.0 - 1.0e-6 else zc
    return P, Z, V, X, Y

def _load_flash_table(dead_oil: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cached :func:`flash_table` (native or dead-oil base), persisted next to this module."""
    if dead_oil in _FLASH_CACHE:
        return _FLASH_CACHE[dead_oil]
    path = _FLASH_DEAD_CACHE_FILE if dead_oil else _FLASH_CACHE_FILE
    fingerprint = _eos_fingerprint() + (":dead" if dead_oil else "")
    if path.is_file():
        try:
            z = np.load(path)
            if str(z["fingerprint"]) == fingerprint:
                _FLASH_CACHE[dead_oil] = (z["P"], z["Z"], z["V"], z["X"], z["Y"])
                return _FLASH_CACHE[dead_oil]
        except Exception:
            pass
    P, Z, V, X, Y = flash_table(dead_oil=dead_oil)
    try:
        np.savez(path, P=P, Z=Z, V=V, X=X, Y=Y, fingerprint=np.array(fingerprint))
    except Exception:
        pass
    _FLASH_CACHE[dead_oil] = (P, Z, V, X, Y)
    return _FLASH_CACHE[dead_oil]

# PR EOS parameters for the direct (iterative) flash at the reservoir temperature.
# Building ``aij``/``b`` is the expensive part of a flash, and they depend only on
# ``T``, so cache them here for the per-cell Newton loop in the forward model.
_AB_DIRECT: tuple[tuple[np.ndarray, np.ndarray], float] | None = None

def _cubic_roots_vec(a0, a1, a2):
    """Vectorized cubic roots of ``Z³ + a2·Z² + a1·Z + a0 = 0``.

    Mirrors MRST's ``cubicPositive``: the depressed-cubic parameters
    ``Q = (a²−3b)/9``, ``R = (2a³−9ab+27c)/54`` and the numerically-stable
    single-root branch ``S = −sign(R)·(|R|+√M)^(1/3)``, ``T = Q/S`` with
    ``T(~isfinite)=0``. The naive ``cbrt(−q/2±√disc)`` form suffers catastrophic
    cancellation near the triple root / critical point and produced NaN roots
    (and hence ``log(Z−B)`` NaN) at the phase boundary. Returns ``(n, 3)`` real
    roots sorted ascending, ``NaN`` filling unused slots.
    """
    a0 = np.atleast_1d(np.asarray(a0, dtype=float))
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    a2 = np.atleast_1d(np.asarray(a2, dtype=float))
    n = a0.size
    Q = (a2 ** 2 - 3.0 * a1) / 9.0
    R = (2.0 * a2 ** 3 - 9.0 * a2 * a1 + 27.0 * a0) / 54.0
    M = R ** 2 - Q ** 3
    roots = np.full((n, 3), np.nan)
    neg = M < 0.0
    pos = ~neg
    if neg.any():
        Qn = np.maximum(Q[neg], 0.0)
        theta = np.arccos(np.clip(R[neg] / np.maximum(Qn, 1.0e-300) ** 1.5, -1.0, 1.0))
        sq = 2.0 * np.sqrt(Qn)
        roots[neg, 0] = -sq * np.cos(theta / 3.0) - a2[neg] / 3.0
        roots[neg, 1] = -sq * np.cos((theta + 2.0 * np.pi) / 3.0) - a2[neg] / 3.0
        roots[neg, 2] = -sq * np.cos((theta - 2.0 * np.pi) / 3.0) - a2[neg] / 3.0
    if pos.any():
        Mp = np.maximum(M[pos], 0.0)
        S = -np.sign(R[pos]) * (np.abs(R[pos]) + np.sqrt(Mp)) ** (1.0 / 3.0)
        T = Q[pos] / S
        T = np.where(np.isfinite(T), T, 0.0)
        roots[pos, 0] = S + T - a2[pos] / 3.0
    roots.sort(axis=1)  # ascending, NaN last
    return roots

def _z_factor_vec(x, aij, b, P, T, phase):
    """Vectorized compressibility factor for ``x`` (n×14); liq = smallest root."""
    amix = np.einsum("ij,jk,ik->i", x, aij, x)
    bmix = x @ b
    A = amix * P / (_R * T) ** 2
    B = bmix * P / (_R * T)
    roots = _cubic_roots_vec(-(A * B - B ** 2 - B ** 3), A - 3.0 * B ** 2 - 2.0 * B, -(1.0 - B))
    Z = np.nanmin(roots, axis=1) if phase == "liq" else np.nanmax(roots, axis=1)
    return Z, A, B, amix, bmix

def _fugacity_vec(x, aij, b, P, T, phase):
    """Vectorized log-fugacity coefficients ``ln(phi_i)`` for ``x`` (n×14)."""
    Z, A, B, amix, bmix = _z_factor_vec(x, aij, b, P, T, phase)
    s = (2.0 * (x @ aij)) / amix[:, None] - b[None, :] / bmix[:, None]
    Zc = Z[:, None]
    Bc = B[:, None]
    Ac = A[:, None]
    return (b[None, :] / bmix[:, None]) * (Zc - 1.0) - np.log(Zc - Bc) - Ac / (
        2.0 * np.sqrt(2.0) * Bc
    ) * s * np.log((Zc + (1.0 + np.sqrt(2.0)) * Bc) / (Zc + (1.0 - np.sqrt(2.0)) * Bc))

def phase_molar_volumes(x, y, P, T=393.0):
    """EOS phase molar volumes ``(v_l, v_g)`` (m3/mol) with the Peneloux shift.

    ``x``/``y`` are the (n,14) liquid/vapor mole-fraction matrices returned by the
    flash; ``P`` the per-cell pressure (Pa). Each phase molar volume is
    ``v = Z(P,T,comp)*R*T/P - sum(x_i s_i)``, so the gas/liquid densities follow the
    EOS compressibility (with the volume shift fixing the supercritical-CO2 density).
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    Z_l = _z_factor_vec(x, aij, b, P_arr, T, "liq")[0]
    Z_g = _z_factor_vec(y, aij, b, P_arr, T, "vap")[0]
    v_l = Z_l * _R * T / P_arr - x @ _VOL_SHIFT
    v_g = Z_g * _R * T / P_arr - y @ _VOL_SHIFT
    return v_l, v_g


def phase_molar_volume_deriv(x, P, T=393.0, phase="liq", *, aij=None, b=None):
    """Peneloux-shifted molar volume ``v`` and its derivatives ``(∂v/∂x_j, ∂v/∂p)``.

    ``v = Z·RT/p − Σ x_i s_i``, so ``∂v/∂x_j = (∂Z/∂x_j)·RT/p − s_j`` and
    ``∂v/∂p = (∂Z/∂p)·RT/p − Z·RT/p²``. The cubic-root derivatives reuse the
    ``∂Z/∂x_j = −(∂F/∂x_j)/(∂F/∂Z)`` of :func:`_fugacity_frac_deriv_analytic`
    (with ``∂Z/∂p = −(∂F/∂p)/(∂F/∂Z)`` added). Returns ``(v (n,), dvdx (n, ncomp),
    dvdp (n,))`` — the building block for the analytic accumulation Jacobian.
    """
    global _AB_DIRECT
    if aij is None or b is None:
        if _AB_DIRECT is None or _AB_DIRECT[1] != T:
            _AB_DIRECT = (_ab(T), T)
        aij, b = _AB_DIRECT[0]
    x = np.atleast_2d(np.asarray(x, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float)).astype(float)
    Z, A, B, amix, bmix = _z_factor_vec(x, aij, b, P, T, phase)
    n, ncomp = x.shape
    psi = x @ aij
    RT = _R * T
    Fz = 3.0 * Z ** 2 + 2.0 * (B - 1.0) * Z + (A - 3.0 * B ** 2 - 2.0 * B)
    dFdx = ((Z - B)[:, None] * (2.0 * psi) * P[:, None] / (RT * RT)
            + (Z ** 2 - (6.0 * B + 2.0) * Z - A + 2.0 * B + 3.0 * B ** 2)[:, None]
            * b[None, :] * P[:, None] / RT)
    dZdx = -dFdx / Fz[:, None]
    dFdp = ((Z - B) * A / P
            + (Z ** 2 - (6.0 * B + 2.0) * Z - A + 2.0 * B + 3.0 * B ** 2) * B / P)
    dZdp = -dFdp / Fz
    v = Z * RT / P - x @ _VOL_SHIFT
    dvdx = dZdx * RT / P[:, None] - _VOL_SHIFT[None, :]
    dvdp = dZdp * RT / P - Z * RT / (P ** 2)
    return v, dvdx, dvdp

def phase_mass_densities(x, y, v_l, v_g):
    """EOS phase mass densities ``(rho_l, rho_g)`` (kg/m3): ``(x·MW)/v`` per phase.

    ``x``/``y`` are the (n,14) flashed phase compositions and ``v_l``/``v_g`` their
    :func:`phase_molar_volumes`. An absent phase (zero composition row, ``v = 0``)
    gets density 0; callers weight densities by saturation.
    """
    m_l = np.atleast_2d(x) @ _MW / 1000.0  # kg/mol
    m_g = np.atleast_2d(y) @ _MW / 1000.0
    v_l = np.atleast_1d(np.asarray(v_l, dtype=float))
    v_g = np.atleast_1d(np.asarray(v_g, dtype=float))
    rho_l = np.divide(m_l, v_l, out=np.zeros_like(v_l), where=v_l > 1.0e-30)
    rho_g = np.divide(m_g, v_g, out=np.zeros_like(v_g), where=v_g > 1.0e-30)
    return rho_l, rho_g

def co2_molar_volume(P, T=393.0):
    """Pure-CO2 EOS molar volume (m3/mol, Peneloux-shifted) at pressure ``P`` (Pa).

    The reservoir molar volume of the injected CO2, i.e. the EOS counterpart of
    ``Bg * _V_CO2_STD``.
    """
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    pure = np.zeros((P_arr.size, _MW.size))
    pure[:, _CO2_IDX] = 1.0
    return phase_molar_volumes(pure, pure, P_arr, T)[1]

def _flash_vec(z_cells, P_cells, T, aij, b, tol=1.0e-5):
    """Vectorized Rachford-Rice flash over cells; returns ``(V, x, y)``.

    ``z_cells`` is an ``(n, 14)`` overall mole-fraction matrix and ``P_cells`` the
    per-cell pressures; the K-value (Wilson init) / Rachford-Rice / fugacity update
    loops run once for the whole cell batch, so the per-cell cost drops from the
    scalar flash's Python-loop overhead to vectorized numpy.
    """
    n = P_cells.size
    K = (_PC[None, :] / P_cells[:, None]) * np.exp(
        5.373 * (1.0 + _ACENTRIC)[None, :] * (1.0 - _TC[None, :] / T)
    )
    for _ in range(60):
        V = np.full(n, 0.5)
        for _ in range(40):
            denom = 1.0 + V[:, None] * (K - 1.0)
            f = np.sum(z_cells * (K - 1.0) / denom, axis=1)
            # Converged: |f|≈0 (two-phase root), or V pinned at a boundary with f
            # pointing outward — single-phase liquid (V=0, f≤0) / vapor (V=1, f≥0)
            # has no Rachford-Rice root inside (0,1), so it would otherwise spin the
            # full 40 iterations at the boundary.
            conv = ((np.abs(f) < 1.0e-10)
                    | ((V <= 1.0e-12) & (f <= 0.0))
                    | ((V >= 1.0 - 1.0e-12) & (f >= 0.0)))
            if conv.all():
                break
            df = np.sum(-z_cells * (K - 1.0) ** 2 / denom ** 2, axis=1)
            step = np.where(np.abs(df) > 1.0e-12, f / np.where(np.abs(df) > 1.0e-12, df, 1.0),
                            np.where(f > 0.0, 0.01, -0.01))
            V = np.clip(V - step, 0.0, 1.0)
        x = z_cells / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        Knew = np.exp(_fugacity_vec(x, aij, b, P_cells, T, "liq")
                      - _fugacity_vec(y, aij, b, P_cells, T, "vap"))
        if np.max(np.abs(Knew - K)) < tol:
            break
        K = Knew
    return _collapse_trivial_flash(V, x, y, z_cells, P_cells, T, aij, b)

def _collapse_trivial_flash(V, x, y, z_cells, P_cells, T, aij, b, gap: float = 1.0e-4):
    """Replace the ``x = y = z`` trivial root with the stable single phase.

    Near-pure CO2 (and any other mixture the successive-substitution flash does
    not split) converges to ``x = y`` with an arbitrary vapor fraction. The two
    phases then have the same composition, so the lower Gibbs energy
    ``sum z_i ln phi_i`` picks liquid (``V = 0``) or vapor (``V = 1``). A real
    two-phase split has ``max|y-x|`` well above ``gap`` and is left alone.
    """
    diff = np.max(np.abs(y - x), axis=1)
    triv = (diff < gap) & (V > 1.0e-8) & (V < 1.0 - 1.0e-8)
    if not np.any(triv):
        return V, x, y
    z_t = z_cells[triv]
    p_t = P_cells[triv]
    g_liq = np.sum(z_t * _fugacity_vec(z_t, aij, b, p_t, T, "liq"), axis=1)
    g_vap = np.sum(z_t * _fugacity_vec(z_t, aij, b, p_t, T, "vap"), axis=1)
    # An invalid compressibility root (Z <= B) makes ln(phi) non-finite; keep
    # the phase that still has a Gibbs energy, and the vapor when it is lower
    # (supercritical CO2).
    both = np.isfinite(g_liq) & np.isfinite(g_vap)
    vapor = np.where(both, g_vap <= g_liq, np.isfinite(g_vap) & ~np.isfinite(g_liq))
    vapor = np.where(np.isfinite(g_liq) | np.isfinite(g_vap), vapor, True)
    idx = np.flatnonzero(triv)
    iv = idx[vapor]
    il = idx[~vapor]
    V = np.array(V, dtype=float, copy=True)
    x = np.array(x, dtype=float, copy=True)
    y = np.array(y, dtype=float, copy=True)
    V[iv] = 1.0
    y[iv] = z_cells[iv]
    x[iv] = 0.0
    V[il] = 0.0
    x[il] = z_cells[il]
    y[il] = 0.0
    return V, x, y

def flash_direct(
    pressure: float | np.ndarray,
    z_co2: float | np.ndarray,
    T: float = 393.0,
    tol: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterative Rachford-Rice flash (no interpolation) → ``(V, x_CO2, y_CO2)``.

    Evaluates the full Peng-Robinson two-phase flash at every requested
    ``(pressure, z_co2)`` point instead of interpolating the precomputed table, so
    the phase split has no interpolation kink at the bubble point — the formulation
    CMG/OPM use. ``pressure`` (Pa) and ``z_co2`` (overall CO2 mole fraction in the
    hydrocarbon) are broadcast against each other; returns scalars only when both
    inputs are scalars.

    The K-value tolerance ``tol`` (default 1e-5) is tight enough that the vapor
    fraction ``V`` is smooth to well below the 1e-6 finite-difference step the
    forward model uses to build its Jacobian, so the derivatives stay clean without
    the extra outer iterations a 1e-8 tolerance would cost.
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    scalar = np.ndim(pressure) == 0 and np.ndim(z_co2) == 0
    p_arr = np.atleast_1d(np.asarray(pressure, dtype=float))
    z_arr = np.atleast_1d(np.asarray(z_co2, dtype=float))
    n = max(p_arr.size, z_arr.size)
    pp = np.broadcast_to(p_arr if p_arr.size == n else p_arr[0], (n,)).astype(float)
    zz = np.broadcast_to(z_arr if z_arr.size == n else z_arr[0], (n,)).astype(float)
    z_cells = np.outer(1.0 - zz, _Z_OIL)
    z_cells[:, _CO2_IDX] += zz
    # Phase classification from the cached table (bilinear V is good enough to say
    # *whether* a phase exists), then run the exact iterative flash only on the
    # two-phase cells. This keeps the full-grid cost near the table look-up while
    # the two-phase cells (the only ones with a nontrivial V/x/y) get the exact
    # smooth flash — no interpolation kink where it matters.
    P, Z, Vtab, _, _ = _load_flash_table()
    ip = np.clip(np.searchsorted(P, pp, side="right") - 1, 0, len(P) - 2)
    iz = np.clip(np.searchsorted(Z, zz, side="right") - 1, 0, len(Z) - 2)
    tp = (pp - P[ip]) / (P[ip + 1] - P[ip])
    tz = (zz - Z[iz]) / (Z[iz + 1] - Z[iz])
    w00 = (1.0 - tp) * (1.0 - tz)
    w10 = tp * (1.0 - tz)
    w01 = (1.0 - tp) * tz
    w11 = tp * tz
    Vclass = (Vtab[ip, iz] * w00 + Vtab[ip + 1, iz] * w10
              + Vtab[ip, iz + 1] * w01 + Vtab[ip + 1, iz + 1] * w11)
    liquid = Vclass <= 1.0e-8
    vapor = Vclass >= 1.0 - 1.0e-8
    two_phase = ~(liquid | vapor)
    V = np.zeros(n)
    x = np.zeros((n, _Z_OIL.size))
    y = np.zeros((n, _Z_OIL.size))
    V[vapor] = 1.0
    x[liquid] = z_cells[liquid]
    y[vapor] = z_cells[vapor]
    if two_phase.any():
        V[two_phase], x[two_phase], y[two_phase] = _flash_vec(
            z_cells[two_phase], pp[two_phase], T, aij, b, tol=tol)
    x_co2 = np.where(V > 1.0e-6, x[:, _CO2_IDX], zz)
    y_co2 = np.where(V < 1.0 - 1.0e-6, y[:, _CO2_IDX], zz)
    if scalar:
        return float(V[0]), float(x_co2[0]), float(y_co2[0])
    return V, x_co2, y_co2

def flash_direct_volumes(
    pressure: float | np.ndarray,
    z_co2: float | np.ndarray,
    T: float = 393.0,
    tol: float = 1.0e-5,
    *,
    with_density: bool = False,
) -> tuple[np.ndarray, ...]:
    """Dead-oil two-component flash + EOS phase molar volumes.

    Returns ``(V, x_co2, y_co2, v_l, v_g)``. Unlike :func:`flash_direct` (which
    flashes on the native 0.03-CO2 ``_Z_OIL``), this flashes on the CO2-free
    ``_Z_OIL_DEAD`` base so ``z_co2`` is the true total CO2 mole fraction, and it
    also returns the EOS (Peneloux-shifted) phase molar volumes ``v_l``/``v_g`` for
    the reservoir volume balance and buoyancy (see :func:`phase_molar_volumes`).
    ``with_density`` appends the phase mass densities ``(rho_l, rho_g)`` (kg/m3)
    of the full 14-component phase compositions (:func:`phase_mass_densities`).
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    scalar = np.ndim(pressure) == 0 and np.ndim(z_co2) == 0
    p_arr = np.atleast_1d(np.asarray(pressure, dtype=float))
    z_arr = np.atleast_1d(np.asarray(z_co2, dtype=float))
    n = max(p_arr.size, z_arr.size)
    pp = np.broadcast_to(p_arr if p_arr.size == n else p_arr[0], (n,)).astype(float)
    zz = np.broadcast_to(z_arr if z_arr.size == n else z_arr[0], (n,)).astype(float)
    z_cells = np.outer(1.0 - zz, _Z_OIL_DEAD)
    z_cells[:, _CO2_IDX] += zz
    # Phase classification from the cached *dead-oil* table (the native-oil table
    # puts the bubble point at a different z), then the exact iterative flash only
    # on the two-phase cells.
    P, Z, Vtab, _, _ = _load_flash_table(dead_oil=True)
    ip = np.clip(np.searchsorted(P, pp, side="right") - 1, 0, len(P) - 2)
    iz = np.clip(np.searchsorted(Z, zz, side="right") - 1, 0, len(Z) - 2)
    tp = (pp - P[ip]) / (P[ip + 1] - P[ip])
    tz = (zz - Z[iz]) / (Z[iz + 1] - Z[iz])
    w00 = (1.0 - tp) * (1.0 - tz)
    w10 = tp * (1.0 - tz)
    w01 = (1.0 - tp) * tz
    w11 = tp * tz
    Vclass = (Vtab[ip, iz] * w00 + Vtab[ip + 1, iz] * w10
              + Vtab[ip, iz + 1] * w01 + Vtab[ip + 1, iz + 1] * w11)
    liquid = Vclass <= 1.0e-8
    vapor = Vclass >= 1.0 - 1.0e-8
    two_phase = ~(liquid | vapor)
    V = np.zeros(n)
    x = np.zeros((n, _Z_OIL.size))
    y = np.zeros((n, _Z_OIL.size))
    V[vapor] = 1.0
    x[liquid] = z_cells[liquid]
    y[vapor] = z_cells[vapor]
    if two_phase.any():
        V[two_phase], x[two_phase], y[two_phase] = _flash_vec(
            z_cells[two_phase], pp[two_phase], T, aij, b, tol=tol)
    x_co2 = np.where(V > 1.0e-6, x[:, _CO2_IDX], zz)
    y_co2 = np.where(V < 1.0 - 1.0e-6, y[:, _CO2_IDX], zz)
    v_l, v_g = phase_molar_volumes(x, y, pp, T)
    out = (V, x_co2, y_co2, v_l, v_g)
    if with_density:
        out = out + phase_mass_densities(x, y, v_l, v_g)
    if scalar:
        return tuple(float(a[0]) for a in out)
    return out

def flash_direct_full(
    z_full: np.ndarray,
    pressure: float | np.ndarray,
    T: float = 393.0,
    tol: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full 14-component flash → ``(V, x, y)``.

    ``z_full`` is an ``(n, 14)`` overall mole-fraction matrix; returns the vapor
    mole fraction ``V`` (n,) and the liquid/gas compositions ``x``/``y`` (n, 14).

    The forward model's composition stays on the CO2-on-dead-oil mixing line
    (initial ``_Z_OIL_DEAD`` + injected CO2), so the cached dead-oil flash table
    classifies each cell as single/two-phase from its CO2 mole fraction alone.
    Only the two-phase cells run the iterative flash; the single-phase cells skip
    the ~60 fugacity iterations (the dominant cost of a full-grid flash). Cells
    whose composition leaves the mixing line still flash correctly because the
    table's single-phase bounds (V ≈ 0 or 1) are only used to short-circuit cells
    that are unequivocally single-phase.
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    z = np.atleast_2d(np.asarray(z_full, dtype=float))
    pp = np.atleast_1d(np.asarray(pressure, dtype=float)).astype(float)
    if pp.size == 1:
        pp = np.full(z.shape[0], float(pp[0]))
    n = z.shape[0]
    zz = z[:, _CO2_IDX]
    # Phase classification from the cached dead-oil table (bilinear V over (p, z_CO2)),
    # then run the exact iterative flash only on the two-phase cells.
    P, Z, Vtab, _, _ = _load_flash_table(dead_oil=True)
    ip = np.clip(np.searchsorted(P, pp, side="right") - 1, 0, len(P) - 2)
    iz = np.clip(np.searchsorted(Z, zz, side="right") - 1, 0, len(Z) - 2)
    tp = (pp - P[ip]) / (P[ip + 1] - P[ip])
    tz = (zz - Z[iz]) / (Z[iz + 1] - Z[iz])
    w00 = (1.0 - tp) * (1.0 - tz)
    w10 = tp * (1.0 - tz)
    w01 = (1.0 - tp) * tz
    w11 = tp * tz
    Vclass = (Vtab[ip, iz] * w00 + Vtab[ip + 1, iz] * w10
              + Vtab[ip, iz + 1] * w01 + Vtab[ip + 1, iz + 1] * w11)
    liquid = Vclass <= 1.0e-8
    vapor = Vclass >= 1.0 - 1.0e-8
    two_phase = ~(liquid | vapor)
    V = np.zeros(n)
    x = np.zeros((n, _Z_OIL.size))
    y = np.zeros((n, _Z_OIL.size))
    V[vapor] = 1.0
    x[liquid] = z[liquid]
    y[vapor] = z[vapor]
    if two_phase.any():
        V[two_phase], x[two_phase], y[two_phase] = _flash_vec(
            z[two_phase], pp[two_phase], T, aij, b, tol=tol)
    return V, x, y


def _stability_check(z, K, p, T, aij, b, is_vapor_test, tol_equil, tol_trivial, max_iter):
    """Successive substitution for one Michelsen-TPD stationary point.

    ``is_vapor_test``: reference ``z`` is liquid, trial ``w`` is vapor, update
    ``Y_i = z_i·φ^L_i(z)/φ^V_i(w)``. Otherwise reference is vapor, trial liquid,
    ``X_i = z_i·φ^V_i(z)/φ^L_i(w)``. Returns ``(w, S)`` with ``S = ΣY``; ``S > 1``
    means the incipient phase is present (the mixture splits).
    """
    n, ncomp = z.shape
    phase_w = "vap" if is_vapor_test else "liq"
    lnphi_z = _fugacity_vec(z, aij, b, p, T, "liq" if is_vapor_test else "vap")
    if is_vapor_test:
        w = K * z
    else:
        w = z / np.maximum(K, 1.0e-30)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-300)
    S = np.ones(n)
    for _ in range(max_iter):
        lnphi_w = _fugacity_vec(w, aij, b, p, T, phase_w)
        if not np.isfinite(lnphi_w).all():
            break  # invalid trial composition (cubic root singular)
        R = np.exp(lnphi_z - lnphi_w)
        Y = z * R
        S = Y.sum(axis=1)
        w_new = Y / np.maximum(S, 1.0e-300)[:, None]
        if np.all(np.max(np.abs(w_new - w), axis=1) < tol_equil):
            w = w_new
            break
        w = w_new
    return w, S


def phase_stability_test(z, p, T=393.0, *, aij=None, b=None,
                         tol_equil=1.0e-10, tol_trivial=1.0e-5, max_iter=200):
    """Michelsen tangent-plane-distance phase stability test (MRST ``phaseStabilityTest``).

    Determines whether a single-phase mixture ``z`` stays single-phase or splits, by
    minimising ``TPD(w) = Σ w_i(ln w_i + ln φ_i(w) − ln z_i − ln φ_i(z))``. Two
    successive-substitution passes locate the vapor-like and liquid-like stationary
    points; a phase is present when its ``S = Σ w_i > 1``. Returns ``(stable (n,),
    x (n,ncomp), y (n,ncomp))`` — ``x = y = z`` for stable cells, otherwise the
    incipient phase compositions. This is the single→two-phase trigger in MRST's
    ``flashPhases`` (via ``performPhaseStabilityTest``).
    """
    global _AB_DIRECT
    if aij is None or b is None:
        if _AB_DIRECT is None or _AB_DIRECT[1] != T:
            _AB_DIRECT = (_ab(T), T)
        aij, b = _AB_DIRECT[0]
    z = np.atleast_2d(np.asarray(z, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float)).astype(float)
    if p.size == 1:
        p = np.full(z.shape[0], float(p[0]))
    z = np.maximum(z, 1.0e-300)
    z = z / z.sum(axis=1, keepdims=True)
    K = (_PC[None, :] / p[:, None]) * np.exp(
        5.373 * (1.0 + _ACENTRIC)[None, :] * (1.0 - _TC[None, :] / T)
    )
    y, S_v = _stability_check(z, K, p, T, aij, b, True, tol_equil, tol_trivial, max_iter)
    x, S_l = _stability_check(z, K, p, T, aij, b, False, tol_equil, tol_trivial, max_iter)
    stable = (S_v <= 1.0 + tol_trivial) & (S_l <= 1.0 + tol_trivial)
    x = np.where(stable[:, None], z, x)
    y = np.where(stable[:, None], z, y)
    return stable, x, y


def _fugacity_frac_deriv_analytic(x, aij, b, P, T, phase):
    """Analytic unconstrained mole-fraction derivative ``∂ln φ_i/∂x_j``.

    Closed-form derivative of the PR fugacity coefficient, differentiated with
    every ``x_j`` an independent variable (``Σx`` free). This is the same
    derivative MRST's AD backend produces in ``getPhaseFractionDerivativesPTZ``,
    but in closed form: the ``ncomp²`` entries come from one pass over the
    EOS state (``Z, ψ, a, b``). Verified against central differences to ~1e-8.

    ``ln φ_i`` depends on ``x`` through ``Z`` (via the cubic), ``a``/``b``, and
    ``ψ_i = Σ_j a_ij x_j``; the total derivative is an explicit part at fixed ``Z``
    plus the implicit part through ``∂Z/∂x_j``:

        ``∂lnφ_i/∂x_j = g_ij + (∂lnφ_i/∂Z)·(∂Z/∂x_j)``,
        ``∂Z/∂x_j = −(∂F/∂x_j)/(∂F/∂Z)``  with ``F(Z) = 0`` the PR cubic.
    """
    Z, A, B, amix, bmix = _z_factor_vec(x, aij, b, P, T, phase)
    n, ncomp = x.shape
    psi = x @ aij  # partial a: psi_i = sum_j a_ij x_j
    sq2 = np.sqrt(2.0)
    d1 = 1.0 + sq2
    d2 = 1.0 - sq2
    RT = _R * T
    P_RT = P / RT
    P_RT2 = P / (RT * RT)
    S = 2.0 * psi / amix[:, None] - b[None, :] / bmix[:, None]
    L = np.log((Z + d1 * B) / (Z + d2 * B))
    # ∂lnφ_i/∂Z at fixed (a, b, psi)
    dLdZ = 1.0 / (Z + d1 * B) - 1.0 / (Z + d2 * B)
    dlnphi_dZ = (b[None, :] / bmix[:, None] - 1.0 / (Z[:, None] - B[:, None])
                 - (A / (2.0 * sq2 * B))[:, None] * S * dLdZ[:, None])
    # ∂Z/∂x_j from the cubic F(Z) = Z³ + (B−1)Z² + (A−3B²−2B)Z − (AB−B²−B³) = 0
    Fz = 3.0 * Z ** 2 + 2.0 * (B - 1.0) * Z + (A - 3.0 * B ** 2 - 2.0 * B)
    dFdx = ((Z - B)[:, None] * (2.0 * psi) * P_RT2[:, None]
            + (Z ** 2 - (6.0 * B + 2.0) * Z - A + 2.0 * B + 3.0 * B ** 2)[:, None]
            * b[None, :] * P_RT[:, None])
    dZdx = -dFdx / Fz[:, None]
    # explicit part at fixed Z
    C = amix / (2.0 * sq2 * bmix * RT)  # = A/(2√2 B)
    dCdx = (1.0 / (2.0 * sq2 * RT)) * (
        2.0 * psi / bmix[:, None] - amix[:, None] * b[None, :] / bmix[:, None] ** 2)
    dSdx = (2.0 * aij[None, :, :] / amix[:, None, None]
            - 4.0 * psi[:, :, None] * psi[:, None, :] / (amix[:, None, None] ** 2)
            + b[None, :, None] * b[None, None, :] / (bmix[:, None, None] ** 2))
    dterm1 = -(b[None, :, None] * (Z - 1.0)[:, None, None] * b[None, None, :]) / (bmix[:, None, None] ** 2)
    dterm2 = b[None, None, :] * P_RT[:, None, None] / (Z - B)[:, None, None]
    dLdx = (d1 / (Z + d1 * B) - d2 / (Z + d2 * B))[:, None] * b[None, :] * P_RT[:, None]
    dterm3 = (-dCdx[:, None, :] * S[:, :, None] * L[:, None, None]
              - C[:, None, None] * dSdx * L[:, None, None]
              - C[:, None, None] * S[:, :, None] * dLdx[:, None, :])
    return dterm1 + dterm2 + dterm3 + dlnphi_dZ[:, :, None] * dZdx[:, None, :]


def fugacity_mole_deriv(
    x: np.ndarray,
    P: float | np.ndarray,
    T: float = 393.0,
    phase: str = "liq",
    h: float = 1.0e-6,
) -> np.ndarray:
    """``d(ln φ_i)/d(ln n_j)`` — mole-number derivative of the log-fugacity coefficient.

    Derived from the unconstrained mole-fraction derivative
    :func:`_fugacity_frac_deriv_analytic` via the chain rule
    ``d lnφ_i/d ln n_j = x_j·(∂lnφ_i/∂x_j − G_i)`` with the radial gauge
    ``G_i = Σ_k x_k ∂lnφ_i/∂x_k``. This satisfies the Euler identity
    ``Σ_j d lnφ_i/d ln n_j = 0`` exactly (the fugacity coefficient is degree-0
    homogeneous in the mole numbers). ``x`` is an ``(n, ncomp)`` mole-fraction
    matrix; returns ``(n, ncomp, ncomp)`` with ``out[:, i, j] = d lnφ_i/d ln n_j``.
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    x = np.atleast_2d(np.asarray(x, dtype=float))
    dgdx = _fugacity_frac_deriv_analytic(x, aij, b, P, T, phase)
    G = np.sum(x[:, None, :] * dgdx, axis=2)
    return x[:, None, :] * (dgdx - G[:, :, None])


def _natural_variables_residual(L, x, y, z, P, T, aij, b):
    """Equilibrium residual from precomputed ``aij``/``b`` (see :func:`natural_variables_residual`)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    z = np.atleast_2d(np.asarray(z, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float)).astype(float)
    L = np.asarray(L, dtype=float)
    fL = _fugacity_vec(x, aij, b, P, T, "liq")
    fV = _fugacity_vec(y, aij, b, P, T, "vap")
    mass = L[:, None] * x + (1.0 - L)[:, None] * y - z
    fug = (np.log(np.maximum(y, 1.0e-300)) + fV) - (np.log(np.maximum(x, 1.0e-300)) + fL)
    close = (x - y).sum(axis=1)
    return np.concatenate([mass, fug, close[:, None]], axis=1)


def natural_variables_residual(
    L: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    P: float | np.ndarray,
    T: float = 393.0,
) -> np.ndarray:
    """Equilibrium residual of the natural-variables two-phase split.

    Returns the ``2·ncomp + 1`` residuals that vanish at the ``(L, x, y)``
    equilibrium for overall composition ``z`` at pressure ``P`` (the natural
    variables formulation of MRST's ``equationsEquilibrium``):

    - mass balance ``z_i − L·x_i − (1−L)·y_i`` (ncomp),
    - fugacity equality ``ln(y_i·φ_i^V) − ln(x_i·φ_i^L)`` (ncomp),
    - closure ``Σx − Σy`` (1).

    :func:`flash_direct_full` satisfies this to its convergence tolerance, so the
    residual is a ready-to-use Newton residual for a fugacity-driven phase split
    (which stays smooth through the bubble point, unlike the ``V(z)`` flash kink).
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    return _natural_variables_residual(L, x, y, z, P, T, aij, b)


# Minimum mole fraction floor. MRST's ``ensureMinimumFraction`` clips phase
# compositions to ``minimumComposition`` so the fugacity ``ln x_i`` term and the
# ``1/x_i`` Jacobian blocks stay finite at the single-phase boundary.
_MIN_COMPOSITION = 1.0e-12


def phase_flags(L: np.ndarray, tol: float = 1.0e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell phase flags from the liquid mole fraction ``L`` (MRST ``getFlag``).

    Returns ``(pure_liquid, pure_vapor, two_phase)`` boolean masks: ``L == 1`` is
    pure liquid (flag 1), ``L == 0`` pure vapor (flag 2), otherwise two-phase
    (flag 0), following MRST ``EquationOfStateModel.setFlag``.
    """
    L = np.asarray(L, dtype=float)
    pure_liquid = L >= 1.0 - tol
    pure_vapor = L <= tol
    return pure_liquid, pure_vapor, ~(pure_liquid | pure_vapor)


def natural_variables_state(sw, L, x, y, P, T=393.0):
    """Saturations and molar densities from the natural-variables state.

    Given the liquid mole fraction ``L``, phase compositions ``x``/``y`` and the
    water saturation ``sw``, returns ``(sO, sG, rho_l, rho_g)``: the EOS phase
    molar densities ``rho = 1/v`` and the volumetric saturations
    ``sO = (1−L)·N·v_l``, ``sG = L·N·v_g`` where ``N = (1−sw)/(L·v_g + (1−L)·v_l)``
    is the total hydrocarbon moles per pore volume. This is the bridge from the
    natural variables to the molar FIM accumulation ``m_c = sO·x_c/v_l + sG·y_c/v_g``
    (Step 5), and it reduces to ``m_c = z_c·N`` at the flash equilibrium.
    """
    v_l, v_g = phase_molar_volumes(x, y, P, T)
    sw = np.asarray(sw, dtype=float)
    L = np.asarray(L, dtype=float)
    denom = L * v_l + (1.0 - L) * v_g
    N = (1.0 - sw) / np.maximum(denom, 1.0e-30)
    sO = L * N * v_l
    sG = (1.0 - L) * N * v_g
    return sO, sG, 1.0 / np.maximum(v_l, 1.0e-30), 1.0 / np.maximum(v_g, 1.0e-30)


def natural_variables_jacobian(
    L: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    P: float | np.ndarray,
    T: float = 393.0,
    *,
    aij: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> np.ndarray:
    """Analytic Jacobian of :func:`natural_variables_residual` w.r.t. ``(L, x, y)``.

    This is MRST ``EquationOfStateModel.equationsEquilibrium`` differentiated
    analytically (the derivative its automatic-differentiation backend produces in
    ``getPhaseFractionDerivativesPTZ``). Per cell the variable order is
    ``[L, x_1..x_n, y_1..y_n]`` and the equation order is ``[mass, fugacity, closure]``:

    - mass balance ``L·x + (1−L)·y − z``:  ``∂/∂L = x − y``, ``∂/∂x = L·I``, ``∂/∂y = (1−L)·I``.
    - fugacity ``ln y + lnφ^V − ln x − lnφ^L``:  ``∂/∂x = −diag(1/x) − ∂lnφ^L/∂x``,
      ``∂/∂y = +diag(1/y) + ∂lnφ^V/∂y``. The ``∂lnφ/∂x_j`` blocks are the *unconstrained*
      mole-fraction derivatives (:func:`_fugacity_frac_deriv`) — the same derivative
      MRST's AD produces by treating each ``x_j`` as independent — not the simplex
      mole-number derivative ``d lnφ/d ln n_j`` (they differ by a nonzero radial gauge).

    - closure ``Σx − Σy``:  ``∂/∂x = 1ᵀ``, ``∂/∂y = −1ᵀ``.

    Returns ``(n_cells, 2·ncomp+1, 2·ncomp+1)``.
    """
    global _AB_DIRECT
    if aij is None or b is None:
        if _AB_DIRECT is None or _AB_DIRECT[1] != T:
            _AB_DIRECT = (_ab(T), T)
        aij, b = _AB_DIRECT[0]
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float)).astype(float)
    L = np.asarray(L, dtype=float)
    n_cells, ncomp = x.shape
    m = 2 * ncomp + 1
    dL = _fugacity_frac_deriv_analytic(x, aij, b, P, T, "liq")
    dV = _fugacity_frac_deriv_analytic(y, aij, b, P, T, "vap")
    x_safe = np.maximum(x, _MIN_COMPOSITION)
    y_safe = np.maximum(y, _MIN_COMPOSITION)
    eye = np.eye(ncomp)
    J = np.zeros((n_cells, m, m))
    # mass-balance rows (0 .. ncomp-1)
    J[:, :ncomp, 0] = x - y
    J[:, :ncomp, 1:1 + ncomp] = L[:, None, None] * eye[None]
    J[:, :ncomp, 1 + ncomp:1 + 2 * ncomp] = (1.0 - L)[:, None, None] * eye[None]
    # fugacity rows (ncomp .. 2*ncomp-1)
    J[:, ncomp:2 * ncomp, 1:1 + ncomp] = -eye[None] / x_safe[:, :, None] - dL
    J[:, ncomp:2 * ncomp, 1 + ncomp:1 + 2 * ncomp] = eye[None] / y_safe[:, :, None] + dV
    # closure row (2*ncomp)
    J[:, 2 * ncomp, 1:1 + ncomp] = 1.0
    J[:, 2 * ncomp, 1 + ncomp:1 + 2 * ncomp] = -1.0
    return J


def flash_derivatives_ptz(z, p, T=393.0, *, aij=None, b=None):
    """Analytic flash derivatives ``∂(V, x, y)/∂(p, z)`` (MRST ``getPhaseFractionDerivativesPTZ``).

    At the two-phase flash solution the equilibrium residual ``F(L, x, y, z, p) = 0``
    (mass balance + fugacity equality + closure), so the implicit-function theorem gives
    ``∂(L,x,y)/∂(p,z) = −(∂F/∂(L,x,y))⁻¹·(∂F/∂(p,z))`` — the same Schur step MRST's
    ``getPhaseFractionDerivativesPTZ`` performs (``dsdp = -(dFds \\ dFdp)``). This replaces
    the ``ncomp`` extra flash evaluations of a finite-difference Jacobian with one
    ``(2·ncomp+1)``-solve per two-phase cell. Single-phase cells (``V≈0``/``V≈1``) get the
    pinned-phase derivatives (``x=z``/``y=z``), matching :func:`flash_direct_full`. Returns
    ``(dV_dp (n,), dV_dz (n,ncomp), dx_dp (n,ncomp), dx_dz (n,ncomp,ncomp),
    dy_dp (n,ncomp), dy_dz (n,ncomp,ncomp))``.
    """
    global _AB_DIRECT
    if aij is None or b is None:
        if _AB_DIRECT is None or _AB_DIRECT[1] != T:
            _AB_DIRECT = (_ab(T), T)
        aij, b = _AB_DIRECT[0]
    z = np.atleast_2d(np.asarray(z, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float)).astype(float)
    if p.size == 1:
        p = np.full(z.shape[0], float(p[0]))
    n, ncomp = z.shape
    V, x, y = flash_direct_full(z, p, T)
    L = 1.0 - V
    liquid = V <= 1.0e-8
    vapor = V >= 1.0 - 1.0e-8
    two = ~(liquid | vapor)

    dV_dp = np.zeros(n)
    dV_dz = np.zeros((n, ncomp))
    dx_dp = np.zeros((n, ncomp))
    dx_dz = np.zeros((n, ncomp, ncomp))
    dy_dp = np.zeros((n, ncomp))
    dy_dz = np.zeros((n, ncomp, ncomp))
    if liquid.any():
        dx_dz[liquid] = np.eye(ncomp)      # x = z, V = 0
    if vapor.any():
        dy_dz[vapor] = np.eye(ncomp)       # y = z, V = 1

    if two.any():
        nt = int(two.sum())
        Lt = L[two]
        xt = x[two]
        yt = y[two]
        pt = p[two]
        m = 2 * ncomp + 1
        J = natural_variables_jacobian(Lt, xt, yt, pt, T, aij=aij, b=b)   # (nt, m, m)
        # ∂F/∂z: only the mass-balance block depends on z (mass = L·x + (1−L)·y − z → −I)
        dFdz = np.zeros((nt, m, ncomp))
        dFdz[:, :ncomp, :ncomp] = -np.eye(ncomp)
        # ∂F/∂p: only the fugacity block depends on p (fug = ln y + lnφᵛ − ln x − lnφˡ)
        dpf = 1.0
        dlnphiL_dp = (_fugacity_vec(xt, aij, b, pt + dpf, T, "liq")
                      - _fugacity_vec(xt, aij, b, pt - dpf, T, "liq")) / (2.0 * dpf)
        dlnphiV_dp = (_fugacity_vec(yt, aij, b, pt + dpf, T, "vap")
                      - _fugacity_vec(yt, aij, b, pt - dpf, T, "vap")) / (2.0 * dpf)
        dFdp = np.zeros((nt, m))
        dFdp[:, ncomp:2 * ncomp] = dlnphiV_dp - dlnphiL_dp
        dsd_z = np.zeros((nt, m, ncomp))
        dsd_p = np.zeros((nt, m))
        for i in range(nt):
            Ji = J[i]
            dsd_z[i] = -np.linalg.solve(Ji, dFdz[i])
            dsd_p[i] = -np.linalg.solve(Ji, dFdp[i])
        dL_dz = dsd_z[:, 0, :]
        dL_dp = dsd_p[:, 0]
        dV_dz[two] = -dL_dz
        dV_dp[two] = -dL_dp
        dx_dz[two] = dsd_z[:, 1:1 + ncomp, :]
        dy_dz[two] = dsd_z[:, 1 + ncomp:1 + 2 * ncomp, :]
        dx_dp[two] = dsd_p[:, 1:1 + ncomp]
        dy_dp[two] = dsd_p[:, 1 + ncomp:1 + 2 * ncomp]
    return dV_dp, dV_dz, dx_dp, dx_dz, dy_dp, dy_dz


def _flash_natural_variables(z_cells, P_cells, T, aij, b, tol=1.0e-9, max_iter=40):
    """Natural-variables Newton flash → ``(V, x, y)`` (drop-in replacement for ``_flash_vec``).

    Initializes from Wilson K-values + Rachford-Rice, then Newton-solves the
    ``(L, x, y)`` natural variables driving :func:`natural_variables_residual` to
    zero with the analytic :func:`natural_variables_jacobian`. Converges
    quadratically and stays smooth through the bubble point (no ``V(z)`` kink).
    Falls back to the successive-substitution :func:`_flash_vec` if Newton stalls,
    so the flash never returns a worse result than the loop it replaces.

    Note: the finite-difference Jacobian costs 2·(ncomp+1) fugacity evaluations
    per Newton step, so this is ~5× slower than :func:`_flash_vec`; it is validated
    here but the production flash stays on SSI until the natural-variables FIM
    (Step 5) embeds the Jacobian in the global system instead of per-cell.
    """
    n = P_cells.size
    ncomp = z_cells.shape[1]
    K = (_PC[None, :] / P_cells[:, None]) * np.exp(
        5.373 * (1.0 + _ACENTRIC)[None, :] * (1.0 - _TC[None, :] / T))
    V = np.full(n, 0.5)
    for _ in range(40):  # Rachford-Rice
        denom = 1.0 + V[:, None] * (K - 1.0)
        f = np.sum(z_cells * (K - 1.0) / denom, axis=1)
        conv = ((np.abs(f) < 1.0e-10)
                | ((V <= 1.0e-12) & (f <= 0.0))
                | ((V >= 1.0 - 1.0e-12) & (f >= 0.0)))
        if conv.all():
            break
        df = np.sum(-z_cells * (K - 1.0) ** 2 / denom ** 2, axis=1)
        step = np.where(np.abs(df) > 1.0e-12, f / np.where(np.abs(df) > 1.0e-12, df, 1.0),
                        np.where(f > 0.0, 0.01, -0.01))
        V = np.clip(V - step, 0.0, 1.0)
    x = z_cells / (1.0 + V[:, None] * (K - 1.0))
    y = K * x
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1.0e-300)
    y = y / np.maximum(y.sum(axis=1, keepdims=True), 1.0e-300)
    L = 1.0 - V
    converged = False
    for _ in range(max_iter):
        res = _natural_variables_residual(L, x, y, z_cells, P_cells, T, aij, b)
        if np.max(np.abs(res)) < tol:
            converged = True
            break
        J = natural_variables_jacobian(L, x, y, P_cells, T, aij=aij, b=b)
        try:
            du = np.linalg.solve(J, -res[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break  # singular Jacobian (near-critical): fall back to SSI
        dL = du[:, 0]
        dx = du[:, 1:1 + ncomp]
        dy = du[:, 1 + ncomp:]
        alpha = 1.0
        for _ in range(24):  # backtracking: keep L in (0,1) and x/y positive
            Ln = L + alpha * dL
            xn = x + alpha * dx
            yn = y + alpha * dy
            if (np.all(Ln > _MIN_COMPOSITION) and np.all(Ln < 1.0 - _MIN_COMPOSITION)
                    and np.all(xn > _MIN_COMPOSITION) and np.all(yn > _MIN_COMPOSITION)):
                break
            alpha *= 0.5
        L, x, y = Ln, xn, yn
    if not converged:
        return _flash_vec(z_cells, P_cells, T, aij, b, tol=tol)
    return _collapse_trivial_flash(1.0 - L, x, y, z_cells, P_cells, T, aij, b)
