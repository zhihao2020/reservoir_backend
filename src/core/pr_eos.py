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

# 3-component lumping: CO2 (idx 4) / light C1-NC6 (volatile, goes to the gas) /
# heavy C7+ (the oil). The light components are the ones that co-evaporate into
# the CO2-rich gas phase, which is exactly what the 2-component model (pure-CO2
# gas) misses.
_LIGHT_IDX = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9])
_HEAVY_IDX = np.array([10, 11, 12, 13])
_Z_LIGHT_NORM = _Z_OIL[_LIGHT_IDX] / _Z_OIL[_LIGHT_IDX].sum()
_Z_HEAVY_NORM = _Z_OIL[_HEAVY_IDX] / _Z_OIL[_HEAVY_IDX].sum()

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
_MW_OIL = float(np.sum(_Z_OIL * _MW))  # ~85 g/mol


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


def _bubble_point_co2(P: float, T: float, aij=None, b=None) -> float:
    """CO2 mole fraction in the liquid at the bubble point (binary search).

    The bubble point is the smallest overall CO2 fraction where a vapor phase
    appears (V > 0). A binary search needs ~log2(1/1e-3) ~ 10 flashes instead of
    the ~180 of a linear scan.
    """
    if aij is None:
        aij, b = _ab(T)
    lo, hi = 0.0, 0.95
    for _ in range(11):
        zc = 0.5 * (lo + hi)
        z = (1.0 - zc) * _Z_OIL
        z[_CO2_IDX] += zc
        V, x, _ = _flash(z, P, T, aij, b)
        if V > 1.0e-4:
            hi = zc
        else:
            lo = zc
    z = (1.0 - hi) * _Z_OIL
    z[_CO2_IDX] += hi
    _, x, _ = _flash(z, P, T, aij, b)
    return float(x[_CO2_IDX])


def rs_sat(P: float | np.ndarray, T: float = 393.0) -> float | np.ndarray:
    """Equilibrium CO2 solubility ``Rs`` (surface m3 gas / surface m3 oil).

    ``P`` is pressure (Pa), ``T`` temperature (K, default 120 C = 393.15). The
    bubble-point CO2 mole fraction ``x`` converts to a surface-volume ratio via
    the standard molar volumes: ``Rs = x/(1-x) * (V_co2 / V_oil)``.
    """
    scalar = np.ndim(P) == 0
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    out = np.empty_like(P_arr)
    cache: dict[float, float] = {}
    aij, b = _ab(T)  # PR EOS parameters depend on T only — compute once.
    v_oil = _MW_OIL / 1000.0 / _RHO_OIL_STD  # m3/mol oil at surface
    for i, p in enumerate(P_arr):
        p = float(p)
        if p not in cache:
            x = _bubble_point_co2(p, T, aij, b)
            cache[p] = (x / (1.0 - x)) * (_V_CO2_STD / v_oil) if np.isfinite(x) else float(np.nan)
        out[i] = cache[p]
    return float(out[0]) if scalar else out


def rs_sat_table(pmin: float = 15.0e6, pmax: float = 22.0e6, n: int = 36, T: float = 393.0):
    """Precompute ``(pressure, Rs_sat)`` points over ``[pmin, pmax]`` for
    interpolation in the forward model."""
    P = np.linspace(pmin, pmax, n)
    return P, rs_sat(P, T)


_TABLE_CACHE: tuple[np.ndarray, np.ndarray] | None = None
_CACHE_FILE = Path(__file__).parent / ".rs_sat_cache.npz"


def _eos_fingerprint() -> str:
    """Checksum of the EOS inputs, so the disk cache is invalidated whenever a
    critical property, composition, binary-interaction coefficient, or standard
    density/volume changes."""
    h = hashlib.sha256()
    for arr in (_MW, _TC, _PC, _ACENTRIC, _Z_OIL, _BIN):
        h.update(np.asarray(arr, dtype=float).tobytes())
    h.update(np.array([_R, _RHO_OIL_STD, _V_CO2_STD], dtype=float).tobytes())
    return h.hexdigest()


def _load_table() -> tuple[np.ndarray, np.ndarray]:
    """Return the cached Rs_sat table, from disk when available.

    The PR-EOS flash that builds the table is the dominant one-time cost (~13s
    after the Cardano cubic roots), so the table is persisted to a ``.npz`` file
    next to this module and reused on the next run. Falls back to a fresh build
    (and tries to write the cache) on the first run or a read-only install.
    """
    global _TABLE_CACHE
    if _TABLE_CACHE is not None:
        return _TABLE_CACHE
    if _CACHE_FILE.is_file():
        try:
            z = np.load(_CACHE_FILE)
            if str(z["fingerprint"]) == _eos_fingerprint():
                _TABLE_CACHE = (np.asarray(z["P"], dtype=float), np.asarray(z["Rs"], dtype=float))
                return _TABLE_CACHE
        except Exception:
            pass  # corrupt / incompatible cache: rebuild below
    P, Rs = rs_sat_table()
    try:
        np.savez(_CACHE_FILE, P=P, Rs=Rs, fingerprint=np.array(_eos_fingerprint()))
    except Exception:
        pass  # read-only install: keep the in-memory cache only
    _TABLE_CACHE = (P, Rs)
    return _TABLE_CACHE


def rs_sat_interp(pressure: float | np.ndarray, T: float = 393.0) -> float | np.ndarray:
    """Linearly interpolate a cached Rs_sat(p) table (computes it once)."""
    if T != 393.0:
        # non-default temperature: don't use the default-temperature disk cache.
        P, Rs = rs_sat_table(T=T)
    else:
        P, Rs = _load_table()
    scalar = np.ndim(pressure) == 0
    p = np.atleast_1d(np.asarray(pressure, dtype=float))
    out = np.interp(p, P, Rs)
    return float(out[0]) if scalar else out


# --- two-phase flash table (vapor fraction V, liquid/gas CO2 mole fractions) ---
#
# The compositional phase split (GEM-style) uses the full Peng-Robinson flash
# ``V(z), x_CO2(z), y_CO2(z)`` rather than the black-oil bubble-point solubility
# ``Rs(p)``. At the injector the flash keeps most injected CO2 as a *gas phase*
# (V ≈ 0.86 at the accumulated state z ≈ 0.81), whereas the black-oil Rs dissolves
# it. The flash is per-cell and expensive, so it is tabulated over (p, z) and
# interpolated. ``x_CO2`` is the liquid (oil) phase CO2 mole fraction, ``y_CO2``
# the gas (vapor) phase CO2 mole fraction.
_FLASH_CACHE: tuple[np.ndarray, ...] | None = None
_FLASH_CACHE_FILE = Path(__file__).parent / ".flash_table_cache.npz"


def flash_table(
    pmin: float = 15.0e6,
    pmax: float = 22.0e6,
    np_: int = 36,
    zmin: float = 0.0,
    zmax: float = 0.999,
    nz: int = 50,
    T: float = 393.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute the two-phase flash over a ``(pressure, CO2 mole fraction)`` grid.

    Returns ``(P, Z, V, X, Y)`` where ``V[i, j]`` is the vapor (gas) mole fraction,
    ``X[i, j]`` the liquid (oil) CO2 mole fraction and ``Y[i, j]`` the gas (vapor)
    CO2 mole fraction at ``(P[i], Z[j])``. ``X``/``Y`` are the *overall* CO2 mole
    fraction ``Z`` in the single-phase regions (X = Z for V = 0, Y = Z for V = 1)
    so the composition fields stay continuous across the phase boundary.
    """
    P = np.linspace(pmin, pmax, np_)
    Z = np.linspace(zmin, zmax, nz)
    aij, b = _ab(T)
    V = np.zeros((np_, nz))
    X = np.zeros((np_, nz))
    Y = np.zeros((np_, nz))
    for i, p in enumerate(P):
        for j, zc in enumerate(Z):
            z = (1.0 - zc) * _Z_OIL
            z[_CO2_IDX] += zc
            v, x, y = _flash(z, float(p), T, aij, b)
            V[i, j] = v
            X[i, j] = x[_CO2_IDX] if v > 1.0e-6 else zc
            Y[i, j] = y[_CO2_IDX] if v < 1.0 - 1.0e-6 else zc
    return P, Z, V, X, Y


def _load_flash_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global _FLASH_CACHE
    if _FLASH_CACHE is not None:
        return _FLASH_CACHE
    if _FLASH_CACHE_FILE.is_file():
        try:
            z = np.load(_FLASH_CACHE_FILE)
            if str(z["fingerprint"]) == _eos_fingerprint():
                _FLASH_CACHE = (z["P"], z["Z"], z["V"], z["X"], z["Y"])
                return _FLASH_CACHE
        except Exception:
            pass
    P, Z, V, X, Y = flash_table()
    try:
        np.savez(_FLASH_CACHE_FILE, P=P, Z=Z, V=V, X=X, Y=Y,
                 fingerprint=np.array(_eos_fingerprint()))
    except Exception:
        pass
    _FLASH_CACHE = (P, Z, V, X, Y)
    return _FLASH_CACHE


def flash_interp(pressure: float | np.ndarray, z_co2: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly interpolate the flash table → ``(vapor_fraction, liquid_x_co2, gas_y_co2)``."""
    P, Z, V, X, Y = _load_flash_table()
    p = np.asarray(pressure, dtype=float)
    zc = np.asarray(z_co2, dtype=float)
    ip = np.clip(np.searchsorted(P, p, side="right") - 1, 0, len(P) - 2)
    iz = np.clip(np.searchsorted(Z, zc, side="right") - 1, 0, len(Z) - 2)
    tp = (p - P[ip]) / (P[ip + 1] - P[ip])
    tz = (zc - Z[iz]) / (Z[iz + 1] - Z[iz])
    w00 = (1.0 - tp) * (1.0 - tz)
    w10 = tp * (1.0 - tz)
    w01 = (1.0 - tp) * tz
    w11 = tp * tz
    v = V[ip, iz] * w00 + V[ip + 1, iz] * w10 + V[ip, iz + 1] * w01 + V[ip + 1, iz + 1] * w11
    x = X[ip, iz] * w00 + X[ip + 1, iz] * w10 + X[ip, iz + 1] * w01 + X[ip + 1, iz + 1] * w11
    y = Y[ip, iz] * w00 + Y[ip + 1, iz] * w10 + Y[ip, iz + 1] * w01 + Y[ip + 1, iz + 1] * w11
    return v, x, y


# PR EOS parameters for the direct (iterative) flash at the reservoir temperature.
# Building ``aij``/``b`` is the expensive part of a flash, and they depend only on
# ``T``, so cache them here for the per-cell Newton loop in the forward model.
_AB_DIRECT: tuple[tuple[np.ndarray, np.ndarray], float] | None = None


def _cubic_roots_vec(a0, a1, a2):
    """Vectorized Cardano roots of ``Z³ + a2·Z² + a1·Z + a0 = 0``.

    ``a0``/``a1``/``a2`` are arrays of the same length ``n``; returns an
    ``(n, 3)`` array of real roots sorted ascending, with ``NaN`` filling the
    unused slots when a cubic has a single real root (``NaN`` sorts last).
    """
    a0 = np.atleast_1d(np.asarray(a0, dtype=float))
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    a2 = np.atleast_1d(np.asarray(a2, dtype=float))
    n = a0.size
    p = a1 - a2 ** 2 / 3.0
    q = 2.0 * a2 ** 3 / 27.0 - a2 * a1 / 3.0 + a0
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    roots = np.full((n, 3), np.nan)
    one = disc >= 0.0
    three = disc < 0.0
    if one.any():
        s = np.sqrt(np.maximum(disc[one], 0.0))
        u = np.cbrt(-q[one] / 2.0 + s)
        v = np.cbrt(-q[one] / 2.0 - s)
        roots[one, 0] = u + v - a2[one] / 3.0
    if three.any():
        r = 2.0 * np.sqrt(-p[three] / 3.0)
        arg = 3.0 * q[three] / (2.0 * p[three]) * np.sqrt(-3.0 / p[three])
        arg = np.clip(arg, -1.0, 1.0)
        theta = np.arccos(arg) / 3.0
        for k in range(3):
            roots[three, k] = r * np.cos(theta + 2.0 * np.pi * k / 3.0) - a2[three] / 3.0
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


def flash_direct_3comp(
    pressure: float | np.ndarray,
    z_co2: float | np.ndarray,
    z_light: float | np.ndarray,
    T: float = 393.0,
    tol: float = 1.0e-5,
) -> tuple[np.ndarray, ...]:
    """3-component lumped flash (CO2 / light C1-NC6 / heavy C7+) → phase split.

    ``z_co2`` and ``z_light`` are the CO2 and light-component *overall* mole
    fractions (``z_heavy = 1 − z_co2 − z_light``); they are distributed over the
    14 GEM components (light/heavy proportional to ``_Z_OIL``) and the full PR
    flash is run, then lumped back. Returns ``(V, x_co2, x_light, y_co2, y_light)``
    — the vapor mole fraction and the CO2/light mole fractions in the liquid (x)
    and gas (y). The heavy fraction is ``1 − x_co2 − x_light`` / ``1 − y_co2 −
    y_light``.
    """
    global _AB_DIRECT
    if _AB_DIRECT is None or _AB_DIRECT[1] != T:
        _AB_DIRECT = (_ab(T), T)
    aij, b = _AB_DIRECT[0]
    scalar = (np.ndim(pressure) == 0 and np.ndim(z_co2) == 0 and np.ndim(z_light) == 0)
    p_arr = np.atleast_1d(np.asarray(pressure, dtype=float))
    zc_arr = np.atleast_1d(np.asarray(z_co2, dtype=float))
    zl_arr = np.atleast_1d(np.asarray(z_light, dtype=float))
    n = max(p_arr.size, zc_arr.size, zl_arr.size)
    pp = np.broadcast_to(p_arr if p_arr.size == n else p_arr[0], (n,)).astype(float)
    zc = np.broadcast_to(zc_arr if zc_arr.size == n else zc_arr[0], (n,)).astype(float)
    zl = np.broadcast_to(zl_arr if zl_arr.size == n else zl_arr[0], (n,)).astype(float)
    zh = 1.0 - zc - zl
    z14 = np.zeros((n, 14))
    z14[:, _CO2_IDX] = zc
    z14[:, _LIGHT_IDX] = np.outer(zl, _Z_LIGHT_NORM)
    z14[:, _HEAVY_IDX] = np.outer(zh, _Z_HEAVY_NORM)
    V, x, y = _flash_vec(z14, pp, T, aij, b, tol=tol)
    x_co2 = x[:, _CO2_IDX]
    x_light = x[:, _LIGHT_IDX].sum(axis=1)
    y_co2 = y[:, _CO2_IDX]
    y_light = y[:, _LIGHT_IDX].sum(axis=1)
    if scalar:
        return (float(V[0]), float(x_co2[0]), float(x_light[0]),
                float(y_co2[0]), float(y_light[0]))
    return V, x_co2, x_light, y_co2, y_light
