"""Lohrenz–Bray–Clark viscosity. Numbers from the published correlation.

Adapted from the MRST compositional property model (ideas only; product does
not import ``references/``). GEM ``*VISCOR *HZYT`` uses the same Jossi/LBC
polynomial as ``*VISCOEFF`` on the physical_3d deck.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.pr import PengRobinson

# GEM *VISCOEFF / LBC paper (Jossi et al.), last offset from the LBC form.
_LBC = (0.1023, 0.023364, 0.058533, -0.040758, 0.0093324, -1.0e-4)
_PSIA = 6894.757293168  # Pa
_CP = 1.0e-3  # Pa·s


def lbc_viscosity(
    eos: PengRobinson,
    temperature_k: float,
    x: NDArray[np.float64],
    molar_density: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Phase viscosity (Pa·s) from composition and molar density (mol/m³)."""
    if eos.vcrit is None:
        raise ValueError("LBC viscosity needs vcrit on the EOS card")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = np.maximum(x, 0.0)
    x = x / np.maximum(x.sum(axis=1, keepdims=True), 1.0e-30)
    rho = np.asarray(molar_density, dtype=float).ravel()
    n, nc = x.shape
    if rho.size != n:
        raise ValueError(f"molar_density size {rho.size} != {n}")
    t = float(temperature_k)
    tc = np.asarray(eos.tc, dtype=float)
    pc = np.asarray(eos.pc, dtype=float)
    mw = np.asarray(eos.mw, dtype=float) * 1.0e3  # g/mol
    vc = np.asarray(eos.vcrit, dtype=float)
    tr = t / tc
    tc_r = tc * 1.8
    pc_psia = pc / _PSIA
    sqrt_mw = np.sqrt(np.maximum(mw, 1.0e-12))
    e_i = (5.4402 * np.power(tc_r, 1.0 / 6.0)) / (sqrt_mw * np.power(pc_psia, 2.0 / 3.0) * _CP)
    hi = tr > 1.5
    mu_st = 34.0e-5 * np.power(np.maximum(tr, 1.0e-8), 0.94)
    mu_st = np.where(hi, 17.78e-5 * np.power(np.maximum(4.58 * tr - 1.67, 1.0e-12), 0.625), mu_st)
    mu_i = mu_st / e_i
    w = x * sqrt_mw[None, :]
    mu_atm = np.sum(x * mu_i[None, :] * sqrt_mw[None, :], axis=1) / np.maximum(w.sum(axis=1), 1.0e-30)
    t_pc = x @ tc
    p_pc = x @ pc
    mw_mix = x @ (np.asarray(eos.mw, dtype=float))
    vc_mix = x @ vc
    tc_r_m = t_pc * 1.8
    pc_psia_m = p_pc / _PSIA
    e_mix = (5.4402 * np.power(tc_r_m, 1.0 / 6.0)) / (
        np.sqrt(np.maximum(mw_mix * 1.0e3, 1.0e-12)) * np.power(pc_psia_m, 2.0 / 3.0) * _CP
    )
    rhor = np.maximum(vc_mix * rho, 0.0)
    poly = _LBC[0] + _LBC[1] * rhor + _LBC[2] * rhor**2 + _LBC[3] * rhor**3 + _LBC[4] * rhor**4
    mu = mu_atm + (np.power(poly, 4.0) + _LBC[5]) / np.maximum(e_mix, 1.0e-30)
    return np.clip(mu, 1.0e-8, 10.0)
