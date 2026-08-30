"""Flash + Corey gas–oil mobility. Saturations come from the flash, not Rs/Sg switch."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.eos.flash import flash_tp

_LAST_FLASH_S = 0.0


def last_flash_seconds() -> float:
    """Wall time of the most recent ``flash_state`` cell loop."""
    return float(_LAST_FLASH_S)


@dataclass
class PhaseProps:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    xi_l: NDArray[np.float64]
    xi_v: NDArray[np.float64]
    sl: NDArray[np.float64]
    sv: NDArray[np.float64]
    sw: NDArray[np.float64]
    v_mix: NDArray[np.float64]
    vw: NDArray[np.float64]
    lam_l: NDArray[np.float64]
    lam_v: NDArray[np.float64]
    lam_w: NDArray[np.float64]
    xi_w: NDArray[np.float64]
    vapor_frac: NDArray[np.float64]
    two_phase: NDArray[np.bool_]
    has_water: bool = False
    k_flash: NDArray[np.float64] | None = None
    p_flash: NDArray[np.float64] | None = None
    z_flash: NDArray[np.float64] | None = None

    def copy(self) -> PhaseProps:
        return PhaseProps(
            x=self.x.copy(),
            y=self.y.copy(),
            xi_l=self.xi_l.copy(),
            xi_v=self.xi_v.copy(),
            sl=self.sl.copy(),
            sv=self.sv.copy(),
            sw=self.sw.copy(),
            v_mix=self.v_mix.copy(),
            vw=self.vw.copy(),
            lam_l=self.lam_l.copy(),
            lam_v=self.lam_v.copy(),
            lam_w=self.lam_w.copy(),
            xi_w=self.xi_w.copy(),
            vapor_frac=self.vapor_frac.copy(),
            two_phase=self.two_phase.copy(),
            has_water=self.has_water,
            k_flash=None if self.k_flash is None else self.k_flash.copy(),
            p_flash=None if self.p_flash is None else self.p_flash.copy(),
            z_flash=None if self.z_flash is None else self.z_flash.copy(),
        )


def _corey_og(sv: NDArray[np.float64], spec: CompSpec) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    denom = max(1.0 - spec.sorg - spec.sgr, 1.0e-12)
    se = np.clip((sv - spec.sgr) / denom, 0.0, 1.0)
    krg = spec.krg0 * np.power(se, spec.ng)
    kro = spec.kro0 * np.power(1.0 - se, spec.no)
    return kro, krg


def _corey_three(
    sw: NDArray[np.float64], sg: NDArray[np.float64], spec: CompSpec
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    denom_w = max(1.0 - spec.swc - spec.sorg, 1.0e-12)
    se_w = np.clip((sw - spec.swc) / denom_w, 0.0, 1.0)
    denom_g = max(1.0 - spec.sgr - spec.sorg, 1.0e-12)
    se_g = np.clip((sg - spec.sgr) / denom_g, 0.0, 1.0)
    krw = spec.krw0 * np.power(se_w, spec.nw)
    krg = spec.krg0 * np.power(se_g, spec.ng)
    kro = spec.kro0 * np.power(1.0 - se_w, spec.no) * np.power(1.0 - se_g, spec.no)
    return krw, kro, krg


def flash_state(
    spec: CompSpec,
    pressure: NDArray[np.float64],
    moles: NDArray[np.float64],
    *,
    cells: NDArray[np.int64] | None = None,
    out: PhaseProps | None = None,
) -> PhaseProps:
    """PT flash each listed cell. ``moles`` is (n_cells, spec.nc)."""
    p = np.asarray(pressure, dtype=float).ravel()
    n = np.asarray(moles, dtype=float)
    n_cells = p.size
    n_hc = spec.n_hc
    if n.ndim != 2 or n.shape[0] != n_cells or n.shape[1] != spec.nc:
        raise ValueError(f"moles shape {n.shape} != {(n_cells, spec.nc)}")
    if out is None:
        out = PhaseProps(
            x=np.zeros((n_cells, n_hc)),
            y=np.zeros((n_cells, n_hc)),
            xi_l=np.zeros(n_cells),
            xi_v=np.zeros(n_cells),
            sl=np.zeros(n_cells),
            sv=np.zeros(n_cells),
            sw=np.zeros(n_cells),
            v_mix=np.zeros(n_cells),
            vw=np.zeros(n_cells),
            lam_l=np.zeros(n_cells),
            lam_v=np.zeros(n_cells),
            lam_w=np.zeros(n_cells),
            xi_w=np.zeros(n_cells),
            vapor_frac=np.zeros(n_cells),
            two_phase=np.zeros(n_cells, dtype=bool),
            has_water=bool(spec.has_water),
            k_flash=np.ones((n_cells, n_hc)),
            p_flash=np.zeros(n_cells),
            z_flash=np.zeros((n_cells, n_hc)),
        )
    out.has_water = bool(spec.has_water)
    if out.k_flash is None:
        out.k_flash = np.ones((n_cells, n_hc))
    if out.p_flash is None:
        out.p_flash = np.zeros(n_cells)
    if out.z_flash is None:
        out.z_flash = np.zeros((n_cells, n_hc))
    idx = np.arange(n_cells, dtype=np.int64) if cells is None else np.asarray(cells, dtype=np.int64).ravel()
    t = float(spec.temperature_k)
    vw = spec.water_vw(p) if spec.has_water else np.zeros(n_cells)
    jac_fd = cells is not None
    t_flash = time.perf_counter()
    for c in idx:
        nh = n[c, :n_hc]
        tot = float(np.sum(nh))
        z = nh / tot if tot > 1.0e-18 else spec.z_init
        prev_p = float(out.p_flash[c])
        k_guess = None
        skip = False
        single_vapor = None
        if (not jac_fd) and prev_p > 0.0:
            rel_p = abs(float(p[c]) - prev_p) / max(abs(float(p[c])), 1.0)
            rel_z = float(np.max(np.abs(z - out.z_flash[c])))
            if bool(out.two_phase[c]) and rel_p < 0.05:
                k_guess = out.k_flash[c]
            elif (not bool(out.two_phase[c])) and 1.0e-4 < rel_p < 2.0e-3 and rel_z < 1.0e-3:
                skip = True
                single_vapor = float(out.vapor_frac[c]) > 0.5
        fl = flash_tp(
            spec.eos,
            float(p[c]),
            t,
            z,
            k_guess=k_guess,
            skip_stability=skip,
            single_vapor=single_vapor,
        )
        out.x[c] = fl.x
        out.y[c] = fl.y
        out.v_mix[c] = max(fl.v_mix, 1.0e-12)
        out.xi_l[c] = 1.0 / max(fl.v_liq, 1.0e-12)
        out.xi_v[c] = 1.0 / max(fl.v_vap, 1.0e-12)
        sl = fl.sl
        sv = fl.sv
        sw = 0.0
        if spec.has_water:
            out.vw[c] = float(vw[c]) if vw.ndim else float(vw)
            n_w = float(n[c, n_hc])
            # Sw from water occupancy; HC saturations fill the rest.
            sw = float(np.clip(n_w * out.vw[c] / max(out.v_mix[c] * tot + n_w * out.vw[c], 1.0e-18), 0.0, 0.999))
            sl = sl * (1.0 - sw)
            sv = sv * (1.0 - sw)
            out.xi_w[c] = 1.0 / max(out.vw[c], 1.0e-12)
        out.sl[c] = sl
        out.sv[c] = sv
        out.sw[c] = sw
        out.vapor_frac[c] = fl.vapor_frac
        out.two_phase[c] = fl.two_phase
        out.p_flash[c] = float(p[c])
        out.z_flash[c] = z[:n_hc]
        if fl.k is not None:
            out.k_flash[c] = np.asarray(fl.k, dtype=float).ravel()[:n_hc]
    global _LAST_FLASH_S
    _LAST_FLASH_S = time.perf_counter() - t_flash
    if spec.has_water:
        krw, kro, krg = _corey_three(out.sw, out.sv, spec)
        out.lam_w = krw / spec.mu_water
        out.lam_l = kro / spec.mu_liquid
        out.lam_v = krg / spec.mu_vapor
    else:
        kro, krg = _corey_og(out.sv, spec)
        out.lam_l = kro / spec.mu_liquid
        out.lam_v = krg / spec.mu_vapor
        out.lam_w[:] = 0.0
        out.sw[:] = 0.0
    return out


def moles_from_z(
    spec: CompSpec,
    pressure: NDArray[np.float64],
    z: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    """n_i such that hydrocarbon + water occupy V_pore."""
    p = np.asarray(pressure, dtype=float).ravel()
    pv = np.asarray(pore_volume, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    n_cells = p.size
    n = np.zeros((n_cells, spec.nc))
    t = float(spec.temperature_k)
    zz = z[: spec.n_hc] / float(np.sum(z[: spec.n_hc]))
    sw0 = float(spec.sw_init) if spec.has_water else 0.0
    vw = spec.water_vw(p) if spec.has_water else np.zeros(n_cells)
    hc_vol = pv * (1.0 - sw0)

    def _fill(c: int, vm: float, vw_c: float) -> None:
        n_hc_tot = float(hc_vol[c]) / max(vm, 1.0e-12)
        n[c, : spec.n_hc] = zz * n_hc_tot
        if spec.has_water:
            n[c, spec.n_hc] = float(pv[c]) * sw0 / max(vw_c, 1.0e-12)

    if n_cells and float(np.max(p) - np.min(p)) < 1.0e-6 * max(float(np.mean(p)), 1.0):
        fl = flash_tp(spec.eos, float(p[0]), t, zz)
        vm = max(fl.v_mix, 1.0e-12)
        vw0 = float(vw[0]) if spec.has_water else 1.0
        for c in range(n_cells):
            _fill(c, vm, vw0)
        return n
    for c in range(n_cells):
        fl = flash_tp(spec.eos, float(p[c]), t, zz)
        _fill(c, max(fl.v_mix, 1.0e-12), float(vw[c]) if spec.has_water else 1.0)
    return n
