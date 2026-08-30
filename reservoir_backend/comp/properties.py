"""Flash + Corey gas–oil mobility. Saturations come from the flash, not Rs/Sg switch."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.eos.flash_backend import get_flash_backend
from reservoir_backend.eos.flash_batch import wilson_k_batch
from reservoir_backend.eos.flash_counters import bump

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
    t_flash = time.perf_counter()
    n_idx = int(idx.size)
    z_all = np.zeros((n_idx, n_hc))
    tot = np.sum(n[idx, :n_hc], axis=1)
    use = tot > 1.0e-18
    z_all[use] = n[idx[use], :n_hc] / tot[use, None]
    if spec.z_init is not None:
        z_all[~use] = np.asarray(spec.z_init, dtype=float).ravel()[:n_hc]
    jac_fd = cells is not None
    k_guess = None
    n_warm = 0
    if (not jac_fd) and out.k_flash is not None and out.p_flash is not None:
        prev_p = np.asarray(out.p_flash[idx], dtype=float)
        rel_p = np.abs(p[idx] - prev_p) / np.maximum(np.abs(p[idx]), 1.0)
        warm = (prev_p > 0.0) & np.asarray(out.two_phase[idx], dtype=bool) & (rel_p < 0.05)
        if np.any(warm):
            k_guess = np.clip(wilson_k_batch(spec.eos, p[idx], t), 1.0e-8, 1.0e8)
            k_guess[warm] = out.k_flash[idx[warm]]
            n_warm = int(np.count_nonzero(warm))
    backend = get_flash_backend()
    arr = backend.evaluate_batch(spec.eos, p[idx], t, z_all, k_guess=k_guess)
    n_fallback = 0
    if np.any(~arr.converged):
        bad = np.where(~arr.converged)[0]
        cold = backend.evaluate_batch(spec.eos, p[idx[bad]], t, z_all[bad])
        arr.vapor_frac[bad] = cold.vapor_frac
        arr.x[bad] = cold.x
        arr.y[bad] = cold.y
        arr.z_liq[bad] = cold.z_liq
        arr.z_vap[bad] = cold.z_vap
        arr.v_liq[bad] = cold.v_liq
        arr.v_vap[bad] = cold.v_vap
        arr.two_phase[bad] = cold.two_phase
        arr.k[bad] = cold.k
        arr.converged[bad] = cold.converged
        arr.fallback_used[bad] = True
        n_fallback = int(bad.size)
    out.x[idx] = arr.x
    out.y[idx] = arr.y
    out.v_mix[idx] = np.maximum(arr.v_mix, 1.0e-12)
    out.xi_l[idx] = 1.0 / np.maximum(arr.v_liq, 1.0e-12)
    out.xi_v[idx] = 1.0 / np.maximum(arr.v_vap, 1.0e-12)
    sl = (1.0 - arr.vapor_frac) * arr.v_liq / np.maximum(arr.v_mix, 1.0e-30)
    sv = 1.0 - sl
    sw = np.zeros(n_idx)
    if spec.has_water:
        vw_c = np.asarray(vw, dtype=float).ravel()
        out.vw[idx] = vw_c[idx] if vw_c.size == n_cells else float(vw_c.flat[0])
        n_w = n[idx, n_hc]
        den = np.maximum(out.v_mix[idx] * tot + n_w * out.vw[idx], 1.0e-18)
        sw = np.clip(n_w * out.vw[idx] / den, 0.0, 0.999)
        sl = sl * (1.0 - sw)
        sv = sv * (1.0 - sw)
        out.xi_w[idx] = 1.0 / np.maximum(out.vw[idx], 1.0e-12)
    out.sl[idx] = sl
    out.sv[idx] = sv
    out.sw[idx] = sw
    out.vapor_frac[idx] = arr.vapor_frac
    out.two_phase[idx] = arr.two_phase
    out.p_flash[idx] = p[idx]
    out.z_flash[idx] = z_all
    out.k_flash[idx] = arr.k[:, :n_hc]
    bump(n_cells=n_idx, n_warm_start=n_warm, n_warm_fallback=n_fallback)
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


def flash_compressibility(
    spec: CompSpec,
    p: NDArray[np.float64],
    n: NDArray[np.float64],
    props: PhaseProps,
    *,
    rel: float = 1.0e-5,
) -> NDArray[np.float64]:
    """c_t = -(1/v) ∂v/∂p at frozen moles, from one extra flash with K reuse."""
    p = np.asarray(p, dtype=float).ravel()
    p2 = np.maximum(p * (1.0 + float(rel)), p + 1.0)
    trial = flash_state(spec, p2, n, out=props.copy())
    dv = trial.v_mix - props.v_mix
    dp = np.maximum(p2 - p, 1.0)
    ct = -dv / np.maximum(props.v_mix * dp, 1.0e-30)
    return np.clip(np.asarray(ct, dtype=float), 1.0e-12, 1.0e-6)


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
    backend = get_flash_backend()
    arr = backend.evaluate_batch(spec.eos, p, t, np.broadcast_to(zz, (n_cells, zz.size)).copy())
    vm = np.maximum(arr.v_mix, 1.0e-12)
    for c in range(n_cells):
        _fill(c, float(vm[c]), float(vw[c]) if spec.has_water else 1.0)
    return n
