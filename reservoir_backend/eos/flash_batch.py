"""Vectorized isothermal PT flash. FastPR path; scalar ``flash_tp`` stays the truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.flash import FlashResult, _RR_EPS, flash_tp
from reservoir_backend.eos.pr import R_GAS, PengRobinson, _SQRT2, _frac

_SS_MAX = 20
_SS_TOL = 1.0e-8


def _cbrt(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sign(x) * np.abs(x) ** (1.0 / 3.0)


def pr_z_factors_batch(A: NDArray[np.float64], B: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorized (Z_liquid, Z_vapor). Matches scalar ``pr_z_factors`` branches."""
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    a = -(1.0 - B)
    b = A - 3.0 * B * B - 2.0 * B
    c = -(A * B - B * B - B * B * B)
    aa = a * a
    p = b - aa / 3.0
    q = c + (2.0 * a * aa - 9.0 * a * b) / 27.0
    disc = (q * 0.5) * (q * 0.5) + (p / 3.0) ** 3
    shift = a / 3.0
    n = A.size
    r0 = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    pos = disc > 1.0e-16
    zero = (~pos) & (disc >= -1.0e-16)
    neg = ~(pos | zero)
    sd = np.sqrt(np.maximum(disc, 0.0))
    r0[pos] = _cbrt(-0.5 * q[pos] + sd[pos]) + _cbrt(-0.5 * q[pos] - sd[pos]) - shift[pos]
    u = _cbrt(-0.5 * q)
    r0[zero] = 2.0 * u[zero] - shift[zero]
    r1[zero] = -u[zero] - shift[zero]
    amp = np.sqrt(np.maximum(-p / 3.0, 0.0))
    den = amp * amp * amp
    arg = np.clip(np.divide(-0.5 * q, den, out=np.zeros_like(q), where=den > 0.0), -1.0, 1.0)
    phi = np.arccos(arg)
    r0[neg] = 2.0 * amp[neg] * np.cos((phi[neg] - 0.0) / 3.0) - shift[neg]
    r1[neg] = 2.0 * amp[neg] * np.cos((phi[neg] - 2.0 * np.pi) / 3.0) - shift[neg]
    r2[neg] = 2.0 * amp[neg] * np.cos((phi[neg] - 4.0 * np.pi) / 3.0) - shift[neg]
    amp0 = neg & (amp <= 0.0)
    r0[amp0] = -shift[amp0]
    r1[amp0] = np.nan
    r2[amp0] = np.nan
    roots = np.stack((r0, r1, r2), axis=1)
    valid = np.isfinite(roots) & (roots > B[:, None] + 1.0e-12)
    masked = np.where(valid, roots, np.nan)
    with np.errstate(all="ignore"):
        zmin = np.nanmin(masked, axis=1)
        zmax = np.nanmax(masked, axis=1)
    fallback = ~np.isfinite(zmin)
    zfb = np.maximum(B + 1.0e-8, 1.0)
    zmin = np.where(fallback, zfb, zmin)
    zmax = np.where(fallback, zfb, zmax)
    return zmin, zmax


def wilson_k_batch(eos: PengRobinson, pressure: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    pack = eos._t_pack(temperature)
    p = np.maximum(np.asarray(pressure, dtype=float).ravel(), 1.0)
    return np.asarray(pack[4], dtype=float)[None, :] / p[:, None]


def _mix_ab_batch(
    eos: PengRobinson, temperature: float, z: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    pack = eos._t_pack(temperature)
    a_i, b_i, aij = pack[1], pack[2], pack[3]
    a = np.einsum("ni,ij,nj->n", z, aij, z)
    b = z @ b_i
    return a, b, a_i, b_i, aij


def ln_phi_batch(
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    *,
    vapor: bool | NDArray[np.bool_],
) -> NDArray[np.float64]:
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    z = z / np.maximum(z.sum(axis=1, keepdims=True), 1.0e-30)
    p = np.asarray(pressure, dtype=float).ravel()
    t = float(temperature)
    a, b, a_i, b_i, aij = _mix_ab_batch(eos, t, z)
    A = a * p / (R_GAS * R_GAS * t * t)
    B = b * p / (R_GAS * t)
    zl, zv = pr_z_factors_batch(A, B)
    if np.isscalar(vapor) or isinstance(vapor, (bool, np.bool_)):
        zz = zv if bool(vapor) else zl
    else:
        zz = np.where(np.asarray(vapor, dtype=bool), zv, zl)
    sum_a = z @ aij
    b = np.maximum(b, 1.0e-18)
    B = np.maximum(B, 1.0e-18)
    bi_b = b_i[None, :] / b[:, None]
    a_term = 2.0 * sum_a / np.maximum(a[:, None], 1.0e-30) - bi_b
    log_arg = (zz + (1.0 + _SQRT2) * B) / np.maximum(zz + (1.0 - _SQRT2) * B, 1.0e-18)
    ln_phi = (
        bi_b * (zz[:, None] - 1.0)
        - np.log(np.maximum(zz[:, None] - B[:, None], 1.0e-18))
        - (A / (2.0 * _SQRT2 * B))[:, None] * a_term * np.log(np.maximum(log_arg, 1.0e-18))[:, None]
    )
    return ln_phi


def z_roots_batch(
    eos: PengRobinson, pressure: NDArray[np.float64], temperature: float, z: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    z = np.maximum(np.asarray(z, dtype=float), 0.0)
    z = z / np.maximum(z.sum(axis=1, keepdims=True), 1.0e-30)
    p = np.asarray(pressure, dtype=float).ravel()
    t = float(temperature)
    a, b, *_ = _mix_ab_batch(eos, t, z)
    A = a * p / (R_GAS * R_GAS * t * t)
    B = b * p / (R_GAS * t)
    return pr_z_factors_batch(A, B)


def rachford_rice_batch(k: NDArray[np.float64], z: NDArray[np.float64], eps: float = _RR_EPS) -> NDArray[np.float64]:
    k = np.asarray(k, dtype=float)
    z = np.asarray(z, dtype=float)
    k1 = k - 1.0
    kmax = np.max(k, axis=1)
    kmin = np.min(k, axis=1)
    lo = 1.0 / (1.0 - kmax) + eps
    hi = 1.0 / (1.0 - kmin) - eps
    swap = lo > hi
    lo, hi = np.where(swap, hi, lo), np.where(swap, lo, hi)
    bad = (~np.isfinite(lo)) | (~np.isfinite(hi))
    if k.shape[1] == 2:
        den = k1[:, 0] * k1[:, 1]
        use = np.abs(den) > 1.0e-18
        v_bin = np.divide(-(z[:, 0] * k1[:, 0] + z[:, 1] * k1[:, 1]), den, out=np.full(k.shape[0], 0.5), where=use)
        v = np.clip(v_bin, lo, hi)
        v = np.where(use, v, 0.5 * (lo + hi))
        v = np.where(bad, 0.5, v)
        return v
    v = 0.5 * (lo + hi)
    for _ in range(12):
        denom = v[:, None] * k1 + 1.0
        r = np.sum(z * k1 / denom, axis=1)
        dr = -np.sum(z * k1 * k1 / (denom * denom), axis=1)
        done = np.abs(r) < 1.0e-12
        v_n = v - np.divide(r, dr, out=np.zeros_like(v), where=np.abs(dr) > 1.0e-18)
        inside = (v_n > lo) & (v_n < hi) & (np.abs(dr) > 1.0e-18) & (~done)
        v = np.where(inside, v_n, v)
        v = np.where(done, np.clip(v, 0.0, 1.0), v)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = np.sum(z * k1 / (mid[:, None] * k1 + 1.0), axis=1)
        lo = np.where(r > 0.0, mid, lo)
        hi = np.where(r > 0.0, hi, mid)
        v = mid
        if float(np.max(np.abs(r))) < 1.0e-12:
            break
    v = np.where(bad, 0.5, v)
    return np.clip(v, 0.0, 1.0)


def tpd_batch(
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    y: NDArray[np.float64],
    ln_phi_z: NDArray[np.float64],
) -> NDArray[np.float64]:
    y = np.maximum(y, 1.0e-16)
    y = y / y.sum(axis=1, keepdims=True)
    ln_phi_y = ln_phi_batch(eos, pressure, temperature, y, vapor=True)
    zl, zv = z_roots_batch(eos, pressure, temperature, y)
    liquid_like = (zv - zl > 1.0e-6) & ((y * eos.tc[None, :]).sum(axis=1) > 0.6 * (z * eos.tc[None, :]).sum(axis=1))
    if np.any(liquid_like):
        ln_phi_liq = ln_phi_batch(eos, pressure, temperature, y, vapor=False)
        ln_phi_y = np.where(liquid_like[:, None], ln_phi_liq, ln_phi_y)
    di = np.log(np.maximum(z, 1.0e-16)) + ln_phi_z
    return np.sum(y * (np.log(y) + ln_phi_y - di), axis=1)


def is_unstable_batch(
    eos: PengRobinson, pressure: NDArray[np.float64], temperature: float, z: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    k = np.clip(wilson_k_batch(eos, pressure, temperature), 1.0e-8, 1.0e8)
    ln_phi_z = ln_phi_batch(eos, pressure, temperature, z, vapor=True)
    unstable = np.zeros(z.shape[0], dtype=bool)
    margin = np.full(z.shape[0], np.inf)
    for w in (z * k, z / np.maximum(k, 1.0e-16)):
        y = np.maximum(w, 1.0e-16)
        y = y / y.sum(axis=1, keepdims=True)
        for _ in range(4):
            ln_phi_y = ln_phi_batch(eos, pressure, temperature, y, vapor=True)
            y_new = z * np.exp(np.clip(ln_phi_z - ln_phi_y, -20.0, 20.0))
            y_new = np.maximum(y_new, 1.0e-16)
            y_new = y_new / y_new.sum(axis=1, keepdims=True)
            if float(np.max(np.abs(y_new - y))) < 1.0e-8:
                y = y_new
                break
            y = y_new
        tpd = tpd_batch(eos, pressure, temperature, z, y, ln_phi_z)
        margin = np.minimum(margin, tpd)
        unstable |= tpd < -1.0e-8
    return unstable, margin


@dataclass
class FlashArrays:
    vapor_frac: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z_liq: NDArray[np.float64]
    z_vap: NDArray[np.float64]
    v_liq: NDArray[np.float64]
    v_vap: NDArray[np.float64]
    two_phase: NDArray[np.bool_]
    k: NDArray[np.float64]
    converged: NDArray[np.bool_]
    iterations: NDArray[np.int32]
    fugacity_error: NDArray[np.float64]
    stability_checked: NDArray[np.bool_]
    stability_margin: NDArray[np.float64]
    fallback_used: NDArray[np.bool_]

    @property
    def v_mix(self) -> NDArray[np.float64]:
        v = self.vapor_frac
        return v * self.v_vap + (1.0 - v) * self.v_liq


def _single_arrays(
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    vapor: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    zl, zv = z_roots_batch(eos, pressure, temperature, z)
    zz = np.where(vapor, zv, zl)
    vol = zz * R_GAS * float(temperature) / np.maximum(pressure, 1.0e-12)
    v_liq = zl * R_GAS * float(temperature) / np.maximum(pressure, 1.0e-12)
    v_vap = zv * R_GAS * float(temperature) / np.maximum(pressure, 1.0e-12)
    v_liq = np.where(vapor, v_liq, vol)
    v_vap = np.where(vapor, vol, v_vap)
    return zl, zv, v_liq, v_vap, np.where(vapor, 1.0, 0.0)


def flash_batch(
    eos: PengRobinson,
    pressure: NDArray[np.float64],
    temperature: float,
    z: NDArray[np.float64],
    *,
    k_guess: NDArray[np.float64] | None = None,
    skip_stability: bool | NDArray[np.bool_] = False,
    single_vapor: NDArray[np.bool_] | None = None,
    max_iter: int = _SS_MAX,
    tol: float = _SS_TOL,
) -> FlashArrays:
    """Vectorized PT flash. Same equilibrium map as scalar ``flash_tp``."""
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = np.maximum(z, 0.0)
    z = z / np.maximum(z.sum(axis=1, keepdims=True), 1.0e-30)
    p = np.asarray(pressure, dtype=float).ravel()
    n = p.size
    nc = int(eos.nc)
    if z.shape != (n, nc):
        raise ValueError(f"z shape {z.shape} != {(n, nc)}")
    t = float(temperature)
    skip = np.broadcast_to(np.asarray(skip_stability, dtype=bool), (n,)).copy()
    if single_vapor is not None:
        vap = np.broadcast_to(np.asarray(single_vapor, dtype=bool), (n,))
        force = skip
    else:
        vap = np.zeros(n, dtype=bool)
        force = np.zeros(n, dtype=bool)
    if k_guess is not None:
        k = np.clip(np.asarray(k_guess, dtype=float), 1.0e-8, 1.0e8)
        if k.ndim == 1:
            k = np.broadcast_to(k, (n, nc)).copy()
    else:
        k = np.clip(wilson_k_batch(eos, p, t), 1.0e-8, 1.0e8)
    two = (np.max(k, axis=1) > 1.0 + 1.0e-8) & (np.min(k, axis=1) < 1.0 - 1.0e-8)
    v = np.full(n, 0.5)
    if np.any(two):
        v[two] = rachford_rice_batch(k[two], z[two])
        two[two] = (v[two] > 1.0e-6) & (v[two] < 1.0 - 1.0e-6)
    need_stab = (~two) & (~force)
    margin = np.zeros(n)
    checked = np.zeros(n, dtype=bool)
    if np.any(need_stab & (~skip)):
        idx = np.where(need_stab & (~skip))[0]
        uns, mar = is_unstable_batch(eos, p[idx], t, z[idx])
        two[idx] = uns
        margin[idx] = mar
        checked[idx] = True
    single = (~two) | force
    out_v = np.zeros(n)
    out_x = z.copy()
    out_y = z.copy()
    out_zl = np.ones(n)
    out_zv = np.ones(n)
    out_vl = np.zeros(n)
    out_vv = np.zeros(n)
    two_phase = np.zeros(n, dtype=bool)
    conv = np.ones(n, dtype=bool)
    n_it = np.zeros(n, dtype=np.int32)
    err = np.zeros(n)
    fallback = np.zeros(n, dtype=bool)
    if np.any(single):
        idx = np.where(single)[0]
        if single_vapor is None:
            vap_s = np.sum(z[idx] * k[idx], axis=1) > 1.0
        else:
            vap_s = vap[idx]
        zl, zv, vl, vv, vf = _single_arrays(eos, p[idx], t, z[idx], vap_s)
        out_v[idx] = vf
        out_x[idx] = z[idx]
        out_y[idx] = z[idx]
        out_zl[idx] = zl
        out_zv[idx] = zv
        out_vl[idx] = vl
        out_vv[idx] = vv
    active = two & (~force)
    if np.any(active):
        ia = np.where(active)[0]
        ka = k[ia].copy()
        za = z[ia]
        pa = p[ia]
        va = v[ia]
        xa = za.copy()
        ya = za.copy()
        erra = np.full(ia.size, np.inf)
        ita = np.zeros(ia.size, dtype=np.int32)
        live = np.ones(ia.size, dtype=bool)
        for it in range(1, int(max_iter) + 1):
            if not np.any(live):
                break
            collapsed = (np.max(ka, axis=1) < 1.0 + 1.0e-10) & (np.min(ka, axis=1) > 1.0 - 1.0e-10)
            live = live & (~collapsed)
            if not np.any(live):
                ita = np.where(ita == 0, it, ita)
                break
            va[live] = rachford_rice_batch(ka[live], za[live])
            xa[live] = za[live] / (1.0 + va[live, None] * (ka[live] - 1.0))
            xa[live] = np.maximum(xa[live], 1.0e-16)
            xa[live] = xa[live] / xa[live].sum(axis=1, keepdims=True)
            ya[live] = ka[live] * xa[live]
            ya[live] = np.maximum(ya[live], 1.0e-16)
            ya[live] = ya[live] / ya[live].sum(axis=1, keepdims=True)
            ln_l = ln_phi_batch(eos, pa[live], t, xa[live], vapor=False)
            ln_v = ln_phi_batch(eos, pa[live], t, ya[live], vapor=True)
            k_new = np.exp(np.clip(ln_l - ln_v, -20.0, 20.0))
            k_new = np.clip(k_new, 1.0e-8, 1.0e8)
            e = np.max(np.abs(np.log(np.maximum(k_new, 1.0e-30) / np.maximum(ka[live], 1.0e-30))), axis=1)
            erra[live] = e
            ita[live] = it
            done = e < float(tol)
            sel = live.copy()
            sel[live] = done
            ka[sel] = k_new[done]
            still = live.copy()
            still[live] = ~done
            ka[still] = 0.6 * k_new[~done] + 0.4 * ka[still]
            live[live] = ~done
        va = np.clip(va, 0.0, 1.0)
        liq = va <= 1.0e-8
        vap_end = va >= 1.0 - 1.0e-8
        mid = ~(liq | vap_end)
        if np.any(liq | vap_end):
            idx2 = ia[liq | vap_end]
            vap_s = vap_end[liq | vap_end]
            zl, zv, vl, vv, vf = _single_arrays(eos, p[idx2], t, z[idx2], vap_s)
            out_v[idx2] = vf
            out_x[idx2] = z[idx2]
            out_y[idx2] = z[idx2]
            out_zl[idx2] = zl
            out_zv[idx2] = zv
            out_vl[idx2] = vl
            out_vv[idx2] = vv
            conv[idx2] = True
            n_it[idx2] = ita[liq | vap_end]
            err[idx2] = np.where(np.isfinite(erra[liq | vap_end]), erra[liq | vap_end], 0.0)
            k[idx2] = ka[liq | vap_end]
        if np.any(mid):
            im = ia[mid]
            zl, _ = z_roots_batch(eos, p[im], t, xa[mid])
            _, zv = z_roots_batch(eos, p[im], t, ya[mid])
            out_v[im] = va[mid]
            out_x[im] = xa[mid]
            out_y[im] = ya[mid]
            out_zl[im] = zl
            out_zv[im] = zv
            out_vl[im] = zl * R_GAS * t / np.maximum(p[im], 1.0e-12)
            out_vv[im] = zv * R_GAS * t / np.maximum(p[im], 1.0e-12)
            two_phase[im] = True
            e_m = erra[mid]
            conv[im] = np.isfinite(e_m) & (e_m < float(tol))
            n_it[im] = ita[mid]
            err[im] = np.where(np.isfinite(e_m), e_m, 1.0)
            k[im] = ka[mid]
    return FlashArrays(
        vapor_frac=out_v,
        x=out_x,
        y=out_y,
        z_liq=out_zl,
        z_vap=out_zv,
        v_liq=out_vl,
        v_vap=out_vv,
        two_phase=two_phase,
        k=k,
        converged=conv,
        iterations=n_it,
        fugacity_error=err,
        stability_checked=checked,
        stability_margin=margin,
        fallback_used=fallback,
    )


def flash_arrays_to_result(arr: FlashArrays, i: int = 0) -> FlashResult:
    return FlashResult(
        vapor_frac=float(arr.vapor_frac[i]),
        x=arr.x[i].copy(),
        y=arr.y[i].copy(),
        z_liq=float(arr.z_liq[i]),
        z_vap=float(arr.z_vap[i]),
        v_liq=float(arr.v_liq[i]),
        v_vap=float(arr.v_vap[i]),
        two_phase=bool(arr.two_phase[i]),
        k=arr.k[i].copy(),
        converged=bool(arr.converged[i]),
        iterations=int(arr.iterations[i]),
        fugacity_error=float(arr.fugacity_error[i]),
        stability_checked=bool(arr.stability_checked[i]),
        fallback_used=bool(arr.fallback_used[i]),
        stability_margin=float(arr.stability_margin[i]),
    )


def flash_tp_fast(
    eos: PengRobinson,
    pressure: float,
    temperature: float,
    z: NDArray[np.float64],
    **kwargs,
) -> FlashResult:
    """Single-state FastPR. Falls back to scalar reference if not converged."""
    z = _frac(z, eos.nc)
    k_guess = kwargs.get("k_guess")
    arr = flash_batch(
        eos,
        np.array([float(pressure)]),
        float(temperature),
        z.reshape(1, -1),
        k_guess=None if k_guess is None else np.asarray(k_guess, dtype=float).reshape(1, -1),
        skip_stability=bool(kwargs.get("skip_stability", False)),
        single_vapor=None if kwargs.get("single_vapor") is None else np.array([bool(kwargs["single_vapor"])]),
        max_iter=int(kwargs.get("max_iter", _SS_MAX)),
        tol=float(kwargs.get("tol", _SS_TOL)),
    )
    if not bool(arr.converged[0]):
        fl = flash_tp(eos, pressure, temperature, z, **{k: v for k, v in kwargs.items() if k != "k_guess"})
        fl.fallback_used = True
        return fl
    return flash_arrays_to_result(arr, 0)
