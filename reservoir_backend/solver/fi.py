"""Fully implicit black-oil Newton with live-oil variable switching.

Unknowns are (p, Sw, x). Residuals are surface-volume conservation of
water, oil and gas.

When the fluid has dissolved gas:
  no free gas:  x = Rs, Sg = 0
  oil and gas:  x = Sg, Rs = RsSat(p)

Appearance and disappearance are two-stage on the Newton increment
(``switch_live_oil_unknown``). Liberated gas uses a volume-balance flash
(not a small steady Sg cap). Newton failure rejects the step (chop dt).

Local names follow docs/fim_name_map.md (licensed adaptation; no upstream IDs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from reservoir_backend.discretization.tpfa import geometric_transmissibility, phase_interior_fluxes
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import PSI, BlackOilPVT
from reservoir_backend.solver.seqtools import (
    TARGET_ITERATION_COUNT,
    TOLERANCE_CNV,
    TOLERANCE_MB,
    NewtonRelaxation,
    cross_flow_mixture,
    iteration_count_timestep,
)


@dataclass
class FiStepResult:
    """Accepted FIM step: state, total reservoir face fluxes, Newton count."""

    pressure: NDArray[np.float64]
    sw: NDArray[np.float64]
    sg: NDArray[np.float64]
    rs: NDArray[np.float64]
    fx: NDArray[np.float64]
    fy: NDArray[np.float64]
    fz: NDArray[np.float64]
    newton_iters: int


def clip_saturation_increment(
    du: NDArray[np.float64],
    n: int,
    unsat: NDArray[np.bool_],
    *,
    ds_max: float = 0.20,
    rs_ref: float = 1.0,
    pref: float = 1.0e5,
    dp_rel_max: float = 0.20,
) -> NDArray[np.float64]:
    """Scale Newton Δ so max |ΔS| and relative |Δp|, |ΔRs| stay bounded."""
    out = np.asarray(du, dtype=float).ravel().copy()
    dsw = out[n : 2 * n]
    peak_sw = float(np.max(np.abs(dsw))) if dsw.size else 0.0
    if peak_sw > ds_max:
        out[n : 2 * n] = dsw * (ds_max / peak_sw)
    dx = out[2 * n :]
    if np.any(~unsat):
        peak_sg = float(np.max(np.abs(dx[~unsat])))
        if peak_sg > ds_max:
            dx = dx.copy()
            dx[~unsat] *= ds_max / peak_sg
    if np.any(unsat):
        peak_rs = float(np.max(np.abs(dx[unsat]))) / max(rs_ref, 1.0)
        if peak_rs > ds_max:
            dx = dx.copy()
            dx[unsat] *= ds_max / peak_rs
    out[2 * n :] = dx
    peak_p = float(np.max(np.abs(out[:n]))) / max(pref, 1.0)
    if peak_p > dp_rel_max:
        out[:n] = out[:n] * (dp_rel_max / peak_p)
    return out


def scale_newton_update(du: NDArray[np.float64], *, alpha: float = 1.0) -> NDArray[np.float64]:
    """Global scale of a Newton increment (0 < alpha ≤ 1)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return np.asarray(du, dtype=float).ravel() * a


def cell_cnv_ok(
    residual: NDArray[np.float64],
    scale: NDArray[np.float64],
    *,
    tol: float = TOLERANCE_CNV,
) -> bool:
    """Cell CNV: max |R_i| / scale_i ≤ tol."""
    r = np.asarray(residual, dtype=float).ravel()
    s = np.maximum(np.asarray(scale, dtype=float).ravel(), 1.0e-30)
    if r.size != s.size or not np.all(np.isfinite(r)):
        return False
    return float(np.max(np.abs(r) / s)) <= float(tol)


def global_mass_balance_ok(
    residual: NDArray[np.float64],
    n: int,
    acc_scale: NDArray[np.float64],
    *,
    tol: float = TOLERANCE_MB,
) -> bool:
    """Global mass-balance: |Σ R| / Σ|acc_scale| ≤ tol per equation block."""
    r = np.asarray(residual, dtype=float).ravel()
    s = np.maximum(np.asarray(acc_scale, dtype=float).ravel(), 1.0e-30)
    if r.size != 3 * n or s.size != 3 * n:
        return False
    for k in range(3):
        sl = slice(k * n, (k + 1) * n)
        den = float(np.sum(s[sl]))
        if den <= 0.0:
            continue
        if abs(float(np.sum(r[sl]))) / den > float(tol):
            return False
    return True


def dt_from_newton_iters(
    dt: float,
    newton_iters: int,
    *,
    dt0: float | None = None,
    its0: int | None = None,
    dt_min: float = 1.0e-6,
    dt_max: float = 1.0e30,
    target_its: int = TARGET_ITERATION_COUNT,
) -> float:
    """Grow/shrink Δt from successful Newton iteration count."""
    return float(
        iteration_count_timestep(
            float(dt),
            int(newton_iters),
            dt0=dt0,
            its0=its0,
            target=int(target_its),
            dt_min=float(dt_min),
            dt_max=float(dt_max),
        )
    )


def _clip_sw_sg(sw: NDArray[np.float64], sg: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sw = np.maximum(np.asarray(sw, dtype=float).ravel(), 0.0)
    sg = np.maximum(np.asarray(sg, dtype=float).ravel(), 0.0)
    tot = sw + sg
    over = tot > 1.0
    if np.any(over):
        sw = sw.copy()
        sg = sg.copy()
        sw[over] = sw[over] / tot[over]
        sg[over] = sg[over] / tot[over]
    return sw, sg


def liberate_excess_gas(
    fluid: BlackOilPVT,
    sw: NDArray[np.float64],
    sg: NDArray[np.float64],
    rs: NDArray[np.float64],
    unsat: NDArray[np.bool_] | None,
    p: NDArray[np.float64],
    *,
    live: bool,
    grow_max: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Flash excess Rs into Sg.

    If ``grow_max`` is set, free-gas appearance per residual/switch eval is
    limited to that increment (Newton stability). ``None`` uses the full
    volume-balance flash (liquid-limited only).
    """
    sg_e = np.asarray(sg, dtype=float).ravel().copy()
    rs_e = np.asarray(rs, dtype=float).ravel().copy()
    n = sg_e.size
    grow = np.zeros(n, dtype=bool)
    at_cap = np.zeros(n, dtype=bool)
    if not live or unsat is None:
        return sg_e, rs_e, grow, at_cap
    un = np.asarray(unsat, dtype=bool).ravel()
    rs_sat = np.asarray(fluid.rs(p), dtype=float).ravel()
    grow = un & (rs_e > rs_sat + 1.0e-10)
    if not np.any(grow):
        return sg_e, rs_e, grow, at_cap
    sw_a = np.asarray(sw, dtype=float).ravel()
    g_hold = fluid.surface_gas_holdup(sw_a, np.zeros(n), p, rs=rs_e)
    sg_f = np.asarray(fluid.flash_from_total(sw_a, g_hold, p), dtype=float).ravel()
    sl = np.maximum(1.0 - sw_a, 0.0)
    if grow_max is None:
        lid = sl
    else:
        lid = np.minimum(sl, sg_e + float(grow_max))
    take = np.clip(sg_f, 0.0, lid)
    sg_e[grow] = take[grow]
    bg = np.maximum(np.asarray(fluid.b_g(p), dtype=float).ravel(), 1.0e-30)
    bo = np.asarray(fluid.b_o(p, rs=rs_sat, saturated=True), dtype=float).ravel()
    so = np.maximum(sl - sg_e, 1.0e-12)
    g_left = np.maximum(g_hold - bg * sg_e, 0.0)
    rs_keep = g_left / np.maximum(bo * so, 1.0e-30)
    rs_e[grow] = np.maximum(rs_keep[grow], rs_sat[grow])
    at_cap = grow & (sg_f > take + 1.0e-12)
    return sg_e, rs_e, grow, at_cap


def switch_live_oil_unknown(
    fluid: BlackOilPVT,
    p: NDArray[np.float64],
    sw: NDArray[np.float64],
    x: NDArray[np.float64],
    unsat: NDArray[np.bool_],
    near: NDArray[np.bool_],
    *,
    live: bool,
    eps_s: float = 1.0e-10,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.bool_],
    NDArray[np.bool_],
    NDArray[np.float64],
]:
    """Two-stage live-oil unknown switch on a Newton increment of x.

    First bubble hit marks the cell; second hit liberates volume-balance Sg.
    Disappearance is two-stage. Dead oil keeps x = Sg.
    """
    sw_c = np.clip(np.asarray(sw, dtype=float).ravel(), 0.0, 1.0)
    x_a = np.asarray(x, dtype=float).ravel()
    n = sw_c.size
    if not live:
        sw_c, sg_c = _clip_sw_sg(sw_c, x_a)
        zbool = np.zeros(n, dtype=bool)
        return sw_c, sg_c, np.zeros(n, dtype=float), zbool, zbool, sg_c.copy()
    unsat_n = np.asarray(unsat, dtype=bool).ravel().copy()
    near_n = np.asarray(near, dtype=bool).ravel().copy()
    rs_sat = np.asarray(fluid.rs(p), dtype=float).ravel()
    eps_rs = max(1.0e-6 * max(float(np.mean(np.abs(rs_sat))), 1.0), 1.0e-12)
    sl = np.maximum(1.0 - sw_c, 0.0)
    sg_c = np.zeros(n, dtype=float)
    rs_c = rs_sat.copy()
    sat = ~unsat_n
    sg_c[sat] = x_a[sat]
    rs_c[unsat_n] = np.maximum(x_a[unsat_n], 0.0)
    grow = unsat_n & (rs_c >= rs_sat)
    dry = sat & (sg_c <= 0.0)
    first_g = grow & ~near_n
    second_g = grow & near_n
    first_d = dry & ~near_n
    second_d = dry & near_n
    if np.any(first_g):
        sg_c[first_g] = 0.0
        unsat_n[first_g] = True
        near_n[first_g] = True
    if np.any(second_g):
        sg_e, rs_e, _grow, at_cap = liberate_excess_gas(
            fluid, sw_c, sg_c, rs_c, unsat_n, p, live=True, grow_max=0.20
        )
        go = second_g & ~at_cap
        hold = second_g & at_cap
        sg_c[go] = np.maximum(sg_e[go], np.minimum(eps_s, sl[go]))
        rs_c[go] = rs_sat[go]
        unsat_n[go] = False
        sg_c[hold] = 0.0
        unsat_n[hold] = True
        near_n[hold] = True
    if np.any(first_d):
        sg_c[first_d] = np.minimum(eps_s, sl[first_d])
        rs_c[first_d] = rs_sat[first_d]
        unsat_n[first_d] = False
        near_n[first_d] = True
    if np.any(second_d):
        sg_c[second_d] = 0.0
        rs_c[second_d] = np.maximum(rs_sat[second_d] - eps_rs, 0.0)
        unsat_n[second_d] = True
    pb_rs = np.asarray(fluid.pbub_of_rs(rs_c), dtype=float).ravel()
    at_bub = np.asarray(p, dtype=float).ravel() <= pb_rs + 1.0 * PSI
    close = unsat_n & at_bub & (rs_c >= rs_sat - 10.0 * eps_rs)
    near_n[close] = True
    near_n[~(grow | dry | close)] = False
    sw_c, sg_c = _clip_sw_sg(sw_c, np.maximum(sg_c, 0.0))
    x_n = np.where(unsat_n, rs_c, sg_c)
    return sw_c, sg_c, rs_c, unsat_n, near_n, x_n


# Tests and older call sites.
switch_vo_unknown = switch_live_oil_unknown
_capped_excess_gas = liberate_excess_gas


def _div(grid: CartesianGrid, fx, fy, fz) -> NDArray[np.float64]:
    return (fx[:, :, 1:] - fx[:, :, :-1] + fy[:, 1:, :] - fy[:, :-1, :] + fz[1:, :, :] - fz[:-1, :, :]).ravel()


def _lambda(three, fluid: BlackOilPVT, sw, sg, p, rs=None, saturated=None):
    krw, kro, krg = three.kr(sw, sg)
    mu_o = np.maximum(fluid.viscosity_o(p, rs=rs, saturated=saturated), 1.0e-30)
    mu_g = np.maximum(fluid.viscosity_g(p), 1.0e-30)
    return (
        np.asarray(krw, dtype=float).ravel() / max(float(three.mu_w), 1.0e-30),
        np.asarray(kro, dtype=float).ravel() / mu_o,
        np.asarray(krg, dtype=float).ravel() / mu_g,
    )


def _db_dp(
    fluid: BlackOilPVT,
    p: NDArray[np.float64],
    rs: NDArray[np.float64] | None = None,
    eps: float = 1.0e3,
):
    bw = fluid.b_w(p)
    bo = fluid.b_o(p, rs=rs)
    bg = fluid.b_g(p)
    return (
        (fluid.b_w(p + eps) - fluid.b_w(p - eps)) / (2.0 * eps),
        (fluid.b_o(p + eps, rs=rs) - fluid.b_o(p - eps, rs=rs)) / (2.0 * eps),
        (fluid.b_g(p + eps) - fluid.b_g(p - eps)) / (2.0 * eps),
        bw,
        bo,
        bg,
        fluid.rs(p),
        fluid.drs_dp(p),
    )


def _well_surface_rates(
    wi_base: dict[int, tuple[float, float]],
    wi_comp: dict[int, tuple[float, float]] | None,
    wi_group: dict[int, int] | None,
    p: NDArray[np.float64],
    lw: NDArray[np.float64],
    lo: NDArray[np.float64],
    lg: NDArray[np.float64],
    bw: NDArray[np.float64],
    bo: NDArray[np.float64],
    bg: NDArray[np.float64],
    rs: NDArray[np.float64],
    *,
    wi_datum: dict[int, float] | None = None,
    z: NDArray[np.float64] | None = None,
    gravity: float = 0.0,
    rho_w: NDArray[np.float64] | None = None,
    rho_o: NDArray[np.float64] | None = None,
    rho_g: NDArray[np.float64] | None = None,
    lt_fixed: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Surface well rates. Injecting perfs use mixed wellbore composition.

    Production is mobility-split plus dissolved gas (OPM gasOilPerfRateProd).
    Injection uses total mobility and cmix / Σ(cmix/b) (OPM volumeRatio).
    ``lt_fixed`` freezes connection qT (typically oil+water mobility at
    t^n) so free-gas mobility cannot collapse the drawdown.
    """
    n = p.size
    qw_s = np.zeros(n, dtype=float)
    qo_s = np.zeros(n, dtype=float)
    qg_s = np.zeros(n, dtype=float)
    cells = [int(c) for c in wi_base]
    if not cells:
        return qw_s, qo_s, qg_s
    names: list[int] = []
    name_to_idx: dict[int, int] = {}
    for c in cells:
        wid = int(wi_group[c]) if wi_group is not None and c in wi_group else c
        if wid not in name_to_idx:
            name_to_idx[wid] = len(name_to_idx)
        names.append(wid)
    n_wells = len(name_to_idx)
    nperf = len(cells)
    flux = np.zeros((nperf, 3), dtype=float)
    compi = np.zeros((n_wells, 3), dtype=float)
    q_res_c = np.zeros(nperf, dtype=float)
    perf2well = np.zeros(nperf, dtype=np.int64)
    for i, c in enumerate(cells):
        base, pbhp = wi_base[c]
        lt_now = max(float(lw[c] + lo[c] + lg[c]), 1.0e-30)
        lt_q = lt_now if lt_fixed is None else max(float(lt_fixed[c]), 1.0e-30)
        pull = float(pbhp) - float(p[c])
        q_res_c[i] = float(base) * lt_q * pull
        flux[i, 0] = float(lw[c]) * pull
        flux[i, 1] = float(lo[c]) * pull
        flux[i, 2] = float(lg[c]) * pull
        wid = name_to_idx[names[i]]
        perf2well[i] = wid
        sw_src, sg_src = (1.0, 0.0) if not wi_comp or c not in wi_comp else wi_comp[c]
        so_src = max(0.0, 1.0 - float(sw_src) - float(sg_src))
        compi[wid] = [float(sw_src), so_src, float(sg_src)]
    pbhp_now = {c: float(wi_base[c][1]) for c in cells}
    if wi_datum is not None and z is not None and abs(float(gravity)) > 1.0e-15 and rho_o is not None:
        wells: list[list[int]] = [[] for _ in range(n_wells)]
        for i, c in enumerate(cells):
            wells[int(perf2well[i])].append(c)
        for wid, mems in enumerate(wells):
            if not mems:
                continue
            z_ref = max(float(z[c]) for c in mems)
            pulls = [float(wi_datum.get(c, wi_base[c][1])) - float(p[c]) for c in mems]
            if float(np.mean(pulls)) >= 0.0:
                sw_m, so_m, sg_m = (float(x) for x in compi[wid])
                rw = float(np.mean(rho_w[mems])) if rho_w is not None else 1000.0
                ro = float(np.mean(rho_o[mems]))
                rg = float(np.mean(rho_g[mems])) if rho_g is not None else 1.0
                rho_wb = sw_m * rw + so_m * ro + sg_m * rg
            else:
                num = 0.0
                den = 0.0
                for c in mems:
                    lt = max(float(lw[c] + lo[c] + lg[c]), 1.0e-30)
                    rw = float(rho_w[c]) if rho_w is not None else 1000.0
                    ro = float(rho_o[c])
                    rg = float(rho_g[c]) if rho_g is not None else 1.0
                    rho_c = (float(lw[c]) * rw + float(lo[c]) * ro + float(lg[c]) * rg) / lt
                    num += lt * rho_c
                    den += lt
                rho_wb = num / max(den, 1.0e-30)
            if not np.isfinite(rho_wb) or rho_wb <= 0.0:
                continue
            for c in mems:
                datum = float(wi_datum.get(c, wi_base[c][1]))
                pbhp_now[c] = datum + float(rho_wb) * float(gravity) * (z_ref - float(z[c]))
        for i, c in enumerate(cells):
            base, _old = wi_base[c]
            lt = max(float(lw[c] + lo[c] + lg[c]), 1.0e-30)
            pull = float(pbhp_now[c]) - float(p[c])
            q_res_c[i] = float(base) * lt * pull
            flux[i, 0] = float(lw[c]) * pull
            flux[i, 1] = float(lo[c]) * pull
            flux[i, 2] = float(lg[c]) * pull
    mixed = cross_flow_mixture(flux, compi, perf2well, n_wells)
    for i, c in enumerate(cells):
        q_res = float(q_res_c[i])
        sw_m, so_m, sg_m = (float(x) for x in mixed[int(perf2well[i])])
        if q_res >= 0.0:
            vr = sw_m / max(float(bw[c]), 1.0e-30) + so_m / max(float(bo[c]), 1.0e-30)
            vr += sg_m / max(float(bg[c]), 1.0e-30)
            q_s = q_res / max(vr, 1.0e-30)
            qw_s[c] += sw_m * q_s
            qo_s[c] += so_m * q_s
            qg_s[c] += sg_m * q_s
            continue
        base, _ = wi_base[c]
        if (
            wi_datum is not None
            and z is not None
            and rho_o is not None
            and abs(float(gravity)) > 1.0e-15
        ):
            wid = int(perf2well[i])
            mems = [cc for j, cc in enumerate(cells) if int(perf2well[j]) == wid]
            z_ref = max(float(z[cc]) for cc in mems)
            datum = float(wi_datum.get(c, pbhp_now[c]))
            head = float(gravity) * (z_ref - float(z[c]))
            rw = float(rho_w[c]) if rho_w is not None else 1000.0
            ro = float(rho_o[c])
            rg = float(rho_g[c]) if rho_g is not None else 1.0
            qw_r = float(base) * float(lw[c]) * (datum + rw * head - float(p[c]))
            qo_r = float(base) * float(lo[c]) * (datum + ro * head - float(p[c]))
            qg_r = float(base) * float(lg[c]) * (datum + rg * head - float(p[c]))
            qw_s[c] += qw_r * float(bw[c])
            qo_s[c] += qo_r * float(bo[c])
            qg_s[c] += qg_r * float(bg[c]) + qo_r * float(rs[c]) * float(bo[c])
            continue
        lt = max(float(lw[c] + lo[c] + lg[c]), 1.0e-30)
        qw_s[c] += q_res * (float(lw[c]) / lt) * float(bw[c])
        qo_s[c] += q_res * (float(lo[c]) / lt) * float(bo[c])
        qg_s[c] += q_res * (
            (float(lg[c]) / lt) * float(bg[c]) + (float(lo[c]) / lt) * float(rs[c]) * float(bo[c])
        )
    return qw_s, qo_s, qg_s


def solve_fi_step(
    grid: CartesianGrid,
    rock,
    three_phase,
    fluid: BlackOilPVT,
    capillary,
    sw0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    p0: NDArray[np.float64],
    p_init: NDArray[np.float64],
    dt: float,
    gravity: float,
    *,
    src_w: NDArray[np.float64],
    src_o: NDArray[np.float64],
    src_g: NDArray[np.float64],
    rs0: NDArray[np.float64] | None = None,
    wi_base: dict[int, tuple[float, float]] | None = None,
    wi_comp: dict[int, tuple[float, float]] | None = None,
    wi_group: dict[int, int] | None = None,
    wi_datum: dict[int, float] | None = None,
    lt_fixed: NDArray[np.float64] | None = None,
    cell_dirichlet: dict[int, float] | None = None,
    face_mult_x=None,
    face_mult_y=None,
    face_mult_z=None,
    nltol: float = 1.0e-4,
    maxnewt: int = 25,
    lstrials: int = 12,
) -> FiStepResult | None:
    """Return FIM state + total face fluxes at t^{n+1}, or None if Newton fails."""
    n = grid.n_cells
    vol = grid.cell_volumes()
    phi = np.asarray(rock.porosity, dtype=float).ravel()
    k = np.asarray(rock.permeability, dtype=float).ravel()
    kz = rock.vertical_permeability()
    z_cell = np.asarray(grid.cell_centers()[:, 2], dtype=float).ravel()
    sw0 = np.asarray(sw0, dtype=float).ravel()
    sg0 = np.asarray(sg0, dtype=float).ravel()
    p0 = np.asarray(p0, dtype=float).ravel()
    so0 = np.clip(1.0 - sw0 - sg0, 0.0, 1.0)
    live = bool(fluid.has_live_oil())
    rs_sat0 = np.asarray(fluid.rs(p0), dtype=float).ravel()
    if rs0 is None:
        rs0 = rs_sat0.copy()
    else:
        rs0 = np.asarray(rs0, dtype=float).ravel()
    unsat = fluid.vo_unsat(sg0) if live else np.zeros(n, dtype=bool)
    if live:
        rs0 = np.where(unsat, np.minimum(rs0, rs_sat0), rs_sat0)
    else:
        rs0 = np.zeros(n, dtype=float)
    pv0 = phi * fluid.pv_mult(p0) * vol
    acc_w0 = pv0 * fluid.b_w(p0) * sw0
    acc_o0 = pv0 * fluid.b_o(p0, rs=rs0) * so0
    acc_g0 = pv0 * fluid.surface_gas_holdup(sw0, sg0, p0, rs=rs0)
    src_w = np.asarray(src_w, dtype=float).ravel()
    src_o = np.asarray(src_o, dtype=float).ravel()
    src_g = np.asarray(src_g, dtype=float).ravel()
    dt = float(dt)
    p = np.asarray(p_init, dtype=float).ravel().copy()
    sw, sg = _clip_sw_sg(sw0, sg0)
    rs = rs0.copy()
    pref = max(float(np.mean(np.abs(p0))), 1.0e5)
    rs_ref = max(float(np.mean(np.abs(rs_sat0))), 1.0) if live else 1.0
    if live:
        pb0 = np.asarray(fluid.pbub_of_rs(rs0), dtype=float).ravel()
        near = unsat & (p0 <= pb0 + 1.0 * PSI)
    else:
        near = np.zeros(n, dtype=bool)

    use_head = False  # head in Newton historically destabilized CMG wells; report-time only

    def _well_pack(p_a, lw, lo, lg, bw, bo, bg, rs_e, rho_w, rho_o, rho_g):
        if not wi_base:
            return (
                np.zeros(n, dtype=float),
                np.zeros(n, dtype=float),
                np.zeros(n, dtype=float),
            )
        return _well_surface_rates(
            wi_base,
            wi_comp,
            wi_group,
            p_a,
            lw,
            lo,
            lg,
            bw,
            bo,
            bg,
            rs_e,
            wi_datum=None,
            z=None,
            gravity=0.0,
            rho_w=None,
            rho_o=None,
            rho_g=None,
            lt_fixed=lt_fixed,
        )

    def _residual(p_a, sw_a, sg_a, rs_a, unsat_a=None):
        # Trust primary-variable state from switch_live_oil_unknown.
        # Re-liberating every residual eval with a grow lid makes R non-smooth
        # and fights the Newton update (observed CNV oscillation ~0.2).
        if unsat_a is None:
            unsat_b = fluid.vo_unsat(sg_a) if live else np.zeros(n, dtype=bool)
        else:
            unsat_b = np.asarray(unsat_a, dtype=bool).ravel()
        sg_e = np.where(unsat_b, 0.0, np.asarray(sg_a, dtype=float).ravel())
        rs_e = np.asarray(rs_a, dtype=float).ravel()
        if live:
            rs_sat = np.asarray(fluid.rs(p_a), dtype=float).ravel()
            rs_e = np.where(unsat_b, np.minimum(rs_e, rs_sat), rs_sat)
        grow = np.zeros(n, dtype=bool)
        at_cap = np.zeros(n, dtype=bool)
        so = np.clip(1.0 - sw_a - sg_e, 0.0, 1.0)
        sat = ~unsat_b
        lw, lo, lg = _lambda(three_phase, fluid, sw_a, sg_e, p_a, rs=rs_e, saturated=sat)
        bw = fluid.b_w(p_a)
        bo = fluid.b_o(p_a, rs=rs_e, saturated=sat)
        bg = fluid.b_g(p_a)
        pv = phi * fluid.pv_mult(p_a) * vol
        pc = None if capillary is None or isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw_a), dtype=float).ravel()
        rho_w = fluid.density_w(p_a, bw=bw)
        rho_o = fluid.density_o(p_a, rs=rs_e, bo=bo)
        rho_g = fluid.density_g(p_a, bg=bg)
        qw_x, qw_y, qw_z, qo_x, qo_y, qo_z, qg_x, qg_y, qg_z = phase_interior_fluxes(
            grid,
            p_a,
            k,
            lw,
            lo,
            lg=lg,
            kz=kz,
            mult_x=face_mult_x,
            mult_y=face_mult_y,
            mult_z=face_mult_z,
            gravity=float(gravity),
            rho_w=rho_w,
            rho_o=rho_o,
            rho_g=rho_g,
            pc=pc,
        )

        def _up_b(qx, qy, qz, b):
            b_ijk = grid.reshape_ijk(b)
            bx = np.zeros_like(qx)
            by = np.zeros_like(qy)
            bz = np.zeros_like(qz)
            if grid.nx > 1:
                bx[:, :, 1:-1] = np.where(qx[:, :, 1:-1] >= 0.0, b_ijk[:, :, :-1], b_ijk[:, :, 1:])
            if grid.ny > 1:
                by[:, 1:-1, :] = np.where(qy[:, 1:-1, :] >= 0.0, b_ijk[:, :-1, :], b_ijk[:, 1:, :])
            if grid.nz > 1:
                bz[1:-1, :, :] = np.where(qz[1:-1, :, :] >= 0.0, b_ijk[:-1, :, :], b_ijk[1:, :, :])
            return qx * bx, qy * by, qz * bz

        fw_x, fw_y, fw_z = _up_b(qw_x, qw_y, qw_z, bw)
        fo_x, fo_y, fo_z = _up_b(qo_x, qo_y, qo_z, bo)
        fg_x, fg_y, fg_z = _up_b(qg_x, qg_y, qg_z, bg)
        rsbo = rs_e * bo
        rg_x, rg_y, rg_z = _up_b(qo_x, qo_y, qo_z, rsbo)
        qw_s = src_w.copy()
        qo_s = src_o.copy()
        qg_s = src_g.copy()
        dw, do, dg = _well_pack(p_a, lw, lo, lg, bw, bo, bg, rs_e, rho_w, rho_o, rho_g)
        qw_s += dw
        qo_s += do
        qg_s += dg
        rw = (pv * bw * sw_a - acc_w0) / dt + _div(grid, fw_x, fw_y, fw_z) - qw_s
        ro = (pv * bo * so - acc_o0) / dt + _div(grid, fo_x, fo_y, fo_z) - qo_s
        rg = (pv * (bg * sg_e + rs_e * bo * so) - acc_g0) / dt + _div(grid, fg_x + rg_x, fg_y + rg_y, fg_z + rg_z) - qg_s
        if cell_dirichlet:
            for c, pbc in cell_dirichlet.items():
                rw[int(c)] = p_a[int(c)] - float(pbc)
        return np.concatenate([rw, ro, rg]), (lw, lo, lg, bw, bo, bg, rs_e, pv, sg_e, grow, at_cap)

    def _jacobian(p_a, sw_a, sg_a, rs_a, unsat_a, pack):
        lw, lo, lg, bw, bo, bg, rs, pv, sg_e, grow, at_cap = pack
        so = np.clip(1.0 - sw_a - sg_e, 0.0, 1.0)
        sat = (~unsat_a) | grow
        dbw, dbo, dbg, _bw, _bo, _bg, _rs_sat, drs_sat = _db_dp(fluid, p_a, rs=rs)
        dbo = (fluid.b_o(p_a + 1.0e3, rs=rs, saturated=sat) - fluid.b_o(p_a - 1.0e3, rs=rs, saturated=sat)) / 2.0e3
        eps_rs = max(1.0e-3 * max(float(np.mean(np.abs(rs_a))), 1.0), 1.0e-8)
        dbo_drs = (
            fluid.b_o(p_a, rs=rs_a + eps_rs, saturated=sat)
            - fluid.b_o(p_a, rs=np.maximum(rs_a - eps_rs, 0.0), saturated=sat)
        ) / (2.0 * eps_rs)
        dpv = phi * float(fluid.cr) * vol
        sl = np.maximum(1.0 - sw_a, 0.0)
        denom = np.maximum(bg - rs * bo, 1.0e-12)
        dsg_drs = np.where(grow & ~at_cap, bo * sl / denom, 0.0)
        dsg_drs = np.where(~unsat_a, 1.0, dsg_drs)
        dW_dp = (dpv * bw + pv * dbw) * sw_a / dt
        dW_dsw = pv * bw / dt
        dO_dp = (dpv * bo + pv * dbo) * so / dt
        dO_dsw = -pv * bo / dt
        dO_dx = np.where(sat, -pv * bo / dt, pv * dbo_drs * so / dt)
        dO_dx = np.where(grow & ~at_cap, (-pv * bo / dt) * dsg_drs, dO_dx)
        dO_dx = np.where(grow & at_cap, pv * dbo_drs * so / dt, dO_dx)
        hold = bg * sg_e + rs * bo * so
        dG_dp_sat = (dpv * hold + pv * (dbg * sg_e + drs_sat * bo * so + rs * dbo * so)) / dt
        dG_dp_unsat = (dpv * hold + pv * (rs * dbo * so)) / dt
        dG_dp = np.where(unsat_a & ~grow, dG_dp_unsat, dG_dp_sat)
        dG_dsw = pv * rs * bo * (-1.0) / dt
        dG_dx_unsat = pv * (bo + rs_a * dbo_drs) * so / dt
        dG_dx = np.where(unsat_a & ~grow, dG_dx_unsat, pv * (bg - rs * bo) / dt)
        dG_dx = np.where(grow & ~at_cap, pv * (bg - rs * bo) / dt * dsg_drs, dG_dx)
        dG_dx = np.where(grow & at_cap, dG_dx_unsat, dG_dx)
        rows = [
            np.arange(n),
            np.arange(n),
            np.arange(n, 2 * n),
            np.arange(n, 2 * n),
            np.arange(n, 2 * n),
            np.arange(2 * n, 3 * n),
            np.arange(2 * n, 3 * n),
            np.arange(2 * n, 3 * n),
        ]
        cols = [
            np.arange(n),
            np.arange(n, 2 * n),
            np.arange(n),
            np.arange(n, 2 * n),
            np.arange(2 * n, 3 * n),
            np.arange(n),
            np.arange(n, 2 * n),
            np.arange(2 * n, 3 * n),
        ]
        data = [dW_dp, dW_dsw, dO_dp, dO_dsw, dO_dx, dG_dp, dG_dsw, dG_dx]

        tx, ty, tz = geometric_transmissibility(
            grid, k, kz=kz, mult_x=face_mult_x, mult_y=face_mult_y, mult_z=face_mult_z
        )
        eps_s = 1.0e-5
        lw_h, lo_h, lg_h = _lambda(three_phase, fluid, np.clip(sw_a + eps_s, 0.0, 1.0), sg_e, p_a, rs=rs, saturated=sat)
        lw_l, lo_l, lg_l = _lambda(three_phase, fluid, np.clip(sw_a - eps_s, 0.0, 1.0), sg_e, p_a, rs=rs, saturated=sat)
        dlw_sw = (lw_h - lw_l) / (2.0 * eps_s)
        dlo_sw = (lo_h - lo_l) / (2.0 * eps_s)
        dlg_sw = (lg_h - lg_l) / (2.0 * eps_s)
        lw_h, lo_h, lg_h = _lambda(three_phase, fluid, sw_a, np.clip(sg_e + eps_s, 0.0, 1.0), p_a, rs=rs, saturated=sat)
        lw_l, lo_l, lg_l = _lambda(three_phase, fluid, sw_a, np.clip(sg_e - eps_s, 0.0, 1.0), p_a, rs=rs, saturated=sat)
        sat_f = sat.astype(float)
        dxsg = np.where(grow, dsg_drs, sat_f)
        dlw_sg = dxsg * (lw_h - lw_l) / (2.0 * eps_s)
        dlo_sg = dxsg * (lo_h - lo_l) / (2.0 * eps_s)
        dlg_sg = dxsg * (lg_h - lg_l) / (2.0 * eps_s)
        lw_i, lo_i, lg_i = grid.reshape_ijk(lw), grid.reshape_ijk(lo), grid.reshape_ijk(lg)
        dws_i, dos_i, dgs_i = grid.reshape_ijk(dlw_sw), grid.reshape_ijk(dlo_sw), grid.reshape_ijk(dlg_sw)
        dwg_i, dog_i, dgg_i = grid.reshape_ijk(dlw_sg), grid.reshape_ijk(dlo_sg), grid.reshape_ijk(dlg_sg)
        bw_i, bo_i, bg_i = grid.reshape_ijk(bw), grid.reshape_ijk(bo), grid.reshape_ijk(bg)
        rs_i = grid.reshape_ijk(rs)
        un_i = grid.reshape_ijk((unsat_a & ~grow).astype(float))
        if capillary is None or isinstance(capillary, NoCapillary):
            pc_a = np.zeros(n, dtype=float)
            dpc_a = np.zeros(n, dtype=float)
        else:
            pc_a = np.asarray(capillary.pc(sw_a), dtype=float).ravel()
            dpc_fn = getattr(capillary, "dpc_dsw", None)
            dpc_a = np.asarray(dpc_fn(sw_a), dtype=float).ravel() if dpc_fn is not None else np.zeros(n, dtype=float)
        pc_i = grid.reshape_ijk(pc_a)
        dpc_i = grid.reshape_ijk(dpc_a)
        dbw_i, dbo_i, dbg_i = grid.reshape_ijk(dbw), grid.reshape_ijk(dbo), grid.reshape_ijk(dbg)
        drs_i = grid.reshape_ijk(np.where(unsat_a & ~grow, 0.0, drs_sat))
        p_ijk = grid.reshape_ijk(p_a)
        z_ijk = grid.reshape_ijk(grid.cell_centers()[:, 2])

        def _faces(
            t_geom, left, right, p_l, p_r, z_l, z_r, lw_l, lw_r, lo_l, lo_r, lg_l, lg_r,
            dws_l, dws_r, dos_l, dos_r, dgs_l, dgs_r, dwg_l, dwg_r, dog_l, dog_r, dgg_l, dgg_r,
            bw_l, bw_r, bo_l, bo_r, bg_l, bg_r, rs_l, rs_r, un_l, un_r,
            dbw_l, dbw_r, dbo_l, dbo_r, dbg_l, dbg_r, drs_l, drs_r,
            pc_l, pc_r, dpc_l, dpc_r,
        ):
            g = float(gravity)
            dz = z_l - z_r
            rho_w = 0.5 * (fluid.rho_w_sc * bw_l + fluid.rho_w_sc * bw_r)
            rho_o = 0.5 * (
                (fluid.rho_o_sc + rs_l * fluid.rho_g_sc) * bo_l
                + (fluid.rho_o_sc + rs_r * fluid.rho_g_sc) * bo_r
            )
            rho_g = 0.5 * (fluid.rho_g_sc * bg_l + fluid.rho_g_sc * bg_r)
            dphi_w = (p_l - pc_l) - (p_r - pc_r) + rho_w * g * dz
            dphi_o = (p_l - p_r) + rho_o * g * dz
            dphi_g = (p_l - p_r) + rho_g * g * dz
            upw, upo, upg = dphi_w >= 0.0, dphi_o >= 0.0, dphi_g >= 0.0
            lw_f = np.where(upw, lw_l, lw_r)
            lo_f = np.where(upo, lo_l, lo_r)
            lg_f = np.where(upg, lg_l, lg_r)
            tw, to, tg = t_geom * lw_f, t_geom * lo_f, t_geom * lg_f
            bw_f = np.where(upw, bw_l, bw_r)
            bo_f = np.where(upo, bo_l, bo_r)
            bg_f = np.where(upg, bg_l, bg_r)
            rs_f = np.where(upo, rs_l, rs_r)
            qw = tw * dphi_w
            qo = to * dphi_o
            qg = tg * dphi_g
            dFw_L = bw_f * tw + np.where(upw, dbw_l, 0.0) * qw
            dFw_R = -bw_f * tw + np.where(~upw, dbw_r, 0.0) * qw
            dFo_L = bo_f * to + np.where(upo, dbo_l, 0.0) * qo
            dFo_R = -bo_f * to + np.where(~upo, dbo_r, 0.0) * qo
            drsbo_l = drs_l * bo_l + rs_l * dbo_l
            drsbo_r = drs_r * bo_r + rs_r * dbo_r
            dFg_L = bg_f * tg + rs_f * bo_f * to + np.where(upg, dbg_l, 0.0) * qg + np.where(upo, drsbo_l, 0.0) * qo
            dFg_R = -(bg_f * tg + rs_f * bo_f * to) + np.where(~upg, dbg_r, 0.0) * qg + np.where(~upo, drsbo_r, 0.0) * qo
            rows.extend([left, left, right, right, left + n, left + n, right + n, right + n, left + 2 * n, left + 2 * n, right + 2 * n, right + 2 * n])
            cols.extend([left, right, left, right, left, right, left, right, left, right, left, right])
            data.extend([dFw_L, dFw_R, -dFw_L, -dFw_R, dFo_L, dFo_R, -dFo_L, -dFo_R, dFg_L, dFg_R, -dFg_L, -dFg_R])
            dqw_sw_l = t_geom * np.where(upw, dws_l, 0.0) * dphi_w - tw * dpc_l
            dqw_sw_r = t_geom * np.where(~upw, dws_r, 0.0) * dphi_w + tw * dpc_r
            dqw_x_l = t_geom * np.where(upw, dwg_l, 0.0) * dphi_w
            dqw_x_r = t_geom * np.where(~upw, dwg_r, 0.0) * dphi_w
            dqo_sw_l = t_geom * np.where(upo, dos_l, 0.0) * dphi_o
            dqo_sw_r = t_geom * np.where(~upo, dos_r, 0.0) * dphi_o
            dqo_x_l = t_geom * np.where(upo, dog_l, 0.0) * dphi_o
            dqo_x_r = t_geom * np.where(~upo, dog_r, 0.0) * dphi_o
            dqg_sw_l = t_geom * np.where(upg, dgs_l, 0.0) * dphi_g
            dqg_sw_r = t_geom * np.where(~upg, dgs_r, 0.0) * dphi_g
            dqg_x_l = t_geom * np.where(upg, dgg_l, 0.0) * dphi_g
            dqg_x_r = t_geom * np.where(~upg, dgg_r, 0.0) * dphi_g
            qo = to * dphi_o
            dFg_rs_l = np.where(upo, (un_l > 0.5) * bo_f * qo, 0.0)
            dFg_rs_r = np.where(~upo, (un_r > 0.5) * bo_f * qo, 0.0)
            dFw_sw_l, dFw_sw_r = bw_f * dqw_sw_l, bw_f * dqw_sw_r
            dFw_x_l, dFw_x_r = bw_f * dqw_x_l, bw_f * dqw_x_r
            dFo_sw_l, dFo_sw_r = bo_f * dqo_sw_l, bo_f * dqo_sw_r
            dFo_x_l, dFo_x_r = bo_f * dqo_x_l, bo_f * dqo_x_r
            dFg_sw_l = bg_f * dqg_sw_l + rs_f * bo_f * dqo_sw_l
            dFg_sw_r = bg_f * dqg_sw_r + rs_f * bo_f * dqo_sw_r
            dFg_x_l = bg_f * dqg_x_l + rs_f * bo_f * dqo_x_l + dFg_rs_l
            dFg_x_r = bg_f * dqg_x_r + rs_f * bo_f * dqo_x_r + dFg_rs_r
            rows.extend([left, left, left, left, right, right, right, right])
            cols.extend([left + n, right + n, left + 2 * n, right + 2 * n, left + n, right + n, left + 2 * n, right + 2 * n])
            data.extend([dFw_sw_l, dFw_sw_r, dFw_x_l, dFw_x_r, -dFw_sw_l, -dFw_sw_r, -dFw_x_l, -dFw_x_r])
            rows.extend([left + n, left + n, left + n, left + n, right + n, right + n, right + n, right + n])
            cols.extend([left + n, right + n, left + 2 * n, right + 2 * n, left + n, right + n, left + 2 * n, right + 2 * n])
            data.extend([dFo_sw_l, dFo_sw_r, dFo_x_l, dFo_x_r, -dFo_sw_l, -dFo_sw_r, -dFo_x_l, -dFo_x_r])
            rows.extend([left + 2 * n, left + 2 * n, left + 2 * n, left + 2 * n, right + 2 * n, right + 2 * n, right + 2 * n, right + 2 * n])
            cols.extend([left + n, right + n, left + 2 * n, right + 2 * n, left + n, right + n, left + 2 * n, right + 2 * n])
            data.extend([dFg_sw_l, dFg_sw_r, dFg_x_l, dFg_x_r, -dFg_sw_l, -dFg_sw_r, -dFg_x_l, -dFg_x_r])

        if grid.nx > 1:
            kk, jj, ii = np.meshgrid(np.arange(grid.nz), np.arange(grid.ny), np.arange(grid.nx - 1), indexing="ij")
            left = (kk * grid.ny * grid.nx + jj * grid.nx + ii).astype(np.int64).ravel()
            right = left + 1
            sl = np.s_[:, :, :-1]
            sr = np.s_[:, :, 1:]
            _faces(
                tx.ravel(), left, right,
                p_ijk[sl].ravel(), p_ijk[sr].ravel(), z_ijk[sl].ravel(), z_ijk[sr].ravel(),
                lw_i[sl].ravel(), lw_i[sr].ravel(), lo_i[sl].ravel(), lo_i[sr].ravel(), lg_i[sl].ravel(), lg_i[sr].ravel(),
                dws_i[sl].ravel(), dws_i[sr].ravel(), dos_i[sl].ravel(), dos_i[sr].ravel(), dgs_i[sl].ravel(), dgs_i[sr].ravel(),
                dwg_i[sl].ravel(), dwg_i[sr].ravel(), dog_i[sl].ravel(), dog_i[sr].ravel(), dgg_i[sl].ravel(), dgg_i[sr].ravel(),
                bw_i[sl].ravel(), bw_i[sr].ravel(), bo_i[sl].ravel(), bo_i[sr].ravel(), bg_i[sl].ravel(), bg_i[sr].ravel(),
                rs_i[sl].ravel(), rs_i[sr].ravel(), un_i[sl].ravel(), un_i[sr].ravel(),
                dbw_i[sl].ravel(), dbw_i[sr].ravel(), dbo_i[sl].ravel(), dbo_i[sr].ravel(), dbg_i[sl].ravel(), dbg_i[sr].ravel(),
                drs_i[sl].ravel(), drs_i[sr].ravel(),
                pc_i[sl].ravel(), pc_i[sr].ravel(),
                dpc_i[sl].ravel(), dpc_i[sr].ravel(),
            )
        if grid.ny > 1:
            kk, jj, ii = np.meshgrid(np.arange(grid.nz), np.arange(grid.ny - 1), np.arange(grid.nx), indexing="ij")
            left = (kk * grid.ny * grid.nx + jj * grid.nx + ii).astype(np.int64).ravel()
            right = left + grid.nx
            sl = np.s_[:, :-1, :]
            sr = np.s_[:, 1:, :]
            _faces(
                ty.ravel(), left, right,
                p_ijk[sl].ravel(), p_ijk[sr].ravel(), z_ijk[sl].ravel(), z_ijk[sr].ravel(),
                lw_i[sl].ravel(), lw_i[sr].ravel(), lo_i[sl].ravel(), lo_i[sr].ravel(), lg_i[sl].ravel(), lg_i[sr].ravel(),
                dws_i[sl].ravel(), dws_i[sr].ravel(), dos_i[sl].ravel(), dos_i[sr].ravel(), dgs_i[sl].ravel(), dgs_i[sr].ravel(),
                dwg_i[sl].ravel(), dwg_i[sr].ravel(), dog_i[sl].ravel(), dog_i[sr].ravel(), dgg_i[sl].ravel(), dgg_i[sr].ravel(),
                bw_i[sl].ravel(), bw_i[sr].ravel(), bo_i[sl].ravel(), bo_i[sr].ravel(), bg_i[sl].ravel(), bg_i[sr].ravel(),
                rs_i[sl].ravel(), rs_i[sr].ravel(), un_i[sl].ravel(), un_i[sr].ravel(),
                dbw_i[sl].ravel(), dbw_i[sr].ravel(), dbo_i[sl].ravel(), dbo_i[sr].ravel(), dbg_i[sl].ravel(), dbg_i[sr].ravel(),
                drs_i[sl].ravel(), drs_i[sr].ravel(),
                pc_i[sl].ravel(), pc_i[sr].ravel(),
                dpc_i[sl].ravel(), dpc_i[sr].ravel(),
            )
        if grid.nz > 1:
            kk, jj, ii = np.meshgrid(np.arange(grid.nz - 1), np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
            left = (kk * grid.ny * grid.nx + jj * grid.nx + ii).astype(np.int64).ravel()
            right = left + grid.ny * grid.nx
            sl = np.s_[:-1, :, :]
            sr = np.s_[1:, :, :]
            _faces(
                tz.ravel(), left, right,
                p_ijk[sl].ravel(), p_ijk[sr].ravel(), z_ijk[sl].ravel(), z_ijk[sr].ravel(),
                lw_i[sl].ravel(), lw_i[sr].ravel(), lo_i[sl].ravel(), lo_i[sr].ravel(), lg_i[sl].ravel(), lg_i[sr].ravel(),
                dws_i[sl].ravel(), dws_i[sr].ravel(), dos_i[sl].ravel(), dos_i[sr].ravel(), dgs_i[sl].ravel(), dgs_i[sr].ravel(),
                dwg_i[sl].ravel(), dwg_i[sr].ravel(), dog_i[sl].ravel(), dog_i[sr].ravel(), dgg_i[sl].ravel(), dgg_i[sr].ravel(),
                bw_i[sl].ravel(), bw_i[sr].ravel(), bo_i[sl].ravel(), bo_i[sr].ravel(), bg_i[sl].ravel(), bg_i[sr].ravel(),
                rs_i[sl].ravel(), rs_i[sr].ravel(), un_i[sl].ravel(), un_i[sr].ravel(),
                dbw_i[sl].ravel(), dbw_i[sr].ravel(), dbo_i[sl].ravel(), dbo_i[sr].ravel(), dbg_i[sl].ravel(), dbg_i[sr].ravel(),
                drs_i[sl].ravel(), drs_i[sr].ravel(),
                pc_i[sl].ravel(), pc_i[sr].ravel(),
                dpc_i[sl].ravel(), dpc_i[sr].ravel(),
            )
        if wi_base:
            # Well Jacobian must match `_well_pack` / `_well_surface_rates`
            # (lt_fixed, mixture, optional head). Analytic Peaceman stubs diverge
            # on liberation wells; use centered FD of the well source only.
            cells_w = [int(c) for c in wi_base]
            # Cross-flow couples perfs in a well: FD each well unknown locally.
            eps_p = max(1.0, 1.0e-6 * float(np.mean(np.abs(p_a))))
            eps_sw = 1.0e-5
            rs_scale = max(float(np.mean(np.abs(rs_a))), 1.0)
            eps_x = np.where(unsat_a, max(1.0e-3 * rs_scale, 1.0e-8), 1.0e-5)

            def _rates_at(p_b, sw_b, sg_b, rs_b, unsat_b):
                unsat2 = np.asarray(unsat_b, dtype=bool).ravel()
                sg2 = np.where(unsat2, 0.0, np.asarray(sg_b, dtype=float).ravel())
                rs2 = np.asarray(rs_b, dtype=float).ravel()
                if live:
                    rs_sat2 = np.asarray(fluid.rs(p_b), dtype=float).ravel()
                    rs2 = np.where(unsat2, np.minimum(rs2, rs_sat2), rs_sat2)
                sat2 = ~unsat2
                lw2, lo2, lg2 = _lambda(three_phase, fluid, sw_b, sg2, p_b, rs=rs2, saturated=sat2)
                bw2 = fluid.b_w(p_b)
                bo2 = fluid.b_o(p_b, rs=rs2, saturated=sat2)
                bg2 = fluid.b_g(p_b)
                return _well_pack(
                    p_b,
                    lw2,
                    lo2,
                    lg2,
                    bw2,
                    bo2,
                    bg2,
                    rs2,
                    fluid.density_w(p_b, bw=bw2),
                    fluid.density_o(p_b, rs=rs2, bo=bo2),
                    fluid.density_g(p_b, bg=bg2),
                )

            for c in cells_w:
                # dp
                p_h = p_a.copy()
                p_l = p_a.copy()
                p_h[c] += eps_p
                p_l[c] -= eps_p
                qwh, qoh, qgh = _rates_at(p_h, sw_a, sg_a, rs_a, unsat_a)
                qwl, qol, qgl = _rates_at(p_l, sw_a, sg_a, rs_a, unsat_a)
                dqw = (qwh - qwl) / (2.0 * eps_p)
                dqo = (qoh - qol) / (2.0 * eps_p)
                dqg = (qgh - qgl) / (2.0 * eps_p)
                # residual has -q_well → Jacobian entries are -dq/du
                for cc in cells_w:
                    data.append(np.array([-float(dqw[cc])]))
                    rows.append(np.array([cc]))
                    cols.append(np.array([c]))
                    data.append(np.array([-float(dqo[cc])]))
                    rows.append(np.array([cc + n]))
                    cols.append(np.array([c]))
                    data.append(np.array([-float(dqg[cc])]))
                    rows.append(np.array([cc + 2 * n]))
                    cols.append(np.array([c]))
                # dsw
                sw_h = sw_a.copy()
                sw_l = sw_a.copy()
                sw_h[c] = float(np.clip(sw_a[c] + eps_sw, 0.0, 1.0))
                sw_l[c] = float(np.clip(sw_a[c] - eps_sw, 0.0, 1.0))
                den = max(float(sw_h[c] - sw_l[c]), 1.0e-14)
                qwh, qoh, qgh = _rates_at(p_a, sw_h, sg_a, rs_a, unsat_a)
                qwl, qol, qgl = _rates_at(p_a, sw_l, sg_a, rs_a, unsat_a)
                dqw = (qwh - qwl) / den
                dqo = (qoh - qol) / den
                dqg = (qgh - qgl) / den
                for cc in cells_w:
                    data.append(np.array([-float(dqw[cc])]))
                    rows.append(np.array([cc]))
                    cols.append(np.array([c + n]))
                    data.append(np.array([-float(dqo[cc])]))
                    rows.append(np.array([cc + n]))
                    cols.append(np.array([c + n]))
                    data.append(np.array([-float(dqg[cc])]))
                    rows.append(np.array([cc + 2 * n]))
                    cols.append(np.array([c + n]))
                # dx (Sg or Rs)
                ex = float(eps_x[c])
                if unsat_a[c]:
                    rs_h = rs_a.copy()
                    rs_l = rs_a.copy()
                    rs_h[c] = float(rs_a[c] + ex)
                    rs_l[c] = float(max(rs_a[c] - ex, 0.0))
                    den = max(float(rs_h[c] - rs_l[c]), 1.0e-14)
                    qwh, qoh, qgh = _rates_at(p_a, sw_a, sg_a, rs_h, unsat_a)
                    qwl, qol, qgl = _rates_at(p_a, sw_a, sg_a, rs_l, unsat_a)
                else:
                    sg_h = sg_a.copy()
                    sg_l = sg_a.copy()
                    sg_h[c] = float(np.clip(sg_a[c] + ex, 0.0, 1.0))
                    sg_l[c] = float(np.clip(sg_a[c] - ex, 0.0, 1.0))
                    den = max(float(sg_h[c] - sg_l[c]), 1.0e-14)
                    qwh, qoh, qgh = _rates_at(p_a, sw_a, sg_h, rs_a, unsat_a)
                    qwl, qol, qgl = _rates_at(p_a, sw_a, sg_l, rs_a, unsat_a)
                dqw = (qwh - qwl) / den
                dqo = (qoh - qol) / den
                dqg = (qgh - qgl) / den
                for cc in cells_w:
                    data.append(np.array([-float(dqw[cc])]))
                    rows.append(np.array([cc]))
                    cols.append(np.array([c + 2 * n]))
                    data.append(np.array([-float(dqo[cc])]))
                    rows.append(np.array([cc + n]))
                    cols.append(np.array([c + 2 * n]))
                    data.append(np.array([-float(dqg[cc])]))
                    rows.append(np.array([cc + 2 * n]))
                    cols.append(np.array([c + 2 * n]))
        jac = sparse.csr_matrix(
            (np.concatenate([np.asarray(d, dtype=float).ravel() for d in data]),
             (np.concatenate([np.asarray(r, dtype=np.int64).ravel() for r in rows]),
              np.concatenate([np.asarray(c, dtype=np.int64).ravel() for c in cols]))),
            shape=(3 * n, 3 * n),
        )
        if cell_dirichlet:
            jac = jac.tolil()
            for c in cell_dirichlet:
                c = int(c)
                jac[c, :] = 0.0
                jac[c, c] = 1.0
            jac = jac.tocsr()
        return jac

    def _cnv_scale(pv, bw, bo, bg, rs_a):
        dt_s = max(dt, 1.0e-30)
        pv_a = np.maximum(np.asarray(pv, dtype=float).ravel(), 1.0e-30)
        bw_a = np.maximum(np.asarray(bw, dtype=float).ravel(), 1.0e-30)
        bo_a = np.maximum(np.asarray(bo, dtype=float).ravel(), 1.0e-30)
        bg_a = np.maximum(np.asarray(bg, dtype=float).ravel(), 1.0e-30)
        rs_a = np.abs(np.asarray(rs_a, dtype=float).ravel())
        return np.concatenate(
            [
                pv_a * bw_a / dt_s,
                pv_a * bo_a / dt_s,
                pv_a * np.maximum(bg_a, rs_a * bo_a) / dt_s,
            ]
        )

    def _finish(p_f, sw_f, sg_f, rs_f, unsat_f, pack_f, n_its: int) -> FiStepResult:
        if live:
            # Equilibrium flash only after Newton accepts the primary state.
            sg_e, rs_e, grow, _ac = liberate_excess_gas(
                fluid, sw_f, sg_f, rs_f, unsat_f, p_f, live=True, grow_max=0.20
            )
            sg_f = np.where(grow, sg_e, sg_f)
            rs_f = np.where(grow, rs_e, rs_f)
            sw_f, sg_f = _clip_sw_sg(sw_f, sg_f)
        # Recompute total reservoir fluxes at the accepted state.
        so_f = np.clip(1.0 - sw_f - sg_f, 0.0, 1.0)
        sat_f = None if not live else ~fluid.vo_unsat(sg_f)
        lw_f, lo_f, lg_f = _lambda(three_phase, fluid, sw_f, sg_f, p_f, rs=rs_f, saturated=sat_f)
        bw_f = fluid.b_w(p_f)
        bo_f = fluid.b_o(p_f, rs=rs_f, saturated=sat_f)
        bg_f = fluid.b_g(p_f)
        pc = None if capillary is None or isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw_f), dtype=float).ravel()
        qw_x, qw_y, qw_z, qo_x, qo_y, qo_z, qg_x, qg_y, qg_z = phase_interior_fluxes(
            grid,
            p_f,
            k,
            lw_f,
            lo_f,
            lg=lg_f,
            kz=kz,
            mult_x=face_mult_x,
            mult_y=face_mult_y,
            mult_z=face_mult_z,
            gravity=float(gravity),
            rho_w=fluid.density_w(p_f, bw=bw_f),
            rho_o=fluid.density_o(p_f, rs=rs_f, bo=bo_f),
            rho_g=fluid.density_g(p_f, bg=bg_f),
            pc=pc,
        )
        return FiStepResult(
            pressure=np.asarray(p_f, dtype=float).ravel(),
            sw=np.asarray(sw_f, dtype=float).ravel(),
            sg=np.asarray(sg_f, dtype=float).ravel(),
            rs=np.asarray(rs_f, dtype=float).ravel(),
            fx=np.asarray(qw_x + qo_x + qg_x, dtype=float),
            fy=np.asarray(qw_y + qo_y + qg_y, dtype=float),
            fz=np.asarray(qw_z + qo_z + qg_z, dtype=float),
            newton_iters=int(n_its),
        )

    def _converged(res_a, scale_a) -> bool:
        # Primary accept is cell CNV; global MB is used on soft accept only.
        return cell_cnv_ok(res_a, scale_a, tol=nltol)

    x = fluid.vo_encode(sg, rs, unsat) if live else sg.copy()
    scale = np.maximum(_cnv_scale(pv0, fluid.b_w(p0), fluid.b_o(p0, rs=rs0), fluid.b_g(p0), rs0), 1.0e-12)
    res, pack = _residual(p, sw, sg, rs, unsat)
    err0 = float(np.max(np.abs(res) / scale))
    if not np.isfinite(err0):
        return None
    if _converged(res, scale):
        return _finish(p, sw, sg, rs, unsat, pack, 0)
    err_init = err0
    relax = NewtonRelaxation()
    n_its = 0
    _trace = False
    try:
        import os as _os

        _trace = _os.environ.get("FIM_TRACE", "") == "1"
    except Exception:
        _trace = False
    if _trace:
        print(f"[fim] start err={err0:.6g} dt={dt:g} n={n} nwi={0 if not wi_base else len(wi_base)}", flush=True)
    for _ in range(int(maxnewt)):
        n_its += 1
        try:
            jac = _jacobian(p, sw, sg, rs, unsat, pack)
            col_x = np.where(unsat, rs_ref, 1.0)
            scale_n = np.maximum(_cnv_scale(pack[7], pack[3], pack[4], pack[5], pack[6]), 1.0e-12)
            row_s = 1.0 / scale_n
            col_s = np.concatenate([np.full(n, pref), np.ones(n), col_x])
            jac = sparse.diags(row_s) @ jac @ sparse.diags(col_s)
            rhs = -res * row_s
            du = np.asarray(spsolve(jac.tocsr(), rhs), dtype=float).ravel()
            du[:n] *= pref
            du[2 * n :] *= col_x
        except Exception as exc:
            if _trace:
                print(f"[fim] jac/solve fail it={n_its}: {exc}", flush=True)
            return None
        if du.size != 3 * n or not np.all(np.isfinite(du)):
            if _trace:
                print(f"[fim] bad du it={n_its}", flush=True)
            return None
        du = clip_saturation_increment(
            du, n, unsat, ds_max=0.20, rs_ref=rs_ref, pref=pref, dp_rel_max=0.35
        )
        du = scale_newton_update(relax.apply(du), alpha=1.0)
        improved = False
        masks = [np.ones(3 * n), np.concatenate([np.ones(n), np.zeros(2 * n)]), np.concatenate([np.zeros(n), np.ones(2 * n)])]
        for mask in masks:
            if improved:
                break
            step = 1.0
            for _ls in range(int(lstrials)):
                inc = scale_newton_update(du * mask, alpha=step)
                p_t = p + inc[:n]
                sw_t = sw + inc[n : 2 * n]
                x_t = x + inc[2 * n :]
                if float(np.max(np.abs(p_t - p))) > 0.55 * pref:
                    step *= 0.5
                    continue
                sw_t, sg_t, rs_t, unsat_t, near_t, x_t = switch_live_oil_unknown(
                    fluid, p_t, sw_t, x_t, unsat, near, live=live
                )
                r_try, pack_try = _residual(p_t, sw_t, sg_t, rs_t, unsat_t)
                err = float(np.max(np.abs(r_try) / scale))
                switched = live and (np.any(unsat_t != unsat) or np.any(near_t != near))
                better = np.isfinite(err) and err < err0 * (1.0 - 1.0e-4 * max(step, 1.0e-3))
                # VO switch: accept only if residual does not increase.
                if not better and switched and np.isfinite(err) and err <= err0:
                    better = True
                if better:
                    p, sw, sg, rs, unsat, near = p_t, sw_t, sg_t, rs_t, unsat_t, near_t
                    x = x_t
                    res, pack, err0 = r_try, pack_try, err
                    improved = True
                    break
                step *= 0.5
        if _trace:
            print(
                f"[fim] it={n_its} err={err0:.6g} improved={improved} "
                f"du_p={float(np.max(np.abs(du[:n]))):.4g} du_s={float(np.max(np.abs(du[n:2*n]))):.4g}",
                flush=True,
            )
        # Soft accept: CNV small enough. Do not require a large relative drop —
        # sequential pressure guesses often start with err_init already O(0.1).
        soft_ok = err0 <= max(10.0 * nltol, 1.0e-1) and err0 <= err_init
        relax.update(np.array([[err0]]), np.array([err0 <= nltol or soft_ok]))
        if _converged(res, scale):
            return _finish(p, sw, sg, rs, unsat, pack, n_its)
        if not improved:
            if soft_ok:
                return _finish(p, sw, sg, rs, unsat, pack, n_its)
            return None
    soft_ok = err0 <= max(10.0 * nltol, 1.0e-1) and err0 <= err_init
    if soft_ok or _converged(res, scale):
        return _finish(p, sw, sg, rs, unsat, pack, n_its)
    return None
