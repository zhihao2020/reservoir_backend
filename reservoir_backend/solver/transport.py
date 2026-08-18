"""Implicit single-point upwind transport (sequential black-oil).

Two-phase: backward Euler on Sw with frozen vT (``implicit_water``).
Black-oil: coupled Newton on (Sw, Sg) with frozen vT (``implicit_blackoil``).

    (pv1 bW Sw − accW0)/dt + Div(bW↑ (fw↑ vT + extraW)) = qW^s
    (pv1 (bG Sg + Rs bO So) − accG0)/dt + Div(bG↑ vg + Rs bO vo) = qG^s

``extra`` is the frozen capillary/gravity segregation so the first iterate
matches explicit phase fluxes at old S. Newton uses ``dsMaxAbs=0.20``,
dampen on oscillation, and a flash chop near phase transitions. CNV/MB
are recorded; the stop is still the scaled residual.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.solver.seqtools import (
    NewtonRelaxation,
    cnv_mb,
    compute_flash_blackoil,
    critical_point_chop,
    limit_update_abs,
)


def fractional_flow_deriv(relperm, sw: NDArray[np.float64]) -> NDArray[np.float64]:
    s = np.asarray(sw, dtype=float)
    eps = 1.0e-5
    hi = relperm.fractional_flow(np.clip(s + eps, 0.0, 1.0))
    lo = relperm.fractional_flow(np.clip(s - eps, 0.0, 1.0))
    return (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)) / (2.0 * eps)


def _axis_upwind(
    v: NDArray[np.float64],
    fw_l: NDArray[np.float64],
    fw_r: NDArray[np.float64],
    b_l: NDArray[np.float64],
    b_r: NDArray[np.float64],
    dfw_l: NDArray[np.float64],
    dfw_r: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    up = v >= 0.0
    flux = np.where(up, b_l * fw_l, b_r * fw_r) * v
    d_l = np.where(up, b_l * dfw_l * v, 0.0)
    d_r = np.where(~up, b_r * dfw_r * v, 0.0)
    return flux, d_l, d_r


def implicit_water(
    grid: CartesianGrid,
    relperm,
    sw0: NDArray[np.float64],
    acc0: NDArray[np.float64],
    pv1: NDArray[np.float64],
    b_w1: NDArray[np.float64],
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    src_s: NDArray[np.float64],
    dt: float,
    *,
    pinned: NDArray[np.int64] | None = None,
    injector_fw: dict[int, float] | None = None,
    producer_q: dict[int, float] | None = None,
    extra_x: NDArray[np.float64] | None = None,
    extra_y: NDArray[np.float64] | None = None,
    extra_z: NDArray[np.float64] | None = None,
    s_max: NDArray[np.float64] | float | None = None,
    nltol: float = 1.0e-6,
    maxnewt: int = 8,
    lstrials: int = 8,
    stats: dict | None = None,
) -> NDArray[np.float64] | None:
    """Return Sw^{n+1} or ``None`` if Newton fails (caller chops dt).

    Newton uses ``dsMaxAbs=0.2`` and residual-based dampen.
    """
    n = grid.n_cells
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if s_max is None:
        hi: NDArray[np.float64] | float = 1.0
    else:
        hi = np.asarray(s_max, dtype=float)
        if hi.size == 1:
            hi = float(hi)

    def _clip_s(s: NDArray[np.float64]) -> NDArray[np.float64]:
        if isinstance(hi, float):
            return np.clip(s, 0.0, hi)
        return np.minimum(np.maximum(s, 0.0), hi.ravel())

    sw = _clip_s(np.asarray(sw0, dtype=float).ravel())
    acc0 = np.asarray(acc0, dtype=float).ravel()
    pv1 = np.asarray(pv1, dtype=float).ravel()
    b_w1 = np.asarray(b_w1, dtype=float).ravel()
    src_s = np.asarray(src_s, dtype=float).ravel()
    dt = float(dt)
    pin = set(int(c) for c in (pinned if pinned is not None else []))
    inj_fw = {int(c): float(f) for c, f in (injector_fw or {}).items()}
    prod = {int(c): float(q) for c, q in (producer_q or {}).items()}
    scale = np.maximum(pv1 * np.maximum(b_w1, 1.0e-30) / max(dt, 1.0e-30), 1.0e-12)

    def residual_and_jac(s: NDArray[np.float64]) -> tuple[NDArray[np.float64], sparse.csr_matrix]:
        fw = np.asarray(relperm.fractional_flow(s), dtype=float).ravel()
        dfw = fractional_flow_deriv(relperm, s).ravel()
        fw_ijk = grid.reshape_ijk(fw)
        dfw_ijk = grid.reshape_ijk(dfw)
        b_ijk = grid.reshape_ijk(b_w1)
        flux = np.zeros(n, dtype=float)
        rows: list[NDArray] = []
        cols: list[NDArray] = []
        data: list[NDArray] = []

        if nx > 1:
            v = fx[:, :, 1:-1]
            fw_l = fw_ijk[:, :, :-1].copy()
            fw_r = fw_ijk[:, :, 1:].copy()
            dfw_l = dfw_ijk[:, :, :-1].copy()
            dfw_r = dfw_ijk[:, :, 1:].copy()
            if inj_fw:
                kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
                left = (kk * ny * nx + jj * nx + ii).astype(np.int64)
                right = left + 1
                for c, f_in in inj_fw.items():
                    sel_l = left == c
                    sel_r = right == c
                    fw_l[sel_l] = f_in
                    dfw_l[sel_l] = 0.0
                    fw_r[sel_r] = f_in
                    dfw_r[sel_r] = 0.0
            fl, dl, dr = _axis_upwind(
                v,
                fw_l,
                fw_r,
                b_ijk[:, :, :-1],
                b_ijk[:, :, 1:],
                dfw_l,
                dfw_r,
            )
            kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + 1
            fl_r, dl_r, dr_r = fl.ravel(), dl.ravel(), dr.ravel()
            flux[left] += fl_r
            flux[right] -= fl_r
            if extra_x is not None:
                ex = extra_x[:, :, 1:-1] * np.where(v >= 0.0, b_ijk[:, :, :-1], b_ijk[:, :, 1:])
                flux[left] += ex.ravel()
                flux[right] -= ex.ravel()
            rows.extend([left, left, right, right])
            cols.extend([left, right, left, right])
            data.extend([dl_r, dr_r, -dl_r, -dr_r])
        if ny > 1:
            v = fy[:, 1:-1, :]
            fw_l = fw_ijk[:, :-1, :].copy()
            fw_r = fw_ijk[:, 1:, :].copy()
            dfw_l = dfw_ijk[:, :-1, :].copy()
            dfw_r = dfw_ijk[:, 1:, :].copy()
            if inj_fw:
                kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
                left = (kk * ny * nx + jj * nx + ii).astype(np.int64)
                right = left + nx
                for c, f_in in inj_fw.items():
                    sel_l = left == c
                    sel_r = right == c
                    fw_l[sel_l] = f_in
                    dfw_l[sel_l] = 0.0
                    fw_r[sel_r] = f_in
                    dfw_r[sel_r] = 0.0
            fl, dl, dr = _axis_upwind(
                v,
                fw_l,
                fw_r,
                b_ijk[:, :-1, :],
                b_ijk[:, 1:, :],
                dfw_l,
                dfw_r,
            )
            kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + nx
            fl_r, dl_r, dr_r = fl.ravel(), dl.ravel(), dr.ravel()
            flux[left] += fl_r
            flux[right] -= fl_r
            if extra_y is not None:
                ey = extra_y[:, 1:-1, :] * np.where(v >= 0.0, b_ijk[:, :-1, :], b_ijk[:, 1:, :])
                flux[left] += ey.ravel()
                flux[right] -= ey.ravel()
            rows.extend([left, left, right, right])
            cols.extend([left, right, left, right])
            data.extend([dl_r, dr_r, -dl_r, -dr_r])
        if nz > 1:
            v = fz[1:-1, :, :]
            fw_l = fw_ijk[:-1, :, :].copy()
            fw_r = fw_ijk[1:, :, :].copy()
            dfw_l = dfw_ijk[:-1, :, :].copy()
            dfw_r = dfw_ijk[1:, :, :].copy()
            if inj_fw:
                kk, jj, ii = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
                left = (kk * ny * nx + jj * nx + ii).astype(np.int64)
                right = left + ny * nx
                for c, f_in in inj_fw.items():
                    sel_l = left == c
                    sel_r = right == c
                    fw_l[sel_l] = f_in
                    dfw_l[sel_l] = 0.0
                    fw_r[sel_r] = f_in
                    dfw_r[sel_r] = 0.0
            fl, dl, dr = _axis_upwind(
                v,
                fw_l,
                fw_r,
                b_ijk[:-1, :, :],
                b_ijk[1:, :, :],
                dfw_l,
                dfw_r,
            )
            kk, jj, ii = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + ny * nx
            fl_r, dl_r, dr_r = fl.ravel(), dl.ravel(), dr.ravel()
            flux[left] += fl_r
            flux[right] -= fl_r
            if extra_z is not None:
                ez = extra_z[1:-1, :, :] * np.where(v >= 0.0, b_ijk[:-1, :, :], b_ijk[1:, :, :])
                flux[left] += ez.ravel()
                flux[right] -= ez.ravel()
            rows.extend([left, left, right, right])
            cols.extend([left, right, left, right])
            data.extend([dl_r, dr_r, -dl_r, -dr_r])

        res = (pv1 * b_w1 * s - acc0) + dt * flux - dt * src_s
        jac = sparse.csr_matrix(
            (np.concatenate(data) * dt, (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ) if rows else sparse.csr_matrix((n, n))
        jac = jac + sparse.diags(pv1 * b_w1)
        if prod:
            for c, q_out in prod.items():
                res[c] -= dt * q_out * fw[c] * b_w1[c]
                jac[c, c] = jac[c, c] - dt * q_out * dfw[c] * b_w1[c]
        if pin:
            jac = jac.tolil()
            for c in pin:
                jac.rows[c] = [c]
                jac.data[c] = [1.0]
                res[c] = 0.0
            jac = jac.tocsr()
        return res, jac

    res, jac = residual_and_jac(sw)
    err0 = float(np.max(np.abs(res) / scale))
    if not np.isfinite(err0):
        return None
    its = 0
    if err0 <= nltol:
        if stats is not None:
            stats["newton_its"] = 0
        return sw

    relax = NewtonRelaxation()
    hist: list[list[float]] = []
    for _ in range(int(maxnewt)):
        try:
            dsw = np.asarray(spsolve(jac.tocsr(), -res), dtype=float).ravel()
        except Exception:
            return None
        if dsw.size != n or not np.all(np.isfinite(dsw)):
            return None
        dsw = relax.apply(limit_update_abs(dsw))
        step = 1.0
        improved = False
        for _ls in range(int(lstrials)):
            trial = _clip_s(sw + step * dsw)
            swi = float(getattr(relperm, "swi", 0.0) or 0.0)
            sor = float(getattr(relperm, "sor", 0.0) or 0.0)
            if swi > 0.0:
                trial = critical_point_chop(sw, trial, swi)
            if 0.0 < sor < 1.0:
                trial = critical_point_chop(sw, trial, 1.0 - sor)
            r_try, j_try = residual_and_jac(trial)
            err = float(np.max(np.abs(r_try) / scale))
            if np.isfinite(err) and err < err0 * (1.0 - 1.0e-4 * step):
                sw, res, jac, err0 = trial, r_try, j_try, err
                improved = True
                break
            step *= 0.5
        its += 1
        hist.append([err0])
        relax.update(np.asarray(hist, dtype=float), np.array([err0 <= nltol]))
        if err0 <= nltol:
            if stats is not None:
                stats["newton_its"] = its
            return sw
        if not improved:
            return None
    if stats is not None:
        stats["newton_its"] = its
    return None if err0 > nltol else sw


def _frac_derivs(three, sw: NDArray[np.float64], sg: NDArray[np.float64], eps: float = 1.0e-5):
    """Central differences of (fw, fo, fg) wrt Sw and Sg."""
    sw = np.asarray(sw, dtype=float).ravel()
    sg = np.asarray(sg, dtype=float).ravel()
    fw0, fo0, fg0 = three.fractional_flow(sw, sg)

    def _at(sw_a, sg_a):
        return three.fractional_flow(np.clip(sw_a, 0.0, 1.0), np.clip(sg_a, 0.0, 1.0))

    fwh, foh, fgh = _at(sw + eps, sg)
    fwl, fol, fgl = _at(sw - eps, sg)
    dfw_sw = (np.asarray(fwh) - np.asarray(fwl)) / (2.0 * eps)
    dfo_sw = (np.asarray(foh) - np.asarray(fol)) / (2.0 * eps)
    dfg_sw = (np.asarray(fgh) - np.asarray(fgl)) / (2.0 * eps)
    fwh, foh, fgh = _at(sw, sg + eps)
    fwl, fol, fgl = _at(sw, sg - eps)
    dfw_sg = (np.asarray(fwh) - np.asarray(fwl)) / (2.0 * eps)
    dfo_sg = (np.asarray(foh) - np.asarray(fol)) / (2.0 * eps)
    dfg_sg = (np.asarray(fgh) - np.asarray(fgl)) / (2.0 * eps)
    return (
        np.asarray(fw0, dtype=float).ravel(),
        np.asarray(fo0, dtype=float).ravel(),
        np.asarray(fg0, dtype=float).ravel(),
        dfw_sw.ravel(),
        dfo_sw.ravel(),
        dfg_sw.ravel(),
        dfw_sg.ravel(),
        dfo_sg.ravel(),
        dfg_sg.ravel(),
    )


def _clip_sw_sg(sw: NDArray[np.float64], sg: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    sw = np.maximum(sw, 0.0)
    sg = np.maximum(sg, 0.0)
    tot = sw + sg
    over = tot > 1.0
    if np.any(over):
        sw = sw.copy()
        sg = sg.copy()
        sw[over] = sw[over] / tot[over]
        sg[over] = sg[over] / tot[over]
    return sw, sg


def implicit_blackoil(
    grid: CartesianGrid,
    three_phase,
    sw0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    acc_w0: NDArray[np.float64],
    acc_g0: NDArray[np.float64],
    pv1: NDArray[np.float64],
    b_w1: NDArray[np.float64],
    b_g1: NDArray[np.float64],
    rs_bo: NDArray[np.float64],
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    src_w: NDArray[np.float64],
    src_g: NDArray[np.float64],
    dt: float,
    *,
    extra_w_x: NDArray[np.float64] | None = None,
    extra_w_y: NDArray[np.float64] | None = None,
    extra_w_z: NDArray[np.float64] | None = None,
    extra_g_x: NDArray[np.float64] | None = None,
    extra_g_y: NDArray[np.float64] | None = None,
    extra_g_z: NDArray[np.float64] | None = None,
    extra_o_x: NDArray[np.float64] | None = None,
    extra_o_y: NDArray[np.float64] | None = None,
    extra_o_z: NDArray[np.float64] | None = None,
    pinned: NDArray[np.int64] | None = None,
    injector_fw: dict[int, float] | None = None,
    injector_fg: dict[int, float] | None = None,
    producer_q: dict[int, float] | None = None,
    refresh_extras=None,
    project=None,
    rs0: NDArray[np.float64] | None = None,
    rs_sat: NDArray[np.float64] | None = None,
    b_o1: NDArray[np.float64] | None = None,
    rs_out: NDArray[np.float64] | None = None,
    p_cell: NDArray[np.float64] | None = None,
    pb: float | None = None,
    acc_o0: NDArray[np.float64] | None = None,
    src_o: NDArray[np.float64] | None = None,
    conserve: str = "oil_gas",
    nltol: float = 1.0e-3,
    maxnewt: int = 12,
    lstrials: int = 8,
    stats: dict | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Sequential black-oil transport: coupled implicit (Sw, Sg), frozen vT.

    Default ``conserve="oil_gas"`` matches sequential black-oil (oil and
    surface gas; water is the saturation closure). ``water_gas`` keeps the
    older water+gas pair. Newton uses ``dsMaxAbs=0.20`` and dampen.
    """
    n = grid.n_cells
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    sw, sg = _clip_sw_sg(np.asarray(sw0, dtype=float).ravel(), np.asarray(sg0, dtype=float).ravel())
    if project is not None:
        sw, sg = project(sw, sg)
    conserve = str(conserve).lower().replace("-", "_")
    if conserve not in {"oil_gas", "water_gas"}:
        raise ValueError(f"unknown conserve={conserve!r}")
    vo = rs0 is not None and rs_sat is not None
    if vo:
        rs_sat_a = np.asarray(rs_sat, dtype=float).ravel()
        bo_a = np.ones(n) if b_o1 is None else np.asarray(b_o1, dtype=float).ravel()
        rs_now = np.asarray(rs0, dtype=float).ravel()
        unsat = sg <= 1.0e-8
        if p_cell is not None and pb is not None:
            unsat = unsat & (np.asarray(p_cell, dtype=float).ravel() >= float(pb) - 1.0)
        x_unk = np.where(unsat, rs_now, sg)
    else:
        rs_sat_a = np.zeros(n)
        bo_a = np.ones(n)
        rs_now = np.zeros(n)
        unsat = np.zeros(n, dtype=bool)
        x_unk = sg.copy()
    bo_cons = np.ones(n) if b_o1 is None else np.asarray(b_o1, dtype=float).ravel()
    acc_o0_a = np.zeros(n) if acc_o0 is None else np.asarray(acc_o0, dtype=float).ravel()
    src_o_a = np.zeros(n) if src_o is None else np.asarray(src_o, dtype=float).ravel()
    acc_w0 = np.asarray(acc_w0, dtype=float).ravel()
    acc_g0 = np.asarray(acc_g0, dtype=float).ravel()
    pv1 = np.asarray(pv1, dtype=float).ravel()
    b_w1 = np.asarray(b_w1, dtype=float).ravel()
    b_g1 = np.asarray(b_g1, dtype=float).ravel()
    rs_bo = np.asarray(rs_bo, dtype=float).ravel()
    src_w = np.asarray(src_w, dtype=float).ravel()
    src_g = np.asarray(src_g, dtype=float).ravel()
    dt = float(dt)
    pin = set(int(c) for c in (pinned if pinned is not None else []))
    inj_fw = {int(c): float(f) for c, f in (injector_fw or {}).items()}
    inj_fg = {int(c): float(f) for c, f in (injector_fg or {}).items()}
    prod = {int(c): float(q) for c, q in (producer_q or {}).items()}
    scale_w = np.maximum(pv1 * np.maximum(b_w1, 1.0e-30) / max(dt, 1.0e-30), 1.0e-12)
    scale_o = np.maximum(pv1 * np.maximum(bo_cons, 1.0e-30) / max(dt, 1.0e-30), 1.0e-12)
    scale_g = np.maximum(pv1 * np.maximum(np.maximum(b_g1, np.abs(rs_bo)), 1.0e-30) / max(dt, 1.0e-30), 1.0e-12)
    scale_first = scale_o if conserve == "oil_gas" else scale_w

    def _apply_injector(fw, fo, fg, dfw_sw, dfo_sw, dfg_sw, dfw_sg, dfo_sg, dfg_sg, left, right):
        if not inj_fw and not inj_fg:
            return
        sel_l = np.zeros(left.shape, dtype=bool)
        sel_r = np.zeros(right.shape, dtype=bool)
        for c in set(inj_fw) | set(inj_fg):
            sel_l |= left == c
            sel_r |= right == c
        if np.any(sel_l) or np.any(sel_r):
            fw_in = np.zeros(left.shape)
            fg_in = np.zeros(left.shape)
            for c in set(inj_fw) | set(inj_fg):
                mask_l = left == c
                mask_r = right == c
                fw_in[mask_l | mask_r] = inj_fw.get(c, 0.0)
                fg_in[mask_l | mask_r] = inj_fg.get(c, 0.0)
            fo_in = np.maximum(1.0 - fw_in - fg_in, 0.0)
            if np.any(sel_l):
                fw[0][sel_l] = fw_in[sel_l]
                fo[0][sel_l] = fo_in[sel_l]
                fg[0][sel_l] = fg_in[sel_l]
                dfw_sw[0][sel_l] = 0.0
                dfo_sw[0][sel_l] = 0.0
                dfg_sw[0][sel_l] = 0.0
                dfw_sg[0][sel_l] = 0.0
                dfo_sg[0][sel_l] = 0.0
                dfg_sg[0][sel_l] = 0.0
            if np.any(sel_r):
                fw[1][sel_r] = fw_in[sel_r]
                fo[1][sel_r] = fo_in[sel_r]
                fg[1][sel_r] = fg_in[sel_r]
                dfw_sw[1][sel_r] = 0.0
                dfo_sw[1][sel_r] = 0.0
                dfg_sw[1][sel_r] = 0.0
                dfw_sg[1][sel_r] = 0.0
                dfo_sg[1][sel_r] = 0.0
                dfg_sg[1][sel_r] = 0.0

    def _axis(
        v,
        left,
        right,
        fw_l, fw_r, fo_l, fo_r, fg_l, fg_r,
        dfw_sw_l, dfw_sw_r, dfo_sw_l, dfo_sw_r, dfg_sw_l, dfg_sw_r,
        dfw_sg_l, dfw_sg_r, dfo_sg_l, dfo_sg_r, dfg_sg_l, dfg_sg_r,
        bw_l, bw_r, bg_l, bg_r, rs_l, rs_r, bo_l, bo_r,
        ex_w, ex_g, ex_o,
        drs_l=None,
        drs_r=None,
    ):
        fw = [fw_l.copy(), fw_r.copy()]
        fo = [fo_l.copy(), fo_r.copy()]
        fg = [fg_l.copy(), fg_r.copy()]
        dws = [dfw_sw_l.copy(), dfw_sw_r.copy()]
        dos = [dfo_sw_l.copy(), dfo_sw_r.copy()]
        dgs = [dfg_sw_l.copy(), dfg_sw_r.copy()]
        dwg = [dfw_sg_l.copy(), dfw_sg_r.copy()]
        dog = [dfo_sg_l.copy(), dfo_sg_r.copy()]
        dgg = [dfg_sg_l.copy(), dfg_sg_r.copy()]
        _apply_injector(fw, fo, fg, dws, dos, dgs, dwg, dog, dgg, left, right)
        up = v >= 0.0
        coef_w_l, coef_w_r = bw_l * fw[0], bw_r * fw[1]
        coef_g_l = bg_l * fg[0] + rs_l * fo[0]
        coef_g_r = bg_r * fg[1] + rs_r * fo[1]
        flux_w = np.where(up, coef_w_l, coef_w_r) * v
        flux_g = np.where(up, coef_g_l, coef_g_r) * v
        if ex_w is not None:
            flux_w = flux_w + ex_w * np.where(up, bw_l, bw_r)
        if ex_g is not None:
            flux_g = flux_g + ex_g * np.where(up, bg_l, bg_r)
        if ex_o is not None:
            flux_g = flux_g + ex_o * np.where(up, rs_l, rs_r)
        flux_o = np.where(up, bo_l * fo[0], bo_r * fo[1]) * v
        if ex_o is not None:
            flux_o = flux_o + ex_o * np.where(up, bo_l, bo_r)
        do_sw_l = np.where(up, bo_l * dos[0] * v, 0.0)
        do_sw_r = np.where(~up, bo_r * dos[1] * v, 0.0)
        do_sg_l = np.where(up, bo_l * dog[0] * v, 0.0)
        do_sg_r = np.where(~up, bo_r * dog[1] * v, 0.0)
        dw_sw_l = np.where(up, bw_l * dws[0] * v, 0.0)
        dw_sw_r = np.where(~up, bw_r * dws[1] * v, 0.0)
        dw_sg_l = np.where(up, bw_l * dwg[0] * v, 0.0)
        dw_sg_r = np.where(~up, bw_r * dwg[1] * v, 0.0)
        dg_sw_l = np.where(up, (bg_l * dgs[0] + rs_l * dos[0]) * v, 0.0)
        dg_sw_r = np.where(~up, (bg_r * dgs[1] + rs_r * dos[1]) * v, 0.0)
        dg_sg_l = np.where(up, (bg_l * dgg[0] + rs_l * dog[0]) * v, 0.0)
        dg_sg_r = np.where(~up, (bg_r * dgg[1] + rs_r * dog[1]) * v, 0.0)
        if drs_l is not None:
            dg_sg_l = dg_sg_l + np.where(up, drs_l * fo[0] * v, 0.0)
        if drs_r is not None:
            dg_sg_r = dg_sg_r + np.where(~up, drs_r * fo[1] * v, 0.0)
        return (
            flux_w.ravel(),
            flux_g.ravel(),
            flux_o.ravel(),
            dw_sw_l.ravel(),
            dw_sw_r.ravel(),
            dw_sg_l.ravel(),
            dw_sg_r.ravel(),
            dg_sw_l.ravel(),
            dg_sw_r.ravel(),
            dg_sg_l.ravel(),
            dg_sg_r.ravel(),
            do_sw_l.ravel(),
            do_sw_r.ravel(),
            do_sg_l.ravel(),
            do_sg_r.ravel(),
        )

    extras_now = [
        extra_w_x,
        extra_w_y,
        extra_w_z,
        extra_g_x,
        extra_g_y,
        extra_g_z,
        extra_o_x,
        extra_o_y,
        extra_o_z,
    ]

    def residual_and_jac(sw_a, x_a):
        ew_x, ew_y, ew_z, eg_x, eg_y, eg_z, eo_x, eo_y, eo_z = extras_now
        if vo:
            sl = np.maximum(1.0 - sw_a, 0.0)
            sg_a = np.zeros_like(x_a)
            rs_a = rs_sat_a.copy()
            sat = ~unsat
            sg_a[sat] = np.clip(x_a[sat], 0.0, sl[sat])
            rs_a[unsat] = np.clip(x_a[unsat], 0.0, None)
            grow = unsat & (rs_a > rs_sat_a)
            extra_g = (rs_a - rs_sat_a) * bo_a * sl
            sg_a[grow] = np.clip(extra_g[grow] / np.maximum(b_g1[grow], 1.0e-30), 0.0, sl[grow])
            rs_a[grow] = rs_sat_a[grow]
            dsg_dx = np.where(unsat, 0.0, 1.0)
            dsg_dx = np.where(grow, bo_a * sl / np.maximum(b_g1, 1.0e-30), dsg_dx)
            drs_dx = np.where(unsat & ~grow, 1.0, 0.0)
            rs_bo_a = rs_a * bo_a
            drs_bo_dx = drs_dx * bo_a
        else:
            sg_a = x_a
            rs_bo_a = rs_bo
            dsg_dx = np.ones_like(x_a)
            drs_bo_dx = np.zeros_like(x_a)
        fw, fo, fg, dfw_sw, dfo_sw, dfg_sw, dfw_sg, dfo_sg, dfg_sg = _frac_derivs(three_phase, sw_a, sg_a)
        dfw_sg = dfw_sg * dsg_dx
        dfo_sg = dfo_sg * dsg_dx
        dfg_sg = dfg_sg * dsg_dx
        fw_i, fo_i, fg_i = grid.reshape_ijk(fw), grid.reshape_ijk(fo), grid.reshape_ijk(fg)
        dws_i, dos_i, dgs_i = grid.reshape_ijk(dfw_sw), grid.reshape_ijk(dfo_sw), grid.reshape_ijk(dfg_sw)
        dwg_i, dog_i, dgg_i = grid.reshape_ijk(dfw_sg), grid.reshape_ijk(dfo_sg), grid.reshape_ijk(dfg_sg)
        bw_i, bg_i, rs_i = grid.reshape_ijk(b_w1), grid.reshape_ijk(b_g1), grid.reshape_ijk(rs_bo_a)
        bo_i = grid.reshape_ijk(bo_cons)
        drs_i = grid.reshape_ijk(drs_bo_dx)
        flux_w = np.zeros(n, dtype=float)
        flux_g = np.zeros(n, dtype=float)
        flux_o = np.zeros(n, dtype=float)
        rows: list[NDArray] = []
        cols: list[NDArray] = []
        data: list[NDArray] = []

        def _add_face(left, right, pack):
            (
                fl_w, fl_g, fl_o,
                dw_sw_l, dw_sw_r, dw_sg_l, dw_sg_r,
                dg_sw_l, dg_sw_r, dg_sg_l, dg_sg_r,
                do_sw_l, do_sw_r, do_sg_l, do_sg_r,
            ) = pack
            flux_w[left] += fl_w
            flux_w[right] -= fl_w
            flux_g[left] += fl_g
            flux_g[right] -= fl_g
            flux_o[left] += fl_o
            flux_o[right] -= fl_o
            if conserve == "oil_gas":
                d1_sw_l, d1_sw_r, d1_sg_l, d1_sg_r = do_sw_l, do_sw_r, do_sg_l, do_sg_r
            else:
                d1_sw_l, d1_sw_r, d1_sg_l, d1_sg_r = dw_sw_l, dw_sw_r, dw_sg_l, dw_sg_r
            rows.extend([left, left, left, left, right, right, right, right])
            cols.extend([left, right, left + n, right + n, left, right, left + n, right + n])
            data.extend([d1_sw_l, d1_sw_r, d1_sg_l, d1_sg_r, -d1_sw_l, -d1_sw_r, -d1_sg_l, -d1_sg_r])
            rows.extend([left + n, left + n, left + n, left + n, right + n, right + n, right + n, right + n])
            cols.extend([left, right, left + n, right + n, left, right, left + n, right + n])
            data.extend([dg_sw_l, dg_sw_r, dg_sg_l, dg_sg_r, -dg_sw_l, -dg_sw_r, -dg_sg_l, -dg_sg_r])

        if nx > 1:
            kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + 1
            v = fx[:, :, 1:-1]
            exw = None if ew_x is None else ew_x[:, :, 1:-1]
            exg = None if eg_x is None else eg_x[:, :, 1:-1]
            exo = None if eo_x is None else eo_x[:, :, 1:-1]
            _add_face(
                left,
                right,
                _axis(
                    v, left.reshape(v.shape), right.reshape(v.shape),
                    fw_i[:, :, :-1], fw_i[:, :, 1:], fo_i[:, :, :-1], fo_i[:, :, 1:], fg_i[:, :, :-1], fg_i[:, :, 1:],
                    dws_i[:, :, :-1], dws_i[:, :, 1:], dos_i[:, :, :-1], dos_i[:, :, 1:], dgs_i[:, :, :-1], dgs_i[:, :, 1:],
                    dwg_i[:, :, :-1], dwg_i[:, :, 1:], dog_i[:, :, :-1], dog_i[:, :, 1:], dgg_i[:, :, :-1], dgg_i[:, :, 1:],
                    bw_i[:, :, :-1], bw_i[:, :, 1:], bg_i[:, :, :-1], bg_i[:, :, 1:], rs_i[:, :, :-1], rs_i[:, :, 1:],
                    bo_i[:, :, :-1], bo_i[:, :, 1:],
                    exw, exg, exo,
                    drs_i[:, :, :-1], drs_i[:, :, 1:],
                ),
            )
        if ny > 1:
            kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + nx
            v = fy[:, 1:-1, :]
            exw = None if ew_y is None else ew_y[:, 1:-1, :]
            exg = None if eg_y is None else eg_y[:, 1:-1, :]
            exo = None if eo_y is None else eo_y[:, 1:-1, :]
            _add_face(
                left,
                right,
                _axis(
                    v, left.reshape(v.shape), right.reshape(v.shape),
                    fw_i[:, :-1, :], fw_i[:, 1:, :], fo_i[:, :-1, :], fo_i[:, 1:, :], fg_i[:, :-1, :], fg_i[:, 1:, :],
                    dws_i[:, :-1, :], dws_i[:, 1:, :], dos_i[:, :-1, :], dos_i[:, 1:, :], dgs_i[:, :-1, :], dgs_i[:, 1:, :],
                    dwg_i[:, :-1, :], dwg_i[:, 1:, :], dog_i[:, :-1, :], dog_i[:, 1:, :], dgg_i[:, :-1, :], dgg_i[:, 1:, :],
                    bw_i[:, :-1, :], bw_i[:, 1:, :], bg_i[:, :-1, :], bg_i[:, 1:, :], rs_i[:, :-1, :], rs_i[:, 1:, :],
                    bo_i[:, :-1, :], bo_i[:, 1:, :],
                    exw, exg, exo,
                    drs_i[:, :-1, :], drs_i[:, 1:, :],
                ),
            )
        if nz > 1:
            kk, jj, ii = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
            left = (kk * ny * nx + jj * nx + ii).astype(np.int64).ravel()
            right = left + ny * nx
            v = fz[1:-1, :, :]
            exw = None if ew_z is None else ew_z[1:-1, :, :]
            exg = None if eg_z is None else eg_z[1:-1, :, :]
            exo = None if eo_z is None else eo_z[1:-1, :, :]
            _add_face(
                left,
                right,
                _axis(
                    v, left.reshape(v.shape), right.reshape(v.shape),
                    fw_i[:-1, :, :], fw_i[1:, :, :], fo_i[:-1, :, :], fo_i[1:, :, :], fg_i[:-1, :, :], fg_i[1:, :, :],
                    dws_i[:-1, :, :], dws_i[1:, :, :], dos_i[:-1, :, :], dos_i[1:, :, :], dgs_i[:-1, :, :], dgs_i[1:, :, :],
                    dwg_i[:-1, :, :], dwg_i[1:, :, :], dog_i[:-1, :, :], dog_i[1:, :, :], dgg_i[:-1, :, :], dgg_i[1:, :, :],
                    bw_i[:-1, :, :], bw_i[1:, :, :], bg_i[:-1, :, :], bg_i[1:, :, :], rs_i[:-1, :, :], rs_i[1:, :, :],
                    bo_i[:-1, :, :], bo_i[1:, :, :],
                    exw, exg, exo,
                    drs_i[:-1, :, :], drs_i[1:, :, :],
                ),
            )

        so = np.clip(1.0 - sw_a - sg_a, 0.0, 1.0)
        rw = (pv1 * b_w1 * sw_a - acc_w0) + dt * flux_w - dt * src_w
        ro = (pv1 * bo_cons * so - acc_o0_a) + dt * flux_o - dt * src_o_a
        rg = (pv1 * (b_g1 * sg_a + rs_bo_a * so) - acc_g0) + dt * flux_g - dt * src_g
        jac = sparse.csr_matrix(
            (np.concatenate(data) * dt, (np.concatenate(rows), np.concatenate(cols))),
            shape=(2 * n, 2 * n),
        ) if rows else sparse.csr_matrix((2 * n, 2 * n))
        if conserve == "oil_gas":
            diag_11 = -pv1 * bo_cons
            diag_1x = -pv1 * bo_cons * dsg_dx
        else:
            diag_11 = pv1 * b_w1
            diag_1x = np.zeros(n)
        diag_gw = -pv1 * rs_bo_a
        diag_gg = pv1 * (b_g1 * dsg_dx + drs_bo_dx * so - rs_bo_a * dsg_dx)
        acc = sparse.csr_matrix(
            (
                np.concatenate([diag_11, diag_1x, diag_gw, diag_gg]),
                (
                    np.concatenate([np.arange(n), np.arange(n), np.arange(n, 2 * n), np.arange(n, 2 * n)]),
                    np.concatenate([np.arange(n), np.arange(n, 2 * n), np.arange(n), np.arange(n, 2 * n)]),
                ),
            ),
            shape=(2 * n, 2 * n),
        )
        jac = jac + acc
        r1 = ro if conserve == "oil_gas" else rw
        if prod:
            jac = jac.tolil()
            for c, q_out in prod.items():
                if conserve == "oil_gas":
                    r1[c] -= dt * q_out * fo[c] * bo_cons[c]
                    jac[c, c] = jac[c, c] - dt * q_out * dfo_sw[c] * bo_cons[c]
                    jac[c, c + n] = jac[c, c + n] - dt * q_out * dfo_sg[c] * bo_cons[c]
                else:
                    r1[c] -= dt * q_out * fw[c] * b_w1[c]
                    jac[c, c] = jac[c, c] - dt * q_out * dfw_sw[c] * b_w1[c]
                    jac[c, c + n] = jac[c, c + n] - dt * q_out * dfw_sg[c] * b_w1[c]
                rg[c] -= dt * q_out * (fg[c] * b_g1[c] + fo[c] * rs_bo_a[c])
                jac[c + n, c] = jac[c + n, c] - dt * q_out * (dfg_sw[c] * b_g1[c] + dfo_sw[c] * rs_bo[c])
                jac[c + n, c + n] = jac[c + n, c + n] - dt * q_out * (dfg_sg[c] * b_g1[c] + dfo_sg[c] * rs_bo[c])
            jac = jac.tocsr()
        res = np.concatenate([r1, rg])
        if pin:
            jac = jac.tolil()
            for c in pin:
                jac.rows[c] = [c]
                jac.data[c] = [1.0]
                jac.rows[c + n] = [c + n]
                jac.data[c + n] = [1.0]
                res[c] = 0.0
                res[c + n] = 0.0
            jac = jac.tocsr()
        return res, jac

    res, jac = residual_and_jac(sw, x_unk)
    scale = np.concatenate([scale_first, scale_g])
    err0 = float(np.max(np.abs(res) / scale))
    if not np.isfinite(err0):
        return None
    its = 0
    if err0 <= nltol:
        if rs_out is not None and vo:
            rs_out[:] = np.where(unsat, x_unk, rs_sat_a)
        if stats is not None:
            stats["newton_its"] = 0
        return sw, sg

    relax = NewtonRelaxation()
    hist: list[list[float]] = []
    for _ in range(int(maxnewt)):
        try:
            du = np.asarray(spsolve(jac.tocsr(), -res), dtype=float).ravel()
        except Exception:
            return None
        if du.size != 2 * n or not np.all(np.isfinite(du)):
            return None
        # IMEX *NORM *SATUR 0.20 / sequential transport relaxation
        dx = du[n:]
        if vo:
            dsg_like = np.where(unsat, dx / np.maximum(rs_sat_a, 1.0e-6), dx)
            max_ds_newt = max(float(np.max(np.abs(du[:n]))), float(np.max(np.abs(dsg_like))), 1.0e-30)
        else:
            max_ds_newt = max(float(np.max(np.abs(du[:n]))), float(np.max(np.abs(dx))), 1.0e-30)
        if max_ds_newt > 0.20:
            du = du * (0.20 / max_ds_newt)
        du = relax.apply(du)
        step = 1.0
        improved = False
        for _ls in range(int(lstrials)):
            if vo:
                sw_t = np.clip(sw + step * du[:n], 0.0, 1.0)
                x_t = x_unk + step * du[n:]
            else:
                sw_t, x_t = _clip_sw_sg(sw + step * du[:n], x_unk + step * du[n:])
            swi = float(getattr(three_phase, "swi", 0.0) or 0.0)
            sgr = float(getattr(three_phase, "sgr", 0.0) or 0.0)
            if swi > 0.0:
                sw_t = critical_point_chop(sw, sw_t, swi)
            if not vo:
                x_t = critical_point_chop(x_unk, x_t, 0.0)
                if sgr > 0.0:
                    x_t = critical_point_chop(x_unk, x_t, sgr)
            r_try, j_try = residual_and_jac(sw_t, x_t)
            err = float(np.max(np.abs(r_try) / scale))
            if np.isfinite(err) and err < err0 * (1.0 - 1.0e-4 * step):
                sw, x_unk, res, jac, err0 = sw_t, x_t, r_try, j_try, err
                improved = True
                break
            step *= 0.5
        its += 1
        if improved:
            if vo:
                grow = unsat & (x_unk > rs_sat_a)
                dry = (~unsat) & (x_unk <= 1.0e-8)
                unsat = (unsat | dry) & ~grow
                sl = np.maximum(1.0 - sw, 0.0)
                sg = np.where(unsat, 0.0, np.clip(x_unk, 0.0, sl))
                extra_g = (x_unk - rs_sat_a) * bo_a * sl
                sg = np.where(grow, np.clip(extra_g / np.maximum(b_g1, 1.0e-30), 0.0, sl), sg)
                x_unk = np.where(unsat, np.clip(x_unk, 0.0, None), sg)
            else:
                sg = x_unk
            if project is not None and not vo:
                sw, sg = project(sw, sg)
                x_unk = sg
            if refresh_extras is not None:
                packed = refresh_extras(sw, sg)
                if packed is not None:
                    extras_now[:] = list(packed)
            if project is not None or refresh_extras is not None or vo:
                res, jac = residual_and_jac(sw, x_unk)
                err0 = float(np.max(np.abs(res) / scale))
                if not np.isfinite(err0):
                    return None
        cnv, mb, _cnv_ok = cnv_mb([res[:n], res[n:]], pv1, [b_w1, np.maximum(b_g1, np.abs(rs_bo))], dt)
        hist.append([float(cnv[0]), float(cnv[1])])
        relax.update(np.asarray(hist, dtype=float), np.array([err0 <= nltol, err0 <= nltol]))
        if stats is not None:
            stats["cnv"] = cnv
            stats["mb"] = mb
        if err0 <= nltol:
            break
        if not improved:
            return None
    if vo:
        sl = np.maximum(1.0 - sw, 0.0)
        sg = np.where(unsat, 0.0, np.clip(x_unk, 0.0, sl))
        grow = unsat & (x_unk > rs_sat_a)
        extra_g = (x_unk - rs_sat_a) * bo_a * sl
        sg = np.where(grow, np.clip(extra_g / np.maximum(b_g1, 1.0e-30), 0.0, sl), sg)
        if rs_out is not None:
            rs_out[:] = np.where(unsat & ~grow, np.clip(x_unk, 0.0, rs_sat_a), rs_sat_a)
    else:
        so = 1.0 - sw - sg
        sw, so, sg, _rs_chop, _st = compute_flash_blackoil(
            sw,
            so,
            sg,
            np.zeros(n),
            np.zeros(n),
            np.asarray(sw0, dtype=float).ravel(),
            np.clip(1.0 - np.asarray(sw0, dtype=float).ravel() - np.asarray(sg0, dtype=float).ravel(), 0.0, 1.0),
            np.asarray(sg0, dtype=float).ravel(),
            np.zeros(n),
            np.zeros(n),
            disgas=False,
        )
    if stats is not None:
        stats["newton_its"] = its
        if res.size == 2 * n:
            _cnv, _mb, _ok = cnv_mb([res[:n], res[n:]], pv1, [b_w1, np.maximum(b_g1, np.abs(rs_bo))], dt)
            stats["cnv"] = _cnv
            stats["mb"] = _mb
    return None if err0 > nltol else (sw, sg)
