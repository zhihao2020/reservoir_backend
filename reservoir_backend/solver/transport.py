"""Implicit single-point upwind transport (MRST ``implicitTransport``).

Backward Euler + Newton-Raphson with line search and the same
surface-volume residual as the explicit step:

    (pv1 bW Sw − pv0 bW0 Sw0)/dt + Div(bW↑ fw↑ vT) = qW^s

Total Darcy flux ``vT`` is frozen from the pressure step (sequential).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from reservoir_backend.grid.cartesian import CartesianGrid


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
    nltol: float = 1.0e-6,
    maxnewt: int = 8,
    lstrials: int = 8,
) -> NDArray[np.float64] | None:
    """Return Sw^{n+1} or ``None`` if Newton fails (caller chops dt)."""
    n = grid.n_cells
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    sw = np.clip(np.asarray(sw0, dtype=float).ravel(), 0.0, 1.0)
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
    if err0 <= nltol:
        return sw

    for _ in range(int(maxnewt)):
        try:
            dsw = np.asarray(spsolve(jac.tocsr(), -res), dtype=float).ravel()
        except Exception:
            return None
        if dsw.size != n or not np.all(np.isfinite(dsw)):
            return None
        step = 1.0
        improved = False
        for _ls in range(int(lstrials)):
            trial = np.clip(sw + step * dsw, 0.0, 1.0)
            r_try, j_try = residual_and_jac(trial)
            err = float(np.max(np.abs(r_try) / scale))
            if np.isfinite(err) and err < err0 * (1.0 - 1.0e-4 * step):
                sw, res, jac, err0 = trial, r_try, j_try, err
                improved = True
                break
            step *= 0.5
        if err0 <= nltol:
            return sw
        if not improved:
            return None
    return None if err0 > nltol else sw
