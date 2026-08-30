"""Local thermo FD + explicit TPFA/transfer/well sparse Jacobian.

Does not colour a global residual. Flash derivatives are cell-local.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.properties import PhaseProps, flash_state, last_flash_seconds
from reservoir_backend.grid.cartesian import CartesianGrid


@dataclass
class CellThermoJac:
    dv_mix: NDArray[np.float64]
    dvw: NDArray[np.float64]
    dlam_l: NDArray[np.float64]
    dlam_v: NDArray[np.float64]
    dlam_w: NDArray[np.float64]
    dxi_l: NDArray[np.float64]
    dxi_v: NDArray[np.float64]
    dxi_w: NDArray[np.float64]
    dx: NDArray[np.float64]
    dy: NDArray[np.float64]


def cell_thermo_fd(
    spec: CompSpec,
    pressure: NDArray[np.float64],
    moles: NDArray[np.float64],
    props: PhaseProps,
    n_scale: float,
    p_scale: float,
) -> tuple[CellThermoJac, float]:
    """One slot at a time over the whole grid. Reuses K from ``props``."""
    p = np.asarray(pressure, dtype=float).ravel()
    n = np.asarray(moles, dtype=float)
    n_cells, nc = n.shape
    nu = nc + 1
    n_hc = spec.n_hc
    eps_n = 1.0e-8 * max(float(n_scale), 1.0)
    eps_p = 1.0e-8 * max(float(p_scale), 1.0e5)
    dv_mix = np.zeros((n_cells, nu))
    dvw = np.zeros((n_cells, nu))
    dlam_l = np.zeros((n_cells, nu))
    dlam_v = np.zeros((n_cells, nu))
    dlam_w = np.zeros((n_cells, nu))
    dxi_l = np.zeros((n_cells, nu))
    dxi_v = np.zeros((n_cells, nu))
    dxi_w = np.zeros((n_cells, nu))
    dx = np.zeros((n_cells, n_hc, nu))
    dy = np.zeros((n_cells, n_hc, nu))
    t_flash = 0.0
    for slot in range(nu):
        n2 = n.copy()
        p2 = p.copy()
        eps = eps_n if slot < nc else eps_p
        if slot < nc:
            n2[:, slot] = n2[:, slot] + eps
        else:
            p2 = p2 + eps
        # Same flash map as the residual (Wilson). Reusing K here makes J ≠ dR.
        trial = flash_state(spec, p2, n2)
        t_flash += last_flash_seconds()
        inv = 1.0 / eps
        dv_mix[:, slot] = (trial.v_mix - props.v_mix) * inv
        dvw[:, slot] = (trial.vw - props.vw) * inv
        dlam_l[:, slot] = (trial.lam_l - props.lam_l) * inv
        dlam_v[:, slot] = (trial.lam_v - props.lam_v) * inv
        dlam_w[:, slot] = (trial.lam_w - props.lam_w) * inv
        dxi_l[:, slot] = (trial.xi_l - props.xi_l) * inv
        dxi_v[:, slot] = (trial.xi_v - props.xi_v) * inv
        dxi_w[:, slot] = (trial.xi_w - props.xi_w) * inv
        dx[:, :, slot] = (trial.x - props.x) * inv
        dy[:, :, slot] = (trial.y - props.y) * inv
    return (
        CellThermoJac(dv_mix, dvw, dlam_l, dlam_v, dlam_w, dxi_l, dxi_v, dxi_w, dx, dy),
        t_flash,
    )


def _uid(cont: int, cell: int, slot: int, n_cells: int, nu: int) -> int:
    return int(cont) * n_cells * nu + int(cell) * nu + int(slot)


def _faces(grid: CartesianGrid):
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def ids(ii, jj, kk):
        return (kk * ny * nx + jj * nx + ii).astype(np.int64)

    out: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    if nx > 1:
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
        out.append((ids(i, j, k).ravel(), ids(i + 1, j, k).ravel()))
    if ny > 1:
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
        out.append((ids(i, j, k).ravel(), ids(i, j + 1, k).ravel()))
    if nz > 1:
        k, j, i = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
        out.append((ids(i, j, k).ravel(), ids(i, j, k + 1).ravel()))
    return out


def _add_acc_vol(
    rows,
    cols,
    data,
    cont: int,
    moles: NDArray[np.float64],
    props: PhaseProps,
    th: CellThermoJac,
    spec: CompSpec,
    n_cells: int,
):
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    hc = np.sum(moles[:, :n_hc], axis=1)
    for c in range(n_cells):
        r_vol = _uid(cont, c, nc, n_cells, nu)
        for s in range(nu):
            val = hc[c] * th.dv_mix[c, s]
            if s < n_hc:
                val += props.v_mix[c]
            if spec.has_water:
                val += moles[c, n_hc] * th.dvw[c, s]
                if s == n_hc:
                    val += props.vw[c]
            rows.append(r_vol)
            cols.append(_uid(cont, c, s, n_cells, nu))
            data.append(float(val))
        for i in range(nc):
            rows.append(_uid(cont, c, i, n_cells, nu))
            cols.append(_uid(cont, c, i, n_cells, nu))
            data.append(1.0)


def _add_faces_jac(
    rows,
    cols,
    data,
    cont: int,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
    t: NDArray[np.float64],
    p: NDArray[np.float64],
    props: PhaseProps,
    th: CellThermoJac,
    spec: CompSpec,
    n_cells: int,
    dt: float,
):
    if t.size == 0:
        return
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    t = np.asarray(t, dtype=float).ravel()
    for k in range(t.size):
        L = int(left[k])
        R = int(right[k])
        T = float(t[k])
        dphi = float(p[L] - p[R])
        up = L if dphi >= 0.0 else R
        q_l = T * float(props.lam_l[up]) * dphi
        q_v = T * float(props.lam_v[up]) * dphi
        up_l = L if q_l >= 0.0 else R
        up_v = L if q_v >= 0.0 else R
        q_w = 0.0
        up_w = L
        if spec.has_water:
            q_w = T * float(props.lam_w[up]) * dphi
            up_w = L if q_w >= 0.0 else R
        for src in (L, R):
            for s in range(nu):
                d_pot = 0.0
                if s == nc:
                    d_pot = 1.0 if src == L else -1.0
                dql = T * float(props.lam_l[up]) * d_pot
                dqv = T * float(props.lam_v[up]) * d_pot
                dqw = T * float(props.lam_w[up]) * d_pot if spec.has_water else 0.0
                if src == up:
                    dql += T * dphi * float(th.dlam_l[src, s])
                    dqv += T * dphi * float(th.dlam_v[src, s])
                    if spec.has_water:
                        dqw += T * dphi * float(th.dlam_w[src, s])
                col = _uid(cont, src, s, n_cells, nu)
                for i in range(n_hc):
                    dfi = float(props.xi_l[up_l]) * float(props.x[up_l, i]) * dql
                    dfi += float(props.xi_v[up_v]) * float(props.y[up_v, i]) * dqv
                    if src == up_l:
                        dfi += q_l * (
                            float(th.dxi_l[src, s]) * float(props.x[up_l, i])
                            + float(props.xi_l[up_l]) * float(th.dx[src, i, s])
                        )
                    if src == up_v:
                        dfi += q_v * (
                            float(th.dxi_v[src, s]) * float(props.y[up_v, i])
                            + float(props.xi_v[up_v]) * float(th.dy[src, i, s])
                        )
                    rows.append(_uid(cont, L, i, n_cells, nu))
                    cols.append(col)
                    data.append(float(dt) * dfi)
                    rows.append(_uid(cont, R, i, n_cells, nu))
                    cols.append(col)
                    data.append(-float(dt) * dfi)
                if spec.has_water:
                    dfw = float(props.xi_w[up_w]) * dqw
                    if src == up_w:
                        dfw += q_w * float(th.dxi_w[src, s])
                    rows.append(_uid(cont, L, n_hc, n_cells, nu))
                    cols.append(col)
                    data.append(float(dt) * dfw)
                    rows.append(_uid(cont, R, n_hc, n_cells, nu))
                    cols.append(col)
                    data.append(-float(dt) * dfw)


def _add_transfer_jac(
    rows,
    cols,
    data,
    pm,
    pf,
    vol,
    km,
    transfer,
    props_m: PhaseProps,
    props_f: PhaseProps,
    th_m: CellThermoJac,
    th_f: CellThermoJac,
    spec: CompSpec,
    n_cells: int,
    dt: float,
):
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    cond = float(transfer.shape_factor) * np.asarray(km, dtype=float).ravel() * np.asarray(vol, dtype=float).ravel()
    dphi = np.asarray(pm, dtype=float).ravel() - np.asarray(pf, dtype=float).ravel()
    for c in range(n_cells):
        from_m = bool(dphi[c] >= 0.0)
        lam_l = float(props_m.lam_l[c] if from_m else props_f.lam_l[c])
        lam_v = float(props_m.lam_v[c] if from_m else props_f.lam_v[c])
        q_l = float(cond[c]) * lam_l * float(dphi[c])
        q_v = float(cond[c]) * lam_v * float(dphi[c])
        up_l = 1 if q_l >= 0.0 else 0
        up_v = 1 if q_v >= 0.0 else 0
        props_ul = props_m if up_l else props_f
        props_uv = props_m if up_v else props_f
        th_ul = th_m if up_l else th_f
        th_uv = th_m if up_v else th_f
        th_pot = th_m if from_m else th_f
        for cont_src, d_pot_p in ((1, 1.0), (0, -1.0)):
            th_s = th_m if cont_src == 1 else th_f
            for s in range(nu):
                d_pot = d_pot_p if s == nc else 0.0
                dql = float(cond[c]) * lam_l * d_pot
                dqv = float(cond[c]) * lam_v * d_pot
                if cont_src == (1 if from_m else 0):
                    dql += float(cond[c]) * float(dphi[c]) * float(th_pot.dlam_l[c, s])
                    dqv += float(cond[c]) * float(dphi[c]) * float(th_pot.dlam_v[c, s])
                col = _uid(cont_src, c, s, n_cells, nu)
                for i in range(n_hc):
                    dNi = float(props_ul.xi_l[c]) * float(props_ul.x[c, i]) * dql
                    dNi += float(props_uv.xi_v[c]) * float(props_uv.y[c, i]) * dqv
                    if cont_src == up_l:
                        dNi += q_l * (
                            float(th_ul.dxi_l[c, s]) * float(props_ul.x[c, i])
                            + float(props_ul.xi_l[c]) * float(th_ul.dx[c, i, s])
                        )
                    if cont_src == up_v:
                        dNi += q_v * (
                            float(th_uv.dxi_v[c, s]) * float(props_uv.y[c, i])
                            + float(props_uv.xi_v[c]) * float(th_uv.dy[c, i, s])
                        )
                    rows.append(_uid(0, c, i, n_cells, nu))
                    cols.append(col)
                    data.append(-float(dt) * dNi)
                    rows.append(_uid(1, c, i, n_cells, nu))
                    cols.append(col)
                    data.append(float(dt) * dNi)
                _ = th_s


def assemble_block_jacobian(
    grid: CartesianGrid,
    spec: CompSpec,
    dual_rock,
    state,
    dt: float,
    transfer,
    t_f,
    t_m,
    props_f: PhaseProps,
    props_m: PhaseProps,
    n_scale: float,
    p_scale: float,
    q_f: NDArray[np.float64] | None = None,
    q_m: NDArray[np.float64] | None = None,
) -> tuple[sparse.csc_matrix, float]:
    """Accumulation + flux + transfer. Wells are handled by the caller if needed."""
    n_cells = grid.n_cells
    nc = spec.nc
    nu = nc + 1
    n_u = 2 * n_cells * nu
    th_f, fl_f = cell_thermo_fd(spec, state.fracture.pressure, state.fracture.moles, props_f, n_scale, p_scale)
    th_m, fl_m = cell_thermo_fd(spec, state.matrix.pressure, state.matrix.moles, props_m, n_scale, p_scale)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    _add_acc_vol(rows, cols, data, 0, state.fracture.moles, props_f, th_f, spec, n_cells)
    _add_acc_vol(rows, cols, data, 1, state.matrix.moles, props_m, th_m, spec, n_cells)
    face_pairs = _faces(grid)
    tf_axes: list[NDArray[np.float64]] = []
    tm_axes: list[NDArray[np.float64]] = []
    if grid.nx > 1:
        tf_axes.append(t_f[0])
        tm_axes.append(t_m[0])
    if grid.ny > 1:
        tf_axes.append(t_f[1])
        tm_axes.append(t_m[1])
    if grid.nz > 1:
        tf_axes.append(t_f[2])
        tm_axes.append(t_m[2])
    for (left, right), tf, tm in zip(face_pairs, tf_axes, tm_axes):
        _add_faces_jac(rows, cols, data, 0, left, right, tf, state.fracture.pressure, props_f, th_f, spec, n_cells, dt)
        _add_faces_jac(rows, cols, data, 1, left, right, tm, state.matrix.pressure, props_m, th_m, spec, n_cells, dt)
    vol = grid.cell_volumes()
    km = np.asarray(dual_rock.matrix.permeability, dtype=float).ravel()
    _add_transfer_jac(
        rows, cols, data, state.matrix.pressure, state.fracture.pressure, vol, km, transfer, props_m, props_f, th_m, th_f, spec, n_cells, dt
    )
    jac = sparse.csc_matrix((data, (rows, cols)), shape=(n_u, n_u))
    return jac, fl_f + fl_m
