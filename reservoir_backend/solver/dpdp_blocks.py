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


def _acc_coo(
    cont: int,
    moles: NDArray[np.float64],
    props: PhaseProps,
    th: CellThermoJac,
    spec: CompSpec,
    n_cells: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    hc = np.sum(moles[:, :n_hc], axis=1)
    c = np.arange(n_cells, dtype=np.int64)
    r_vol = cont * n_cells * nu + c * nu + nc
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []
    for s in range(nu):
        val = hc * th.dv_mix[:, s]
        if s < n_hc:
            val = val + props.v_mix
        if spec.has_water:
            val = val + moles[:, n_hc] * th.dvw[:, s]
            if s == n_hc:
                val = val + props.vw
        rows.append(r_vol)
        cols.append(cont * n_cells * nu + c * nu + s)
        data.append(val)
    for i in range(nc):
        rows.append(cont * n_cells * nu + c * nu + i)
        cols.append(cont * n_cells * nu + c * nu + i)
        data.append(np.ones(n_cells))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


def _faces_coo(
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
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, np.zeros(0)
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    L = np.asarray(left, dtype=np.int64).ravel()
    R = np.asarray(right, dtype=np.int64).ravel()
    dphi = p[L] - p[R]
    up = np.where(dphi >= 0.0, L, R)
    q_l = t * props.lam_l[up] * dphi
    q_v = t * props.lam_v[up] * dphi
    up_l = np.where(q_l >= 0.0, L, R)
    up_v = np.where(q_v >= 0.0, L, R)
    q_w = t * props.lam_w[up] * dphi if spec.has_water else np.zeros_like(t)
    up_w = np.where(q_w >= 0.0, L, R) if spec.has_water else L
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []
    for src, pot_sign in ((L, 1.0), (R, -1.0)):
        is_up = src == up
        is_up_l = src == up_l
        is_up_v = src == up_v
        is_up_w = src == up_w if spec.has_water else np.zeros(src.size, dtype=bool)
        for s in range(nu):
            d_pot = np.full(src.size, pot_sign) if s == nc else np.zeros(src.size)
            dql = t * props.lam_l[up] * d_pot
            dqv = t * props.lam_v[up] * d_pot
            dqw = t * props.lam_w[up] * d_pot if spec.has_water else np.zeros(src.size)
            dql = dql + np.where(is_up, t * dphi * th.dlam_l[src, s], 0.0)
            dqv = dqv + np.where(is_up, t * dphi * th.dlam_v[src, s], 0.0)
            if spec.has_water:
                dqw = dqw + np.where(is_up, t * dphi * th.dlam_w[src, s], 0.0)
            col = cont * n_cells * nu + src * nu + s
            for i in range(n_hc):
                dfi = props.xi_l[up_l] * props.x[up_l, i] * dql + props.xi_v[up_v] * props.y[up_v, i] * dqv
                extra_l = q_l * (th.dxi_l[src, s] * props.x[up_l, i] + props.xi_l[up_l] * th.dx[src, i, s])
                extra_v = q_v * (th.dxi_v[src, s] * props.y[up_v, i] + props.xi_v[up_v] * th.dy[src, i, s])
                dfi = dfi + np.where(is_up_l, extra_l, 0.0) + np.where(is_up_v, extra_v, 0.0)
                rows.append(cont * n_cells * nu + L * nu + i)
                cols.append(col)
                data.append(float(dt) * dfi)
                rows.append(cont * n_cells * nu + R * nu + i)
                cols.append(col)
                data.append(-float(dt) * dfi)
            if spec.has_water:
                dfw = props.xi_w[up_w] * dqw
                dfw = dfw + np.where(is_up_w, q_w * th.dxi_w[src, s], 0.0)
                rows.append(cont * n_cells * nu + L * nu + n_hc)
                cols.append(col)
                data.append(float(dt) * dfw)
                rows.append(cont * n_cells * nu + R * nu + n_hc)
                cols.append(col)
                data.append(-float(dt) * dfw)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


def _transfer_coo(
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
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    nc = spec.nc
    nu = nc + 1
    n_hc = spec.n_hc
    c = np.arange(n_cells, dtype=np.int64)
    cond = float(transfer.shape_factor) * np.asarray(km, dtype=float).ravel() * np.asarray(vol, dtype=float).ravel()
    dphi = np.asarray(pm, dtype=float).ravel() - np.asarray(pf, dtype=float).ravel()
    from_m = dphi >= 0.0
    lam_l = np.where(from_m, props_m.lam_l, props_f.lam_l)
    lam_v = np.where(from_m, props_m.lam_v, props_f.lam_v)
    q_l = cond * lam_l * dphi
    q_v = cond * lam_v * dphi
    up_l = np.where(q_l >= 0.0, 1, 0)
    up_v = np.where(q_v >= 0.0, 1, 0)
    pot_cont = np.where(from_m, 1, 0)
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []
    for cont_src, d_pot_p in ((1, 1.0), (0, -1.0)):
        th_s = th_m if cont_src == 1 else th_f
        is_pot = cont_src == pot_cont
        is_up_l = cont_src == up_l
        is_up_v = cont_src == up_v
        for s in range(nu):
            d_pot = np.full(n_cells, d_pot_p) if s == nc else np.zeros(n_cells)
            dql = cond * lam_l * d_pot
            dqv = cond * lam_v * d_pot
            dlam_l = np.where(from_m, th_m.dlam_l[:, s], th_f.dlam_l[:, s])
            dlam_v = np.where(from_m, th_m.dlam_v[:, s], th_f.dlam_v[:, s])
            dql = dql + np.where(is_pot, cond * dphi * dlam_l, 0.0)
            dqv = dqv + np.where(is_pot, cond * dphi * dlam_v, 0.0)
            col = cont_src * n_cells * nu + c * nu + s
            xi_l = np.where(up_l == 1, props_m.xi_l, props_f.xi_l)
            xi_v = np.where(up_v == 1, props_m.xi_v, props_f.xi_v)
            dxi_l = np.where(up_l == 1, th_m.dxi_l[:, s], th_f.dxi_l[:, s])
            dxi_v = np.where(up_v == 1, th_m.dxi_v[:, s], th_f.dxi_v[:, s])
            for i in range(n_hc):
                x_l = np.where(up_l == 1, props_m.x[:, i], props_f.x[:, i])
                y_v = np.where(up_v == 1, props_m.y[:, i], props_f.y[:, i])
                dx_l = np.where(up_l == 1, th_m.dx[:, i, s], th_f.dx[:, i, s])
                dy_v = np.where(up_v == 1, th_m.dy[:, i, s], th_f.dy[:, i, s])
                dNi = xi_l * x_l * dql + xi_v * y_v * dqv
                extra_l = q_l * (dxi_l * x_l + xi_l * dx_l)
                extra_v = q_v * (dxi_v * y_v + xi_v * dy_v)
                dNi = dNi + np.where(is_up_l, extra_l, 0.0) + np.where(is_up_v, extra_v, 0.0)
                rows.append(c * nu + i)
                cols.append(col)
                data.append(-float(dt) * dNi)
                rows.append(n_cells * nu + c * nu + i)
                cols.append(col)
                data.append(float(dt) * dNi)
            _ = th_s
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


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
    parts_r: list[NDArray[np.int64]] = []
    parts_c: list[NDArray[np.int64]] = []
    parts_d: list[NDArray[np.float64]] = []
    for piece in (
        _acc_coo(0, state.fracture.moles, props_f, th_f, spec, n_cells),
        _acc_coo(1, state.matrix.moles, props_m, th_m, spec, n_cells),
    ):
        parts_r.append(piece[0])
        parts_c.append(piece[1])
        parts_d.append(piece[2])
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
        fr = _faces_coo(0, left, right, tf, state.fracture.pressure, props_f, th_f, spec, n_cells, dt)
        mr = _faces_coo(1, left, right, tm, state.matrix.pressure, props_m, th_m, spec, n_cells, dt)
        parts_r.extend((fr[0], mr[0]))
        parts_c.extend((fr[1], mr[1]))
        parts_d.extend((fr[2], mr[2]))
    vol = grid.cell_volumes()
    km = np.asarray(dual_rock.matrix.permeability, dtype=float).ravel()
    tr = _transfer_coo(
        state.matrix.pressure,
        state.fracture.pressure,
        vol,
        km,
        transfer,
        props_m,
        props_f,
        th_m,
        th_f,
        spec,
        n_cells,
        dt,
    )
    parts_r.append(tr[0])
    parts_c.append(tr[1])
    parts_d.append(tr[2])
    rows = np.concatenate(parts_r)
    cols = np.concatenate(parts_c)
    data = np.concatenate(parts_d)
    jac = _csc_cached(rows, cols, data, n_u)
    return jac, fl_f + fl_m


_JAC_CACHE: dict[tuple[int, int], tuple] = {}


def _csc_cached(rows, cols, data, n_u: int):
    """Reuse CSC indptr/indices; fill values with the same COO pattern."""
    key = (int(n_u), int(rows.size))
    cached = _JAC_CACHE.get(key)
    if cached is None:
        jac = sparse.csc_matrix((data, (rows, cols)), shape=(n_u, n_u))
        indptr = np.asarray(jac.indptr, dtype=np.int64)
        indices = np.asarray(jac.indices, dtype=np.int64)
        col_csc = np.repeat(np.arange(n_u, dtype=np.int64), np.diff(indptr))
        keys_csc = col_csc * int(n_u) + indices.astype(np.int64)
        order = np.argsort(keys_csc, kind="mergesort")
        sorted_keys = keys_csc[order]
        keys_coo = cols.astype(np.int64) * int(n_u) + rows.astype(np.int64)
        mapping = order[np.searchsorted(sorted_keys, keys_coo)]
        _JAC_CACHE[key] = (indptr, indices, mapping)
        return jac
    indptr, indices, mapping = cached
    acc = np.zeros(indices.size, dtype=float)
    np.add.at(acc, mapping, data)
    return sparse.csc_matrix((acc, indices, indptr), shape=(n_u, n_u))
