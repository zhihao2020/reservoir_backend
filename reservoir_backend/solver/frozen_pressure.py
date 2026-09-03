"""Short-window DPDP pressure with frozen mobility (no flash).

Accumulation uses c_t frozen from the last full compositional flash.
Wells use live controls; λ is held.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.ports.flow import FlowPort, half_cell_wi, peaceman_wi
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp import _control_map


def _face_ids(grid: CartesianGrid, axis: str):
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def ids(ii, jj, kk):
        return (kk * ny * nx + jj * nx + ii).astype(np.int64)

    if axis == "x" and nx > 1:
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
        return ids(i, j, k).ravel(), ids(i + 1, j, k).ravel()
    if axis == "y" and ny > 1:
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
        return ids(i, j, k).ravel(), ids(i, j + 1, k).ravel()
    if axis == "z" and nz > 1:
        k, j, i = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
        return ids(i, j, k).ravel(), ids(i, j, k + 1).ravel()
    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)


def _add_tpfa(rows, cols, data, left, right, t, lam, pressure, offset: int) -> None:
    if t.size == 0:
        return
    dphi = pressure[left] - pressure[right]
    cond = np.asarray(t, dtype=float).ravel() * np.where(dphi >= 0.0, lam[left], lam[right])
    rows.extend((offset + left).tolist())
    cols.extend((offset + left).tolist())
    data.extend(cond.tolist())
    rows.extend((offset + left).tolist())
    cols.extend((offset + right).tolist())
    data.extend((-cond).tolist())
    rows.extend((offset + right).tolist())
    cols.extend((offset + right).tolist())
    data.extend(cond.tolist())
    rows.extend((offset + right).tolist())
    cols.extend((offset + left).tolist())
    data.extend((-cond).tolist())


def _wi(grid: CartesianGrid, k: float, port: FlowPort, cell: int) -> float:
    if port.use_productivity:
        return float(
            peaceman_wi(
                grid,
                int(cell),
                float(k),
                rw_m=port.rw_m,
                skin=port.skin,
                geofac=port.geofac,
                axis=getattr(port, "axis", "k"),
            )
        )
    return float(half_cell_wi(grid, int(cell), float(k))) * float(port.wi_multiplier)


def _well_terms(
    grid: CartesianGrid,
    dual_rock: DualRock,
    ports: list[FlowPort],
    cmap: dict,
    pf: NDArray[np.float64],
    pm: NDArray[np.float64],
    lf: NDArray[np.float64],
    lm: NDArray[np.float64],
    v_f: NDArray[np.float64],
    v_m: NDArray[np.float64],
    t_eval: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n = grid.n_cells
    diag_f = np.zeros(n)
    diag_m = np.zeros(n)
    rhs_f = np.zeros(n)
    rhs_m = np.zeros(n)
    for port in ports:
        coupling = str(getattr(port, "continuum_coupling", "fracture"))
        frac = float(getattr(port, "fracture_fraction", 1.0))
        cells = np.asarray(port.cell_ids, dtype=np.int64)
        if coupling == "matrix":
            rock, _, lam, vmix, diag, rhs, wgt = dual_rock.matrix, pm, lm, v_m, diag_m, rhs_m, 0.0
        else:
            rock, _, lam, vmix, diag, rhs, wgt = dual_rock.fracture, pf, lf, v_f, diag_f, rhs_f, 1.0 if coupling == "fracture" else frac
        k = np.asarray(rock.permeability, dtype=float).ravel()
        wi = np.array([_wi(grid, float(k[int(c)]), port, int(c)) for c in cells], dtype=float)
        w = np.maximum(wi * lam[cells], 1.0e-30)
        series_p = cmap.get((port.name, "pressure"))
        series_q = cmap.get((port.name, "rate"))
        if port.control == "rate" or (series_q is not None and series_p is None):
            q_mol = float(series_q.value_at(t_eval)) if series_q is not None else 0.0
            share = w / max(float(np.sum(w)), 1.0e-30)
            q_vol = q_mol * share * np.maximum(vmix[cells], 1.0e-12)
            if coupling == "split":
                rhs_f[cells] += wgt * q_vol
                rhs_m[cells] += (1.0 - wgt) * q_vol
            else:
                rhs[cells] += q_vol
            continue
        if series_p is None:
            continue
        p_wf = float(series_p.value_at(t_eval))
        if coupling == "split":
            diag_f[cells] += wgt * w
            rhs_f[cells] += wgt * w * p_wf
            diag_m[cells] += (1.0 - wgt) * w
            rhs_m[cells] += (1.0 - wgt) * w * p_wf
        else:
            diag[cells] += w
            rhs[cells] += w * p_wf
    return diag_f, rhs_f, diag_m, rhs_m


def _control_signature(ports: list[FlowPort] | None, cmap: dict) -> tuple:
    if not ports:
        return ()
    rows = []
    for port in ports:
        has_p = (port.name, "pressure") in cmap
        has_q = (port.name, "rate") in cmap
        rows.append((port.name, str(port.control), str(getattr(port, "continuum_coupling", "fracture")), has_p, has_q))
    return tuple(rows)


@dataclass
class FrozenPressureContext:
    """Reuse the frozen-λ pressure LU between slow-loop refreshes."""

    lu: object | None = None
    acc_f: NDArray[np.float64] | None = None
    acc_m: NDArray[np.float64] | None = None
    n: int = 0
    key: tuple | None = None
    n_reuse: int = 0
    n_factor: int = 0

    def matches(self, key: tuple) -> bool:
        return self.lu is not None and self.key == key


def step_frozen_pressure(
    grid: CartesianGrid,
    ctx: DPDPModelContext,
    dual_rock: DualRock,
    transfer: ComponentTransfer,
    p_fracture: NDArray[np.float64],
    p_matrix: NDArray[np.float64],
    lam_fracture: NDArray[np.float64],
    lam_matrix: NDArray[np.float64],
    dt: float,
    *,
    ct_fracture: NDArray[np.float64] | None = None,
    ct_matrix: NDArray[np.float64] | None = None,
    ports: list[FlowPort] | None = None,
    controls: list[ControlSeries] | dict | None = None,
    t_eval: float = 0.0,
    v_mix_fracture: NDArray[np.float64] | None = None,
    v_mix_matrix: NDArray[np.float64] | None = None,
    factor: FrozenPressureContext | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Implicit pressure step. Mobility and composition are frozen."""
    n = grid.n_cells
    pf = np.asarray(p_fracture, dtype=float).ravel()
    pm = np.asarray(p_matrix, dtype=float).ravel()
    lf = np.maximum(np.asarray(lam_fracture, dtype=float).ravel(), 1.0e-18)
    lm = np.maximum(np.asarray(lam_matrix, dtype=float).ravel(), 1.0e-18)
    cmap = {}
    if ports:
        cmap = controls if isinstance(controls, dict) else _control_map(list(controls or []))
    ct_f_arr = np.asarray(ct_fracture if ct_fracture is not None else np.full(n, 1.0e-9), dtype=float).ravel()
    ct_m_arr = np.asarray(ct_matrix if ct_matrix is not None else np.full(n, 1.0e-9), dtype=float).ravel()
    key = (
        n,
        float(dt),
        float(np.sum(lf)),
        float(np.sum(lm)),
        float(np.sum(ct_f_arr)),
        float(np.sum(ct_m_arr)),
        _control_signature(list(ports or []), cmap),
    )
    if factor is not None and factor.matches(key):
        rhs_f = factor.acc_f * pf
        rhs_m = factor.acc_m * pm
        if ports:
            vf = np.asarray(v_mix_fracture if v_mix_fracture is not None else np.full(n, 1.0e-4), dtype=float).ravel()
            vm = np.asarray(v_mix_matrix if v_mix_matrix is not None else np.full(n, 1.0e-4), dtype=float).ravel()
            _d_f, r_f, _d_m, r_m = _well_terms(
                grid, dual_rock, list(ports), cmap, pf, pm, lf, lm, vf, vm, float(t_eval)
            )
            rhs_f = rhs_f + r_f
            rhs_m = rhs_m + r_m
        x = np.asarray(factor.lu.solve(np.concatenate([rhs_f, rhs_m])), dtype=float).ravel()
        factor.n_reuse += 1
        if x.size != 2 * n or not np.all(np.isfinite(x)):
            return pf.copy(), pm.copy()
        return x[:n], x[n:]
    t_f, t_m = ctx.transmissibilities(dual_rock)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for t, lam, p, off, axis in (
        (t_f[0], lf, pf, 0, "x"),
        (t_f[1], lf, pf, 0, "y"),
        (t_f[2], lf, pf, 0, "z"),
        (t_m[0], lm, pm, n, "x"),
        (t_m[1], lm, pm, n, "y"),
        (t_m[2], lm, pm, n, "z"),
    ):
        left, right = _face_ids(grid, axis)
        _add_tpfa(rows, cols, data, left, right, t, lam, p, off)
    vol = ctx.cell_volumes
    dt = max(float(dt), 1.0e-12)
    ct_f = ct_f_arr
    ct_m = ct_m_arr
    acc_f = np.asarray(dual_rock.fracture.porosity, dtype=float).ravel() * vol * ct_f / dt
    acc_m = np.asarray(dual_rock.matrix.porosity, dtype=float).ravel() * vol * ct_m / dt
    idx = np.arange(n, dtype=np.int64)
    rows.extend(idx.tolist())
    cols.extend(idx.tolist())
    data.extend(acc_f.tolist())
    rows.extend((n + idx).tolist())
    cols.extend((n + idx).tolist())
    data.extend(acc_m.tolist())
    km = np.asarray(dual_rock.matrix.permeability, dtype=float).ravel()
    dphi = pm - pf
    lam_up = np.where(dphi >= 0.0, lm, lf)
    tau = float(transfer.shape_factor) * km * vol * lam_up * float(getattr(transfer, "transfer_multiplier", 1.0))
    rows.extend(idx.tolist())
    cols.extend(idx.tolist())
    data.extend(tau.tolist())
    rows.extend(idx.tolist())
    cols.extend((n + idx).tolist())
    data.extend((-tau).tolist())
    rows.extend((n + idx).tolist())
    cols.extend((n + idx).tolist())
    data.extend(tau.tolist())
    rows.extend((n + idx).tolist())
    cols.extend(idx.tolist())
    data.extend((-tau).tolist())
    rhs_f = acc_f * pf
    rhs_m = acc_m * pm
    if ports:
        vf = np.asarray(v_mix_fracture if v_mix_fracture is not None else np.full(n, 1.0e-4), dtype=float).ravel()
        vm = np.asarray(v_mix_matrix if v_mix_matrix is not None else np.full(n, 1.0e-4), dtype=float).ravel()
        d_f, r_f, d_m, r_m = _well_terms(grid, dual_rock, list(ports), cmap, pf, pm, lf, lm, vf, vm, float(t_eval))
        rows.extend(idx.tolist())
        cols.extend(idx.tolist())
        data.extend(d_f.tolist())
        rows.extend((n + idx).tolist())
        cols.extend((n + idx).tolist())
        data.extend(d_m.tolist())
        rhs_f = rhs_f + r_f
        rhs_m = rhs_m + r_m
    a = sparse.csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))
    rhs = np.concatenate([rhs_f, rhs_m])
    a_csc = a.tocsc()
    if factor is not None:
        factor.lu = splu(a_csc)
        factor.acc_f = acc_f
        factor.acc_m = acc_m
        factor.n = n
        factor.key = key
        factor.n_factor += 1
        x = np.asarray(factor.lu.solve(rhs), dtype=float).ravel()
    else:
        x = np.asarray(spsolve(a_csc, rhs), dtype=float).ravel()
    if x.size != 2 * n or not np.all(np.isfinite(x)):
        return pf.copy(), pm.copy()
    return x[:n], x[n:]
