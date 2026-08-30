"""Short-window DPDP pressure with frozen mobility (no flash).

Accumulation uses a small compressibility so 1 s steps stay local.
λ comes from the last full compositional flash.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext

_C_PA = 1.0e-9


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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Implicit pressure step. Mobility is frozen; composition is not updated."""
    n = grid.n_cells
    pf = np.asarray(p_fracture, dtype=float).ravel()
    pm = np.asarray(p_matrix, dtype=float).ravel()
    lf = np.maximum(np.asarray(lam_fracture, dtype=float).ravel(), 1.0e-18)
    lm = np.maximum(np.asarray(lam_matrix, dtype=float).ravel(), 1.0e-18)
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
    acc_f = np.asarray(dual_rock.fracture.porosity, dtype=float).ravel() * vol * _C_PA / dt
    acc_m = np.asarray(dual_rock.matrix.porosity, dtype=float).ravel() * vol * _C_PA / dt
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
    tau = float(transfer.shape_factor) * km * vol * lam_up
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
    a = sparse.csr_matrix((data, (rows, cols)), shape=(2 * n, 2 * n))
    rhs = np.concatenate([acc_f * pf, acc_m * pm])
    x = np.asarray(spsolve(a.tocsc(), rhs), dtype=float).ravel()
    if x.size != 2 * n or not np.all(np.isfinite(x)):
        return pf.copy(), pm.copy()
    return x[:n], x[n:]
