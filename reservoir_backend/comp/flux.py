"""TPFA molar component flux. Phase-potential upwind, JutulDarcy-style split.

q_i = ξ_L x_i q_L + ξ_V y_i q_V. No molecular diffusion in this cut.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.properties import PhaseProps
from reservoir_backend.grid.cartesian import CartesianGrid


def _add_faces(
    div: NDArray[np.float64],
    t: NDArray[np.float64],
    p: NDArray[np.float64],
    props: PhaseProps,
    left: NDArray[np.int64],
    right: NDArray[np.int64],
) -> None:
    if t.size == 0:
        return
    dphi = p[left] - p[right]
    q_l = t * np.where(dphi >= 0.0, props.lam_l[left], props.lam_l[right]) * dphi
    q_v = t * np.where(dphi >= 0.0, props.lam_v[left], props.lam_v[right]) * dphi
    x_up = np.where(q_l[:, None] >= 0.0, props.x[left], props.x[right])
    y_up = np.where(q_v[:, None] >= 0.0, props.y[left], props.y[right])
    xi_l = np.where(q_l >= 0.0, props.xi_l[left], props.xi_l[right])
    xi_v = np.where(q_v >= 0.0, props.xi_v[left], props.xi_v[right])
    n_hc = props.x.shape[1]
    flux = xi_l[:, None] * x_up * q_l[:, None] + xi_v[:, None] * y_up * q_v[:, None]
    np.add.at(div[:, :n_hc], left, flux)
    np.add.at(div[:, :n_hc], right, -flux)
    if props.has_water and div.shape[1] > n_hc:
        q_w = t * np.where(dphi >= 0.0, props.lam_w[left], props.lam_w[right]) * dphi
        xi_w = np.where(q_w >= 0.0, props.xi_w[left], props.xi_w[right])
        fw = xi_w * q_w
        np.add.at(div[:, n_hc], left, fw)
        np.add.at(div[:, n_hc], right, -fw)


def molar_divergence(
    grid: CartesianGrid,
    t_x: NDArray[np.float64],
    t_y: NDArray[np.float64],
    t_z: NDArray[np.float64],
    pressure: NDArray[np.float64],
    props: PhaseProps,
) -> NDArray[np.float64]:
    """Net molar outflow per cell, shape (n_cells, n_hc[+1 water])."""
    n = grid.n_cells
    n_hc = props.x.shape[1]
    n_comp = n_hc + (1 if props.has_water else 0)
    div = np.zeros((n, n_comp))
    p = np.asarray(pressure, dtype=float).ravel()
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def ids(ii, jj, kk):
        return (kk * ny * nx + jj * nx + ii).astype(np.int64)

    if nx > 1 and t_x.size:
        ii = np.arange(nx - 1)
        jj = np.arange(ny)
        kk = np.arange(nz)
        k, j, i = np.meshgrid(kk, jj, ii, indexing="ij")
        left = ids(i, j, k).ravel()
        right = ids(i + 1, j, k).ravel()
        _add_faces(div, np.asarray(t_x, dtype=float).ravel(), p, props, left, right)
    if ny > 1 and t_y.size:
        ii = np.arange(nx)
        jj = np.arange(ny - 1)
        kk = np.arange(nz)
        k, j, i = np.meshgrid(kk, jj, ii, indexing="ij")
        left = ids(i, j, k).ravel()
        right = ids(i, j + 1, k).ravel()
        _add_faces(div, np.asarray(t_y, dtype=float).ravel(), p, props, left, right)
    if nz > 1 and t_z.size:
        ii = np.arange(nx)
        jj = np.arange(ny)
        kk = np.arange(nz - 1)
        k, j, i = np.meshgrid(kk, jj, ii, indexing="ij")
        left = ids(i, j, k).ravel()
        right = ids(i, j, k + 1).ravel()
        _add_faces(div, np.asarray(t_z, dtype=float).ravel(), p, props, left, right)
    return div
