"""Two-point flux approximation on Cartesian K-orthogonal grids.

Geometric half-transmissibility follows MRST ``computeTrans``.
Phase-potential upwind of mobility follows ``getFluxAndProps*_BO``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve

from reservoir_backend.exceptions import LinearSolveFailure
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import as_cell_field


def _harmonic(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.zeros(a.shape, dtype=float)
    ok = (a > 0.0) & (b > 0.0)
    out[ok] = 2.0 * a[ok] * b[ok] / (a[ok] + b[ok])
    return out


@dataclass
class TpfaSystem:
    """Assembled two-point operator plus last face fluxes."""

    matrix: sparse.csr_matrix
    rhs: NDArray[np.float64]
    t_x: NDArray[np.float64]
    t_y: NDArray[np.float64]
    t_z: NDArray[np.float64]
    g_x: NDArray[np.float64] | None = None
    g_y: NDArray[np.float64] | None = None
    g_z: NDArray[np.float64] | None = None


def geometric_transmissibility(
    grid: CartesianGrid,
    kx: NDArray[np.float64],
    *,
    ky: NDArray[np.float64] | None = None,
    kz: NDArray[np.float64] | None = None,
    mult_x: NDArray[np.float64] | None = None,
    mult_y: NDArray[np.float64] | None = None,
    mult_z: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Face T_geom such that Q_α = T_geom λ_α (Φ_L − Φ_R)."""
    n = grid.n_cells
    kx_ijk = grid.reshape_ijk(as_cell_field(kx, n, "kx"))
    ky_ijk = grid.reshape_ijk(as_cell_field(ky if ky is not None else kx, n, "ky"))
    kz_ijk = grid.reshape_ijk(as_cell_field(kz if kz is not None else kx, n, "kz"))
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    if nx > 1:
        k_f = _harmonic(kx_ijk[:, :, :-1], kx_ijk[:, :, 1:])
        area = grid.dz[:, None, None] * grid.dy[None, :, None]
        dist = grid.center_distance_x()[None, None, :]
        t_x = k_f * area / np.maximum(dist, 1.0e-30)
        if mult_x is not None:
            t_x = t_x * np.asarray(mult_x, dtype=float)
    else:
        t_x = np.zeros((nz, ny, 0), dtype=float)

    if ny > 1:
        k_f = _harmonic(ky_ijk[:, :-1, :], ky_ijk[:, 1:, :])
        area = grid.dz[:, None, None] * grid.dx[None, None, :]
        dist = grid.center_distance_y()[None, :, None]
        t_y = k_f * area / np.maximum(dist, 1.0e-30)
        if mult_y is not None:
            t_y = t_y * np.asarray(mult_y, dtype=float)
    else:
        t_y = np.zeros((nz, 0, nx), dtype=float)

    if nz > 1:
        k_f = _harmonic(kz_ijk[:-1, :, :], kz_ijk[1:, :, :])
        area = grid.dy[None, :, None] * grid.dx[None, None, :]
        dist = grid.center_distance_z()[:, None, None]
        t_z = k_f * area / np.maximum(dist, 1.0e-30)
        if mult_z is not None:
            t_z = t_z * np.asarray(mult_z, dtype=float)
    else:
        t_z = np.zeros((0, ny, nx), dtype=float)
    return t_x, t_y, t_z


def _upwind_pair(
    mob_l: NDArray[np.float64],
    mob_r: NDArray[np.float64],
    dphi: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(dphi >= 0.0, mob_l, mob_r)


def _phase_face_ops(
    t_geom: NDArray[np.float64],
    lw_l: NDArray[np.float64],
    lw_r: NDArray[np.float64],
    lo_l: NDArray[np.float64],
    lo_r: NDArray[np.float64],
    p_l: NDArray[np.float64],
    p_r: NDArray[np.float64],
    z_l: NDArray[np.float64],
    z_r: NDArray[np.float64],
    gravity: float,
    rho_w: float,
    rho_o: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Conductivity and gravity flux L→R from phase-potential upwind."""
    g = float(gravity)
    dphi_w = (p_l - p_r) + float(rho_w) * g * (z_l - z_r)
    dphi_o = (p_l - p_r) + float(rho_o) * g * (z_l - z_r)
    lw_f = _upwind_pair(lw_l, lw_r, dphi_w)
    lo_f = _upwind_pair(lo_l, lo_r, dphi_o)
    cond = t_geom * (lw_f + lo_f)
    gflux = t_geom * (lw_f * float(rho_w) + lo_f * float(rho_o)) * g * (z_l - z_r)
    return cond, gflux


def interior_transmissibility(
    grid: CartesianGrid,
    k: NDArray[np.float64],
    mobility: NDArray[np.float64],
    *,
    kz: NDArray[np.float64] | None = None,
    ky: NDArray[np.float64] | None = None,
    mult_x: NDArray[np.float64] | None = None,
    mult_y: NDArray[np.float64] | None = None,
    mult_z: NDArray[np.float64] | None = None,
    lw: NDArray[np.float64] | None = None,
    lo: NDArray[np.float64] | None = None,
    pressure: NDArray[np.float64] | None = None,
    gravity: float = 0.0,
    rho_w: float = 1000.0,
    rho_o: float = 800.0,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Face conductivity T λ and gravity flux. Q = Tλ Δp + gflux."""
    n = grid.n_cells
    t_gx, t_gy, t_gz = geometric_transmissibility(grid, k, ky=ky, kz=kz, mult_x=mult_x, mult_y=mult_y, mult_z=mult_z)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    use_upwind = lw is not None and lo is not None and pressure is not None
    if use_upwind:
        lw_ijk = grid.reshape_ijk(as_cell_field(lw, n, "lw"))
        lo_ijk = grid.reshape_ijk(as_cell_field(lo, n, "lo"))
        p_ijk = grid.reshape_ijk(as_cell_field(pressure, n, "pressure"))
        z_ijk = grid.reshape_ijk(grid.cell_centers()[:, 2])
    else:
        m_ijk = grid.reshape_ijk(as_cell_field(mobility, n, "mobility"))

    if nx > 1:
        if use_upwind:
            t_x, g_x = _phase_face_ops(
                t_gx,
                lw_ijk[:, :, :-1],
                lw_ijk[:, :, 1:],
                lo_ijk[:, :, :-1],
                lo_ijk[:, :, 1:],
                p_ijk[:, :, :-1],
                p_ijk[:, :, 1:],
                z_ijk[:, :, :-1],
                z_ijk[:, :, 1:],
                gravity,
                rho_w,
                rho_o,
            )
        else:
            t_x = t_gx * 0.5 * (m_ijk[:, :, :-1] + m_ijk[:, :, 1:])
            g_x = np.zeros_like(t_x)
    else:
        t_x = np.zeros((nz, ny, 0), dtype=float)
        g_x = np.zeros((nz, ny, 0), dtype=float)

    if ny > 1:
        if use_upwind:
            t_y, g_y = _phase_face_ops(
                t_gy,
                lw_ijk[:, :-1, :],
                lw_ijk[:, 1:, :],
                lo_ijk[:, :-1, :],
                lo_ijk[:, 1:, :],
                p_ijk[:, :-1, :],
                p_ijk[:, 1:, :],
                z_ijk[:, :-1, :],
                z_ijk[:, 1:, :],
                gravity,
                rho_w,
                rho_o,
            )
        else:
            t_y = t_gy * 0.5 * (m_ijk[:, :-1, :] + m_ijk[:, 1:, :])
            g_y = np.zeros_like(t_y)
    else:
        t_y = np.zeros((nz, 0, nx), dtype=float)
        g_y = np.zeros((nz, 0, nx), dtype=float)

    if nz > 1:
        if use_upwind:
            t_z, g_z = _phase_face_ops(
                t_gz,
                lw_ijk[:-1, :, :],
                lw_ijk[1:, :, :],
                lo_ijk[:-1, :, :],
                lo_ijk[1:, :, :],
                p_ijk[:-1, :, :],
                p_ijk[1:, :, :],
                z_ijk[:-1, :, :],
                z_ijk[1:, :, :],
                gravity,
                rho_w,
                rho_o,
            )
        else:
            t_z = t_gz * 0.5 * (m_ijk[:-1, :, :] + m_ijk[1:, :, :])
            g_z = np.zeros_like(t_z)
    else:
        t_z = np.zeros((0, ny, nx), dtype=float)
        g_z = np.zeros((0, ny, nx), dtype=float)
    return t_x, t_y, t_z, g_x, g_y, g_z


def _face_pairs(t: NDArray[np.float64], a: NDArray[np.int64], b: NDArray[np.int64]) -> tuple[NDArray, NDArray, NDArray]:
    t = t.ravel()
    a = a.ravel()
    b = b.ravel()
    keep = t > 0.0
    t, a, b = t[keep], a[keep], b[keep]
    rows = np.concatenate([a, a, b, b])
    cols = np.concatenate([a, b, b, a])
    data = np.concatenate([t, -t, t, -t])
    return rows, cols, data


def _add_gravity_rhs(
    rhs: NDArray[np.float64],
    gflux: NDArray[np.float64],
    a: NDArray[np.int64],
    b: NDArray[np.int64],
) -> None:
    gf = np.asarray(gflux, dtype=float).ravel()
    aa = a.ravel()
    bb = b.ravel()
    rhs[aa] -= gf
    rhs[bb] += gf


def _add_face_dirichlet(
    grid: CartesianGrid,
    k: NDArray[np.float64],
    mobility: NDArray[np.float64],
    side: str,
    pressure: float,
    rhs: NDArray[np.float64],
    *,
    kz: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    k_ijk = grid.reshape_ijk(k)
    kz_ijk = k_ijk if kz is None else grid.reshape_ijk(kz)
    m_ijk = grid.reshape_ijk(mobility)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if side == "left":
        t = k_ijk[:, :, 0] * m_ijk[:, :, 0] * (grid.dz[:, None] * grid.dy[None, :]) / (0.5 * grid.dx[0])
        kk, jj = np.meshgrid(np.arange(nz), np.arange(ny), indexing="ij")
        cells = (kk * ny * nx + jj * nx).astype(np.int64).ravel()
        t = t.ravel()
    elif side == "right":
        t = k_ijk[:, :, -1] * m_ijk[:, :, -1] * (grid.dz[:, None] * grid.dy[None, :]) / (0.5 * grid.dx[-1])
        kk, jj = np.meshgrid(np.arange(nz), np.arange(ny), indexing="ij")
        cells = (kk * ny * nx + jj * nx + (nx - 1)).astype(np.int64).ravel()
        t = t.ravel()
    elif side == "front":
        t = k_ijk[:, 0, :] * m_ijk[:, 0, :] * (grid.dz[:, None] * grid.dx[None, :]) / (0.5 * grid.dy[0])
        kk, ii = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
        cells = (kk * ny * nx + ii).astype(np.int64).ravel()
        t = t.ravel()
    elif side == "back":
        t = k_ijk[:, -1, :] * m_ijk[:, -1, :] * (grid.dz[:, None] * grid.dx[None, :]) / (0.5 * grid.dy[-1])
        kk, ii = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
        cells = (kk * ny * nx + (ny - 1) * nx + ii).astype(np.int64).ravel()
        t = t.ravel()
    elif side == "bottom":
        t = kz_ijk[0, :, :] * m_ijk[0, :, :] * (grid.dy[:, None] * grid.dx[None, :]) / (0.5 * grid.dz[0])
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        cells = (jj * nx + ii).astype(np.int64).ravel()
        t = t.ravel()
    elif side == "top":
        t = kz_ijk[-1, :, :] * m_ijk[-1, :, :] * (grid.dy[:, None] * grid.dx[None, :]) / (0.5 * grid.dz[-1])
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        cells = ((nz - 1) * ny * nx + jj * nx + ii).astype(np.int64).ravel()
        t = t.ravel()
    else:
        raise ValueError(f"unknown boundary side {side}")
    rhs[cells] += t * float(pressure)
    return cells, t


def assemble_pressure(
    grid: CartesianGrid,
    k: NDArray[np.float64],
    mobility: NDArray[np.float64],
    *,
    cell_dirichlet: dict[int, float] | None = None,
    face_dirichlet: dict[str, float] | None = None,
    cell_rate: NDArray[np.float64] | None = None,
    well_index: dict[int, tuple[float, float]] | None = None,
    storage: NDArray[np.float64] | None = None,
    pressure_prev: NDArray[np.float64] | None = None,
    kz: NDArray[np.float64] | None = None,
    ky: NDArray[np.float64] | None = None,
    mult_x: NDArray[np.float64] | None = None,
    mult_y: NDArray[np.float64] | None = None,
    mult_z: NDArray[np.float64] | None = None,
    lw: NDArray[np.float64] | None = None,
    lo: NDArray[np.float64] | None = None,
    gravity: float = 0.0,
    rho_w: float = 1000.0,
    rho_o: float = 800.0,
) -> TpfaSystem:
    """Assemble ∇·(λ K ∇Φ) = q + s (p − p_prev)."""
    n = grid.n_cells
    k = as_cell_field(k, n, "k")
    mobility = as_cell_field(mobility, n, "mobility")
    t_x, t_y, t_z, g_x, g_y, g_z = interior_transmissibility(
        grid,
        k,
        mobility,
        kz=kz,
        ky=ky,
        mult_x=mult_x,
        mult_y=mult_y,
        mult_z=mult_z,
        lw=lw,
        lo=lo,
        pressure=pressure_prev,
        gravity=gravity,
        rho_w=rho_w,
        rho_o=rho_o,
    )
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    chunks_r: list[NDArray] = []
    chunks_c: list[NDArray] = []
    chunks_d: list[NDArray] = []
    rhs = np.zeros(n, dtype=float)

    if nx > 1:
        kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij")
        a = (kk * ny * nx + jj * nx + ii).astype(np.int64)
        r, c, d = _face_pairs(t_x, a, a + 1)
        chunks_r.append(r)
        chunks_c.append(c)
        chunks_d.append(d)
        _add_gravity_rhs(rhs, g_x, a, a + 1)
    if ny > 1:
        kk, jj, ii = np.meshgrid(np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij")
        a = (kk * ny * nx + jj * nx + ii).astype(np.int64)
        r, c, d = _face_pairs(t_y, a, a + nx)
        chunks_r.append(r)
        chunks_c.append(c)
        chunks_d.append(d)
        _add_gravity_rhs(rhs, g_y, a, a + nx)
    if nz > 1:
        kk, jj, ii = np.meshgrid(np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij")
        a = (kk * ny * nx + jj * nx + ii).astype(np.int64)
        r, c, d = _face_pairs(t_z, a, a + ny * nx)
        chunks_r.append(r)
        chunks_c.append(c)
        chunks_d.append(d)
        _add_gravity_rhs(rhs, g_z, a, a + ny * nx)

    if face_dirichlet:
        for side, p_bc in face_dirichlet.items():
            cells, t = _add_face_dirichlet(grid, k, mobility, side, float(p_bc), rhs, kz=kz)
            chunks_r.append(cells)
            chunks_c.append(cells)
            chunks_d.append(t)

    if cell_rate is not None:
        rhs += np.asarray(cell_rate, dtype=float).ravel()

    if well_index:
        cells = np.fromiter(well_index.keys(), dtype=np.int64, count=len(well_index))
        wis = np.fromiter((well_index[int(c)][0] for c in cells), dtype=float, count=cells.size)
        pbhp = np.fromiter((well_index[int(c)][1] for c in cells), dtype=float, count=cells.size)
        ok = wis > 0.0
        if np.any(ok):
            cc = cells[ok]
            ww = wis[ok]
            chunks_r.append(cc)
            chunks_c.append(cc)
            chunks_d.append(ww)
            rhs[cc] += ww * pbhp[ok]

    if storage is not None and pressure_prev is not None:
        s = as_cell_field(storage, n, "storage")
        p0 = as_cell_field(pressure_prev, n, "pressure_prev")
        active = s > 0.0
        if np.any(active):
            idx = np.nonzero(active)[0].astype(np.int64)
            chunks_r.append(idx)
            chunks_c.append(idx)
            chunks_d.append(s[active])
            rhs[active] += s[active] * p0[active]

    if chunks_r:
        matrix = sparse.csr_matrix(
            (np.concatenate(chunks_d), (np.concatenate(chunks_r), np.concatenate(chunks_c))),
            shape=(n, n),
        )
    else:
        matrix = sparse.csr_matrix((n, n))

    if cell_dirichlet:
        matrix = matrix.tolil()
        for cell, p_bc in cell_dirichlet.items():
            c = int(cell)
            matrix.rows[c] = []
            matrix.data[c] = []
            matrix[c, c] = 1.0
            rhs[c] = float(p_bc)
        matrix = matrix.tocsr()

    return TpfaSystem(matrix=matrix, rhs=rhs, t_x=t_x, t_y=t_y, t_z=t_z, g_x=g_x, g_y=g_y, g_z=g_z)


def solve_pressure(system: TpfaSystem) -> NDArray[np.float64]:
    try:
        p = spsolve(system.matrix.tocsr(), system.rhs)
    except Exception as exc:
        raise LinearSolveFailure(str(exc)) from exc
    p = np.asarray(p, dtype=float).ravel()
    if not np.all(np.isfinite(p)):
        raise LinearSolveFailure("pressure solution contains NaN or Inf")
    return p


def face_fluxes(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    t_x: NDArray[np.float64],
    t_y: NDArray[np.float64],
    t_z: NDArray[np.float64],
    *,
    face_dirichlet: dict[str, float] | None = None,
    k: NDArray[np.float64] | None = None,
    mobility: NDArray[np.float64] | None = None,
    g_x: NDArray[np.float64] | None = None,
    g_y: NDArray[np.float64] | None = None,
    g_z: NDArray[np.float64] | None = None,
    kz: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Total volumetric face fluxes, including Dirichlet boundary faces.

    ``fx[k,j,i]`` is +x flow through the face at the left of cell ``(i,j,k)``.
    """
    p = grid.reshape_ijk(pressure)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fx = np.zeros((nz, ny, nx + 1), dtype=float)
    fy = np.zeros((nz, ny + 1, nx), dtype=float)
    fz = np.zeros((nz + 1, ny, nx), dtype=float)
    if nx > 1:
        fx[:, :, 1:-1] = t_x * (p[:, :, :-1] - p[:, :, 1:])
        if g_x is not None:
            fx[:, :, 1:-1] = fx[:, :, 1:-1] + g_x
    if ny > 1:
        fy[:, 1:-1, :] = t_y * (p[:, :-1, :] - p[:, 1:, :])
        if g_y is not None:
            fy[:, 1:-1, :] = fy[:, 1:-1, :] + g_y
    if nz > 1:
        fz[1:-1, :, :] = t_z * (p[:-1, :, :] - p[1:, :, :])
        if g_z is not None:
            fz[1:-1, :, :] = fz[1:-1, :, :] + g_z

    if face_dirichlet and k is not None and mobility is not None:
        k_ijk = grid.reshape_ijk(k)
        kz_ijk = k_ijk if kz is None else grid.reshape_ijk(kz)
        m_ijk = grid.reshape_ijk(mobility)
        if "left" in face_dirichlet:
            t = k_ijk[:, :, 0] * m_ijk[:, :, 0] * (grid.dz[:, None] * grid.dy[None, :]) / (0.5 * grid.dx[0])
            fx[:, :, 0] = t * (face_dirichlet["left"] - p[:, :, 0])
        if "right" in face_dirichlet:
            t = k_ijk[:, :, -1] * m_ijk[:, :, -1] * (grid.dz[:, None] * grid.dy[None, :]) / (0.5 * grid.dx[-1])
            fx[:, :, -1] = t * (p[:, :, -1] - face_dirichlet["right"])
        if "front" in face_dirichlet:
            t = k_ijk[:, 0, :] * m_ijk[:, 0, :] * (grid.dz[:, None] * grid.dx[None, :]) / (0.5 * grid.dy[0])
            fy[:, 0, :] = t * (face_dirichlet["front"] - p[:, 0, :])
        if "back" in face_dirichlet:
            t = k_ijk[:, -1, :] * m_ijk[:, -1, :] * (grid.dz[:, None] * grid.dx[None, :]) / (0.5 * grid.dy[-1])
            fy[:, -1, :] = t * (p[:, -1, :] - face_dirichlet["back"])
        if "bottom" in face_dirichlet:
            t = kz_ijk[0, :, :] * m_ijk[0, :, :] * (grid.dy[:, None] * grid.dx[None, :]) / (0.5 * grid.dz[0])
            fz[0, :, :] = t * (face_dirichlet["bottom"] - p[0, :, :])
        if "top" in face_dirichlet:
            t = kz_ijk[-1, :, :] * m_ijk[-1, :, :] * (grid.dy[:, None] * grid.dx[None, :]) / (0.5 * grid.dz[-1])
            fz[-1, :, :] = t * (p[-1, :, :] - face_dirichlet["top"])
    return fx, fy, fz
