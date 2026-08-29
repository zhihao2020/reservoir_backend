"""Structured corner-point grid (Eclipse COORD/ZCORN) with CPG-TPFA metrics.

COORD / ZCORN order (Eclipse, 0-based ``(i, j, k)`` with ``i`` fastest)::

    COORD length = (nx+1) * (ny+1) * 6
    pillar (i, j) at offset (j * (nx+1) + i) * 6
        = (Xtop, Ytop, Ztop, Xbtm, Ybtm, Zbtm)

    ZCORN length = 8 * nx * ny * nz
    reshaped as (2*nz, 2*ny, 2*nx) with k-plane, then j, then i.
    Cell (i, j, k) corners (local 0..7)::

        0 (i,   j,   k  )  1 (i+1, j,   k  )
        2 (i,   j+1, k  )  3 (i+1, j+1, k  )
        4 (i,   j,   k+1)  5 (i+1, j,   k+1)
        6 (i,   j+1, k+1)  7 (i+1, j+1, k+1)

    z of corner t at ZCORN[2*k + [t//4], 2*j + [(t//2)%2], 2*i + (t%2)]
    XY from the matching pillar, linearly interpolated in Z.

z increases with k, same frame as :class:`~reservoir_backend.grid.cartesian.CartesianGrid`.
Cell volume is the closed-polyhedron formula from the six bilinear faces.
Two-point half-transmissibility is the OPM/MRST form
``T_half = k * (A·A) / (A·(F-C))``, then ``T = 1 / (1/T_L + 1/T_R)``.
Zero-volume / non-finite cells are marked inactive (T half-factors 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import GridError


# Outward quads on the local 0..7 corners (unit-cube check: V = 1, n = ±e).
_FACE_OUT = (
    (0, 4, 6, 2),  # -I
    (1, 3, 7, 5),  # +I
    (0, 1, 5, 4),  # -J
    (2, 6, 7, 3),  # +J
    (0, 2, 3, 1),  # -K
    (4, 5, 7, 6),  # +K
)
_PLUS_I, _PLUS_J, _PLUS_K = 1, 3, 5
_VOL_EPS = 1.0e-30
_DOT_EPS = 1.0e-30


def _as_1d(name: str, raw: object, n: int) -> NDArray[np.float64]:
    arr = np.asarray(raw, dtype=float).ravel()
    if arr.size != n:
        raise GridError(name + " length " + str(int(arr.size)) + " != " + str(n))
    if not np.isfinite(arr).all():
        raise GridError(name + " must be finite")
    return np.ascontiguousarray(arr, dtype=float)


def _quad_area_normal(
    corners: NDArray[np.float64], a: int, b: int, c: int, d: int
) -> NDArray[np.float64]:
    """Bilinear-face area vector from the two diagonals (OPM-style)."""
    return 0.5 * np.cross(corners[:, c] - corners[:, a], corners[:, d] - corners[:, b])


def _hex_volumes(corners: NDArray[np.float64]) -> NDArray[np.float64]:
    vol = np.zeros(corners.shape[0], dtype=float)
    for a, b, c, d in _FACE_OUT:
        nrm = _quad_area_normal(corners, a, b, c, d)
        centroid = 0.25 * (
            corners[:, a] + corners[:, b] + corners[:, c] + corners[:, d]
        )
        vol += np.einsum("ij,ij->i", centroid, nrm)
    return vol / 3.0


def _half_g(
    corners: NDArray[np.float64],
    centers: NDArray[np.float64],
    face_ids: tuple[int, int, int, int],
) -> NDArray[np.float64]:
    a, b, c, d = face_ids
    area = _quad_area_normal(corners, a, b, c, d)
    face_c = 0.25 * (corners[:, a] + corners[:, b] + corners[:, c] + corners[:, d])
    dot = np.einsum("ij,ij->i", area, face_c - centers)
    aa = np.einsum("ij,ij->i", area, area)
    g = np.zeros(corners.shape[0], dtype=float)
    ok = dot > _DOT_EPS
    g[ok] = aa[ok] / dot[ok]
    g[~np.isfinite(g)] = 0.0
    return g


def _pillar_xyz(
    coord: NDArray[np.float64],
    pi: NDArray[np.int64],
    pj: NDArray[np.int64],
    z: NDArray[np.float64],
    npx: int,
) -> NDArray[np.float64]:
    """Interpolate pillar (pi, pj) to depth/height z. coord is (n_pillars, 6)."""
    p = coord[pj * npx + pi]
    zt = p[:, 2]
    zb = p[:, 5]
    den = zb - zt
    t = np.zeros(z.shape, dtype=float)
    good = np.abs(den) > 1.0e-30
    t[good] = (z[good] - zt[good]) / den[good]
    xyz = np.empty(z.shape + (3,), dtype=float)
    xyz[:, 0] = p[:, 0] + t * (p[:, 3] - p[:, 0])
    xyz[:, 1] = p[:, 1] + t * (p[:, 4] - p[:, 1])
    xyz[:, 2] = z
    return xyz


def coord_zcorn_from_cartesian(grid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Eclipse-order COORD/ZCORN for an orthogonal :class:`CartesianGrid`."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    ex, ey, ez = grid.edge_x(), grid.edge_y(), grid.edge_z()
    npx, npy = nx + 1, ny + 1
    coord = np.empty((npy, npx, 6), dtype=float)
    xx, yy = np.meshgrid(ex, ey, indexing="xy")
    coord[:, :, 0] = xx
    coord[:, :, 1] = yy
    coord[:, :, 2] = ez[0]
    coord[:, :, 3] = xx
    coord[:, :, 4] = yy
    coord[:, :, 5] = ez[-1]
    zcorn = np.empty((2 * nz, 2 * ny, 2 * nx), dtype=float)
    for k in range(nz):
        zcorn[2 * k, :, :] = ez[k]
        zcorn[2 * k + 1, :, :] = ez[k + 1]
    return coord.reshape(-1), zcorn.reshape(-1)


def _mean_axis_spacing(extent: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """extent is (nz, ny, nx); reduce over the other two axes."""
    axes = tuple(a for a in (0, 1, 2) if a != axis)
    out = np.mean(extent, axis=axes)
    return np.maximum(np.asarray(out, dtype=float), 1.0e-30)


@dataclass(frozen=True)
class CpgHalfGeom:
    """Geometric half-transmissibility factors (no permeability).

    ``g_*_left[k,j,i]`` is ``(A·A)/(A·(F-C))`` on the minus-side cell of that face.
    """

    g_x_left: NDArray[np.float64]
    g_x_right: NDArray[np.float64]
    g_y_left: NDArray[np.float64]
    g_y_right: NDArray[np.float64]
    g_z_left: NDArray[np.float64]
    g_z_right: NDArray[np.float64]


@dataclass(frozen=True)
class CornerPointGrid:
    """Structured IJK corner-point grid. Public fields match CartesianGrid."""

    nx: int
    ny: int
    nz: int
    dx: NDArray[np.float64]
    dy: NDArray[np.float64]
    dz: NDArray[np.float64]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    corners: NDArray[np.float64] = None  # type: ignore[assignment]
    active: NDArray[np.bool_] = None  # type: ignore[assignment]
    cpg_half_geom: CpgHalfGeom = None  # type: ignore[assignment]
    _volumes: NDArray[np.float64] = None  # type: ignore[assignment]
    _centers: NDArray[np.float64] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise GridError(name + " must be a positive integer")
        n = self.nx * self.ny * self.nz
        if self.corners is None or self._volumes is None or self._centers is None:
            raise GridError("use CornerPointGrid.from_coord_zcorn")
        if self.corners.shape != (n, 8, 3):
            raise GridError("corners must have shape (n_cells, 8, 3)")
        object.__setattr__(self, "dx", np.asarray(self.dx, dtype=float).ravel())
        object.__setattr__(self, "dy", np.asarray(self.dy, dtype=float).ravel())
        object.__setattr__(self, "dz", np.asarray(self.dz, dtype=float).ravel())
        if self.dx.size != self.nx or self.dy.size != self.ny or self.dz.size != self.nz:
            raise GridError("dx/dy/dz lengths must match nx/ny/nz")

    @classmethod
    def from_coord_zcorn(
        cls,
        nx: int,
        ny: int,
        nz: int,
        coord,
        zcorn,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        actnum=None,
    ) -> "CornerPointGrid":
        if min(int(nx), int(ny), int(nz)) <= 0:
            raise GridError("nx, ny, nz must be positive integers")
        nx, ny, nz = int(nx), int(ny), int(nz)
        npx, npy = nx + 1, ny + 1
        coord_a = _as_1d("coord", coord, npx * npy * 6).reshape(npx * npy, 6)
        zc = _as_1d("zcorn", zcorn, 8 * nx * ny * nz).reshape(2 * nz, 2 * ny, 2 * nx)

        kk, jj, ii = np.meshgrid(
            np.arange(nz, dtype=np.int64),
            np.arange(ny, dtype=np.int64),
            np.arange(nx, dtype=np.int64),
            indexing="ij",
        )
        z8 = np.empty((nz, ny, nx, 8), dtype=float)
        z8[..., 0] = zc[2 * kk, 2 * jj, 2 * ii]
        z8[..., 1] = zc[2 * kk, 2 * jj, 2 * ii + 1]
        z8[..., 2] = zc[2 * kk, 2 * jj + 1, 2 * ii]
        z8[..., 3] = zc[2 * kk, 2 * jj + 1, 2 * ii + 1]
        z8[..., 4] = zc[2 * kk + 1, 2 * jj, 2 * ii]
        z8[..., 5] = zc[2 * kk + 1, 2 * jj, 2 * ii + 1]
        z8[..., 6] = zc[2 * kk + 1, 2 * jj + 1, 2 * ii]
        z8[..., 7] = zc[2 * kk + 1, 2 * jj + 1, 2 * ii + 1]

        pi = np.stack(
            [ii, ii + 1, ii, ii + 1, ii, ii + 1, ii, ii + 1], axis=-1
        ).astype(np.int64)
        pj = np.stack(
            [jj, jj, jj + 1, jj + 1, jj, jj, jj + 1, jj + 1], axis=-1
        ).astype(np.int64)

        n = nx * ny * nz
        corners = np.empty((n, 8, 3), dtype=float)
        flat_pi = pi.reshape(n, 8)
        flat_pj = pj.reshape(n, 8)
        flat_z = z8.reshape(n, 8)
        for t in range(8):
            corners[:, t, :] = _pillar_xyz(
                coord_a, flat_pi[:, t], flat_pj[:, t], flat_z[:, t], npx
            )

        volumes = np.abs(_hex_volumes(corners))
        active = np.isfinite(volumes) & (volumes > _VOL_EPS)
        if actnum is not None:
            act = np.asarray(actnum, dtype=float).ravel()
            if act.size != n:
                raise GridError("actnum length " + str(int(act.size)) + " != " + str(n))
            active = active & (act != 0.0)
        volumes = np.where(active, volumes, 0.0)
        centers = corners.mean(axis=1)
        centers[~active] = np.nan

        g_cell = np.zeros((n, 6), dtype=float)
        for f, ids in enumerate(_FACE_OUT):
            g_cell[:, f] = _half_g(corners, centers, ids)
        g_cell[~active, :] = 0.0
        g_ijk = g_cell.reshape(nz, ny, nx, 6)

        if nx > 1:
            g_x_left = g_ijk[:, :, :-1, _PLUS_I].copy()
            g_x_right = g_ijk[:, :, 1:, 0].copy()
        else:
            g_x_left = np.zeros((nz, ny, 0), dtype=float)
            g_x_right = np.zeros((nz, ny, 0), dtype=float)
        if ny > 1:
            g_y_left = g_ijk[:, :-1, :, _PLUS_J].copy()
            g_y_right = g_ijk[:, 1:, :, 2].copy()
        else:
            g_y_left = np.zeros((nz, 0, nx), dtype=float)
            g_y_right = np.zeros((nz, 0, nx), dtype=float)
        if nz > 1:
            g_z_left = g_ijk[:-1, :, :, _PLUS_K].copy()
            g_z_right = g_ijk[1:, :, :, 4].copy()
        else:
            g_z_left = np.zeros((0, ny, nx), dtype=float)
            g_z_right = np.zeros((0, ny, nx), dtype=float)

        half = CpgHalfGeom(
            g_x_left=g_x_left,
            g_x_right=g_x_right,
            g_y_left=g_y_left,
            g_y_right=g_y_right,
            g_z_left=g_z_left,
            g_z_right=g_z_right,
        )

        ext = corners.reshape(nz, ny, nx, 8, 3)
        dx_cell = ext[..., 0].max(axis=-1) - ext[..., 0].min(axis=-1)
        dy_cell = ext[..., 1].max(axis=-1) - ext[..., 1].min(axis=-1)
        dz_cell = ext[..., 2].max(axis=-1) - ext[..., 2].min(axis=-1)
        dx = _mean_axis_spacing(dx_cell, 2)
        dy = _mean_axis_spacing(dy_cell, 1)
        dz = _mean_axis_spacing(dz_cell, 0)
        xmin = float(np.nanmin(corners[:, :, 0])) if n else 0.0
        ymin = float(np.nanmin(corners[:, :, 1])) if n else 0.0
        zmin = float(np.nanmin(corners[:, :, 2])) if n else 0.0
        stored_origin = (
            float(origin[0]) if origin is not None else xmin,
            float(origin[1]) if origin is not None else ymin,
            float(origin[2]) if origin is not None else zmin,
        )
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            origin=stored_origin,
            corners=corners,
            active=active,
            cpg_half_geom=half,
            _volumes=volumes,
            _centers=centers,
        )

    @classmethod
    def from_cartesian(cls, grid) -> "CornerPointGrid":
        coord, zcorn = coord_zcorn_from_cartesian(grid)
        return cls.from_coord_zcorn(
            grid.nx, grid.ny, grid.nz, coord, zcorn, origin=grid.origin
        )

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def shape_ijk(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    def index(self, i: int, j: int, k: int) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise GridError("ijk " + str((i, j, k)) + " out of bounds")
        return k * self.ny * self.nx + j * self.nx + i

    def ijk(self, cell: int) -> tuple[int, int, int]:
        if not isinstance(cell, (int, np.integer)) or not 0 <= int(cell) < self.n_cells:
            raise GridError("cell " + repr(cell) + " out of range")
        cell = int(cell)
        k, rem = divmod(cell, self.nx * self.ny)
        j, i = divmod(rem, self.nx)
        return i, j, k

    def edge_x(self) -> NDArray[np.float64]:
        return self.origin[0] + np.concatenate(([0.0], np.cumsum(self.dx)))

    def edge_y(self) -> NDArray[np.float64]:
        return self.origin[1] + np.concatenate(([0.0], np.cumsum(self.dy)))

    def edge_z(self) -> NDArray[np.float64]:
        return self.origin[2] + np.concatenate(([0.0], np.cumsum(self.dz)))

    def size_m(self) -> tuple[float, float, float]:
        return (float(np.sum(self.dx)), float(np.sum(self.dy)), float(np.sum(self.dz)))

    def total_volume(self) -> float:
        return float(np.sum(self.cell_volumes()))

    def cell_volumes(self) -> NDArray[np.float64]:
        return self._volumes.copy()

    def cell_centers(self) -> NDArray[np.float64]:
        return self._centers.copy()

    def center_axis(self, axis: str) -> NDArray[np.float64]:
        if axis == "x":
            return self.edge_x()[:-1] + 0.5 * self.dx
        if axis == "y":
            return self.edge_y()[:-1] + 0.5 * self.dy
        if axis == "z":
            return self.edge_z()[:-1] + 0.5 * self.dz
        raise GridError("axis must be x, y, or z")

    def neighbors(self, cell: int) -> list[int]:
        i, j, k = self.ijk(cell)
        out: list[int] = []
        for di, dj, dk in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            ii, jj, kk = i + di, j + dj, k + dk
            if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                out.append(self.index(ii, jj, kk))
        return out

    def locate_cell(self, x: float, y: float, z: float) -> int:
        pts = self._centers
        active = self.active
        if not np.any(active):
            raise GridError("no active cells")
        d = pts[active] - np.array([x, y, z], dtype=float)
        idx = np.nonzero(active)[0][int(np.argmin(np.einsum("ij,ij->i", d, d)))]
        return int(idx)

    def reshape_ijk(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(field, dtype=float)
        if arr.size != self.n_cells:
            raise GridError(
                "field size " + str(arr.size) + " != n_cells " + str(self.n_cells)
            )
        return arr.reshape(self.shape_ijk)

    def flatten(self, field_ijk: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(field_ijk, dtype=float)
        if arr.shape != self.shape_ijk:
            raise GridError(
                "ijk field shape " + str(arr.shape) + " != " + str(self.shape_ijk)
            )
        return arr.ravel()

    def face_area_x(self) -> NDArray[np.float64]:
        base = self.dz[:, None] * self.dy[None, :]
        return np.broadcast_to(base[:, :, None], (self.nz, self.ny, self.nx + 1)).copy()

    def face_area_y(self) -> NDArray[np.float64]:
        base = self.dz[:, None] * self.dx[None, :]
        return np.broadcast_to(base[:, None, :], (self.nz, self.ny + 1, self.nx)).copy()

    def face_area_z(self) -> NDArray[np.float64]:
        base = self.dy[:, None] * self.dx[None, :]
        return np.broadcast_to(base[None, :, :], (self.nz + 1, self.ny, self.nx)).copy()

    def center_distance_x(self) -> NDArray[np.float64]:
        if self.nx < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self.dx[:-1] + self.dx[1:])

    def center_distance_y(self) -> NDArray[np.float64]:
        if self.ny < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self.dy[:-1] + self.dy[1:])

    def center_distance_z(self) -> NDArray[np.float64]:
        if self.nz < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self.dz[:-1] + self.dz[1:])
