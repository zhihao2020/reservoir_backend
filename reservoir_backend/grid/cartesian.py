"""Cartesian cell-centered grid. Public fields are flat ``(n_cells,)``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import GridError


def _positive_spacing(name: str, value: float | NDArray[np.float64], count: int) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(count, float(arr), dtype=float)
    elif arr.ndim == 1:
        if arr.size != count:
            raise GridError(f"{name} length {arr.size} != {count}")
        arr = arr.astype(float, copy=True)
    else:
        raise GridError(f"{name} must be a scalar or 1-D array")
    if not np.isfinite(arr).all() or (arr <= 0.0).any():
        raise GridError(f"{name} must contain positive finite values")
    return arr


@dataclass(frozen=True)
class CartesianGrid:
    """Structured orthogonal grid.

    Flattened index: ``k * ny * nx + j * nx + i``. Visualization may reshape
    to ``(nz, ny, nx)`` via :meth:`reshape_ijk`.
    """

    nx: int
    ny: int
    nz: int
    dx: NDArray[np.float64]
    dy: NDArray[np.float64]
    dz: NDArray[np.float64]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    active: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise GridError(f"{name} must be a positive integer")
        object.__setattr__(self, "dx", _positive_spacing("dx", self.dx, self.nx))
        object.__setattr__(self, "dy", _positive_spacing("dy", self.dy, self.ny))
        object.__setattr__(self, "dz", _positive_spacing("dz", self.dz, self.nz))
        if self.active is not None:
            act = np.asarray(self.active, dtype=bool).ravel()
            if act.size != self.nx * self.ny * self.nz:
                raise GridError(
                    "active length " + str(int(act.size)) + " != " + str(self.nx * self.ny * self.nz)
                )
            object.__setattr__(self, "active", act)

    @classmethod
    def uniform(
        cls,
        size_m: tuple[float, float, float],
        spacing_m: float | tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> CartesianGrid:
        lx, ly, lz = (float(size_m[0]), float(size_m[1]), float(size_m[2]))
        if isinstance(spacing_m, (int, float)):
            hx = hy = hz = float(spacing_m)
        else:
            hx, hy, hz = (float(spacing_m[0]), float(spacing_m[1]), float(spacing_m[2]))
        if min(lx, ly, lz, hx, hy, hz) <= 0.0:
            raise GridError("size and spacing must be positive")
        nx = max(1, int(round(lx / hx)))
        ny = max(1, int(round(ly / hy)))
        nz = max(1, int(round(lz / hz)))
        return cls(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=np.full(nx, lx / nx),
            dy=np.full(ny, ly / ny),
            dz=np.full(nz, lz / nz),
            origin=origin,
        )

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def shape_ijk(self) -> tuple[int, int, int]:
        """Visualization layout ``(nz, ny, nx)`` only."""
        return (self.nz, self.ny, self.nx)

    def index(self, i: int, j: int, k: int) -> int:
        if not (0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz):
            raise GridError(f"ijk {(i, j, k)} out of bounds")
        return k * self.ny * self.nx + j * self.nx + i

    def ijk(self, cell: int) -> tuple[int, int, int]:
        if not isinstance(cell, (int, np.integer)) or not 0 <= int(cell) < self.n_cells:
            raise GridError(f"cell {cell!r} out of range")
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
        vol_ijk = self.dz[:, None, None] * self.dy[None, :, None] * self.dx[None, None, :]
        vol = vol_ijk.ravel()
        if self.active is not None:
            vol = np.where(self.active, vol, 0.0)
        return vol

    def cell_centers(self) -> NDArray[np.float64]:
        cx = self.edge_x()[:-1] + 0.5 * self.dx
        cy = self.edge_y()[:-1] + 0.5 * self.dy
        cz = self.edge_z()[:-1] + 0.5 * self.dz
        zz, yy, xx = np.meshgrid(cz, cy, cx, indexing="ij")
        return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

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
        """Containing cell. Points on the max face map into the last cell."""
        ex, ey, ez = self.edge_x(), self.edge_y(), self.edge_z()
        if not (ex[0] <= x <= ex[-1] and ey[0] <= y <= ey[-1] and ez[0] <= z <= ez[-1]):
            raise GridError(f"point {(x, y, z)} lies outside the grid")
        i = int(np.searchsorted(ex, x, side="right") - 1)
        j = int(np.searchsorted(ey, y, side="right") - 1)
        k = int(np.searchsorted(ez, z, side="right") - 1)
        i = min(max(i, 0), self.nx - 1)
        j = min(max(j, 0), self.ny - 1)
        k = min(max(k, 0), self.nz - 1)
        return self.index(i, j, k)

    def reshape_ijk(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(field, dtype=float)
        if arr.size != self.n_cells:
            raise GridError(f"field size {arr.size} != n_cells {self.n_cells}")
        return arr.reshape(self.shape_ijk)

    def flatten(self, field_ijk: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(field_ijk, dtype=float)
        if arr.shape != self.shape_ijk:
            raise GridError(f"ijk field shape {arr.shape} != {self.shape_ijk}")
        return arr.ravel()

    def face_area_x(self) -> NDArray[np.float64]:
        """Areas of x-faces including domain boundaries, shape ``(nz, ny, nx+1)``."""
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
