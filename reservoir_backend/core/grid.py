"""Structured orthogonal mesh utilities (cell-centered Cartesian / tensor spacing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridIndexError, InvalidPhysicalValueError

SpacingInput = float | Sequence[float] | ArrayLike


def _as_positive_spacing(name: str, value: SpacingInput, count: int) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(count, float(arr), dtype=float)
    elif arr.ndim == 1:
        if arr.size != count:
            raise InvalidPhysicalValueError(
                f"{name} length {arr.size} does not match cell count {count}"
            )
        arr = arr.astype(float, copy=True)
    else:
        raise InvalidPhysicalValueError(f"{name} must be a scalar or 1-D array of length {count}")

    if not np.isfinite(arr).all() or (arr <= 0.0).any():
        raise InvalidPhysicalValueError(f"{name} must contain positive finite values")
    return arr


@dataclass(frozen=True)
class Grid3D:
    """Structured orthogonal grid with optional non-uniform axis spacing.

    Field arrays use the shape `(nz, ny, nx)`. Flattened indices are computed as
    `k * ny * nx + j * nx + i`.

    Spacing may be a positive scalar (uniform) or a positive 1-D vector per axis
    (`spacing_i` length `nx`, etc.). Public attributes `dx`, `dy`, `dz` remain
    available for uniform grids as scalars for backward compatibility; for
    non-uniform axes prefer `spacing_i` / `spacing_j` / `spacing_k`.
    """

    nx: int
    ny: int
    nz: int
    dx: float | Sequence[float] | ArrayLike = 1.0
    dy: float | Sequence[float] | ArrayLike = 1.0
    dz: float | Sequence[float] | ArrayLike = 1.0
    active_mask: ArrayLike | None = field(default=None, compare=False, repr=False)
    _active_mask: NDArray[np.bool_] = field(init=False, repr=False, compare=False)
    _spacing_i: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _spacing_j: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _spacing_k: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _dx_public: float | NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _dy_public: float | NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _dz_public: float | NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise InvalidPhysicalValueError(f"{name} must be a positive integer")

        spacing_i = _as_positive_spacing("dx", self.dx, self.nx)
        spacing_j = _as_positive_spacing("dy", self.dy, self.ny)
        spacing_k = _as_positive_spacing("dz", self.dz, self.nz)
        object.__setattr__(self, "_spacing_i", spacing_i)
        object.__setattr__(self, "_spacing_j", spacing_j)
        object.__setattr__(self, "_spacing_k", spacing_k)

        # Backward-compatible public scalars when an axis is uniform.
        object.__setattr__(
            self,
            "_dx_public",
            float(spacing_i[0]) if self._axis_uniform(spacing_i) else spacing_i.copy(),
        )
        object.__setattr__(
            self,
            "_dy_public",
            float(spacing_j[0]) if self._axis_uniform(spacing_j) else spacing_j.copy(),
        )
        object.__setattr__(
            self,
            "_dz_public",
            float(spacing_k[0]) if self._axis_uniform(spacing_k) else spacing_k.copy(),
        )
        object.__setattr__(self, "dx", self._dx_public)
        object.__setattr__(self, "dy", self._dy_public)
        object.__setattr__(self, "dz", self._dz_public)

        if self.active_mask is None:
            mask = np.ones(self.shape, dtype=bool)
        else:
            mask = np.asarray(self.active_mask, dtype=bool)
            if mask.shape != self.shape:
                raise GridIndexError(
                    f"active_mask shape {mask.shape} does not match grid shape {self.shape}"
                )
            mask = mask.copy()

        object.__setattr__(self, "_active_mask", mask)

    @staticmethod
    def _axis_uniform(spacing: NDArray[np.float64]) -> bool:
        return bool(np.allclose(spacing, spacing[0]))

    @property
    def spacing_i(self) -> NDArray[np.float64]:
        """Cell widths along the i (x) axis, shape `(nx,)`."""
        return self._spacing_i.copy()

    @property
    def spacing_j(self) -> NDArray[np.float64]:
        """Cell widths along the j (y) axis, shape `(ny,)`."""
        return self._spacing_j.copy()

    @property
    def spacing_k(self) -> NDArray[np.float64]:
        """Cell widths along the k (z) axis, shape `(nz,)`."""
        return self._spacing_k.copy()

    @property
    def is_uniform(self) -> bool:
        """True when all three axes use constant spacing."""
        return (
            self._axis_uniform(self._spacing_i)
            and self._axis_uniform(self._spacing_j)
            and self._axis_uniform(self._spacing_k)
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return field array shape as `(nz, ny, nx)`."""
        return (self.nz, self.ny, self.nx)

    @property
    def total_cells(self) -> int:
        """Return the total number of cells including inactive cells."""
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float | NDArray[np.float64]:
        """Return cell volume(s) in cubic meters.

        Uniform grids return a scalar; non-uniform grids return `(nz, ny, nx)`.
        """
        volumes = self.cell_volumes
        if self.is_uniform:
            return float(volumes.flat[0])
        return volumes

    @property
    def cell_volumes(self) -> NDArray[np.float64]:
        """Return per-cell volumes with shape `(nz, ny, nx)`."""
        # outer product: V[k,j,i] = di[i] * dj[j] * dk[k]
        return (
            self._spacing_k[:, None, None]
            * self._spacing_j[None, :, None]
            * self._spacing_i[None, None, :]
        )

    @property
    def active_mask_array(self) -> NDArray[np.bool_]:
        """Return a copy of the boolean active-cell mask."""
        return self._active_mask.copy()

    def edge_coordinates_i(self) -> NDArray[np.float64]:
        """Return i-direction node coordinates of length `nx + 1` starting at 0."""
        return np.concatenate(([0.0], np.cumsum(self._spacing_i)))

    def edge_coordinates_j(self) -> NDArray[np.float64]:
        """Return j-direction node coordinates of length `ny + 1` starting at 0."""
        return np.concatenate(([0.0], np.cumsum(self._spacing_j)))

    def edge_coordinates_k(self) -> NDArray[np.float64]:
        """Return k-direction node coordinates of length `nz + 1` starting at 0."""
        return np.concatenate(([0.0], np.cumsum(self._spacing_k)))

    def domain_extents(self) -> tuple[float, float, float]:
        """Return physical domain lengths `(Lx, Ly, Lz)`."""
        return (
            float(np.sum(self._spacing_i)),
            float(np.sum(self._spacing_j)),
            float(np.sum(self._spacing_k)),
        )

    def cell_centers(self) -> NDArray[np.float64]:
        """Return cell-center coordinates with shape `(nz, ny, nx, 3)`."""
        ci = self.edge_coordinates_i()[:-1] + 0.5 * self._spacing_i
        cj = self.edge_coordinates_j()[:-1] + 0.5 * self._spacing_j
        ck = self.edge_coordinates_k()[:-1] + 0.5 * self._spacing_k
        zz, yy, xx = np.meshgrid(ck, cj, ci, indexing="ij")
        return np.stack([xx, yy, zz], axis=-1)

    def center_distances_i(self) -> NDArray[np.float64]:
        """Distances between adjacent cell centers along i, shape `(nx - 1,)`."""
        if self.nx < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self._spacing_i[:-1] + self._spacing_i[1:])

    def center_distances_j(self) -> NDArray[np.float64]:
        """Distances between adjacent cell centers along j, shape `(ny - 1,)`."""
        if self.ny < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self._spacing_j[:-1] + self._spacing_j[1:])

    def center_distances_k(self) -> NDArray[np.float64]:
        """Distances between adjacent cell centers along k, shape `(nz - 1,)`."""
        if self.nz < 2:
            return np.zeros(0, dtype=float)
        return 0.5 * (self._spacing_k[:-1] + self._spacing_k[1:])

    def x_face_areas(self) -> NDArray[np.float64]:
        """Areas of internal x-faces, shape `(nz, ny, nx - 1)` (or empty)."""
        if self.nx < 2:
            return np.zeros((self.nz, self.ny, 0), dtype=float)
        # A[k, j, :] = spacing_j[j] * spacing_k[k]
        base = self._spacing_k[:, None, None] * self._spacing_j[None, :, None]
        return np.broadcast_to(base, (self.nz, self.ny, self.nx - 1)).copy()

    def y_face_areas(self) -> NDArray[np.float64]:
        """Areas of internal y-faces, shape `(nz, ny - 1, nx)`."""
        if self.ny < 2:
            return np.zeros((self.nz, 0, self.nx), dtype=float)
        # A[k, :, i] = spacing_i[i] * spacing_k[k]
        base = self._spacing_k[:, None, None] * self._spacing_i[None, None, :]
        return np.broadcast_to(base, (self.nz, self.ny - 1, self.nx)).copy()

    def z_face_areas(self) -> NDArray[np.float64]:
        """Areas of internal z-faces, shape `(nz - 1, ny, nx)`."""
        if self.nz < 2:
            return np.zeros((0, self.ny, self.nx), dtype=float)
        # A[:, j, i] = spacing_i[i] * spacing_j[j]
        base = self._spacing_j[None, :, None] * self._spacing_i[None, None, :]
        return np.broadcast_to(base, (self.nz - 1, self.ny, self.nx)).copy()

    def locate_cell(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Map a physical point to `(i, j, k)` using edge coordinates."""
        lx, ly, lz = self.domain_extents()
        if x < 0.0 or y < 0.0 or z < 0.0 or x >= lx or y >= ly or z >= lz:
            raise InvalidPhysicalValueError("point lies outside grid domain")
        i = int(np.searchsorted(self.edge_coordinates_i(), x, side="right") - 1)
        j = int(np.searchsorted(self.edge_coordinates_j(), y, side="right") - 1)
        k = int(np.searchsorted(self.edge_coordinates_k(), z, side="right") - 1)
        return (
            min(max(i, 0), self.nx - 1),
            min(max(j, 0), self.ny - 1),
            min(max(k, 0), self.nz - 1),
        )

    def index(self, i: int, j: int, k: int) -> int:
        """Convert `(i, j, k)` coordinates to a flattened cell index."""
        self._validate_ijk(i, j, k)
        return k * self.ny * self.nx + j * self.nx + i

    def ijk(self, index: int) -> tuple[int, int, int]:
        """Convert a flattened cell index to `(i, j, k)` coordinates."""
        self._validate_index(index)
        k, rem = divmod(index, self.nx * self.ny)
        j, i = divmod(rem, self.nx)
        return i, j, k

    def get_neighbors(self, index: int) -> list[int]:
        """Return active face-neighbor indices for the given cell index."""
        i, j, k = self.ijk(index)
        neighbors: list[int] = []
        for di, dj, dk in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            ni, nj, nk = i + di, j + dj, k + dk
            if self._is_inside(ni, nj, nk) and self._active_mask[nk, nj, ni]:
                neighbors.append(self.index(ni, nj, nk))
        return neighbors

    def is_active(self, index: int) -> bool:
        """Return whether a flattened cell index is active."""
        i, j, k = self.ijk(index)
        return bool(self._active_mask[k, j, i])

    def _is_inside(self, i: int, j: int, k: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def _validate_ijk(self, i: int, j: int, k: int) -> None:
        if not self._is_inside(i, j, k):
            raise GridIndexError(
                f"cell coordinates {(i, j, k)} are outside grid bounds "
                f"(nx={self.nx}, ny={self.ny}, nz={self.nz})"
            )

    def _validate_index(self, index: int) -> None:
        if not isinstance(index, int) or not 0 <= index < self.total_cells:
            raise GridIndexError(
                f"cell index {index!r} is outside valid range [0, {self.total_cells})"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid3D):
            return NotImplemented
        return (
            self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and np.allclose(self._spacing_i, other._spacing_i)
            and np.allclose(self._spacing_j, other._spacing_j)
            and np.allclose(self._spacing_k, other._spacing_k)
            and np.array_equal(self._active_mask, other._active_mask)
        )
