"""Cartesian 3D grid utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridIndexError, InvalidPhysicalValueError


@dataclass(frozen=True)
class Grid3D:
    """Regular Cartesian grid with x-fastest flattened indexing.

    Field arrays use the shape `(nz, ny, nx)`. Flattened indices are computed as
    `k * ny * nx + j * nx + i`.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    active_mask: ArrayLike | None = field(default=None, compare=False, repr=False)
    _active_mask: NDArray[np.bool_] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise InvalidPhysicalValueError(f"{name} must be a positive integer")

        for name in ("dx", "dy", "dz"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise InvalidPhysicalValueError(f"{name} must be a positive finite value")
            object.__setattr__(self, name, value)

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

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return field array shape as `(nz, ny, nx)`."""
        return (self.nz, self.ny, self.nx)

    @property
    def total_cells(self) -> int:
        """Return the total number of cells including inactive cells."""
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float:
        """Return volume of each Cartesian cell in cubic meters."""
        return self.dx * self.dy * self.dz

    @property
    def active_mask_array(self) -> NDArray[np.bool_]:
        """Return a copy of the boolean active-cell mask."""
        return self._active_mask.copy()

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
