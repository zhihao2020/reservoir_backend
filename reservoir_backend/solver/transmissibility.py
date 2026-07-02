"""Transmissibility calculations for regular Cartesian grids."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import (
    FieldShapeError,
    GridMismatchError,
    InvalidPhysicalValueError,
    NonNeighborCellError,
)
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D

Direction = Literal["x", "y", "z"]


def harmonic_average(k1: float | ArrayLike, k2: float | ArrayLike) -> float | NDArray[np.float64]:
    """Return harmonic average permeability for adjacent cells.

    Any zero permeability on either side makes the face permeability zero.
    Negative, NaN, or Inf permeability values raise `InvalidPhysicalValueError`.
    """
    k1_arr, k2_arr = np.broadcast_arrays(
        np.asarray(k1, dtype=float),
        np.asarray(k2, dtype=float),
    )
    validate_permeability(k1_arr)
    validate_permeability(k2_arr)

    result = np.zeros(k1_arr.shape, dtype=float)
    nonzero = (k1_arr > 0.0) & (k2_arr > 0.0)
    result[nonzero] = 2.0 * k1_arr[nonzero] * k2_arr[nonzero] / (
        k1_arr[nonzero] + k2_arr[nonzero]
    )

    if result.shape == ():
        return float(result)
    return result


def validate_permeability(k: float | ArrayLike) -> None:
    """Validate permeability values in SI units (`m^2`)."""
    values = np.asarray(k, dtype=float)
    if np.isnan(values).any() or np.isinf(values).any() or (values < 0.0).any():
        raise InvalidPhysicalValueError("permeability must be finite and non-negative")


def validate_viscosity(mu: float) -> None:
    """Validate dynamic viscosity in SI units (`Pa.s`)."""
    value = float(mu)
    if not np.isfinite(value) or value <= 0.0:
        raise InvalidPhysicalValueError("viscosity must be a positive finite value")


def compute_transmissibility_between_cells(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    mu: float,
    cell_a: int,
    cell_b: int,
) -> float:
    """Compute transmissibility between two face-neighbor cells.

    Permeability fields can be scalars, arrays with shape `(nz, ny, nx)`, or
    `Field3D` instances on the same grid.
    """
    validate_viscosity(mu)
    ia, ja, ka = grid.ijk(cell_a)
    ib, jb, kb = grid.ijk(cell_b)
    di, dj, dk = ib - ia, jb - ja, kb - ka
    manhattan_distance = abs(di) + abs(dj) + abs(dk)
    if manhattan_distance != 1:
        raise NonNeighborCellError(
            f"cells {cell_a} and {cell_b} are not face neighbors on this grid"
        )

    if di != 0:
        values = _permeability_values(grid, kx)
        k_face = harmonic_average(values[ka, ja, ia], values[kb, jb, ib])
        return float(k_face) * grid.dy * grid.dz / (float(mu) * grid.dx)
    if dj != 0:
        values = _permeability_values(grid, ky)
        k_face = harmonic_average(values[ka, ja, ia], values[kb, jb, ib])
        return float(k_face) * grid.dx * grid.dz / (float(mu) * grid.dy)

    values = _permeability_values(grid, kz)
    k_face = harmonic_average(values[ka, ja, ia], values[kb, jb, ib])
    return float(k_face) * grid.dx * grid.dy / (float(mu) * grid.dz)


def compute_directional_transmissibility(
    grid: Grid3D,
    k_field: float | ArrayLike | Field3D,
    mu: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Compute transmissibility on all internal faces in one direction.

    Return shapes are:
    - x: `(nz, ny, nx - 1)`
    - y: `(nz, ny - 1, nx)`
    - z: `(nz - 1, ny, nx)`
    """
    validate_viscosity(mu)
    values = _permeability_values(grid, k_field)

    if direction == "x":
        k_face = harmonic_average(values[:, :, :-1], values[:, :, 1:])
        return k_face * grid.dy * grid.dz / (float(mu) * grid.dx)
    if direction == "y":
        k_face = harmonic_average(values[:, :-1, :], values[:, 1:, :])
        return k_face * grid.dx * grid.dz / (float(mu) * grid.dy)
    if direction == "z":
        k_face = harmonic_average(values[:-1, :, :], values[1:, :, :])
        return k_face * grid.dx * grid.dy / (float(mu) * grid.dz)

    raise ValueError("direction must be one of 'x', 'y', or 'z'")


def _permeability_values(
    grid: Grid3D,
    permeability: float | ArrayLike | Field3D,
) -> NDArray[np.float64]:
    if isinstance(permeability, Field3D):
        if permeability.grid != grid:
            raise GridMismatchError("permeability Field3D is defined on a different grid")
        values = permeability.values.astype(float, copy=False)
    else:
        values = np.asarray(permeability, dtype=float)
        if values.shape == ():
            values = np.full(grid.shape, float(values), dtype=float)
        elif values.shape != grid.shape:
            raise FieldShapeError(
                f"permeability shape {values.shape} does not match grid shape {grid.shape}"
            )

    validate_permeability(values)
    return values
