"""Boundary-condition matrix and RHS contribution diagnostics.

The helpers in this module assemble standalone contribution arrays for simple
Cartesian pressure systems. They are intended for diagnostics and tests, not as
a replacement for the pressure solver assembly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.grid import Grid3D


_FACES = {"left", "right", "front", "back", "bottom", "top"}
_TYPES = {"dirichlet", "neumann", "noflow", "no-flow"}


@dataclass(frozen=True)
class BoundaryConditionContribution:
    """Minimal boundary-condition definition for contribution diagnostics."""

    face: str
    boundary_type: str
    value: float = 0.0
    transmissibility: float = 1.0

    def __post_init__(self) -> None:
        face = self.face.lower()
        if face not in _FACES:
            raise ValueError(f"unsupported boundary face: {self.face}")
        boundary_type = self.boundary_type.lower().replace("_", "-")
        if boundary_type not in _TYPES:
            raise ValueError(f"unsupported boundary type: {self.boundary_type}")
        object.__setattr__(self, "face", face)
        object.__setattr__(self, "boundary_type", "noflow" if boundary_type in {"noflow", "no-flow"} else boundary_type)
        value = float(self.value)
        transmissibility = float(self.transmissibility)
        if not np.isfinite(value):
            raise ValueError("boundary value must be finite")
        if not np.isfinite(transmissibility) or transmissibility < 0.0:
            raise ValueError("boundary transmissibility must be nonnegative finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "transmissibility", transmissibility)


def build_boundary_contribution(
    grid: Grid3D | tuple[int, int, int],
    boundary_conditions: Iterable[BoundaryConditionContribution | dict],
) -> dict:
    """Return diagonal and RHS contributions for simple boundary conditions.

    Dirichlet boundaries add `T` to the adjacent-cell diagonal and `T * p_b` to
    the RHS. Neumann boundaries add their value directly to the RHS using the
    convention that positive values inject into the domain. No-flow boundaries
    add no matrix or RHS contribution.
    """
    shape = _grid_shape(grid)
    nz, ny, nx = shape
    total = nx * ny * nz
    diagonal = np.zeros(total, dtype=float)
    rhs = np.zeros(total, dtype=float)
    diagnostics = []

    for raw_condition in boundary_conditions:
        condition = _as_condition(raw_condition)
        cells = _face_cells(shape, condition.face)
        diagonal_before = diagonal.copy()
        rhs_before = rhs.copy()
        if condition.boundary_type == "dirichlet":
            diagonal[cells] += condition.transmissibility
            rhs[cells] += condition.transmissibility * condition.value
        elif condition.boundary_type == "neumann":
            rhs[cells] += condition.value
        diagonal_delta = diagonal - diagonal_before
        rhs_delta = rhs - rhs_before
        diagnostics.append(
            {
                "face": condition.face,
                "boundary_type": condition.boundary_type,
                "value": float(condition.value),
                "transmissibility": float(condition.transmissibility),
                "num_affected_cells": int(len(cells)),
                "diagonal_contribution_sum": float(np.sum(diagonal_delta)),
                "rhs_contribution_sum": float(np.sum(rhs_delta)),
            }
        )

    return {
        "success": True,
        "matrix_diagonal": diagonal,
        "rhs": rhs,
        "matrix_shape": [total, total],
        "rhs_shape": list(rhs.shape),
        "diagnostics": diagnostics,
        "warnings": [],
    }


def apply_source_sink_to_rhs(rhs: ArrayLike, source_sink: ArrayLike) -> NDArray[np.float64]:
    """Return a copy of RHS with source/sink terms added."""
    rhs_array = np.asarray(rhs, dtype=float)
    source_array = np.asarray(source_sink, dtype=float)
    if rhs_array.shape != source_array.shape:
        raise ValueError("rhs and source_sink must have matching shapes")
    if not np.isfinite(rhs_array).all() or not np.isfinite(source_array).all():
        raise ValueError("rhs and source_sink must be finite")
    return rhs_array.copy() + source_array


def build_boundary_diagnostics(contribution: dict) -> dict:
    """Build JSON-serializable diagnostics from a contribution result."""
    diagonal = np.asarray(contribution["matrix_diagonal"], dtype=float)
    rhs = np.asarray(contribution["rhs"], dtype=float)
    return {
        "success": bool(contribution.get("success", False)) and bool(np.isfinite(diagonal).all()) and bool(np.isfinite(rhs).all()),
        "matrix_shape": list(contribution["matrix_shape"]),
        "rhs_shape": list(contribution["rhs_shape"]),
        "num_nonzero_diagonal": int(np.count_nonzero(diagonal)),
        "num_nonzero_rhs": int(np.count_nonzero(rhs)),
        "rhs_sum": float(np.sum(rhs)),
        "diagonal_sum": float(np.sum(diagonal)),
        "has_nan": bool(np.isnan(diagonal).any() or np.isnan(rhs).any()),
        "has_inf": bool(np.isinf(diagonal).any() or np.isinf(rhs).any()),
        "diagnostics": list(contribution.get("diagnostics", [])),
        "warnings": list(contribution.get("warnings", [])),
    }


def _as_condition(condition: BoundaryConditionContribution | dict) -> BoundaryConditionContribution:
    if isinstance(condition, BoundaryConditionContribution):
        return condition
    boundary_type = condition.get("boundary_type", condition.get("type", "noflow"))
    return BoundaryConditionContribution(
        face=str(condition["face"]),
        boundary_type=str(boundary_type),
        value=float(condition.get("value", 0.0)),
        transmissibility=float(condition.get("transmissibility", 1.0)),
    )


def _face_cells(shape: tuple[int, int, int], face: str) -> list[int]:
    nz, ny, nx = shape
    cells: list[int] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (
                    (face == "left" and i == 0)
                    or (face == "right" and i == nx - 1)
                    or (face == "front" and j == 0)
                    or (face == "back" and j == ny - 1)
                    or (face == "bottom" and k == 0)
                    or (face == "top" and k == nz - 1)
                ):
                    cells.append(k * ny * nx + j * nx + i)
    return cells


def _grid_shape(grid: Grid3D | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(grid, Grid3D):
        return grid.shape
    if len(grid) != 3:
        raise ValueError("grid shape must be (nz, ny, nx)")
    nz, ny, nx = (int(v) for v in grid)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("grid shape entries must be positive")
    return nz, ny, nx
