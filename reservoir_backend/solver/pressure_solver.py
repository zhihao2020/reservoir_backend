"""Pressure solvers for staged finite-volume development.

This module currently implements only the 1D x-direction steady-state
Dirichlet case needed before extending the solver to 2D/3D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.transmissibility import (
    compute_directional_transmissibility,
    validate_permeability,
    validate_viscosity,
)


@dataclass(frozen=True)
class PressureSolveResult:
    """Pressure solver output."""

    pressure: Field3D
    report: dict[str, float | str | int]


BoundaryName = str


def solve_steady_state_pressure_1d(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    mu: float,
    left_pressure: float,
    right_pressure: float,
) -> PressureSolveResult:
    """Solve 1D steady single-phase Darcy pressure with Dirichlet boundaries.

    The Dirichlet pressures are applied at the left and right domain faces.
    Returned pressures are cell-center pressures with shape `(1, 1, nx)`.
    """
    if grid.ny != 1 or grid.nz != 1:
        raise NotImplementedError("solve_steady_state_pressure_1d supports only ny=1, nz=1")

    validate_viscosity(mu)
    left_pressure = _validate_pressure(left_pressure, "left_pressure")
    right_pressure = _validate_pressure(right_pressure, "right_pressure")
    kx_values = _permeability_values(grid, kx)

    nx = grid.nx
    matrix = np.zeros((nx, nx), dtype=float)
    rhs = np.zeros(nx, dtype=float)

    if nx > 1:
        tx = compute_directional_transmissibility(grid, kx_values, mu, "x")[0, 0, :]
        for i, transmissibility in enumerate(tx):
            matrix[i, i] += transmissibility
            matrix[i + 1, i + 1] += transmissibility
            matrix[i, i + 1] -= transmissibility
            matrix[i + 1, i] -= transmissibility

    left_t = 2.0 * kx_values[0, 0, 0] * grid.dy * grid.dz / (float(mu) * grid.dx)
    right_t = 2.0 * kx_values[0, 0, -1] * grid.dy * grid.dz / (float(mu) * grid.dx)
    matrix[0, 0] += left_t
    matrix[-1, -1] += right_t
    rhs[0] += left_t * left_pressure
    rhs[-1] += right_t * right_pressure

    pressure_values = np.linalg.solve(matrix, rhs)
    residual = matrix @ pressure_values - rhs
    pressure = Field3D(
        grid=grid,
        values=pressure_values.reshape(grid.shape),
        name="pressure",
        unit="Pa",
    )
    return PressureSolveResult(
        pressure=pressure,
        report={
            "solver": "numpy.linalg.solve",
            "status": "converged",
            "unknowns": nx,
            "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
        },
    )


def solve_steady_state_pressure_2d_no_flow_y(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    mu: float,
    left_pressure: float,
    right_pressure: float,
) -> PressureSolveResult:
    """Solve 2D steady pressure with left/right Dirichlet and y no-flow boundaries.

    This staged solver supports `nz=1`. Top and bottom y-boundaries are no-flow
    because no boundary flux terms are assembled there.
    """
    result = solve_steady_state_pressure_2d(
        grid=grid,
        kx=kx,
        ky=ky,
        mu=mu,
        dirichlet_boundaries={"left": left_pressure, "right": right_pressure},
    )
    result.report["boundary_y"] = "no_flow"
    return result


def solve_steady_state_pressure_2d(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    mu: float,
    dirichlet_boundaries: dict[BoundaryName, float] | None = None,
    wells: list[Well] | None = None,
    reference_pressure: float = 0.0,
) -> PressureSolveResult:
    """Solve 2D steady single-phase Darcy pressure on an `nz=1` Cartesian grid.

    `dirichlet_boundaries` may contain `left`, `right`, `bottom`, and `top`.
    Omitted boundaries are treated as no-flow. If all boundaries are no-flow,
    one cell is fixed to `reference_pressure` to remove the constant-pressure
    nullspace of the pure Neumann system.
    """
    if grid.nz != 1 or grid.nx <= 1 or grid.ny <= 1:
        raise NotImplementedError("solve_steady_state_pressure_2d supports nx>1, ny>1, nz=1")

    validate_viscosity(mu)
    boundaries = _validate_boundaries(dirichlet_boundaries)
    reference_pressure = _validate_pressure(reference_pressure, "reference_pressure")
    kx_values = _permeability_values(grid, kx)
    ky_values = _permeability_values(grid, ky)
    wells = [] if wells is None else list(wells)

    n = grid.total_cells
    matrix = lil_matrix((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    tx = compute_directional_transmissibility(grid, kx_values, mu, "x")[0]
    for j in range(grid.ny):
        for i in range(grid.nx - 1):
            _add_internal_face(matrix, grid.index(i, j, 0), grid.index(i + 1, j, 0), float(tx[j, i]))

    ty = compute_directional_transmissibility(grid, ky_values, mu, "y")[0]
    for j in range(grid.ny - 1):
        for i in range(grid.nx):
            _add_internal_face(matrix, grid.index(i, j, 0), grid.index(i, j + 1, 0), float(ty[j, i]))

    _apply_2d_dirichlet_boundaries(matrix, rhs, grid, kx_values, ky_values, float(mu), boundaries)

    net_well_rate = 0.0
    for well in wells:
        if well.grid != grid:
            raise GridMismatchError(f"well {well.name} is defined on a different grid")
        assert well.cell_index is not None
        signed_rate = well.signed_rate
        rhs[well.cell_index] += signed_rate
        net_well_rate += signed_rate

    pressure_reference_applied = False
    if not boundaries:
        matrix.rows[0] = [0]
        matrix.data[0] = [1.0]
        rhs[0] = reference_pressure
        pressure_reference_applied = True

    sparse_matrix = matrix.tocsr()
    pressure_values = np.asarray(spsolve(sparse_matrix, rhs), dtype=float)
    if np.isnan(pressure_values).any() or np.isinf(pressure_values).any():
        raise InvalidPhysicalValueError("pressure solve produced NaN or Inf")

    residual = sparse_matrix @ pressure_values - rhs
    pressure = Field3D(
        grid=grid,
        values=pressure_values.reshape(grid.shape),
        name="pressure",
        unit="Pa",
    )
    boundary_outflow = _compute_2d_boundary_outflow(
        grid=grid,
        pressure_values=pressure.values,
        kx_values=kx_values,
        ky_values=ky_values,
        mu=float(mu),
        boundaries=boundaries,
    )
    balance_scale = max(abs(boundary_outflow), abs(net_well_rate), 1.0)
    mass_balance_error = abs(boundary_outflow - net_well_rate) / balance_scale

    return PressureSolveResult(
        pressure=pressure,
        report={
            "solver": "scipy.sparse.linalg.spsolve",
            "status": "converged",
            "unknowns": n,
            "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
            "mass_balance_error": float(mass_balance_error),
            "boundary_outflow_m3_s": float(boundary_outflow),
            "net_well_rate_m3_s": float(net_well_rate),
            "pressure_reference_applied": int(pressure_reference_applied),
            "dimensions": "2d",
        },
    )


def solve_steady_state_pressure_3d(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    mu: float,
    dirichlet_boundaries: dict[BoundaryName, float] | None = None,
    wells: list[Well] | None = None,
    reference_pressure: float = 0.0,
) -> PressureSolveResult:
    """Solve 3D steady single-phase Darcy pressure on a Cartesian grid.

    `dirichlet_boundaries` may contain `left`, `right`, `front`, `back`,
    `bottom`, and `top`. Omitted boundaries are treated as no-flow. If all
    boundaries are no-flow, one reference cell is fixed to remove the constant
    nullspace of the pure Neumann system.
    """
    if grid.nx <= 1 or grid.ny <= 1 or grid.nz <= 1:
        raise NotImplementedError("solve_steady_state_pressure_3d supports nx>1, ny>1, nz>1")

    validate_viscosity(mu)
    boundaries = _validate_3d_boundaries(dirichlet_boundaries)
    reference_pressure = _validate_pressure(reference_pressure, "reference_pressure")
    kx_values = _permeability_values(grid, kx)
    ky_values = _permeability_values(grid, ky)
    kz_values = _permeability_values(grid, kz)
    wells = [] if wells is None else list(wells)

    n = grid.total_cells
    matrix = lil_matrix((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)

    tx = compute_directional_transmissibility(grid, kx_values, mu, "x")
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx - 1):
                _add_internal_face(
                    matrix,
                    grid.index(i, j, k),
                    grid.index(i + 1, j, k),
                    float(tx[k, j, i]),
                )

    ty = compute_directional_transmissibility(grid, ky_values, mu, "y")
    for k in range(grid.nz):
        for j in range(grid.ny - 1):
            for i in range(grid.nx):
                _add_internal_face(
                    matrix,
                    grid.index(i, j, k),
                    grid.index(i, j + 1, k),
                    float(ty[k, j, i]),
                )

    tz = compute_directional_transmissibility(grid, kz_values, mu, "z")
    for k in range(grid.nz - 1):
        for j in range(grid.ny):
            for i in range(grid.nx):
                _add_internal_face(
                    matrix,
                    grid.index(i, j, k),
                    grid.index(i, j, k + 1),
                    float(tz[k, j, i]),
                )

    _apply_3d_dirichlet_boundaries(
        matrix,
        rhs,
        grid,
        kx_values,
        ky_values,
        kz_values,
        float(mu),
        boundaries,
    )

    net_well_rate = 0.0
    for well in wells:
        if well.grid != grid:
            raise GridMismatchError(f"well {well.name} is defined on a different grid")
        assert well.cell_index is not None
        signed_rate = well.signed_rate
        rhs[well.cell_index] += signed_rate
        net_well_rate += signed_rate

    pressure_reference_applied = False
    if not boundaries:
        matrix.rows[0] = [0]
        matrix.data[0] = [1.0]
        rhs[0] = reference_pressure
        pressure_reference_applied = True

    sparse_matrix = matrix.tocsr()
    pressure_values = np.asarray(spsolve(sparse_matrix, rhs), dtype=float)
    if np.isnan(pressure_values).any() or np.isinf(pressure_values).any():
        raise InvalidPhysicalValueError("pressure solve produced NaN or Inf")

    residual = sparse_matrix @ pressure_values - rhs
    pressure = Field3D(
        grid=grid,
        values=pressure_values.reshape(grid.shape),
        name="pressure",
        unit="Pa",
    )
    boundary_outflow = _compute_3d_boundary_outflow(
        grid=grid,
        pressure_values=pressure.values,
        kx_values=kx_values,
        ky_values=ky_values,
        kz_values=kz_values,
        mu=float(mu),
        boundaries=boundaries,
    )
    balance_scale = max(abs(boundary_outflow), abs(net_well_rate), 1.0)
    mass_balance_error = abs(boundary_outflow - net_well_rate) / balance_scale

    return PressureSolveResult(
        pressure=pressure,
        report={
            "solver": "scipy.sparse.linalg.spsolve",
            "status": "converged",
            "unknowns": n,
            "residual_norm": float(np.linalg.norm(residual, ord=np.inf)),
            "mass_balance_error": float(mass_balance_error),
            "boundary_outflow_m3_s": float(boundary_outflow),
            "net_well_rate_m3_s": float(net_well_rate),
            "pressure_reference_applied": pressure_reference_applied,
            "dimensions": "3d",
        },
    )


def compute_hydrostatic_pressure(
    grid: Grid3D,
    datum_pressure: float,
    density: float,
    gravity: float = 9.80665,
    datum_depth: float = 0.0,
) -> Field3D:
    """Return hydrostatic pressure at cell centers for a z-down grid."""
    datum_pressure = _validate_pressure(datum_pressure, "datum_pressure")
    density = float(density)
    gravity = float(gravity)
    datum_depth = float(datum_depth)
    if not np.isfinite(density) or density <= 0.0:
        raise InvalidPhysicalValueError("density must be a positive finite value")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise InvalidPhysicalValueError("gravity must be a positive finite value")
    if not np.isfinite(datum_depth):
        raise ValueError("datum_depth must be finite")

    values = np.empty(grid.shape, dtype=float)
    for k in range(grid.nz):
        depth = datum_depth + (k + 0.5) * grid.dz
        values[k, :, :] = datum_pressure + density * gravity * depth

    return Field3D(grid=grid, values=values, name="pressure", unit="Pa")


def _validate_pressure(value: float, name: str) -> float:
    pressure = float(value)
    if not np.isfinite(pressure):
        raise ValueError(f"{name} must be finite")
    return pressure


def _validate_boundaries(boundaries: dict[BoundaryName, float] | None) -> dict[str, float]:
    if boundaries is None:
        return {}
    allowed = {"left", "right", "bottom", "top"}
    normalized: dict[str, float] = {}
    for name, value in boundaries.items():
        key = name.lower()
        if key not in allowed:
            raise ValueError(f"unsupported 2D boundary name: {name}")
        normalized[key] = _validate_pressure(value, f"{key}_pressure")
    return normalized


def _validate_3d_boundaries(boundaries: dict[BoundaryName, float] | None) -> dict[str, float]:
    if boundaries is None:
        return {}
    allowed = {"left", "right", "front", "back", "bottom", "top"}
    normalized: dict[str, float] = {}
    for name, value in boundaries.items():
        key = name.lower()
        if key not in allowed:
            raise ValueError(f"unsupported 3D boundary name: {name}")
        normalized[key] = _validate_pressure(value, f"{key}_pressure")
    return normalized


def _add_internal_face(matrix, cell_a: int, cell_b: int, transmissibility: float) -> None:
    matrix[cell_a, cell_a] += transmissibility
    matrix[cell_b, cell_b] += transmissibility
    matrix[cell_a, cell_b] -= transmissibility
    matrix[cell_b, cell_a] -= transmissibility


def _apply_2d_dirichlet_boundaries(
    matrix,
    rhs: NDArray[np.float64],
    grid: Grid3D,
    kx_values: NDArray[np.float64],
    ky_values: NDArray[np.float64],
    mu: float,
    boundaries: dict[str, float],
) -> None:
    if "left" in boundaries:
        for j in range(grid.ny):
            cell = grid.index(0, j, 0)
            transmissibility = 2.0 * kx_values[0, j, 0] * grid.dy * grid.dz / (mu * grid.dx)
            matrix[cell, cell] += transmissibility
            rhs[cell] += transmissibility * boundaries["left"]

    if "right" in boundaries:
        for j in range(grid.ny):
            cell = grid.index(grid.nx - 1, j, 0)
            transmissibility = 2.0 * kx_values[0, j, -1] * grid.dy * grid.dz / (mu * grid.dx)
            matrix[cell, cell] += transmissibility
            rhs[cell] += transmissibility * boundaries["right"]

    if "bottom" in boundaries:
        for i in range(grid.nx):
            cell = grid.index(i, 0, 0)
            transmissibility = 2.0 * ky_values[0, 0, i] * grid.dx * grid.dz / (mu * grid.dy)
            matrix[cell, cell] += transmissibility
            rhs[cell] += transmissibility * boundaries["bottom"]

    if "top" in boundaries:
        for i in range(grid.nx):
            cell = grid.index(i, grid.ny - 1, 0)
            transmissibility = 2.0 * ky_values[0, -1, i] * grid.dx * grid.dz / (mu * grid.dy)
            matrix[cell, cell] += transmissibility
            rhs[cell] += transmissibility * boundaries["top"]


def _compute_2d_boundary_outflow(
    grid: Grid3D,
    pressure_values: NDArray[np.float64],
    kx_values: NDArray[np.float64],
    ky_values: NDArray[np.float64],
    mu: float,
    boundaries: dict[str, float],
) -> float:
    outflow = 0.0
    if "left" in boundaries:
        for j in range(grid.ny):
            transmissibility = 2.0 * kx_values[0, j, 0] * grid.dy * grid.dz / (mu * grid.dx)
            outflow += transmissibility * (pressure_values[0, j, 0] - boundaries["left"])
    if "right" in boundaries:
        for j in range(grid.ny):
            transmissibility = 2.0 * kx_values[0, j, -1] * grid.dy * grid.dz / (mu * grid.dx)
            outflow += transmissibility * (pressure_values[0, j, -1] - boundaries["right"])
    if "bottom" in boundaries:
        for i in range(grid.nx):
            transmissibility = 2.0 * ky_values[0, 0, i] * grid.dx * grid.dz / (mu * grid.dy)
            outflow += transmissibility * (pressure_values[0, 0, i] - boundaries["bottom"])
    if "top" in boundaries:
        for i in range(grid.nx):
            transmissibility = 2.0 * ky_values[0, -1, i] * grid.dx * grid.dz / (mu * grid.dy)
            outflow += transmissibility * (pressure_values[0, -1, i] - boundaries["top"])
    return float(outflow)


def _apply_3d_dirichlet_boundaries(
    matrix,
    rhs: NDArray[np.float64],
    grid: Grid3D,
    kx_values: NDArray[np.float64],
    ky_values: NDArray[np.float64],
    kz_values: NDArray[np.float64],
    mu: float,
    boundaries: dict[str, float],
) -> None:
    if "left" in boundaries:
        for k in range(grid.nz):
            for j in range(grid.ny):
                cell = grid.index(0, j, k)
                transmissibility = 2.0 * kx_values[k, j, 0] * grid.dy * grid.dz / (mu * grid.dx)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["left"]

    if "right" in boundaries:
        for k in range(grid.nz):
            for j in range(grid.ny):
                cell = grid.index(grid.nx - 1, j, k)
                transmissibility = 2.0 * kx_values[k, j, -1] * grid.dy * grid.dz / (mu * grid.dx)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["right"]

    if "front" in boundaries:
        for k in range(grid.nz):
            for i in range(grid.nx):
                cell = grid.index(i, 0, k)
                transmissibility = 2.0 * ky_values[k, 0, i] * grid.dx * grid.dz / (mu * grid.dy)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["front"]

    if "back" in boundaries:
        for k in range(grid.nz):
            for i in range(grid.nx):
                cell = grid.index(i, grid.ny - 1, k)
                transmissibility = 2.0 * ky_values[k, -1, i] * grid.dx * grid.dz / (mu * grid.dy)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["back"]

    if "bottom" in boundaries:
        for j in range(grid.ny):
            for i in range(grid.nx):
                cell = grid.index(i, j, 0)
                transmissibility = 2.0 * kz_values[0, j, i] * grid.dx * grid.dy / (mu * grid.dz)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["bottom"]

    if "top" in boundaries:
        for j in range(grid.ny):
            for i in range(grid.nx):
                cell = grid.index(i, j, grid.nz - 1)
                transmissibility = 2.0 * kz_values[-1, j, i] * grid.dx * grid.dy / (mu * grid.dz)
                matrix[cell, cell] += transmissibility
                rhs[cell] += transmissibility * boundaries["top"]


def _compute_3d_boundary_outflow(
    grid: Grid3D,
    pressure_values: NDArray[np.float64],
    kx_values: NDArray[np.float64],
    ky_values: NDArray[np.float64],
    kz_values: NDArray[np.float64],
    mu: float,
    boundaries: dict[str, float],
) -> float:
    outflow = 0.0
    if "left" in boundaries:
        for k in range(grid.nz):
            for j in range(grid.ny):
                transmissibility = 2.0 * kx_values[k, j, 0] * grid.dy * grid.dz / (mu * grid.dx)
                outflow += transmissibility * (pressure_values[k, j, 0] - boundaries["left"])
    if "right" in boundaries:
        for k in range(grid.nz):
            for j in range(grid.ny):
                transmissibility = 2.0 * kx_values[k, j, -1] * grid.dy * grid.dz / (mu * grid.dx)
                outflow += transmissibility * (pressure_values[k, j, -1] - boundaries["right"])
    if "front" in boundaries:
        for k in range(grid.nz):
            for i in range(grid.nx):
                transmissibility = 2.0 * ky_values[k, 0, i] * grid.dx * grid.dz / (mu * grid.dy)
                outflow += transmissibility * (pressure_values[k, 0, i] - boundaries["front"])
    if "back" in boundaries:
        for k in range(grid.nz):
            for i in range(grid.nx):
                transmissibility = 2.0 * ky_values[k, -1, i] * grid.dx * grid.dz / (mu * grid.dy)
                outflow += transmissibility * (pressure_values[k, -1, i] - boundaries["back"])
    if "bottom" in boundaries:
        for j in range(grid.ny):
            for i in range(grid.nx):
                transmissibility = 2.0 * kz_values[0, j, i] * grid.dx * grid.dy / (mu * grid.dz)
                outflow += transmissibility * (pressure_values[0, j, i] - boundaries["bottom"])
    if "top" in boundaries:
        for j in range(grid.ny):
            for i in range(grid.nx):
                transmissibility = 2.0 * kz_values[-1, j, i] * grid.dx * grid.dy / (mu * grid.dz)
                outflow += transmissibility * (pressure_values[-1, j, i] - boundaries["top"])
    return float(outflow)


def _permeability_values(grid: Grid3D, permeability: float | ArrayLike | Field3D) -> NDArray[np.float64]:
    if isinstance(permeability, Field3D):
        if permeability.grid != grid:
            raise GridMismatchError("kx Field3D is defined on a different grid")
        values = permeability.values.astype(float, copy=False)
    else:
        values = np.asarray(permeability, dtype=float)
        if values.shape == ():
            values = np.full(grid.shape, float(values), dtype=float)
        elif values.shape != grid.shape:
            raise FieldShapeError(
                f"kx shape {values.shape} does not match grid shape {grid.shape}"
            )

    validate_permeability(values)
    return values
