"""Darcy face flux and cell-centered velocity calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.transmissibility import (
    compute_directional_transmissibility,
    validate_viscosity,
)


@dataclass(frozen=True)
class FaceFluxes:
    """Finite-volume face fluxes in x, y, and z directions."""

    flux_x: NDArray[np.float64]
    flux_y: NDArray[np.float64]
    flux_z: NDArray[np.float64]


@dataclass(frozen=True)
class DarcyVelocityResult:
    """Darcy velocity calculation output."""

    velocity_x: Field3D
    velocity_y: Field3D
    velocity_z: Field3D
    face_fluxes: FaceFluxes
    report: dict[str, float | bool]


def compute_face_fluxes(
    grid: Grid3D,
    pressure: Field3D | ArrayLike,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    mu: float,
) -> FaceFluxes:
    """Compute Darcy face fluxes with no-flow external boundaries.

    Positive `flux_x` is left-to-right, positive `flux_y` is front-to-back,
    and positive `flux_z` is bottom-to-top in grid index order.
    """
    validate_viscosity(mu)
    pressure_values = _field_values(grid, pressure, "pressure")
    if np.isnan(pressure_values).any() or np.isinf(pressure_values).any():
        raise InvalidPhysicalValueError("pressure must be finite")

    tx = compute_directional_transmissibility(grid, kx, mu, "x")
    ty = compute_directional_transmissibility(grid, ky, mu, "y")
    tz = compute_directional_transmissibility(grid, kz, mu, "z")

    flux_x = np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float)
    flux_y = np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float)
    flux_z = np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float)

    if grid.nx > 1:
        flux_x[:, :, 1:-1] = -tx * (pressure_values[:, :, 1:] - pressure_values[:, :, :-1])
    if grid.ny > 1:
        flux_y[:, 1:-1, :] = -ty * (pressure_values[:, 1:, :] - pressure_values[:, :-1, :])
    if grid.nz > 1:
        flux_z[1:-1, :, :] = -tz * (pressure_values[1:, :, :] - pressure_values[:-1, :, :])

    return FaceFluxes(flux_x=flux_x, flux_y=flux_y, flux_z=flux_z)


def compute_cell_center_velocity(grid: Grid3D, face_fluxes: FaceFluxes) -> tuple[Field3D, Field3D, Field3D]:
    """Compute cell-centered velocity fields by averaging adjacent face fluxes."""
    _validate_flux_shapes(grid, face_fluxes)
    area_x = grid.dy * grid.dz
    area_y = grid.dx * grid.dz
    area_z = grid.dx * grid.dy

    vx = 0.5 * (face_fluxes.flux_x[:, :, :-1] + face_fluxes.flux_x[:, :, 1:]) / area_x
    vy = 0.5 * (face_fluxes.flux_y[:, :-1, :] + face_fluxes.flux_y[:, 1:, :]) / area_y
    vz = 0.5 * (face_fluxes.flux_z[:-1, :, :] + face_fluxes.flux_z[1:, :, :]) / area_z

    return (
        Field3D(grid=grid, values=vx, name="velocity_x", unit="m/s"),
        Field3D(grid=grid, values=vy, name="velocity_y", unit="m/s"),
        Field3D(grid=grid, values=vz, name="velocity_z", unit="m/s"),
    )


def compute_darcy_velocity(
    grid: Grid3D,
    pressure: Field3D | ArrayLike,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    mu: float,
) -> DarcyVelocityResult:
    """Compute face fluxes, cell-centered velocities, and a diagnostic report."""
    face_fluxes = compute_face_fluxes(grid, pressure, kx, ky, kz, mu)
    velocity_x, velocity_y, velocity_z = compute_cell_center_velocity(grid, face_fluxes)
    arrays = [
        face_fluxes.flux_x,
        face_fluxes.flux_y,
        face_fluxes.flux_z,
        velocity_x.values,
        velocity_y.values,
        velocity_z.values,
    ]
    all_values = np.concatenate([array.ravel() for array in arrays])
    boundary_flux = (
        np.sum(face_fluxes.flux_x[:, :, 0])
        + np.sum(face_fluxes.flux_x[:, :, -1])
        + np.sum(face_fluxes.flux_y[:, 0, :])
        + np.sum(face_fluxes.flux_y[:, -1, :])
        + np.sum(face_fluxes.flux_z[0, :, :])
        + np.sum(face_fluxes.flux_z[-1, :, :])
    )
    return DarcyVelocityResult(
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        face_fluxes=face_fluxes,
        report={
            "max_flux": float(max(np.max(face_fluxes.flux_x), np.max(face_fluxes.flux_y), np.max(face_fluxes.flux_z))),
            "min_flux": float(min(np.min(face_fluxes.flux_x), np.min(face_fluxes.flux_y), np.min(face_fluxes.flux_z))),
            "total_boundary_flux": float(boundary_flux),
            "has_nan": bool(np.isnan(all_values).any()),
            "has_inf": bool(np.isinf(all_values).any()),
        },
    )


def _field_values(grid: Grid3D, value: Field3D | ArrayLike, name: str) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        if value.grid != grid:
            raise GridMismatchError(f"{name} Field3D is defined on a different grid")
        values = value.values.astype(float, copy=False)
    else:
        values = np.asarray(value, dtype=float)
        if values.shape != grid.shape:
            raise FieldShapeError(f"{name} shape {values.shape} does not match grid shape {grid.shape}")
    return values


def _validate_flux_shapes(grid: Grid3D, face_fluxes: FaceFluxes) -> None:
    expected_x = (grid.nz, grid.ny, grid.nx + 1)
    expected_y = (grid.nz, grid.ny + 1, grid.nx)
    expected_z = (grid.nz + 1, grid.ny, grid.nx)
    if face_fluxes.flux_x.shape != expected_x:
        raise FieldShapeError(f"flux_x shape {face_fluxes.flux_x.shape} does not match {expected_x}")
    if face_fluxes.flux_y.shape != expected_y:
        raise FieldShapeError(f"flux_y shape {face_fluxes.flux_y.shape} does not match {expected_y}")
    if face_fluxes.flux_z.shape != expected_z:
        raise FieldShapeError(f"flux_z shape {face_fluxes.flux_z.shape} does not match {expected_z}")
