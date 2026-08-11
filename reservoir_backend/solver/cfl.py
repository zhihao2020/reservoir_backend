"""CFL checks for explicit saturation transport."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D


def compute_cfl_number(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Compute cell-wise CFL numbers from connected face flux magnitudes."""
    validate_time_step(dt)
    phi_values = validate_porosity(phi, grid)
    fx, fy, fz = _validate_fluxes(grid, flux_x, flux_y, flux_z)

    connected_flux = (
        np.abs(fx[:, :, :-1])
        + np.abs(fx[:, :, 1:])
        + np.abs(fy[:, :-1, :])
        + np.abs(fy[:, 1:, :])
        + np.abs(fz[:-1, :, :])
        + np.abs(fz[1:, :, :])
    )
    pore_volume = phi_values * grid.cell_volumes
    cfl_field = float(dt) * connected_flux / pore_volume
    has_nan = bool(np.isnan(cfl_field).any())
    has_inf = bool(np.isinf(cfl_field).any())
    flat_index = int(np.nanargmax(cfl_field)) if cfl_field.size else 0
    max_location = tuple(int(v) for v in np.unravel_index(flat_index, cfl_field.shape))
    report: dict[str, object] = {
        "max_cfl": float(np.nanmax(cfl_field)),
        "mean_cfl": float(np.nanmean(cfl_field)),
        "min_cfl": float(np.nanmin(cfl_field)),
        "max_cfl_location": max_location,
        "dt": float(dt),
        "max_cfl_allowed": None,
        "stable": None,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    return cfl_field, report


def check_cfl_condition(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    max_cfl: float = 1.0,
) -> dict[str, object]:
    """Validate the CFL condition and return a stable report."""
    max_allowed = _validate_max_cfl(max_cfl)
    _, report = compute_cfl_number(grid, phi, flux_x, flux_y, flux_z, dt)
    report["max_cfl_allowed"] = max_allowed
    report["stable"] = bool(report["max_cfl"] <= max_allowed and not report["has_nan"] and not report["has_inf"])
    if not report["stable"]:
        raise CFLViolationError(
            f"CFL condition violated: max_cfl={report['max_cfl']} "
            f"max_cfl_allowed={max_allowed}"
        )
    return report


def estimate_stable_dt(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    max_cfl: float = 0.5,
) -> float:
    """Estimate a stable explicit time step.

    If all face fluxes are zero, no advective CFL restriction exists and
    `np.inf` is returned.
    """
    max_allowed = _validate_max_cfl(max_cfl)
    phi_values = validate_porosity(phi, grid)
    fx, fy, fz = _validate_fluxes(grid, flux_x, flux_y, flux_z)
    connected_flux = (
        np.abs(fx[:, :, :-1])
        + np.abs(fx[:, :, 1:])
        + np.abs(fy[:, :-1, :])
        + np.abs(fy[:, 1:, :])
        + np.abs(fz[:-1, :, :])
        + np.abs(fz[1:, :, :])
    )
    max_ratio = float(np.max(connected_flux / (phi_values * grid.cell_volumes)))
    if max_ratio == 0.0:
        return float(np.inf)
    return max_allowed / max_ratio


def validate_porosity(phi: float | ArrayLike | Field3D, grid: Grid3D | None = None) -> NDArray[np.float64]:
    """Validate porosity values and return an array."""
    if isinstance(phi, Field3D):
        if grid is not None and phi.grid != grid:
            raise GridMismatchError("phi Field3D is defined on a different grid")
        values = phi.values.astype(float, copy=False)
        expected_shape = phi.grid.shape if grid is None else grid.shape
    else:
        values = np.asarray(phi, dtype=float)
        expected_shape = None if grid is None else grid.shape
        if grid is not None and values.shape == ():
            values = np.full(grid.shape, float(values), dtype=float)

    if expected_shape is not None and values.shape != expected_shape:
        raise FieldShapeError(f"phi shape {values.shape} does not match grid shape {expected_shape}")
    if np.isnan(values).any() or np.isinf(values).any() or (values <= 0.0).any():
        raise InvalidPhysicalValueError("porosity must be positive and finite")
    return values


def validate_time_step(dt: float) -> None:
    """Validate explicit time step in seconds."""
    value = float(dt)
    if not np.isfinite(value) or value <= 0.0:
        raise InvalidPhysicalValueError("dt must be a positive finite value")


def _validate_max_cfl(max_cfl: float) -> float:
    value = float(max_cfl)
    if not np.isfinite(value) or value <= 0.0:
        raise InvalidPhysicalValueError("max_cfl must be a positive finite value")
    return value


def _validate_fluxes(
    grid: Grid3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    fx = np.asarray(flux_x, dtype=float)
    fy = np.asarray(flux_y, dtype=float)
    fz = np.asarray(flux_z, dtype=float)
    expected_x = (grid.nz, grid.ny, grid.nx + 1)
    expected_y = (grid.nz, grid.ny + 1, grid.nx)
    expected_z = (grid.nz + 1, grid.ny, grid.nx)
    if fx.shape != expected_x:
        raise FieldShapeError(f"flux_x shape {fx.shape} does not match {expected_x}")
    if fy.shape != expected_y:
        raise FieldShapeError(f"flux_y shape {fy.shape} does not match {expected_y}")
    if fz.shape != expected_z:
        raise FieldShapeError(f"flux_z shape {fz.shape} does not match {expected_z}")
    if np.isnan(fx).any() or np.isnan(fy).any() or np.isnan(fz).any():
        raise InvalidPhysicalValueError("flux arrays must not contain NaN")
    if np.isinf(fx).any() or np.isinf(fy).any() or np.isinf(fz).any():
        raise InvalidPhysicalValueError("flux arrays must not contain Inf")
    return fx, fy, fz
