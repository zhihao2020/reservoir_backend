"""One-dimensional oil-water saturation transport solvers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.capillary_flux import compute_capillary_fluxes, compute_capillary_water_flux_1d
from reservoir_backend.solver.cfl import check_cfl_condition
from reservoir_backend.solver.gravity_flux import compute_gravity_fluxes, compute_gravity_water_flux_1d_vertical
from reservoir_backend.solver.relperm import fractional_flow_water, validate_saturation_params
from reservoir_backend.solver.water_flux_composer import compose_water_fluxes_3d, compute_effective_flux_for_cfl


@dataclass(frozen=True)
class SaturationStepResult:
    """Output from one explicit saturation step."""

    sw: Field3D
    report: dict[str, float | str]


DEFAULT_RELPERM_PARAMS = {
    "swi": 0.2,
    "sor": 0.2,
    "krw0": 1.0,
    "kro0": 1.0,
    "nw": 2.0,
    "no": 2.0,
    "mu_w": 1.0e-3,
    "mu_o": 5.0e-3,
}


def compute_upwind_water_flux_1d(
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    relperm_params: dict[str, float],
) -> NDArray[np.float64]:
    """Compute 1D upwind water flux from total face flux.

    Positive total flux moves left-to-right. Boundary injection saturation uses
    optional `injected_sw`, `injected_sw_left`, and `injected_sw_right`
    relperm parameters, defaulting to `1 - Sor`.
    """
    sw_line = _sw_line_from_value(sw)
    nx = sw_line.size
    total_flux = _flux_x_line(flux_x, nx)
    params = _normalized_relperm_params(relperm_params)
    injected_left = float(params.get("injected_sw_left", params.get("injected_sw", 1.0 - params["sor"])))
    injected_right = float(params.get("injected_sw_right", params.get("injected_sw", 1.0 - params["sor"])))

    water_flux = np.zeros(nx + 1, dtype=float)
    for face, total in enumerate(total_flux):
        if total == 0.0:
            water_flux[face] = 0.0
            continue
        if face == 0:
            upwind_sw = injected_left if total > 0.0 else sw_line[0]
        elif face == nx:
            upwind_sw = sw_line[-1] if total > 0.0 else injected_right
        else:
            upwind_sw = sw_line[face - 1] if total > 0.0 else sw_line[face]
        water_flux[face] = float(_fractional_flow(upwind_sw, params)) * total
    return water_flux.reshape(1, 1, nx + 1)


def advance_saturation_1d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 1D water saturation with explicit upwind fractional flow."""
    _validate_1d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    total_flux_x = _validate_flux_x(grid, flux_x)
    params = _normalized_relperm_params(relperm_params)
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=total_flux_x,
        flux_y=np.zeros((1, 2, grid.nx), dtype=float),
        flux_z=np.zeros((2, 1, grid.nx), dtype=float),
        dt=dt,
        max_cfl=max_cfl,
    )

    water_flux_x = compute_upwind_water_flux_1d(sw_values, total_flux_x, params)
    old = sw_values[0, 0, :]
    phi_line = phi_values[0, 0, :]
    raw = old - float(dt) / (phi_line * grid.cell_volume) * (
        water_flux_x[0, 0, 1:] - water_flux_x[0, 0, :-1]
    )
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    clipped_cells = int(np.count_nonzero((raw < lower) | (raw > upper)))
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new.reshape(grid.shape), name="sw", unit="fraction")

    balance = compute_saturation_material_balance_1d(
        grid=grid,
        sw_old=old,
        sw_new=new,
        phi=phi_line,
        water_flux_x=water_flux_x,
        dt=float(dt),
    )
    water_cut = compute_water_cut_1d(new.reshape(grid.shape), total_flux_x, params, producer_boundary="right")
    has_nan = bool(np.isnan(new).any())
    has_inf = bool(np.isinf(new).any())
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "water_cut": float(water_cut),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "relative_balance_error": balance["material_balance_error"],
        "clipped_cells": clipped_cells,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    return SaturationStepResult(sw=field, report=report)


def compute_total_water_flux_1d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    relperm_params: dict[str, float],
    capillary_params: dict[str, float] | None = None,
    kx: float | ArrayLike | Field3D | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Return advective, capillary, and total 1D water face fluxes."""
    _validate_1d_grid(grid)
    total_flux_x = _validate_flux_x(grid, flux_x)
    params = _normalized_relperm_params(relperm_params)
    water_flux_adv = compute_upwind_water_flux_1d(sw, total_flux_x, params)

    if capillary_params is None or not bool(capillary_params.get("enabled", False)) or capillary_params.get("model", "none") == "none":
        cap_flux_x = np.zeros_like(water_flux_adv, dtype=float)
        cap_report: dict[str, object] = {
            "enabled": False,
            "model": "none",
            "max_abs_capillary_flux": 0.0,
            "min_capillary_flux": 0.0,
            "max_capillary_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }
    else:
        if kx is None:
            raise ValueError("kx is required when capillary flux is enabled")
        cap_flux_x, cap_report = compute_capillary_water_flux_1d(
            grid=grid,
            sw=sw,
            kx=kx,
            capillary_params=capillary_params,
            relperm_params=params,
        )

    total_water_flux = water_flux_adv + cap_flux_x
    return water_flux_adv, cap_flux_x, total_water_flux, cap_report


def compute_capillary_saturation_update_1d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    water_flux_x: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
) -> tuple[Field3D, dict[str, float | bool | int]]:
    """Apply a 1D explicit saturation update using a supplied water flux."""
    _validate_1d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    water_flux = _validate_flux_x(grid, water_flux_x)
    params = _normalized_relperm_params(relperm_params)
    old = sw_values[0, 0, :]
    phi_line = phi_values[0, 0, :]
    raw = old - float(dt) / (phi_line * grid.cell_volume) * (
        water_flux[0, 0, 1:] - water_flux[0, 0, :-1]
    )
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    clipped_cells = int(np.count_nonzero((raw < lower) | (raw > upper)))
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new.reshape(grid.shape), name="sw", unit="fraction")
    balance = compute_saturation_material_balance_1d(
        grid=grid,
        sw_old=old,
        sw_new=new,
        phi=phi_line,
        water_flux_x=water_flux,
        dt=float(dt),
    )
    report: dict[str, float | bool | int] = {
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "relative_balance_error": balance["material_balance_error"],
        "clipped_cells": clipped_cells,
        "has_nan": bool(np.isnan(new).any()),
        "has_inf": bool(np.isinf(new).any()),
    }
    return field, report


def advance_saturation_1d_with_capillary(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    capillary_params: dict[str, float] | None,
    kx: float | ArrayLike | Field3D,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 1D saturation with optional capillary water flux coupling."""
    _validate_1d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    total_flux_x = _validate_flux_x(grid, flux_x)
    params = _normalized_relperm_params(relperm_params)
    water_flux_adv, cap_flux_x, total_water_flux, cap_report = compute_total_water_flux_1d(
        grid=grid,
        sw=sw_values,
        flux_x=total_flux_x,
        relperm_params=params,
        capillary_params=capillary_params,
        kx=kx,
    )
    capillary_enabled = bool(cap_report["enabled"])
    cfl_flux_x = (
        np.abs(total_flux_x) + np.abs(cap_flux_x)
        if capillary_enabled
        else total_flux_x
    )
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=cfl_flux_x,
        flux_y=np.zeros((1, 2, grid.nx), dtype=float),
        flux_z=np.zeros((2, 1, grid.nx), dtype=float),
        dt=dt,
        max_cfl=max_cfl,
    )
    field, update_report = compute_capillary_saturation_update_1d(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        water_flux_x=total_water_flux,
        dt=dt,
        relperm_params=params,
    )
    water_cut = compute_water_cut_1d(field.values, total_flux_x, params, producer_boundary="right")
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(update_report["sw_min"]),
        "sw_max": float(update_report["sw_max"]),
        "water_cut": float(water_cut),
        "injected_water_volume": float(update_report["injected_water_volume"]),
        "produced_water_volume": float(update_report["produced_water_volume"]),
        "storage_change": float(update_report["storage_change"]),
        "material_balance_error": float(update_report["material_balance_error"]),
        "relative_balance_error": float(update_report["relative_balance_error"]),
        "clipped_cells": int(update_report["clipped_cells"]),
        "has_nan": bool(update_report["has_nan"]),
        "has_inf": bool(update_report["has_inf"]),
        "capillary_enabled": capillary_enabled,
        "capillary_model": str(cap_report["model"]),
        "max_abs_capillary_flux": float(cap_report["max_abs_capillary_flux"]),
        "max_advective_flux": float(np.max(np.abs(water_flux_adv))),
        "max_capillary_flux": float(np.max(np.abs(cap_flux_x))),
        "max_total_water_flux": float(np.max(np.abs(total_water_flux))),
        "capillary_flux_included": capillary_enabled,
    }
    return SaturationStepResult(sw=field, report=report)


def compute_upwind_water_flux_1d_vertical(
    sw: Field3D | ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
) -> NDArray[np.float64]:
    """Compute vertical 1D upwind water flux from total z-face flux.

    Positive z flux follows the project convention: bottom cell to top cell.
    """
    sw_values = _sw_3d_from_value(sw)
    nz, ny, nx = sw_values.shape
    if nx != 1 or ny != 1 or nz <= 1:
        raise NotImplementedError("vertical 1D saturation transport requires nx=1, ny=1, nz>1")
    fz = _validate_flux_z_vertical_shape((nz, ny, nx), flux_z)
    fx = np.zeros((nz, ny, nx + 1), dtype=float)
    fy = np.zeros((nz, ny + 1, nx), dtype=float)
    return compute_upwind_water_flux_3d(sw_values, fx, fy, fz, relperm_params)[2]


def compute_total_water_flux_1d_vertical(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
    gravity_params: dict[str, float] | None = None,
    kz: float | ArrayLike | Field3D | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Return advective, gravity, and total vertical 1D water face fluxes."""
    _validate_vertical_1d_grid(grid)
    total_flux_z = _validate_flux_z_vertical(grid, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_flux_adv = compute_upwind_water_flux_1d_vertical(sw, total_flux_z, params)

    if gravity_params is None or not bool(gravity_params.get("enabled", False)):
        gravity_flux_z = np.zeros_like(water_flux_adv, dtype=float)
        gravity_report: dict[str, object] = {
            "enabled": False,
            "g": float((gravity_params or {}).get("g", 9.80665)),
            "rho_w": float((gravity_params or {}).get("rho_w", 1000.0)),
            "rho_o": float((gravity_params or {}).get("rho_o", 800.0)),
            "density_difference": float(
                (gravity_params or {}).get("rho_w", 1000.0) - (gravity_params or {}).get("rho_o", 800.0)
            ),
            "max_abs_gravity_flux": 0.0,
            "min_gravity_flux": 0.0,
            "max_gravity_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }
    else:
        if kz is None:
            raise ValueError("kz is required when gravity flux is enabled")
        gravity_flux_z, gravity_report = compute_gravity_water_flux_1d_vertical(
            grid=grid,
            sw=sw,
            kz=kz,
            gravity_params=gravity_params,
            relperm_params=params,
        )

    total_water_flux = water_flux_adv + gravity_flux_z
    return water_flux_adv, gravity_flux_z, total_water_flux, gravity_report


def compute_gravity_saturation_update_1d_vertical(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    water_flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
) -> tuple[Field3D, dict[str, float | bool | int]]:
    """Apply a vertical 1D explicit saturation update from supplied water flux."""
    _validate_vertical_1d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    water_flux = _validate_flux_z_vertical(grid, water_flux_z)
    params = _normalized_relperm_params(relperm_params)
    divergence = water_flux[1:, :, :] - water_flux[:-1, :, :]
    raw = sw_values - float(dt) * divergence / (phi_values * grid.cell_volume)
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    clipped_cells = int(np.count_nonzero((raw < lower) | (raw > upper)))
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new, name="sw", unit="fraction")
    zeros_x = np.zeros((grid.nz, 1, 2), dtype=float)
    zeros_y = np.zeros((grid.nz, 2, 1), dtype=float)
    balance = compute_saturation_material_balance_3d(
        grid=grid,
        sw_old=sw_values,
        sw_new=new,
        phi=phi_values,
        water_flux_x=zeros_x,
        water_flux_y=zeros_y,
        water_flux_z=water_flux,
        dt=float(dt),
    )
    report: dict[str, float | bool | int] = {
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "relative_balance_error": balance["material_balance_error"],
        "clipped_cells": clipped_cells,
        "has_nan": bool(np.isnan(new).any()),
        "has_inf": bool(np.isinf(new).any()),
    }
    return field, report


def advance_saturation_1d_vertical_with_gravity(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    gravity_params: dict[str, float] | None,
    kz: float | ArrayLike | Field3D,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance vertical 1D saturation with optional gravity water flux coupling."""
    _validate_vertical_1d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    total_flux_z = _validate_flux_z_vertical(grid, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_flux_adv, gravity_flux_z, total_water_flux, gravity_report = compute_total_water_flux_1d_vertical(
        grid=grid,
        sw=sw_values,
        flux_z=total_flux_z,
        relperm_params=params,
        gravity_params=gravity_params,
        kz=kz,
    )
    gravity_enabled = bool(gravity_report["enabled"])
    cfl_flux_z = np.abs(total_flux_z) + np.abs(gravity_flux_z) if gravity_enabled else total_flux_z
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=np.zeros((grid.nz, 1, 2), dtype=float),
        flux_y=np.zeros((grid.nz, 2, 1), dtype=float),
        flux_z=cfl_flux_z,
        dt=dt,
        max_cfl=max_cfl,
    )
    field, update_report = compute_gravity_saturation_update_1d_vertical(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        water_flux_z=total_water_flux,
        dt=dt,
        relperm_params=params,
    )
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(update_report["sw_min"]),
        "sw_max": float(update_report["sw_max"]),
        "water_cut": 0.0,
        "injected_water_volume": float(update_report["injected_water_volume"]),
        "produced_water_volume": float(update_report["produced_water_volume"]),
        "storage_change": float(update_report["storage_change"]),
        "material_balance_error": float(update_report["material_balance_error"]),
        "relative_balance_error": float(update_report["relative_balance_error"]),
        "clipped_cells": int(update_report["clipped_cells"]),
        "has_nan": bool(update_report["has_nan"]),
        "has_inf": bool(update_report["has_inf"]),
        "gravity_enabled": gravity_enabled,
        "gravity_flux_included": gravity_enabled,
        "rho_w": float(gravity_report["rho_w"]),
        "rho_o": float(gravity_report["rho_o"]),
        "density_difference": float(gravity_report["density_difference"]),
        "max_abs_gravity_flux": float(gravity_report["max_abs_gravity_flux"]),
        "max_advective_flux": float(np.max(np.abs(water_flux_adv))),
        "max_gravity_flux": float(np.max(np.abs(gravity_flux_z))),
        "max_total_water_flux": float(np.max(np.abs(total_water_flux))),
    }
    return SaturationStepResult(sw=field, report=report)


def compute_water_cut_1d(
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    relperm_params: dict[str, float],
    producer_boundary: str = "right",
) -> float:
    """Compute producer boundary water cut from upwind water flux."""
    sw_line = _sw_line_from_value(sw)
    total_flux = _flux_x_line(flux_x, sw_line.size)
    water_flux = compute_upwind_water_flux_1d(sw_line.reshape(1, 1, sw_line.size), flux_x, relperm_params)[0, 0, :]
    if producer_boundary == "right":
        if total_flux[-1] <= 0.0:
            return 0.0
        return float(np.clip(water_flux[-1] / total_flux[-1], 0.0, 1.0))
    if producer_boundary == "left":
        if total_flux[0] >= 0.0:
            return 0.0
        return float(np.clip(water_flux[0] / total_flux[0], 0.0, 1.0))
    raise ValueError("producer_boundary must be 'left' or 'right'")


def compute_saturation_material_balance_1d(
    grid: Grid3D,
    sw_old: ArrayLike,
    sw_new: ArrayLike,
    phi: ArrayLike,
    water_flux_x: ArrayLike,
    dt: float,
) -> dict[str, float]:
    """Return 1D water material-balance terms for a saturation step."""
    _validate_1d_grid(grid)
    old = np.asarray(sw_old, dtype=float).reshape(grid.nx)
    new = np.asarray(sw_new, dtype=float).reshape(grid.nx)
    phi_line = np.asarray(phi, dtype=float)
    if phi_line.shape == ():
        phi_line = np.full(grid.nx, float(phi_line), dtype=float)
    phi_line = phi_line.reshape(grid.nx)
    water_flux = _flux_x_line(water_flux_x, grid.nx)

    storage_change = float(np.sum((new - old) * phi_line * grid.cell_volume))
    injected_water_volume = float(max(water_flux[0], 0.0) * dt + max(-water_flux[-1], 0.0) * dt)
    produced_water_volume = float(max(water_flux[-1], 0.0) * dt + max(-water_flux[0], 0.0) * dt)
    scale = max(abs(injected_water_volume), abs(produced_water_volume), abs(storage_change), 1.0e-30)
    residual = injected_water_volume - produced_water_volume - storage_change
    error = 0.0 if abs(residual) <= 1.0e-14 else abs(residual) / scale
    return {
        "injected_water_volume": injected_water_volume,
        "produced_water_volume": produced_water_volume,
        "storage_change": storage_change,
        "material_balance_error": float(error),
    }


def compute_upwind_water_flux_3d(
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute 3D upwind water fluxes from total face fluxes."""
    sw_values = _sw_3d_from_value(sw)
    nz, ny, nx = sw_values.shape
    fx, fy, fz = _validate_fluxes_for_shape((nz, ny, nx), flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    injected_sw = float(params.get("injected_sw", 1.0 - params["sor"]))
    fw_cells = np.asarray(_fractional_flow(sw_values, params), dtype=float)
    fw_injected = float(_fractional_flow(injected_sw, params))

    water_x = np.zeros_like(fx, dtype=float)
    water_y = np.zeros_like(fy, dtype=float)
    water_z = np.zeros_like(fz, dtype=float)

    positive = fx[:, :, 0] > 0.0
    negative = fx[:, :, 0] < 0.0
    water_x[:, :, 0] = np.where(positive, fw_injected * fx[:, :, 0], water_x[:, :, 0])
    water_x[:, :, 0] = np.where(negative, fw_cells[:, :, 0] * fx[:, :, 0], water_x[:, :, 0])
    positive = fx[:, :, -1] > 0.0
    negative = fx[:, :, -1] < 0.0
    water_x[:, :, -1] = np.where(positive, fw_cells[:, :, -1] * fx[:, :, -1], water_x[:, :, -1])
    water_x[:, :, -1] = np.where(negative, fw_injected * fx[:, :, -1], water_x[:, :, -1])
    if nx > 1:
        internal = fx[:, :, 1:-1]
        water_x[:, :, 1:-1] = np.where(
            internal > 0.0,
            fw_cells[:, :, :-1] * internal,
            np.where(internal < 0.0, fw_cells[:, :, 1:] * internal, 0.0),
        )

    positive = fy[:, 0, :] > 0.0
    negative = fy[:, 0, :] < 0.0
    water_y[:, 0, :] = np.where(positive, fw_injected * fy[:, 0, :], water_y[:, 0, :])
    water_y[:, 0, :] = np.where(negative, fw_cells[:, 0, :] * fy[:, 0, :], water_y[:, 0, :])
    positive = fy[:, -1, :] > 0.0
    negative = fy[:, -1, :] < 0.0
    water_y[:, -1, :] = np.where(positive, fw_cells[:, -1, :] * fy[:, -1, :], water_y[:, -1, :])
    water_y[:, -1, :] = np.where(negative, fw_injected * fy[:, -1, :], water_y[:, -1, :])
    if ny > 1:
        internal = fy[:, 1:-1, :]
        water_y[:, 1:-1, :] = np.where(
            internal > 0.0,
            fw_cells[:, :-1, :] * internal,
            np.where(internal < 0.0, fw_cells[:, 1:, :] * internal, 0.0),
        )

    positive = fz[0, :, :] > 0.0
    negative = fz[0, :, :] < 0.0
    water_z[0, :, :] = np.where(positive, fw_injected * fz[0, :, :], water_z[0, :, :])
    water_z[0, :, :] = np.where(negative, fw_cells[0, :, :] * fz[0, :, :], water_z[0, :, :])
    positive = fz[-1, :, :] > 0.0
    negative = fz[-1, :, :] < 0.0
    water_z[-1, :, :] = np.where(positive, fw_cells[-1, :, :] * fz[-1, :, :], water_z[-1, :, :])
    water_z[-1, :, :] = np.where(negative, fw_injected * fz[-1, :, :], water_z[-1, :, :])
    if nz > 1:
        internal = fz[1:-1, :, :]
        water_z[1:-1, :, :] = np.where(
            internal > 0.0,
            fw_cells[:-1, :, :] * internal,
            np.where(internal < 0.0, fw_cells[1:, :, :] * internal, 0.0),
        )

    return water_x, water_y, water_z


def advance_saturation_3d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 3D water saturation with explicit upwind fractional flow."""
    _validate_3d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=fx,
        flux_y=fy,
        flux_z=fz,
        dt=dt,
        max_cfl=max_cfl,
    )
    water_x, water_y, water_z = compute_upwind_water_flux_3d(sw_values, fx, fy, fz, params)
    divergence = (
        water_x[:, :, 1:] - water_x[:, :, :-1]
        + water_y[:, 1:, :] - water_y[:, :-1, :]
        + water_z[1:, :, :] - water_z[:-1, :, :]
    )
    raw = sw_values - float(dt) * divergence / (phi_values * grid.cell_volume)
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    clipped_cells = int(np.count_nonzero((raw < lower) | (raw > upper)))
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new, name="sw", unit="fraction")
    balance = compute_saturation_material_balance_3d(
        grid=grid,
        sw_old=sw_values,
        sw_new=new,
        phi=phi_values,
        water_flux_x=water_x,
        water_flux_y=water_y,
        water_flux_z=water_z,
        dt=float(dt),
    )
    water_cut = compute_water_cut_3d(new, fx, fy, fz, params)
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "water_cut": float(water_cut),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "clipped_cells": clipped_cells,
        "has_nan": bool(np.isnan(new).any()),
        "has_inf": bool(np.isinf(new).any()),
    }
    return SaturationStepResult(sw=field, report=report)


def compute_total_water_flux_3d_with_gravity(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
    gravity_params: dict[str, float] | None = None,
    kx: float | ArrayLike | Field3D | None = None,
    ky: float | ArrayLike | Field3D | None = None,
    kz: float | ArrayLike | Field3D | None = None,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    dict[str, object],
]:
    """Return advective, gravity, and total 3D water face fluxes."""
    _validate_3d_grid(grid)
    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_adv = compute_upwind_water_flux_3d(sw, fx, fy, fz, params)

    if gravity_params is None or not bool(gravity_params.get("enabled", False)):
        gravity_flux = (
            np.zeros_like(fx, dtype=float),
            np.zeros_like(fy, dtype=float),
            np.zeros_like(fz, dtype=float),
        )
        gravity_report: dict[str, object] = {
            "enabled": False,
            "g": float((gravity_params or {}).get("g", 9.80665)),
            "rho_w": float((gravity_params or {}).get("rho_w", 1000.0)),
            "rho_o": float((gravity_params or {}).get("rho_o", 800.0)),
            "density_difference": float(
                (gravity_params or {}).get("rho_w", 1000.0) - (gravity_params or {}).get("rho_o", 800.0)
            ),
            "max_abs_gravity_flux": 0.0,
            "min_gravity_flux": 0.0,
            "max_gravity_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }
    else:
        if kx is None or ky is None or kz is None:
            raise ValueError("kx, ky, and kz are required when gravity flux is enabled")
        grav_x, grav_y, grav_z, gravity_report = compute_gravity_fluxes(
            grid=grid,
            sw=sw,
            kx=kx,
            ky=ky,
            kz=kz,
            gravity_params=gravity_params,
            relperm_params=params,
        )
        gravity_flux = (grav_x, grav_y, grav_z)

    total_water = tuple(adv + grav for adv, grav in zip(water_adv, gravity_flux, strict=True))
    return water_adv, gravity_flux, total_water, gravity_report


def compute_gravity_saturation_update_3d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
) -> tuple[Field3D, dict[str, float | bool | int]]:
    """Apply a 3D explicit saturation update using supplied water fluxes."""
    return compute_capillary_saturation_update_3d(
        grid=grid,
        sw=sw,
        phi=phi,
        water_flux_x=water_flux_x,
        water_flux_y=water_flux_y,
        water_flux_z=water_flux_z,
        dt=dt,
        relperm_params=relperm_params,
    )


def advance_saturation_3d_with_gravity(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    gravity_params: dict[str, float] | None,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 3D saturation with optional gravity water flux coupling."""
    _validate_3d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_adv, gravity_flux, total_water, gravity_report = compute_total_water_flux_3d_with_gravity(
        grid=grid,
        sw=sw_values,
        flux_x=fx,
        flux_y=fy,
        flux_z=fz,
        relperm_params=params,
        gravity_params=gravity_params,
        kx=kx,
        ky=ky,
        kz=kz,
    )
    gravity_enabled = bool(gravity_report["enabled"])
    grav_x, grav_y, grav_z = gravity_flux
    cfl_fx = np.abs(fx) + np.abs(grav_x) if gravity_enabled else fx
    cfl_fy = np.abs(fy) + np.abs(grav_y) if gravity_enabled else fy
    cfl_fz = np.abs(fz) + np.abs(grav_z) if gravity_enabled else fz
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=cfl_fx,
        flux_y=cfl_fy,
        flux_z=cfl_fz,
        dt=dt,
        max_cfl=max_cfl,
    )
    field, update_report = compute_gravity_saturation_update_3d(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        water_flux_x=total_water[0],
        water_flux_y=total_water[1],
        water_flux_z=total_water[2],
        dt=dt,
        relperm_params=params,
    )
    water_cut = compute_water_cut_3d(field.values, fx, fy, fz, params)
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(update_report["sw_min"]),
        "sw_max": float(update_report["sw_max"]),
        "water_cut": float(water_cut),
        "injected_water_volume": float(update_report["injected_water_volume"]),
        "produced_water_volume": float(update_report["produced_water_volume"]),
        "storage_change": float(update_report["storage_change"]),
        "material_balance_error": float(update_report["material_balance_error"]),
        "relative_balance_error": float(update_report["relative_balance_error"]),
        "clipped_cells": int(update_report["clipped_cells"]),
        "has_nan": bool(update_report["has_nan"]),
        "has_inf": bool(update_report["has_inf"]),
        "gravity_enabled": gravity_enabled,
        "gravity_flux_included": gravity_enabled,
        "rho_w": float(gravity_report["rho_w"]),
        "rho_o": float(gravity_report["rho_o"]),
        "density_difference": float(gravity_report["density_difference"]),
        "max_abs_gravity_flux": float(gravity_report["max_abs_gravity_flux"]),
        "max_advective_flux": _max_abs_arrays(*water_adv),
        "max_gravity_flux": _max_abs_arrays(*gravity_flux),
        "max_total_water_flux": _max_abs_arrays(*total_water),
    }
    return SaturationStepResult(sw=field, report=report)


def compute_combined_saturation_update_3d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
) -> tuple[Field3D, dict[str, float | bool | int]]:
    """Apply a 3D explicit saturation update from composed water fluxes."""
    return compute_capillary_saturation_update_3d(
        grid=grid,
        sw=sw,
        phi=phi,
        water_flux_x=water_flux_x,
        water_flux_y=water_flux_y,
        water_flux_z=water_flux_z,
        dt=dt,
        relperm_params=relperm_params,
    )


def build_combined_transport_report(
    *,
    dt: float,
    cfl_report: dict[str, object],
    update_report: dict[str, object],
    water_cut: float,
    capillary_enabled: bool,
    gravity_enabled: bool,
    capillary_report: dict[str, object],
    gravity_report: dict[str, object],
    composer_report: dict[str, object],
) -> dict[str, object]:
    """Merge saturation, capillary, gravity, and composer reports."""
    return {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(update_report["sw_min"]),
        "sw_max": float(update_report["sw_max"]),
        "water_cut": float(water_cut),
        "injected_water_volume": float(update_report["injected_water_volume"]),
        "produced_water_volume": float(update_report["produced_water_volume"]),
        "storage_change": float(update_report["storage_change"]),
        "material_balance_error": float(update_report["material_balance_error"]),
        "relative_balance_error": float(update_report["relative_balance_error"]),
        "clipped_cells": int(update_report["clipped_cells"]),
        "has_nan": bool(update_report["has_nan"]) or bool(composer_report["has_nan"]),
        "has_inf": bool(update_report["has_inf"]) or bool(composer_report["has_inf"]),
        "capillary_enabled": bool(capillary_enabled),
        "gravity_enabled": bool(gravity_enabled),
        "capillary_model": str(capillary_report.get("model", "none")),
        "rho_w": float(gravity_report.get("rho_w", 1000.0)),
        "rho_o": float(gravity_report.get("rho_o", 800.0)),
        "density_difference": float(gravity_report.get("density_difference", 200.0)),
        "max_abs_capillary_flux": float(capillary_report.get("max_abs_capillary_flux", 0.0)),
        "max_abs_gravity_flux": float(gravity_report.get("max_abs_gravity_flux", 0.0)),
        "max_advective_flux": float(composer_report["max_advective_flux"]),
        "max_capillary_flux": float(composer_report["max_capillary_flux"]),
        "max_gravity_flux": float(composer_report["max_gravity_flux"]),
        "max_total_water_flux": float(composer_report["max_total_water_flux"]),
        "max_effective_flux": float(composer_report["max_effective_flux"]),
        "capillary_flux_included": bool(capillary_enabled),
        "gravity_flux_included": bool(gravity_enabled),
        "composer_report": composer_report,
    }


def advance_saturation_3d_with_capillary_and_gravity(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    capillary_params: dict[str, float] | None,
    gravity_params: dict[str, float] | None,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 3D saturation with optional combined capillary and gravity fluxes."""
    _validate_3d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_adv_x, water_adv_y, water_adv_z = compute_upwind_water_flux_3d(sw_values, fx, fy, fz, params)

    capillary_enabled = bool(capillary_params and capillary_params.get("enabled", False) and capillary_params.get("model", "none") != "none")
    if capillary_enabled:
        cap_x, cap_y, cap_z, cap_report = compute_capillary_fluxes(
            grid=grid,
            sw=sw_values,
            kx=kx,
            ky=ky,
            kz=kz,
            capillary_params=capillary_params,
            relperm_params=params,
        )
    else:
        cap_x = np.zeros_like(fx, dtype=float)
        cap_y = np.zeros_like(fy, dtype=float)
        cap_z = np.zeros_like(fz, dtype=float)
        cap_report: dict[str, object] = {
            "enabled": False,
            "model": "none",
            "max_abs_capillary_flux": 0.0,
            "min_capillary_flux": 0.0,
            "max_capillary_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }

    gravity_enabled = bool(gravity_params and gravity_params.get("enabled", False))
    if gravity_enabled:
        grav_x, grav_y, grav_z, gravity_report = compute_gravity_fluxes(
            grid=grid,
            sw=sw_values,
            kx=kx,
            ky=ky,
            kz=kz,
            gravity_params=gravity_params,
            relperm_params=params,
        )
    else:
        grav_x = np.zeros_like(fx, dtype=float)
        grav_y = np.zeros_like(fy, dtype=float)
        grav_z = np.zeros_like(fz, dtype=float)
        gravity_report: dict[str, object] = {
            "enabled": False,
            "g": float((gravity_params or {}).get("g", 9.80665)),
            "rho_w": float((gravity_params or {}).get("rho_w", 1000.0)),
            "rho_o": float((gravity_params or {}).get("rho_o", 800.0)),
            "density_difference": float(
                (gravity_params or {}).get("rho_w", 1000.0) - (gravity_params or {}).get("rho_o", 800.0)
            ),
            "max_abs_gravity_flux": 0.0,
            "min_gravity_flux": 0.0,
            "max_gravity_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }

    water_x, water_y, water_z, composer_report = compose_water_fluxes_3d(
        adv_flux_x=water_adv_x,
        adv_flux_y=water_adv_y,
        adv_flux_z=water_adv_z,
        cap_flux_x=cap_x,
        cap_flux_y=cap_y,
        cap_flux_z=cap_z,
        grav_flux_x=grav_x,
        grav_flux_y=grav_y,
        grav_flux_z=grav_z,
        include_capillary=capillary_enabled,
        include_gravity=gravity_enabled,
    )
    eff_x, eff_y, eff_z = compute_effective_flux_for_cfl(
        adv_flux_x=water_adv_x,
        adv_flux_y=water_adv_y,
        adv_flux_z=water_adv_z,
        cap_flux_x=cap_x,
        cap_flux_y=cap_y,
        cap_flux_z=cap_z,
        grav_flux_x=grav_x,
        grav_flux_y=grav_y,
        grav_flux_z=grav_z,
        include_capillary=capillary_enabled,
        include_gravity=gravity_enabled,
    )
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=eff_x,
        flux_y=eff_y,
        flux_z=eff_z,
        dt=dt,
        max_cfl=max_cfl,
    )
    field, update_report = compute_combined_saturation_update_3d(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        water_flux_x=water_x,
        water_flux_y=water_y,
        water_flux_z=water_z,
        dt=dt,
        relperm_params=params,
    )
    water_cut = compute_water_cut_3d(field.values, fx, fy, fz, params)
    report = build_combined_transport_report(
        dt=dt,
        cfl_report=cfl_report,
        update_report=update_report,
        water_cut=water_cut,
        capillary_enabled=capillary_enabled,
        gravity_enabled=gravity_enabled,
        capillary_report=cap_report,
        gravity_report=gravity_report,
        composer_report=composer_report,
    )
    return SaturationStepResult(sw=field, report=report)


def compute_total_water_flux_3d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
    capillary_params: dict[str, float] | None = None,
    kx: float | ArrayLike | Field3D | None = None,
    ky: float | ArrayLike | Field3D | None = None,
    kz: float | ArrayLike | Field3D | None = None,
) -> tuple[
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    dict[str, object],
]:
    """Return advective, capillary, and total 3D water face fluxes."""
    _validate_3d_grid(grid)
    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_adv = compute_upwind_water_flux_3d(sw, fx, fy, fz, params)

    if capillary_params is None or not bool(capillary_params.get("enabled", False)) or capillary_params.get("model", "none") == "none":
        cap_flux = (
            np.zeros_like(fx, dtype=float),
            np.zeros_like(fy, dtype=float),
            np.zeros_like(fz, dtype=float),
        )
        cap_report: dict[str, object] = {
            "enabled": False,
            "model": "none",
            "max_abs_capillary_flux": 0.0,
            "min_capillary_flux": 0.0,
            "max_capillary_flux": 0.0,
            "has_nan": False,
            "has_inf": False,
        }
    else:
        if kx is None or ky is None or kz is None:
            raise ValueError("kx, ky, and kz are required when capillary flux is enabled")
        cap_x, cap_y, cap_z, cap_report = compute_capillary_fluxes(
            grid=grid,
            sw=sw,
            kx=kx,
            ky=ky,
            kz=kz,
            capillary_params=capillary_params,
            relperm_params=params,
        )
        cap_flux = (cap_x, cap_y, cap_z)

    total_water = tuple(adv + cap for adv, cap in zip(water_adv, cap_flux, strict=True))
    return water_adv, cap_flux, total_water, cap_report


def compute_capillary_saturation_update_3d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
) -> tuple[Field3D, dict[str, float | bool | int]]:
    """Apply a 3D explicit saturation update using supplied water fluxes."""
    _validate_3d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    water_x, water_y, water_z = _validate_fluxes_for_shape(grid.shape, water_flux_x, water_flux_y, water_flux_z)
    params = _normalized_relperm_params(relperm_params)
    divergence = (
        water_x[:, :, 1:] - water_x[:, :, :-1]
        + water_y[:, 1:, :] - water_y[:, :-1, :]
        + water_z[1:, :, :] - water_z[:-1, :, :]
    )
    raw = sw_values - float(dt) * divergence / (phi_values * grid.cell_volume)
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    clipped_cells = int(np.count_nonzero((raw < lower) | (raw > upper)))
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new, name="sw", unit="fraction")
    balance = compute_saturation_material_balance_3d(
        grid=grid,
        sw_old=sw_values,
        sw_new=new,
        phi=phi_values,
        water_flux_x=water_x,
        water_flux_y=water_y,
        water_flux_z=water_z,
        dt=float(dt),
    )
    report: dict[str, float | bool | int] = {
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "relative_balance_error": balance["material_balance_error"],
        "clipped_cells": clipped_cells,
        "has_nan": bool(np.isnan(new).any()),
        "has_inf": bool(np.isinf(new).any()),
    }
    return field, report


def advance_saturation_3d_with_capillary(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    capillary_params: dict[str, float] | None,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance 3D saturation with optional capillary water flux coupling."""
    _validate_3d_grid(grid)
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, phi, "phi")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    if np.isnan(phi_values).any() or np.isinf(phi_values).any() or (phi_values <= 0.0).any():
        raise InvalidPhysicalValueError("phi must be positive and finite")

    fx, fy, fz = _validate_fluxes_for_shape(grid.shape, flux_x, flux_y, flux_z)
    params = _normalized_relperm_params(relperm_params)
    water_adv, cap_flux, total_water, cap_report = compute_total_water_flux_3d(
        grid=grid,
        sw=sw_values,
        flux_x=fx,
        flux_y=fy,
        flux_z=fz,
        relperm_params=params,
        capillary_params=capillary_params,
        kx=kx,
        ky=ky,
        kz=kz,
    )
    capillary_enabled = bool(cap_report["enabled"])
    cap_x, cap_y, cap_z = cap_flux
    cfl_fx = np.abs(fx) + np.abs(cap_x) if capillary_enabled else fx
    cfl_fy = np.abs(fy) + np.abs(cap_y) if capillary_enabled else fy
    cfl_fz = np.abs(fz) + np.abs(cap_z) if capillary_enabled else fz
    cfl_report = check_cfl_condition(
        grid=grid,
        phi=phi_values,
        flux_x=cfl_fx,
        flux_y=cfl_fy,
        flux_z=cfl_fz,
        dt=dt,
        max_cfl=max_cfl,
    )
    field, update_report = compute_capillary_saturation_update_3d(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        water_flux_x=total_water[0],
        water_flux_y=total_water[1],
        water_flux_z=total_water[2],
        dt=dt,
        relperm_params=params,
    )
    water_cut = compute_water_cut_3d(field.values, fx, fy, fz, params)
    report: dict[str, float | str | bool | int] = {
        "dt": float(dt),
        "max_cfl": float(cfl_report["max_cfl"]),
        "stable": bool(cfl_report["stable"]),
        "sw_min": float(update_report["sw_min"]),
        "sw_max": float(update_report["sw_max"]),
        "water_cut": float(water_cut),
        "injected_water_volume": float(update_report["injected_water_volume"]),
        "produced_water_volume": float(update_report["produced_water_volume"]),
        "storage_change": float(update_report["storage_change"]),
        "material_balance_error": float(update_report["material_balance_error"]),
        "relative_balance_error": float(update_report["relative_balance_error"]),
        "clipped_cells": int(update_report["clipped_cells"]),
        "has_nan": bool(update_report["has_nan"]),
        "has_inf": bool(update_report["has_inf"]),
        "capillary_enabled": capillary_enabled,
        "capillary_model": str(cap_report["model"]),
        "max_abs_capillary_flux": float(cap_report["max_abs_capillary_flux"]),
        "max_advective_flux": _max_abs_arrays(*water_adv),
        "max_capillary_flux": _max_abs_arrays(*cap_flux),
        "max_total_water_flux": _max_abs_arrays(*total_water),
        "capillary_flux_included": capillary_enabled,
    }
    return SaturationStepResult(sw=field, report=report)


def compute_water_cut_3d(
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
    producer_faces: list[tuple[str, int, int, int]] | None = None,
) -> float:
    """Compute aggregate produced water cut from boundary outflow faces."""
    sw_values = _sw_3d_from_value(sw)
    water_x, water_y, water_z = compute_upwind_water_flux_3d(sw_values, flux_x, flux_y, flux_z, relperm_params)
    fx, fy, fz = _validate_fluxes_for_shape(sw_values.shape, flux_x, flux_y, flux_z)
    if producer_faces is not None:
        water_total = 0.0
        total = 0.0
        for direction, a, b, c in producer_faces:
            if direction == "x":
                face_total = fx[a, b, c]
                face_water = water_x[a, b, c]
            elif direction == "y":
                face_total = fy[a, b, c]
                face_water = water_y[a, b, c]
            elif direction == "z":
                face_total = fz[a, b, c]
                face_water = water_z[a, b, c]
            else:
                raise ValueError("producer face direction must be x, y, or z")
            if face_total > 0.0:
                total += face_total
                water_total += face_water
        return 0.0 if total == 0.0 else float(np.clip(water_total / total, 0.0, 1.0))

    total_out = (
        np.sum(np.maximum(-fx[:, :, 0], 0.0))
        + np.sum(np.maximum(fx[:, :, -1], 0.0))
        + np.sum(np.maximum(-fy[:, 0, :], 0.0))
        + np.sum(np.maximum(fy[:, -1, :], 0.0))
        + np.sum(np.maximum(-fz[0, :, :], 0.0))
        + np.sum(np.maximum(fz[-1, :, :], 0.0))
    )
    water_out = (
        np.sum(np.where(fx[:, :, 0] < 0.0, -water_x[:, :, 0], 0.0))
        + np.sum(np.where(fx[:, :, -1] > 0.0, water_x[:, :, -1], 0.0))
        + np.sum(np.where(fy[:, 0, :] < 0.0, -water_y[:, 0, :], 0.0))
        + np.sum(np.where(fy[:, -1, :] > 0.0, water_y[:, -1, :], 0.0))
        + np.sum(np.where(fz[0, :, :] < 0.0, -water_z[0, :, :], 0.0))
        + np.sum(np.where(fz[-1, :, :] > 0.0, water_z[-1, :, :], 0.0))
    )
    return 0.0 if total_out == 0.0 else float(np.clip(water_out / total_out, 0.0, 1.0))


def compute_saturation_material_balance_3d(
    grid: Grid3D,
    sw_old: ArrayLike,
    sw_new: ArrayLike,
    phi: ArrayLike,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    dt: float,
) -> dict[str, float]:
    """Return 3D water material-balance terms for a saturation step."""
    old = np.asarray(sw_old, dtype=float).reshape(grid.shape)
    new = np.asarray(sw_new, dtype=float).reshape(grid.shape)
    phi_values = np.asarray(phi, dtype=float)
    if phi_values.shape == ():
        phi_values = np.full(grid.shape, float(phi_values), dtype=float)
    phi_values = phi_values.reshape(grid.shape)
    wx, wy, wz = _validate_fluxes_for_shape(grid.shape, water_flux_x, water_flux_y, water_flux_z)
    storage_change = float(np.sum((new - old) * phi_values * grid.cell_volume))
    injected = float(
        (
            np.sum(np.maximum(wx[:, :, 0], 0.0))
            + np.sum(np.maximum(-wx[:, :, -1], 0.0))
            + np.sum(np.maximum(wy[:, 0, :], 0.0))
            + np.sum(np.maximum(-wy[:, -1, :], 0.0))
            + np.sum(np.maximum(wz[0, :, :], 0.0))
            + np.sum(np.maximum(-wz[-1, :, :], 0.0))
        )
        * dt
    )
    produced = float(
        (
            np.sum(np.maximum(-wx[:, :, 0], 0.0))
            + np.sum(np.maximum(wx[:, :, -1], 0.0))
            + np.sum(np.maximum(-wy[:, 0, :], 0.0))
            + np.sum(np.maximum(wy[:, -1, :], 0.0))
            + np.sum(np.maximum(-wz[0, :, :], 0.0))
            + np.sum(np.maximum(wz[-1, :, :], 0.0))
        )
        * dt
    )
    scale = max(abs(injected), abs(produced), abs(storage_change), 1.0e-30)
    residual = injected - produced - storage_change
    error = 0.0 if abs(residual) <= 1.0e-14 else abs(residual) / scale
    return {
        "injected_water_volume": injected,
        "produced_water_volume": produced,
        "storage_change": storage_change,
        "material_balance_error": float(error),
    }


def advance_buckley_leverett_1d(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    velocity_x: float | ArrayLike,
    phi: float | ArrayLike | Field3D,
    dt: float,
    *,
    injected_sw: float = 1.0,
    swi: float = 0.2,
    sor: float = 0.2,
    mu_w: float = 1.0e-3,
    mu_o: float = 5.0e-3,
    krw0: float = 1.0,
    kro0: float = 1.0,
    nw: float = 2.0,
    no: float = 2.0,
    max_cfl: float = 1.0,
) -> SaturationStepResult:
    """Advance a 1D Buckley-Leverett water saturation field by one explicit step."""
    velocity_faces = _velocity_faces(grid, velocity_x)
    area = float(grid.spacing_j[0] * grid.spacing_k[0])
    result = advance_saturation_1d(
        grid=grid,
        sw=sw,
        phi=phi,
        flux_x=velocity_faces.reshape(1, 1, grid.nx + 1) * area,
        dt=dt,
        relperm_params={
            "swi": swi,
            "sor": sor,
            "krw0": krw0,
            "kro0": kro0,
            "nw": nw,
            "no": no,
            "mu_w": mu_w,
            "mu_o": mu_o,
            "injected_sw": injected_sw,
        },
        max_cfl=max_cfl,
    )
    result.report["status"] = "advanced"
    result.report["min_sw"] = result.report["sw_min"]
    result.report["max_sw"] = result.report["sw_max"]
    return result


def effective_saturation(sw: ArrayLike, swi: float, sor: float) -> NDArray[np.float64]:
    """Return clipped effective water saturation."""
    _validate_residual_saturation(swi, sor)
    return np.clip((np.asarray(sw, dtype=float) - swi) / (1.0 - swi - sor), 0.0, 1.0)


def fractional_flow(
    sw: ArrayLike,
    swi: float,
    sor: float,
    mu_w: float,
    mu_o: float,
    krw0: float,
    kro0: float,
    nw: float,
    no: float,
) -> NDArray[np.float64]:
    """Return Corey water fractional flow."""
    for name, value in {
        "mu_w": mu_w,
        "mu_o": mu_o,
        "krw0": krw0,
        "kro0": kro0,
        "nw": nw,
        "no": no,
    }.items():
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise InvalidPhysicalValueError(f"{name} must be positive and finite")

    se = effective_saturation(sw, swi, sor)
    krw = krw0 * se**nw
    kro = kro0 * (1.0 - se) ** no
    lambda_w = krw / mu_w
    lambda_o = kro / mu_o
    denom = lambda_w + lambda_o
    return np.divide(lambda_w, denom, out=np.zeros_like(lambda_w, dtype=float), where=denom > 0.0)


def _validate_1d_grid(grid: Grid3D) -> None:
    if grid.ny != 1 or grid.nz != 1:
        raise NotImplementedError("1D saturation transport supports only ny=1, nz=1")


def _validate_vertical_1d_grid(grid: Grid3D) -> None:
    if grid.nx != 1 or grid.ny != 1 or grid.nz <= 1:
        raise NotImplementedError("vertical 1D saturation transport requires nx=1, ny=1, nz>1")


def _validate_3d_grid(grid: Grid3D) -> None:
    if grid.nx <= 1 or grid.ny <= 1 or grid.nz <= 1:
        raise NotImplementedError("3D saturation transport supports nx>1, ny>1, nz>1")


def _normalized_relperm_params(relperm_params: dict[str, float]) -> dict[str, float]:
    params = dict(DEFAULT_RELPERM_PARAMS)
    params.update(relperm_params)
    validate_saturation_params(params["swi"], params["sor"])
    # Trigger full relperm parameter validation without duplicating relperm.py.
    _fractional_flow(params["swi"], params)
    return params


def _fractional_flow(sw: ArrayLike, params: dict[str, float]) -> float | NDArray[np.float64]:
    return fractional_flow_water(
        sw=sw,
        swi=params["swi"],
        sor=params["sor"],
        krw0=params["krw0"],
        kro0=params["kro0"],
        nw=params["nw"],
        no=params["no"],
        mu_w=params["mu_w"],
        mu_o=params["mu_o"],
    )


def _validate_flux_x(grid: Grid3D, flux_x: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(flux_x, dtype=float)
    expected = (1, 1, grid.nx + 1)
    if values.shape != expected:
        raise FieldShapeError(f"flux_x shape {values.shape} does not match {expected}")
    if np.isnan(values).any() or np.isinf(values).any():
        raise InvalidPhysicalValueError("flux_x must be finite")
    return values


def _validate_flux_z_vertical(grid: Grid3D, flux_z: ArrayLike) -> NDArray[np.float64]:
    return _validate_flux_z_vertical_shape(grid.shape, flux_z)


def _validate_flux_z_vertical_shape(
    shape: tuple[int, int, int],
    flux_z: ArrayLike,
) -> NDArray[np.float64]:
    nz, ny, nx = shape
    values = np.asarray(flux_z, dtype=float)
    expected = (nz + 1, ny, nx)
    if values.shape != expected:
        raise FieldShapeError(f"flux_z shape {values.shape} does not match {expected}")
    if np.isnan(values).any() or np.isinf(values).any():
        raise InvalidPhysicalValueError("flux_z must be finite")
    return values


def _flux_x_line(flux_x: ArrayLike, nx: int) -> NDArray[np.float64]:
    values = np.asarray(flux_x, dtype=float)
    if values.shape == (1, 1, nx + 1):
        return values[0, 0, :].copy()
    if values.shape == (nx + 1,):
        return values.copy()
    raise FieldShapeError(f"flux_x must have shape {(1, 1, nx + 1)} or {(nx + 1,)}")


def _sw_line_from_value(sw: Field3D | ArrayLike) -> NDArray[np.float64]:
    values = sw.values if isinstance(sw, Field3D) else sw
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[1] == 1:
        line = array[0, 0, :]
    elif array.ndim == 1:
        line = array
    elif array.shape == ():
        line = np.asarray([float(array)], dtype=float)
    else:
        raise FieldShapeError("sw must be scalar, 1D array, or shape (1, 1, nx)")
    if np.isnan(line).any() or np.isinf(line).any():
        raise InvalidPhysicalValueError("sw must be finite")
    return line.astype(float, copy=True)


def _sw_3d_from_value(sw: Field3D | ArrayLike) -> NDArray[np.float64]:
    values = sw.values if isinstance(sw, Field3D) else sw
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise FieldShapeError("sw must have shape (nz, ny, nx)")
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError("sw must be finite")
    return array.astype(float, copy=True)


def _validate_fluxes_for_shape(
    shape: tuple[int, int, int],
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    nz, ny, nx = shape
    fx = np.asarray(flux_x, dtype=float)
    fy = np.asarray(flux_y, dtype=float)
    fz = np.asarray(flux_z, dtype=float)
    expected_x = (nz, ny, nx + 1)
    expected_y = (nz, ny + 1, nx)
    expected_z = (nz + 1, ny, nx)
    if fx.shape != expected_x:
        raise FieldShapeError(f"flux_x shape {fx.shape} does not match {expected_x}")
    if fy.shape != expected_y:
        raise FieldShapeError(f"flux_y shape {fy.shape} does not match {expected_y}")
    if fz.shape != expected_z:
        raise FieldShapeError(f"flux_z shape {fz.shape} does not match {expected_z}")
    if np.isnan(fx).any() or np.isnan(fy).any() or np.isnan(fz).any():
        raise InvalidPhysicalValueError("flux arrays must be finite")
    if np.isinf(fx).any() or np.isinf(fy).any() or np.isinf(fz).any():
        raise InvalidPhysicalValueError("flux arrays must be finite")
    return fx, fy, fz


def _validate_residual_saturation(swi: float, sor: float) -> None:
    if not np.isfinite(swi) or not np.isfinite(sor) or swi < 0.0 or sor < 0.0 or swi + sor >= 1.0:
        raise InvalidPhysicalValueError("swi and sor must be non-negative and sum to less than 1")


def _field_values(grid: Grid3D, value: Field3D | ArrayLike, name: str) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        if value.grid != grid:
            raise GridMismatchError(f"{name} Field3D is defined on a different grid")
        return value.values.astype(float, copy=True)
    values = np.asarray(value, dtype=float)
    if values.shape == ():
        return np.full(grid.shape, float(values), dtype=float)
    if values.shape != grid.shape:
        raise FieldShapeError(f"{name} shape {values.shape} does not match grid shape {grid.shape}")
    return values.copy()


def _max_abs_arrays(*arrays: ArrayLike) -> float:
    return float(max(np.max(np.abs(np.asarray(array, dtype=float))) for array in arrays))


def _velocity_faces(grid: Grid3D, velocity_x: float | ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(velocity_x, dtype=float)
    if values.shape == ():
        faces = np.full(grid.nx + 1, float(values), dtype=float)
    elif values.shape == (grid.nx + 1,):
        faces = values.astype(float, copy=True)
    else:
        raise FieldShapeError(f"velocity_x must be scalar or shape {(grid.nx + 1,)}")
    if np.isnan(faces).any() or np.isinf(faces).any():
        raise InvalidPhysicalValueError("velocity_x must be finite")
    return faces
