"""Explicit transport for simplified incompressible three-phase flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import CFLViolationError, FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.solver.three_phase_flux import compute_three_phase_flux_1d, compute_three_phase_fluxes_3d
from reservoir_backend.solver.three_phase_relperm import (
    compute_oil_saturation,
    fractional_flow_three_phase,
    validate_three_phase_params,
    validate_three_phase_saturations,
)


def validate_three_phase_transport_1d_inputs(
    flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
    max_cfl: float = 1.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate 1D three-phase transport inputs and return copied arrays."""
    validate_three_phase_params(params)
    fx = _finite_array(flux_x, "flux_x").copy()
    sw_values = _finite_array(sw, "sw").copy()
    sg_values = _finite_array(sg, "sg").copy()
    if sw_values.ndim != 1 or sg_values.ndim != 1:
        raise FieldShapeError("sw and sg must be 1D arrays")
    if sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg shapes must match")
    if fx.shape != (sw_values.shape[0] + 1,):
        raise FieldShapeError(f"flux_x shape {fx.shape} must be {(sw_values.shape[0] + 1,)}")
    phi_values = _broadcast_positive(phi, sw_values.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw_values.shape, "cell_volume")
    dt_value = float(dt)
    max_cfl_value = float(max_cfl)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    if not np.isfinite(max_cfl_value) or max_cfl_value <= 0.0:
        raise InvalidPhysicalValueError("max_cfl must be positive and finite")
    validate_three_phase_saturations(sw_values, sg_values, params)
    return fx, sw_values, sg_values, phi_values, volume_values


def compute_three_phase_cfl_1d(
    flux_x: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
) -> tuple[NDArray[np.float64], float]:
    """Compute conservative 1D CFL using `abs(qt)` on adjacent faces."""
    fx = _finite_array(flux_x, "flux_x")
    if fx.ndim != 1 or fx.shape[0] < 2:
        raise FieldShapeError("flux_x must have shape (nx + 1,)")
    nx = fx.shape[0] - 1
    phi_values = _broadcast_positive(phi, (nx,), "phi")
    volume_values = _broadcast_positive(cell_volume, (nx,), "cell_volume")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    cfl = dt_value / (phi_values * volume_values) * (np.abs(fx[:-1]) + np.abs(fx[1:]))
    return cfl, float(np.max(cfl))


def compute_three_phase_saturation_update_1d(
    water_flux_x: ArrayLike,
    gas_flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Update `Sw` and `Sg` explicitly from water and gas phase fluxes."""
    validate_three_phase_params(params)
    water_flux = _finite_array(water_flux_x, "water_flux_x")
    gas_flux = _finite_array(gas_flux_x, "gas_flux_x")
    sw_values = _finite_array(sw, "sw").copy()
    sg_values = _finite_array(sg, "sg").copy()
    if sw_values.ndim != 1 or sg_values.ndim != 1 or sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg must be matching 1D arrays")
    if water_flux.shape != (sw_values.shape[0] + 1,) or gas_flux.shape != (sw_values.shape[0] + 1,):
        raise FieldShapeError("phase fluxes must have shape (nx + 1,)")
    phi_values = _broadcast_positive(phi, sw_values.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw_values.shape, "cell_volume")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    validate_three_phase_saturations(sw_values, sg_values, params)
    pore_volume = phi_values * volume_values
    sw_new = sw_values - dt_value / pore_volume * (water_flux[1:] - water_flux[:-1])
    sg_new = sg_values - dt_value / pore_volume * (gas_flux[1:] - gas_flux[:-1])
    validate_three_phase_saturations(sw_new, sg_new, params)
    so_new = np.asarray(compute_oil_saturation(sw_new, sg_new), dtype=float)
    return sw_new, sg_new, so_new


def compute_three_phase_material_balance_1d(
    water_flux_x: ArrayLike,
    oil_flux_x: ArrayLike,
    gas_flux_x: ArrayLike,
    sw_old: ArrayLike,
    sg_old: ArrayLike,
    sw_new: ArrayLike,
    sg_new: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
) -> dict[str, float]:
    """Compute water, gas, and oil material balance from boundary fluxes."""
    validate_three_phase_params(params)
    water_flux = _finite_array(water_flux_x, "water_flux_x")
    oil_flux = _finite_array(oil_flux_x, "oil_flux_x")
    gas_flux = _finite_array(gas_flux_x, "gas_flux_x")
    sw0 = _finite_array(sw_old, "sw_old")
    sg0 = _finite_array(sg_old, "sg_old")
    sw1 = _finite_array(sw_new, "sw_new")
    sg1 = _finite_array(sg_new, "sg_new")
    if sw0.shape != sg0.shape or sw0.shape != sw1.shape or sw0.shape != sg1.shape:
        raise FieldShapeError("old and new saturation arrays must have matching shapes")
    phi_values = _broadcast_positive(phi, sw0.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw0.shape, "cell_volume")
    dt_value = float(dt)
    pore_volume = phi_values * volume_values
    so0 = np.asarray(compute_oil_saturation(sw0, sg0), dtype=float)
    so1 = np.asarray(compute_oil_saturation(sw1, sg1), dtype=float)
    report: dict[str, float] = {}
    _add_phase_balance(report, "water", water_flux, sw0, sw1, pore_volume, dt_value)
    _add_phase_balance(report, "gas", gas_flux, sg0, sg1, pore_volume, dt_value)
    _add_phase_balance(report, "oil", oil_flux, so0, so1, pore_volume, dt_value)
    report["closure_error_max"] = float(np.max(np.abs(sw1 + so1 + sg1 - 1.0)))
    return report


def advance_three_phase_saturation_1d(
    flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
    max_cfl: float = 1.0,
    injected_sw: float | None = None,
    injected_sg: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Advance 1D three-phase saturation by one explicit step."""
    fx, sw_values, sg_values, phi_values, volume_values = validate_three_phase_transport_1d_inputs(
        flux_x, sw, sg, phi, cell_volume, dt, params, max_cfl=max_cfl
    )
    cfl_array, max_cfl_value = compute_three_phase_cfl_1d(fx, phi_values, volume_values, dt)
    if max_cfl_value > float(max_cfl):
        raise CFLViolationError(f"three-phase CFL violation: max_cfl={max_cfl_value}, max_cfl_allowed={max_cfl}")
    water_flux, oil_flux, gas_flux, flux_report = compute_three_phase_flux_1d(fx, sw_values, sg_values, params)
    water_flux = water_flux.copy()
    oil_flux = oil_flux.copy()
    gas_flux = gas_flux.copy()
    _apply_injection_boundary_fluxes(
        fx=fx,
        water_flux=water_flux,
        oil_flux=oil_flux,
        gas_flux=gas_flux,
        params=params,
        injected_sw=injected_sw,
        injected_sg=injected_sg,
    )
    sw_new, sg_new, so_new = compute_three_phase_saturation_update_1d(
        water_flux, gas_flux, sw_values, sg_values, phi_values, volume_values, dt, params
    )
    balance = compute_three_phase_material_balance_1d(
        water_flux,
        oil_flux,
        gas_flux,
        sw_values,
        sg_values,
        sw_new,
        sg_new,
        phi_values,
        volume_values,
        dt,
        params,
    )
    arrays = [sw_new, sg_new, so_new, water_flux, oil_flux, gas_flux]
    report: dict[str, object] = {
        "max_cfl": max_cfl_value,
        "cfl_min": float(np.min(cfl_array)),
        "cfl_max": float(np.max(cfl_array)),
        **balance,
        "sw_min": float(np.min(sw_new)),
        "sw_max": float(np.max(sw_new)),
        "sg_min": float(np.min(sg_new)),
        "sg_max": float(np.max(sg_new)),
        "so_min": float(np.min(so_new)),
        "so_max": float(np.max(so_new)),
        "has_nan": any(np.isnan(array).any() for array in arrays),
        "has_inf": any(np.isinf(array).any() for array in arrays),
        "transport_dimension": "1d",
        "phase_flux_report": flux_report,
    }
    return sw_new, sg_new, so_new, report


def validate_three_phase_transport_3d_inputs(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
    max_cfl: float = 1.0,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate 3D three-phase transport inputs and return copied arrays."""
    validate_three_phase_params(params)
    sw_values = _finite_array(sw, "sw").copy()
    sg_values = _finite_array(sg, "sg").copy()
    if sw_values.ndim != 3 or sg_values.ndim != 3:
        raise FieldShapeError("sw and sg must be 3D arrays with shape (nz, ny, nx)")
    if sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg shapes must match")
    nz, ny, nx = sw_values.shape
    fx = _finite_array(flux_x, "flux_x").copy()
    fy = _finite_array(flux_y, "flux_y").copy()
    fz = _finite_array(flux_z, "flux_z").copy()
    if fx.shape != (nz, ny, nx + 1):
        raise FieldShapeError(f"flux_x shape {fx.shape} must be {(nz, ny, nx + 1)}")
    if fy.shape != (nz, ny + 1, nx):
        raise FieldShapeError(f"flux_y shape {fy.shape} must be {(nz, ny + 1, nx)}")
    if fz.shape != (nz + 1, ny, nx):
        raise FieldShapeError(f"flux_z shape {fz.shape} must be {(nz + 1, ny, nx)}")
    phi_values = _broadcast_positive(phi, sw_values.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw_values.shape, "cell_volume")
    dt_value = float(dt)
    max_cfl_value = float(max_cfl)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    if not np.isfinite(max_cfl_value) or max_cfl_value <= 0.0:
        raise InvalidPhysicalValueError("max_cfl must be positive and finite")
    validate_three_phase_saturations(sw_values, sg_values, params)
    return fx, fy, fz, sw_values, sg_values, phi_values, volume_values


def compute_three_phase_cfl_3d(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
) -> tuple[NDArray[np.float64], float]:
    """Compute conservative 3D CFL using `abs(qt)` on all connected faces."""
    fx = _finite_array(flux_x, "flux_x")
    fy = _finite_array(flux_y, "flux_y")
    fz = _finite_array(flux_z, "flux_z")
    if fx.ndim != 3 or fy.ndim != 3 or fz.ndim != 3:
        raise FieldShapeError("flux_x, flux_y, and flux_z must be 3D face arrays")
    nz, ny, nx_plus_one = fx.shape
    nx = nx_plus_one - 1
    if nx < 1:
        raise FieldShapeError("flux_x must have shape (nz, ny, nx + 1)")
    if fy.shape != (nz, ny + 1, nx):
        raise FieldShapeError(f"flux_y shape {fy.shape} must be {(nz, ny + 1, nx)}")
    if fz.shape != (nz + 1, ny, nx):
        raise FieldShapeError(f"flux_z shape {fz.shape} must be {(nz + 1, ny, nx)}")
    phi_values = _broadcast_positive(phi, (nz, ny, nx), "phi")
    volume_values = _broadcast_positive(cell_volume, (nz, ny, nx), "cell_volume")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    connected_flux = (
        np.abs(fx[:, :, :-1])
        + np.abs(fx[:, :, 1:])
        + np.abs(fy[:, :-1, :])
        + np.abs(fy[:, 1:, :])
        + np.abs(fz[:-1, :, :])
        + np.abs(fz[1:, :, :])
    )
    cfl = dt_value / (phi_values * volume_values) * connected_flux
    return cfl, float(np.max(cfl))


def compute_three_phase_saturation_update_3d(
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    gas_flux_x: ArrayLike,
    gas_flux_y: ArrayLike,
    gas_flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Update 3D `Sw` and `Sg` explicitly from water and gas phase fluxes."""
    validate_three_phase_params(params)
    sw_values = _finite_array(sw, "sw").copy()
    sg_values = _finite_array(sg, "sg").copy()
    if sw_values.ndim != 3 or sg_values.ndim != 3 or sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg must be matching 3D arrays")
    nz, ny, nx = sw_values.shape
    wx = _finite_array(water_flux_x, "water_flux_x")
    wy = _finite_array(water_flux_y, "water_flux_y")
    wz = _finite_array(water_flux_z, "water_flux_z")
    gx = _finite_array(gas_flux_x, "gas_flux_x")
    gy = _finite_array(gas_flux_y, "gas_flux_y")
    gz = _finite_array(gas_flux_z, "gas_flux_z")
    if wx.shape != (nz, ny, nx + 1) or gx.shape != (nz, ny, nx + 1):
        raise FieldShapeError("x phase fluxes must have shape (nz, ny, nx + 1)")
    if wy.shape != (nz, ny + 1, nx) or gy.shape != (nz, ny + 1, nx):
        raise FieldShapeError("y phase fluxes must have shape (nz, ny + 1, nx)")
    if wz.shape != (nz + 1, ny, nx) or gz.shape != (nz + 1, ny, nx):
        raise FieldShapeError("z phase fluxes must have shape (nz + 1, ny, nx)")
    phi_values = _broadcast_positive(phi, sw_values.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw_values.shape, "cell_volume")
    dt_value = float(dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise InvalidPhysicalValueError("dt must be positive and finite")
    validate_three_phase_saturations(sw_values, sg_values, params)
    pore_volume = phi_values * volume_values
    net_water_out = (wx[:, :, 1:] - wx[:, :, :-1]) + (wy[:, 1:, :] - wy[:, :-1, :]) + (wz[1:, :, :] - wz[:-1, :, :])
    net_gas_out = (gx[:, :, 1:] - gx[:, :, :-1]) + (gy[:, 1:, :] - gy[:, :-1, :]) + (gz[1:, :, :] - gz[:-1, :, :])
    sw_new = sw_values - dt_value / pore_volume * net_water_out
    sg_new = sg_values - dt_value / pore_volume * net_gas_out
    validate_three_phase_saturations(sw_new, sg_new, params)
    so_new = np.asarray(compute_oil_saturation(sw_new, sg_new), dtype=float)
    return sw_new, sg_new, so_new


def compute_three_phase_material_balance_3d(
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike,
    water_flux_z: ArrayLike,
    oil_flux_x: ArrayLike,
    oil_flux_y: ArrayLike,
    oil_flux_z: ArrayLike,
    gas_flux_x: ArrayLike,
    gas_flux_y: ArrayLike,
    gas_flux_z: ArrayLike,
    sw_old: ArrayLike,
    sg_old: ArrayLike,
    sw_new: ArrayLike,
    sg_new: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
) -> dict[str, float]:
    """Compute 3D water, gas, and oil material balance across six boundaries."""
    validate_three_phase_params(params)
    sw0 = _finite_array(sw_old, "sw_old")
    sg0 = _finite_array(sg_old, "sg_old")
    sw1 = _finite_array(sw_new, "sw_new")
    sg1 = _finite_array(sg_new, "sg_new")
    if sw0.shape != sg0.shape or sw0.shape != sw1.shape or sw0.shape != sg1.shape or sw0.ndim != 3:
        raise FieldShapeError("old and new saturation arrays must have matching 3D shapes")
    nz, ny, nx = sw0.shape
    phase_fluxes = [
        _finite_array(water_flux_x, "water_flux_x"),
        _finite_array(water_flux_y, "water_flux_y"),
        _finite_array(water_flux_z, "water_flux_z"),
        _finite_array(oil_flux_x, "oil_flux_x"),
        _finite_array(oil_flux_y, "oil_flux_y"),
        _finite_array(oil_flux_z, "oil_flux_z"),
        _finite_array(gas_flux_x, "gas_flux_x"),
        _finite_array(gas_flux_y, "gas_flux_y"),
        _finite_array(gas_flux_z, "gas_flux_z"),
    ]
    for fx_value, fy_value, fz_value in (phase_fluxes[0:3], phase_fluxes[3:6], phase_fluxes[6:9]):
        if fx_value.shape != (nz, ny, nx + 1):
            raise FieldShapeError("x phase fluxes must have shape (nz, ny, nx + 1)")
        if fy_value.shape != (nz, ny + 1, nx):
            raise FieldShapeError("y phase fluxes must have shape (nz, ny + 1, nx)")
        if fz_value.shape != (nz + 1, ny, nx):
            raise FieldShapeError("z phase fluxes must have shape (nz + 1, ny, nx)")
    phi_values = _broadcast_positive(phi, sw0.shape, "phi")
    volume_values = _broadcast_positive(cell_volume, sw0.shape, "cell_volume")
    dt_value = float(dt)
    pore_volume = phi_values * volume_values
    so0 = np.asarray(compute_oil_saturation(sw0, sg0), dtype=float)
    so1 = np.asarray(compute_oil_saturation(sw1, sg1), dtype=float)
    report: dict[str, float] = {}
    _add_phase_balance_3d(report, "water", phase_fluxes[0], phase_fluxes[1], phase_fluxes[2], sw0, sw1, pore_volume, dt_value)
    _add_phase_balance_3d(report, "gas", phase_fluxes[6], phase_fluxes[7], phase_fluxes[8], sg0, sg1, pore_volume, dt_value)
    _add_phase_balance_3d(report, "oil", phase_fluxes[3], phase_fluxes[4], phase_fluxes[5], so0, so1, pore_volume, dt_value)
    report["closure_error_max"] = float(np.max(np.abs(sw1 + so1 + sg1 - 1.0)))
    return report


def advance_three_phase_saturation_3d(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    phi: float | ArrayLike,
    cell_volume: float | ArrayLike,
    dt: float,
    params: dict[str, float],
    max_cfl: float = 1.0,
    injected_sw: float | None = None,
    injected_sg: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Advance 3D three-phase saturation by one explicit finite-volume step."""
    fx, fy, fz, sw_values, sg_values, phi_values, volume_values = validate_three_phase_transport_3d_inputs(
        flux_x, flux_y, flux_z, sw, sg, phi, cell_volume, dt, params, max_cfl=max_cfl
    )
    cfl_array, max_cfl_value = compute_three_phase_cfl_3d(fx, fy, fz, phi_values, volume_values, dt)
    if max_cfl_value > float(max_cfl):
        raise CFLViolationError(f"three-phase CFL violation: max_cfl={max_cfl_value}, max_cfl_allowed={max_cfl}")
    (
        water_flux_x,
        water_flux_y,
        water_flux_z,
        oil_flux_x,
        oil_flux_y,
        oil_flux_z,
        gas_flux_x,
        gas_flux_y,
        gas_flux_z,
        flux_report,
    ) = compute_three_phase_fluxes_3d(fx, fy, fz, sw_values, sg_values, params)
    water_flux_x, water_flux_y, water_flux_z = water_flux_x.copy(), water_flux_y.copy(), water_flux_z.copy()
    oil_flux_x, oil_flux_y, oil_flux_z = oil_flux_x.copy(), oil_flux_y.copy(), oil_flux_z.copy()
    gas_flux_x, gas_flux_y, gas_flux_z = gas_flux_x.copy(), gas_flux_y.copy(), gas_flux_z.copy()
    _apply_injection_boundary_fluxes_3d(
        fx=fx,
        fy=fy,
        fz=fz,
        water_flux_x=water_flux_x,
        water_flux_y=water_flux_y,
        water_flux_z=water_flux_z,
        oil_flux_x=oil_flux_x,
        oil_flux_y=oil_flux_y,
        oil_flux_z=oil_flux_z,
        gas_flux_x=gas_flux_x,
        gas_flux_y=gas_flux_y,
        gas_flux_z=gas_flux_z,
        params=params,
        injected_sw=injected_sw,
        injected_sg=injected_sg,
    )
    sw_new, sg_new, so_new = compute_three_phase_saturation_update_3d(
        water_flux_x,
        water_flux_y,
        water_flux_z,
        gas_flux_x,
        gas_flux_y,
        gas_flux_z,
        sw_values,
        sg_values,
        phi_values,
        volume_values,
        dt,
        params,
    )
    balance = compute_three_phase_material_balance_3d(
        water_flux_x,
        water_flux_y,
        water_flux_z,
        oil_flux_x,
        oil_flux_y,
        oil_flux_z,
        gas_flux_x,
        gas_flux_y,
        gas_flux_z,
        sw_values,
        sg_values,
        sw_new,
        sg_new,
        phi_values,
        volume_values,
        dt,
        params,
    )
    arrays = [
        sw_new,
        sg_new,
        so_new,
        water_flux_x,
        water_flux_y,
        water_flux_z,
        oil_flux_x,
        oil_flux_y,
        oil_flux_z,
        gas_flux_x,
        gas_flux_y,
        gas_flux_z,
    ]
    report: dict[str, object] = {
        "max_cfl": max_cfl_value,
        "cfl_min": float(np.min(cfl_array)),
        "cfl_max": float(np.max(cfl_array)),
        **balance,
        "sw_min": float(np.min(sw_new)),
        "sw_max": float(np.max(sw_new)),
        "sg_min": float(np.min(sg_new)),
        "sg_max": float(np.max(sg_new)),
        "so_min": float(np.min(so_new)),
        "so_max": float(np.min(so_new)),
        "has_nan": any(np.isnan(array).any() for array in arrays),
        "has_inf": any(np.isinf(array).any() for array in arrays),
        "transport_dimension": "3d",
        "phase_flux_report": flux_report,
    }
    report["so_max"] = float(np.max(so_new))
    return sw_new, sg_new, so_new, report


def _apply_injection_boundary_fluxes(
    *,
    fx: NDArray[np.float64],
    water_flux: NDArray[np.float64],
    oil_flux: NDArray[np.float64],
    gas_flux: NDArray[np.float64],
    params: dict[str, float],
    injected_sw: float | None,
    injected_sg: float | None,
) -> None:
    p = {key: float(value) for key, value in params.items()}
    inj_sw = 1.0 - p["sor"] - p["sgc"] if injected_sw is None else float(injected_sw)
    inj_sg = p["sgc"] if injected_sg is None else float(injected_sg)
    validate_three_phase_saturations(inj_sw, inj_sg, params)
    fw, fo, fg = (float(value) for value in fractional_flow_three_phase(inj_sw, inj_sg, params))
    if fx[0] > 0.0:
        water_flux[0], oil_flux[0], gas_flux[0] = fw * fx[0], fo * fx[0], fg * fx[0]
    if fx[-1] < 0.0:
        water_flux[-1], oil_flux[-1], gas_flux[-1] = fw * fx[-1], fo * fx[-1], fg * fx[-1]


def _apply_injection_boundary_fluxes_3d(
    *,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    water_flux_x: NDArray[np.float64],
    water_flux_y: NDArray[np.float64],
    water_flux_z: NDArray[np.float64],
    oil_flux_x: NDArray[np.float64],
    oil_flux_y: NDArray[np.float64],
    oil_flux_z: NDArray[np.float64],
    gas_flux_x: NDArray[np.float64],
    gas_flux_y: NDArray[np.float64],
    gas_flux_z: NDArray[np.float64],
    params: dict[str, float],
    injected_sw: float | None,
    injected_sg: float | None,
) -> None:
    p = {key: float(value) for key, value in params.items()}
    inj_sw = 1.0 - p["sor"] - p["sgc"] if injected_sw is None else float(injected_sw)
    inj_sg = p["sgc"] if injected_sg is None else float(injected_sg)
    validate_three_phase_saturations(inj_sw, inj_sg, params)
    fw, fo, fg = (float(value) for value in fractional_flow_three_phase(inj_sw, inj_sg, params))
    _set_injection_faces(fx[:, :, 0], water_flux_x[:, :, 0], oil_flux_x[:, :, 0], gas_flux_x[:, :, 0], fw, fo, fg, "positive")
    _set_injection_faces(fx[:, :, -1], water_flux_x[:, :, -1], oil_flux_x[:, :, -1], gas_flux_x[:, :, -1], fw, fo, fg, "negative")
    _set_injection_faces(fy[:, 0, :], water_flux_y[:, 0, :], oil_flux_y[:, 0, :], gas_flux_y[:, 0, :], fw, fo, fg, "positive")
    _set_injection_faces(fy[:, -1, :], water_flux_y[:, -1, :], oil_flux_y[:, -1, :], gas_flux_y[:, -1, :], fw, fo, fg, "negative")
    _set_injection_faces(fz[0, :, :], water_flux_z[0, :, :], oil_flux_z[0, :, :], gas_flux_z[0, :, :], fw, fo, fg, "positive")
    _set_injection_faces(fz[-1, :, :], water_flux_z[-1, :, :], oil_flux_z[-1, :, :], gas_flux_z[-1, :, :], fw, fo, fg, "negative")


def _set_injection_faces(
    total_flux: NDArray[np.float64],
    water_flux: NDArray[np.float64],
    oil_flux: NDArray[np.float64],
    gas_flux: NDArray[np.float64],
    fw: float,
    fo: float,
    fg: float,
    sign: str,
) -> None:
    mask = total_flux > 0.0 if sign == "positive" else total_flux < 0.0
    water_flux[mask] = fw * total_flux[mask]
    oil_flux[mask] = fo * total_flux[mask]
    gas_flux[mask] = fg * total_flux[mask]


def _add_phase_balance(
    report: dict[str, float],
    phase: str,
    flux: NDArray[np.float64],
    old_saturation: NDArray[np.float64],
    new_saturation: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
    dt: float,
) -> None:
    inflow = (max(float(flux[0]), 0.0) + max(-float(flux[-1]), 0.0)) * dt
    outflow = (max(-float(flux[0]), 0.0) + max(float(flux[-1]), 0.0)) * dt
    storage = float(np.sum((new_saturation - old_saturation) * pore_volume))
    balance_error = storage - (inflow - outflow)
    report[f"{phase}_inflow"] = inflow
    report[f"{phase}_outflow"] = outflow
    report[f"{phase}_storage_change"] = storage
    report[f"{phase}_balance_error"] = balance_error


def _add_phase_balance_3d(
    report: dict[str, float],
    phase: str,
    flux_x: NDArray[np.float64],
    flux_y: NDArray[np.float64],
    flux_z: NDArray[np.float64],
    old_saturation: NDArray[np.float64],
    new_saturation: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
    dt: float,
) -> None:
    inflow_rate, outflow_rate = _boundary_inflow_outflow_3d(flux_x, flux_y, flux_z)
    storage = float(np.sum((new_saturation - old_saturation) * pore_volume))
    inflow = inflow_rate * dt
    outflow = outflow_rate * dt
    balance_error = storage - (inflow - outflow)
    report[f"{phase}_inflow"] = inflow
    report[f"{phase}_outflow"] = outflow
    report[f"{phase}_storage_change"] = storage
    report[f"{phase}_balance_error"] = balance_error


def _boundary_inflow_outflow_3d(
    flux_x: NDArray[np.float64],
    flux_y: NDArray[np.float64],
    flux_z: NDArray[np.float64],
) -> tuple[float, float]:
    inflow = (
        np.maximum(flux_x[:, :, 0], 0.0).sum()
        + np.maximum(-flux_x[:, :, -1], 0.0).sum()
        + np.maximum(flux_y[:, 0, :], 0.0).sum()
        + np.maximum(-flux_y[:, -1, :], 0.0).sum()
        + np.maximum(flux_z[0, :, :], 0.0).sum()
        + np.maximum(-flux_z[-1, :, :], 0.0).sum()
    )
    outflow = (
        np.maximum(-flux_x[:, :, 0], 0.0).sum()
        + np.maximum(flux_x[:, :, -1], 0.0).sum()
        + np.maximum(-flux_y[:, 0, :], 0.0).sum()
        + np.maximum(flux_y[:, -1, :], 0.0).sum()
        + np.maximum(-flux_z[0, :, :], 0.0).sum()
        + np.maximum(flux_z[-1, :, :], 0.0).sum()
    )
    return float(inflow), float(outflow)


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError(f"{name} must be finite")
    return array


def _broadcast_positive(value: float | ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        array = np.full(shape, float(array), dtype=float)
    elif array.shape != shape:
        raise FieldShapeError(f"{name} shape {array.shape} must be scalar or {shape}")
    if np.isnan(array).any() or np.isinf(array).any() or (array <= 0.0).any():
        raise InvalidPhysicalValueError(f"{name} must be positive and finite")
    return array.copy()
