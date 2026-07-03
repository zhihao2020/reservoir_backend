"""Advective phase fluxes for simplified three-phase water-oil-gas flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, InvalidPhysicalValueError
from reservoir_backend.solver.three_phase_relperm import (
    fractional_flow_three_phase,
    validate_three_phase_params,
    validate_three_phase_saturations,
)


def compute_upwind_fractional_flow_1d(
    flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return 1D face fractional flows selected by total-flux upwinding."""
    fx, sw_values, sg_values = _validate_1d_inputs(flux_x, sw, sg, params)
    nx = sw_values.shape[0]
    upstream_sw = np.empty(nx + 1, dtype=float)
    upstream_sg = np.empty(nx + 1, dtype=float)
    upstream_sw[0], upstream_sg[0] = sw_values[0], sg_values[0]
    upstream_sw[-1], upstream_sg[-1] = sw_values[-1], sg_values[-1]
    for face in range(1, nx):
        upstream = face - 1 if fx[face] >= 0.0 else face
        upstream_sw[face] = sw_values[upstream]
        upstream_sg[face] = sg_values[upstream]
    fw, fo, fg = fractional_flow_three_phase(upstream_sw, upstream_sg, params)
    return np.asarray(fw, dtype=float), np.asarray(fo, dtype=float), np.asarray(fg, dtype=float)


def compute_three_phase_flux_1d(
    flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Return 1D water, oil, and gas phase fluxes plus report."""
    fx, _, _ = _validate_1d_inputs(flux_x, sw, sg, params)
    fw, fo, fg = compute_upwind_fractional_flow_1d(fx, sw, sg, params)
    water_flux_x = fw * fx
    oil_flux_x = fo * fx
    gas_flux_x = fg * fx
    report = build_three_phase_flux_report(
        flux_x=fx,
        flux_y=None,
        flux_z=None,
        water_flux_x=water_flux_x,
        water_flux_y=None,
        water_flux_z=None,
        oil_flux_x=oil_flux_x,
        oil_flux_y=None,
        oil_flux_z=None,
        gas_flux_x=gas_flux_x,
        gas_flux_y=None,
        gas_flux_z=None,
    )
    return water_flux_x, oil_flux_x, gas_flux_x, report


def compute_upwind_fractional_flow_3d(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return face fractional flows for x, y, and z directions."""
    fx, fy, fz, sw_values, sg_values = validate_three_phase_flux_inputs(flux_x, flux_y, flux_z, sw, sg, params)
    sw_x, sg_x = _upwind_cells_x(fx, sw_values, sg_values)
    sw_y, sg_y = _upwind_cells_y(fy, sw_values, sg_values)
    sw_z, sg_z = _upwind_cells_z(fz, sw_values, sg_values)
    fw_x, fo_x, fg_x = (np.asarray(value, dtype=float) for value in fractional_flow_three_phase(sw_x, sg_x, params))
    fw_y, fo_y, fg_y = (np.asarray(value, dtype=float) for value in fractional_flow_three_phase(sw_y, sg_y, params))
    fw_z, fo_z, fg_z = (np.asarray(value, dtype=float) for value in fractional_flow_three_phase(sw_z, sg_z, params))
    return fw_x, fo_x, fg_x, fw_y, fo_y, fg_y, fw_z, fo_z, fg_z


def compute_three_phase_fluxes_3d(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, object],
]:
    """Return 3D water, oil, and gas phase fluxes plus report."""
    fx, fy, fz, _, _ = validate_three_phase_flux_inputs(flux_x, flux_y, flux_z, sw, sg, params)
    fw_x, fo_x, fg_x, fw_y, fo_y, fg_y, fw_z, fo_z, fg_z = compute_upwind_fractional_flow_3d(
        fx, fy, fz, sw, sg, params
    )
    water_flux_x, water_flux_y, water_flux_z = fw_x * fx, fw_y * fy, fw_z * fz
    oil_flux_x, oil_flux_y, oil_flux_z = fo_x * fx, fo_y * fy, fo_z * fz
    gas_flux_x, gas_flux_y, gas_flux_z = fg_x * fx, fg_y * fy, fg_z * fz
    report = build_three_phase_flux_report(
        flux_x=fx,
        flux_y=fy,
        flux_z=fz,
        water_flux_x=water_flux_x,
        water_flux_y=water_flux_y,
        water_flux_z=water_flux_z,
        oil_flux_x=oil_flux_x,
        oil_flux_y=oil_flux_y,
        oil_flux_z=oil_flux_z,
        gas_flux_x=gas_flux_x,
        gas_flux_y=gas_flux_y,
        gas_flux_z=gas_flux_z,
    )
    return (
        water_flux_x,
        water_flux_y,
        water_flux_z,
        oil_flux_x,
        oil_flux_y,
        oil_flux_z,
        gas_flux_x,
        gas_flux_y,
        gas_flux_z,
        report,
    )


def validate_three_phase_flux_inputs(
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate 3D phase-flux inputs and return float arrays."""
    validate_three_phase_params(params)
    sw_values = _finite_array(sw, "sw")
    sg_values = _finite_array(sg, "sg")
    if sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg shapes must match")
    if sw_values.ndim != 3:
        raise FieldShapeError("sw and sg must be 3D arrays with shape (nz, ny, nx)")
    nz, ny, nx = sw_values.shape
    fx = _finite_array(flux_x, "flux_x")
    fy = _finite_array(flux_y, "flux_y")
    fz = _finite_array(flux_z, "flux_z")
    if fx.shape != (nz, ny, nx + 1):
        raise FieldShapeError(f"flux_x shape {fx.shape} must be {(nz, ny, nx + 1)}")
    if fy.shape != (nz, ny + 1, nx):
        raise FieldShapeError(f"flux_y shape {fy.shape} must be {(nz, ny + 1, nx)}")
    if fz.shape != (nz + 1, ny, nx):
        raise FieldShapeError(f"flux_z shape {fz.shape} must be {(nz + 1, ny, nx)}")
    validate_three_phase_saturations(sw_values, sg_values, params)
    return fx, fy, fz, sw_values, sg_values


def build_three_phase_flux_report(
    *,
    flux_x: ArrayLike,
    flux_y: ArrayLike | None,
    flux_z: ArrayLike | None,
    water_flux_x: ArrayLike,
    water_flux_y: ArrayLike | None,
    water_flux_z: ArrayLike | None,
    oil_flux_x: ArrayLike,
    oil_flux_y: ArrayLike | None,
    oil_flux_z: ArrayLike | None,
    gas_flux_x: ArrayLike,
    gas_flux_y: ArrayLike | None,
    gas_flux_z: ArrayLike | None,
) -> dict[str, object]:
    """Build a report for 1D or 3D three-phase advective fluxes."""
    total_fluxes = _arrays_without_none(flux_x, flux_y, flux_z)
    water_fluxes = _arrays_without_none(water_flux_x, water_flux_y, water_flux_z)
    oil_fluxes = _arrays_without_none(oil_flux_x, oil_flux_y, oil_flux_z)
    gas_fluxes = _arrays_without_none(gas_flux_x, gas_flux_y, gas_flux_z)
    closure_errors = [
        np.abs(water + oil + gas - total)
        for total, water, oil, gas in zip(total_fluxes, water_fluxes, oil_fluxes, gas_fluxes, strict=True)
    ]
    all_arrays = [*total_fluxes, *water_fluxes, *oil_fluxes, *gas_fluxes]
    return {
        "max_total_flux": _max_abs(total_fluxes),
        "max_water_flux": _max_abs(water_fluxes),
        "max_oil_flux": _max_abs(oil_fluxes),
        "max_gas_flux": _max_abs(gas_fluxes),
        "min_water_flux": _min_value(water_fluxes),
        "min_oil_flux": _min_value(oil_fluxes),
        "min_gas_flux": _min_value(gas_fluxes),
        "phase_flux_closure_error_max": _max_value(closure_errors),
        "has_nan": any(np.isnan(array).any() for array in all_arrays),
        "has_inf": any(np.isinf(array).any() for array in all_arrays),
        "flux_shape_x": tuple(np.asarray(flux_x).shape),
        "flux_shape_y": None if flux_y is None else tuple(np.asarray(flux_y).shape),
        "flux_shape_z": None if flux_z is None else tuple(np.asarray(flux_z).shape),
    }


def _validate_1d_inputs(
    flux_x: ArrayLike,
    sw: ArrayLike,
    sg: ArrayLike,
    params: dict[str, float],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    validate_three_phase_params(params)
    fx = _finite_array(flux_x, "flux_x")
    sw_values = _finite_array(sw, "sw")
    sg_values = _finite_array(sg, "sg")
    if sw_values.ndim != 1 or sg_values.ndim != 1:
        raise FieldShapeError("1D sw and sg must have shape (nx,)")
    if sw_values.shape != sg_values.shape:
        raise FieldShapeError("sw and sg shapes must match")
    if fx.shape != (sw_values.shape[0] + 1,):
        raise FieldShapeError(f"flux_x shape {fx.shape} must be {(sw_values.shape[0] + 1,)}")
    validate_three_phase_saturations(sw_values, sg_values, params)
    return fx, sw_values, sg_values


def _upwind_cells_x(
    flux_x: NDArray[np.float64],
    sw: NDArray[np.float64],
    sg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nz, ny, nx = sw.shape
    up_sw = np.empty((nz, ny, nx + 1), dtype=float)
    up_sg = np.empty_like(up_sw)
    up_sw[:, :, 0], up_sg[:, :, 0] = sw[:, :, 0], sg[:, :, 0]
    up_sw[:, :, -1], up_sg[:, :, -1] = sw[:, :, -1], sg[:, :, -1]
    positive = flux_x[:, :, 1:nx] >= 0.0
    up_sw[:, :, 1:nx] = np.where(positive, sw[:, :, : nx - 1], sw[:, :, 1:nx])
    up_sg[:, :, 1:nx] = np.where(positive, sg[:, :, : nx - 1], sg[:, :, 1:nx])
    return up_sw, up_sg


def _upwind_cells_y(
    flux_y: NDArray[np.float64],
    sw: NDArray[np.float64],
    sg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nz, ny, nx = sw.shape
    up_sw = np.empty((nz, ny + 1, nx), dtype=float)
    up_sg = np.empty_like(up_sw)
    up_sw[:, 0, :], up_sg[:, 0, :] = sw[:, 0, :], sg[:, 0, :]
    up_sw[:, -1, :], up_sg[:, -1, :] = sw[:, -1, :], sg[:, -1, :]
    positive = flux_y[:, 1:ny, :] >= 0.0
    up_sw[:, 1:ny, :] = np.where(positive, sw[:, : ny - 1, :], sw[:, 1:ny, :])
    up_sg[:, 1:ny, :] = np.where(positive, sg[:, : ny - 1, :], sg[:, 1:ny, :])
    return up_sw, up_sg


def _upwind_cells_z(
    flux_z: NDArray[np.float64],
    sw: NDArray[np.float64],
    sg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nz, ny, nx = sw.shape
    up_sw = np.empty((nz + 1, ny, nx), dtype=float)
    up_sg = np.empty_like(up_sw)
    up_sw[0, :, :], up_sg[0, :, :] = sw[0, :, :], sg[0, :, :]
    up_sw[-1, :, :], up_sg[-1, :, :] = sw[-1, :, :], sg[-1, :, :]
    positive = flux_z[1:nz, :, :] >= 0.0
    up_sw[1:nz, :, :] = np.where(positive, sw[: nz - 1, :, :], sw[1:nz, :, :])
    up_sg[1:nz, :, :] = np.where(positive, sg[: nz - 1, :, :], sg[1:nz, :, :])
    return up_sw, up_sg


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if np.isnan(array).any() or np.isinf(array).any():
        raise InvalidPhysicalValueError(f"{name} must be finite")
    return array


def _arrays_without_none(*values: ArrayLike | None) -> list[NDArray[np.float64]]:
    return [np.asarray(value, dtype=float) for value in values if value is not None]


def _max_abs(values: list[NDArray[np.float64]]) -> float:
    return max(float(np.max(np.abs(value))) for value in values)


def _max_value(values: list[NDArray[np.float64]]) -> float:
    return max(float(np.max(value)) for value in values)


def _min_value(values: list[NDArray[np.float64]]) -> float:
    return min(float(np.min(value)) for value in values)
