"""Standalone oil-water gravity segregation face-flux calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.relperm import corey_relative_permeability, validate_saturation_params, validate_viscosity
from reservoir_backend.solver.transmissibility import harmonic_average, validate_permeability


def gravity_mobility(
    sw: float | ArrayLike | Field3D,
    relperm_params: dict[str, Any],
) -> float | NDArray[np.float64] | Field3D:
    """Return ``Mgrav = lambda_w * lambda_o / (lambda_w + lambda_o)``."""
    params = _normalize_relperm_params(relperm_params)
    krw, kro = corey_relative_permeability(
        sw,
        params["swi"],
        params["sor"],
        params["krw0"],
        params["kro0"],
        params["nw"],
        params["no"],
    )
    lambda_w = np.asarray(krw, dtype=float) / params["mu_w"]
    lambda_o = np.asarray(kro, dtype=float) / params["mu_o"]
    lambda_t = lambda_w + lambda_o
    mobility = np.divide(
        lambda_w * lambda_o,
        lambda_t,
        out=np.zeros_like(lambda_t, dtype=float),
        where=lambda_t > 0.0,
    )
    if np.isnan(mobility).any() or np.isinf(mobility).any() or (mobility < 0.0).any():
        raise InvalidPhysicalValueError("gravity mobility must be finite and non-negative")
    if isinstance(sw, Field3D):
        return Field3D(sw.grid, mobility, name="gravity_mobility", unit="1/(Pa.s)")
    if mobility.shape == ():
        return float(mobility)
    return mobility


def compute_gravity_fluxes(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    gravity_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Compute x/y/z water gravity segregation face fluxes.

    In the current Cartesian convention gravity acts only along z. Positive
    ``flux_z`` is bottom-to-top, so heavier water moving downward gives
    negative z-face water flux.
    """
    params = validate_gravity_flux_inputs(grid, sw, kx, ky, kz, gravity_params, relperm_params)
    flux_x = np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float)
    flux_y = np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float)
    flux_z = np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float)
    sw_values = _field_values(grid, sw, "sw")

    if not params["gravity"]["enabled"]:
        mobility_values = np.zeros(grid.shape, dtype=float)
        return flux_x, flux_y, flux_z, _build_report(params["gravity"], flux_x, flux_y, flux_z, mobility_values)

    mobility = gravity_mobility(sw_values, params["relperm"])
    mobility_values = np.asarray(mobility, dtype=float)
    kz_values = _field_values(grid, kz, "permeability")
    density_difference = params["gravity"]["rho_w"] - params["gravity"]["rho_o"]
    sign = -1.0 if params["gravity"]["depth_positive"] == "down" else 1.0

    if grid.nz > 1 and params["gravity"]["depth_axis"] == "z":
        t_abs = harmonic_average(kz_values[:-1, :, :], kz_values[1:, :, :]) * grid.dx * grid.dy / grid.dz
        m_face = harmonic_average(mobility_values[:-1, :, :], mobility_values[1:, :, :])
        flux_z[1:-1, :, :] = (
            sign
            * t_abs
            * m_face
            * density_difference
            * params["gravity"]["g"]
            * grid.dz
        )

    report = _build_report(params["gravity"], flux_x, flux_y, flux_z, mobility_values)
    return flux_x, flux_y, flux_z, report


def compute_gravity_water_flux_1d_vertical(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    gravity_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Compute z-direction gravity water flux for a vertical 1D grid."""
    if grid.nx != 1 or grid.ny != 1:
        raise ValueError("compute_gravity_water_flux_1d_vertical requires nx=1 and ny=1")
    _, _, flux_z, report = compute_gravity_fluxes(
        grid,
        sw,
        kz,
        kz,
        kz,
        gravity_params,
        relperm_params,
    )
    return flux_z, report


def validate_gravity_params(gravity_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize gravity model parameters."""
    params = dict(gravity_params or {})
    normalized = {
        "enabled": bool(params.get("enabled", False)),
        "g": float(params.get("g", 9.80665)),
        "rho_w": float(params.get("rho_w", 1000.0)),
        "rho_o": float(params.get("rho_o", 800.0)),
        "depth_axis": str(params.get("depth_axis", "z")),
        "depth_positive": str(params.get("depth_positive", "down")),
    }
    if not np.isfinite(normalized["g"]) or normalized["g"] < 0.0:
        raise InvalidPhysicalValueError("gravity acceleration g must be finite and non-negative")
    if not np.isfinite(normalized["rho_w"]) or normalized["rho_w"] <= 0.0:
        raise InvalidPhysicalValueError("rho_w must be positive and finite")
    if not np.isfinite(normalized["rho_o"]) or normalized["rho_o"] <= 0.0:
        raise InvalidPhysicalValueError("rho_o must be positive and finite")
    if normalized["depth_axis"] != "z":
        raise ValueError("gravity depth_axis currently supports only 'z'")
    if normalized["depth_positive"] not in {"down", "up"}:
        raise ValueError("gravity depth_positive must be 'down' or 'up'")
    return normalized


def build_gravity_model_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build gravity model params from a full case config or gravity section."""
    section = config.get("gravity", config)
    return validate_gravity_params(section)


def validate_gravity_flux_inputs(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    gravity_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate inputs for gravity flux calculations."""
    sw_values = _field_values(grid, sw, "sw")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        raise InvalidPhysicalValueError("sw must be finite")
    gravity = validate_gravity_params(gravity_params)
    relperm = _normalize_relperm_params(relperm_params)
    _field_values(grid, kx, "permeability")
    _field_values(grid, ky, "permeability")
    _field_values(grid, kz, "permeability")
    return {"gravity": gravity, "relperm": relperm}


def _normalize_relperm_params(params: dict[str, Any]) -> dict[str, float]:
    relperm = {
        "swi": float(params["swi"]),
        "sor": float(params["sor"]),
        "krw0": float(params["krw0"]),
        "kro0": float(params["kro0"]),
        "nw": float(params["nw"]),
        "no": float(params["no"]),
        "mu_w": float(params["mu_w"]),
        "mu_o": float(params["mu_o"]),
    }
    validate_saturation_params(relperm["swi"], relperm["sor"])
    validate_viscosity(relperm["mu_w"], relperm["mu_o"])
    return relperm


def _field_values(
    grid: Grid3D,
    value: float | ArrayLike | Field3D,
    name: str,
) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        if value.grid != grid:
            raise GridMismatchError(f"{name} Field3D is defined on a different grid")
        array = value.values.astype(float, copy=False)
    else:
        array = np.asarray(value, dtype=float)
        if array.shape == ():
            array = np.full(grid.shape, float(array), dtype=float)
        elif array.shape != grid.shape:
            raise FieldShapeError(f"{name} shape {array.shape} does not match grid shape {grid.shape}")
    if name == "permeability":
        validate_permeability(array)
    return array


def _build_report(
    gravity_params: dict[str, Any],
    flux_x: NDArray[np.float64],
    flux_y: NDArray[np.float64],
    flux_z: NDArray[np.float64],
    mobility_values: NDArray[np.float64],
) -> dict[str, object]:
    all_flux = np.concatenate([flux_x.ravel(), flux_y.ravel(), flux_z.ravel()])
    return {
        "enabled": bool(gravity_params["enabled"]),
        "g": float(gravity_params["g"]),
        "rho_w": float(gravity_params["rho_w"]),
        "rho_o": float(gravity_params["rho_o"]),
        "density_difference": float(gravity_params["rho_w"] - gravity_params["rho_o"]),
        "max_abs_gravity_flux": float(np.max(np.abs(all_flux))),
        "min_gravity_flux": float(np.min(all_flux)),
        "max_gravity_flux": float(np.max(all_flux)),
        "has_nan": bool(np.isnan(all_flux).any() or np.isnan(mobility_values).any()),
        "has_inf": bool(np.isinf(all_flux).any() or np.isinf(mobility_values).any()),
        "mobility_min": float(np.min(mobility_values)),
        "mobility_max": float(np.max(mobility_values)),
    }
