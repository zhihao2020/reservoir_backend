"""Standalone capillary face-flux calculations.

The fluxes computed here are not coupled into the saturation solver. They are
intended as a preparatory building block for future capillary transport work.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError, NonNeighborCellError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.capillary_pressure import capillary_pressure, validate_capillary_params
from reservoir_backend.solver.relperm import corey_relative_permeability, validate_saturation_params, validate_viscosity
from reservoir_backend.solver.transmissibility import harmonic_average, validate_permeability


def capillary_mobility(
    sw: float | ArrayLike | Field3D,
    relperm_params: dict[str, Any],
) -> float | NDArray[np.float64] | Field3D:
    """Return ``Mcap = lambda_w * lambda_o / (lambda_w + lambda_o)``."""
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
        from reservoir_backend.core.exceptions import InvalidPhysicalValueError

        raise InvalidPhysicalValueError("capillary mobility must be finite and non-negative")
    if isinstance(sw, Field3D):
        return Field3D(sw.grid, mobility, name="capillary_mobility", unit="1/(Pa.s)")
    if mobility.shape == ():
        return float(mobility)
    return mobility


def compute_absolute_transmissibility_between_cells(
    grid: Grid3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    cell_a: int,
    cell_b: int,
) -> float:
    """Return absolute face transmissibility ``k_face * A / d`` without viscosity."""
    ia, ja, ka = grid.ijk(cell_a)
    ib, jb, kb = grid.ijk(cell_b)
    di, dj, dk = ib - ia, jb - ja, kb - ka
    if abs(di) + abs(dj) + abs(dk) != 1:
        raise NonNeighborCellError(f"cells {cell_a} and {cell_b} are not face neighbors")

    if di != 0:
        values = _field_values(grid, kx, "permeability")
        return float(harmonic_average(values[ka, ja, ia], values[kb, jb, ib])) * grid.dy * grid.dz / grid.dx
    if dj != 0:
        values = _field_values(grid, ky, "permeability")
        return float(harmonic_average(values[ka, ja, ia], values[kb, jb, ib])) * grid.dx * grid.dz / grid.dy

    values = _field_values(grid, kz, "permeability")
    return float(harmonic_average(values[ka, ja, ia], values[kb, jb, ib])) * grid.dx * grid.dy / grid.dz


def compute_capillary_fluxes(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    capillary_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Compute x/y/z capillary water face fluxes."""
    params = validate_capillary_flux_inputs(grid, sw, kx, ky, kz, capillary_params, relperm_params)
    flux_x = np.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=float)
    flux_y = np.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=float)
    flux_z = np.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=float)
    sw_values = _field_values(grid, sw, "sw")

    if not params["capillary"]["enabled"] or params["capillary"]["model"] == "none":
        pc_values = np.zeros(grid.shape, dtype=float)
        mobility_values = np.zeros(grid.shape, dtype=float)
        return flux_x, flux_y, flux_z, _build_report(False, "none", flux_x, flux_y, flux_z, pc_values, mobility_values)

    pc = capillary_pressure(sw_values, params["capillary"])
    pc_values = np.asarray(pc, dtype=float)
    mobility = capillary_mobility(sw_values, params["relperm"])
    mobility_values = np.asarray(mobility, dtype=float)

    kx_values = _field_values(grid, kx, "permeability")
    ky_values = _field_values(grid, ky, "permeability")
    kz_values = _field_values(grid, kz, "permeability")

    if grid.nx > 1:
        t_abs = harmonic_average(kx_values[:, :, :-1], kx_values[:, :, 1:]) * grid.dy * grid.dz / grid.dx
        m_face = harmonic_average(mobility_values[:, :, :-1], mobility_values[:, :, 1:])
        flux_x[:, :, 1:-1] = t_abs * m_face * (pc_values[:, :, 1:] - pc_values[:, :, :-1])
    if grid.ny > 1:
        t_abs = harmonic_average(ky_values[:, :-1, :], ky_values[:, 1:, :]) * grid.dx * grid.dz / grid.dy
        m_face = harmonic_average(mobility_values[:, :-1, :], mobility_values[:, 1:, :])
        flux_y[:, 1:-1, :] = t_abs * m_face * (pc_values[:, 1:, :] - pc_values[:, :-1, :])
    if grid.nz > 1:
        t_abs = harmonic_average(kz_values[:-1, :, :], kz_values[1:, :, :]) * grid.dx * grid.dy / grid.dz
        m_face = harmonic_average(mobility_values[:-1, :, :], mobility_values[1:, :, :])
        flux_z[1:-1, :, :] = t_abs * m_face * (pc_values[1:, :, :] - pc_values[:-1, :, :])

    report = _build_report(
        True,
        params["capillary"]["model"],
        flux_x,
        flux_y,
        flux_z,
        pc_values,
        mobility_values,
    )
    return flux_x, flux_y, flux_z, report


def compute_capillary_water_flux_1d(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kx: float | ArrayLike | Field3D,
    capillary_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Compute x-direction capillary flux for a 1D ``ny=nz=1`` grid."""
    if grid.ny != 1 or grid.nz != 1:
        raise ValueError("compute_capillary_water_flux_1d requires ny=1 and nz=1")
    flux_x, _, _, report = compute_capillary_fluxes(
        grid,
        sw,
        kx,
        kx,
        kx,
        capillary_params,
        relperm_params,
    )
    return flux_x, report


def validate_capillary_flux_inputs(
    grid: Grid3D,
    sw: float | ArrayLike | Field3D,
    kx: float | ArrayLike | Field3D,
    ky: float | ArrayLike | Field3D,
    kz: float | ArrayLike | Field3D,
    capillary_params: dict[str, Any],
    relperm_params: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate inputs for capillary flux calculations."""
    sw_values = _field_values(grid, sw, "sw")
    if np.isnan(sw_values).any() or np.isinf(sw_values).any():
        from reservoir_backend.core.exceptions import InvalidPhysicalValueError

        raise InvalidPhysicalValueError("sw must be finite")
    capillary = validate_capillary_params(capillary_params)
    relperm = _normalize_relperm_params(relperm_params)
    _field_values(grid, kx, "permeability")
    _field_values(grid, ky, "permeability")
    _field_values(grid, kz, "permeability")
    return {"capillary": capillary, "relperm": relperm}


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
    enabled: bool,
    model: str,
    flux_x: NDArray[np.float64],
    flux_y: NDArray[np.float64],
    flux_z: NDArray[np.float64],
    pc_values: NDArray[np.float64],
    mobility_values: NDArray[np.float64],
) -> dict[str, object]:
    all_flux = np.concatenate([flux_x.ravel(), flux_y.ravel(), flux_z.ravel()])
    has_nan = bool(
        np.isnan(all_flux).any()
        or np.isnan(pc_values).any()
        or np.isnan(mobility_values).any()
    )
    has_inf = bool(
        np.isinf(all_flux).any()
        or np.isinf(pc_values).any()
        or np.isinf(mobility_values).any()
    )
    return {
        "enabled": bool(enabled),
        "model": model,
        "max_abs_capillary_flux": float(np.max(np.abs(all_flux))),
        "min_capillary_flux": float(np.min(all_flux)),
        "max_capillary_flux": float(np.max(all_flux)),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "pc_min": float(np.min(pc_values)),
        "pc_max": float(np.max(pc_values)),
        "mobility_min": float(np.min(mobility_values)),
        "mobility_max": float(np.max(mobility_values)),
    }
