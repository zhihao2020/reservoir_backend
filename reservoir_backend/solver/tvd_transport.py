"""Optional 1D TVD/MUSCL saturation transport helpers.

The existing first-order upwind solver remains the baseline. This module adds a
small opt-in 1D high-resolution path for benchmark hardening and diagnostics.
It does not replace `advance_saturation_1d` and does not implement a fully
implicit simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.exceptions import CFLViolationError
from reservoir_backend.solver.cfl import compute_cfl_number, estimate_stable_dt, validate_time_step
from reservoir_backend.solver.limiters import compute_limited_slopes
from reservoir_backend.solver.relperm import fractional_flow_water, validate_saturation_params
from reservoir_backend.solver.saturation_solver import (
    DEFAULT_RELPERM_PARAMS,
    SaturationStepResult,
    advance_saturation_1d,
    compute_saturation_material_balance_1d,
)
from reservoir_backend.solver.transport_diagnostics import build_transport_diagnostics


@dataclass(frozen=True)
class EnhancedTransportResult:
    """Output from optional saturation transport enhancement path."""

    sw: Field3D
    report: dict


def compute_cfl(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
):
    """Compute cell-wise CFL diagnostics for the enhancement layer."""
    return compute_cfl_number(grid, phi, flux_x, flux_y, flux_z, dt)


def suggest_stable_timestep(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    target_cfl: float = 0.8,
) -> dict[str, object]:
    """Suggest an adapted explicit timestep without changing solver state."""
    validate_time_step(dt)
    cfl_field, cfl_report = compute_cfl_number(grid, phi, flux_x, flux_y, flux_z, dt)
    stable_dt = estimate_stable_dt(grid, phi, flux_x, flux_y, flux_z, max_cfl=target_cfl)
    adapted = float(dt) if np.isinf(stable_dt) else float(min(float(dt), stable_dt))
    return {
        "dt_original": float(dt),
        "dt_suggested": adapted,
        "dt_adapted": adapted,
        "target_cfl": float(target_cfl),
        "max_cfl": float(cfl_report["max_cfl"]),
        "num_limited_cells": int(np.count_nonzero(cfl_field > float(target_cfl))),
        "stable_without_adaptation": bool(cfl_report["max_cfl"] <= float(target_cfl)),
        "warnings": [] if cfl_report["max_cfl"] <= float(target_cfl) else ["CFL exceeds target; smaller timestep suggested"],
    }


def adapt_timestep(
    grid: Grid3D,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    dt: float,
    target_cfl: float = 0.8,
    raise_on_violation: bool = False,
) -> dict[str, object]:
    """Return an adapted timestep and report for optional enhanced transport."""
    report = suggest_stable_timestep(grid, phi, flux_x, flux_y, flux_z, dt, target_cfl=target_cfl)
    report["adapted"] = bool(report["dt_adapted"] < report["dt_original"])
    if raise_on_violation and report["adapted"]:
        raise CFLViolationError(
            f"CFL condition violated: max_cfl={report['max_cfl']} target_cfl={report['target_cfl']}"
        )
    return report


def advance_saturation_1d_enhanced(
    grid: Grid3D,
    sw: Field3D | ArrayLike,
    phi: float | ArrayLike | Field3D,
    flux_x: ArrayLike,
    dt: float,
    relperm_params: dict[str, float],
    max_cfl: float = 1.0,
    method: str = "upwind",
    limiter: str = "minmod",
    fallback: str = "upwind",
) -> EnhancedTransportResult:
    """Advance saturation with optional 1D TVD/MUSCL path.

    `method="upwind"` delegates to the validated baseline solver.
    `method="tvd"` and `method="muscl"` use limited linear reconstruction.
    `method="implicit"` is intentionally deferred and falls back to upwind with
    an explicit warning.
    """
    requested = method.lower()
    warnings: list[str] = []
    if requested == "upwind":
        base = advance_saturation_1d(grid, sw, phi, flux_x, dt, relperm_params, max_cfl=max_cfl)
        report = dict(base.report)
        report.update(
            {
                "method_requested": requested,
                "method_used": "upwind",
                "limiter": None,
                "fallback_used": False,
                "implicit_deferred": False,
                "warnings": [],
            }
        )
        diagnostics = build_transport_diagnostics(
            _field_values(grid, sw),
            base.sw.values,
            lower=_params(relperm_params)["swi"],
            upper=1.0 - _params(relperm_params)["sor"],
            dx=grid.dx,
            max_cfl=float(report["max_cfl"]),
            material_balance_error=float(report["material_balance_error"]),
        )
        report.update(diagnostics)
        return EnhancedTransportResult(sw=base.sw, report=report)

    if requested == "implicit":
        if fallback != "upwind":
            raise NotImplementedError("implicit saturation transport is deferred")
        warnings.append("implicit saturation transport is deferred; used upwind fallback")
        base = advance_saturation_1d(grid, sw, phi, flux_x, dt, relperm_params, max_cfl=max_cfl)
        report = dict(base.report)
        report.update(
            {
                "method_requested": requested,
                "method_used": "upwind",
                "limiter": None,
                "fallback_used": True,
                "implicit_deferred": True,
                "warnings": warnings,
            }
        )
        return EnhancedTransportResult(sw=base.sw, report=report)

    if requested not in {"tvd", "muscl"}:
        raise ValueError("method must be upwind, tvd, muscl, or implicit")

    _validate_1d_grid(grid)
    params = _params(relperm_params)
    sw_values = _field_values(grid, sw)
    phi_values = _field_values(grid, phi)
    flux = _flux_x(grid, flux_x)
    cfl_report = adapt_timestep(
        grid=grid,
        phi=phi_values,
        flux_x=flux,
        flux_y=np.zeros((1, 2, grid.nx), dtype=float),
        flux_z=np.zeros((2, 1, grid.nx), dtype=float),
        dt=dt,
        target_cfl=max_cfl,
    )
    dt_used = float(cfl_report["dt_adapted"])
    if cfl_report["adapted"]:
        warnings.append("CFL exceeded target; adaptive timestep was used")

    water_flux = compute_tvd_water_flux_1d(sw_values, flux, params, limiter=limiter)
    old = sw_values[0, 0, :]
    phi_line = phi_values[0, 0, :]
    raw = old - dt_used / (phi_line * grid.cell_volume) * (water_flux[0, 0, 1:] - water_flux[0, 0, :-1])
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    out_of_bounds = (raw < lower) | (raw > upper)
    num_clipped = int(np.count_nonzero(out_of_bounds))
    if num_clipped > 0:
        if fallback == "upwind":
            warnings.append("TVD/MUSCL produced out-of-bounds saturation; used upwind fallback")
            base = advance_saturation_1d(grid, sw, phi, flux_x, dt_used, relperm_params, max_cfl=max_cfl)
            report = dict(base.report)
            report.update(
                {
                    "method_requested": requested,
                    "method_used": "upwind",
                    "limiter": limiter,
                    "fallback_used": True,
                    "implicit_deferred": False,
                    "dt_original": float(dt),
                    "dt_adapted": dt_used,
                    "num_limited_cells": int(cfl_report["num_limited_cells"]),
                    "num_clipped_cells": num_clipped,
                    "warnings": warnings,
                }
            )
            return EnhancedTransportResult(sw=base.sw, report=report)
        warnings.append("TVD/MUSCL produced out-of-bounds saturation; clipped with warning")
    new = np.clip(raw, lower, upper)
    field = Field3D(grid=grid, values=new.reshape(grid.shape), name="sw", unit="fraction")
    balance = compute_saturation_material_balance_1d(
        grid=grid,
        sw_old=old,
        sw_new=new,
        phi=phi_line,
        water_flux_x=water_flux,
        dt=dt_used,
    )
    diagnostics = build_transport_diagnostics(
        sw_values,
        field.values,
        lower=lower,
        upper=upper,
        dx=grid.dx,
        max_cfl=float(cfl_report["max_cfl"]),
        material_balance_error=float(balance["material_balance_error"]),
    )
    report = {
        "method_requested": requested,
        "method_used": requested,
        "limiter": limiter,
        "fallback_used": False,
        "implicit_deferred": False,
        "dt_original": float(dt),
        "dt": dt_used,
        "dt_adapted": dt_used,
        "max_cfl": float(cfl_report["max_cfl"]),
        "target_cfl": float(cfl_report["target_cfl"]),
        "num_limited_cells": int(cfl_report["num_limited_cells"]),
        "num_clipped_cells": num_clipped,
        "sw_min": float(np.min(new)),
        "sw_max": float(np.max(new)),
        "injected_water_volume": balance["injected_water_volume"],
        "produced_water_volume": balance["produced_water_volume"],
        "storage_change": balance["storage_change"],
        "material_balance_error": balance["material_balance_error"],
        "has_nan": bool(np.isnan(new).any()),
        "has_inf": bool(np.isinf(new).any()),
        "warnings": warnings,
        **diagnostics,
    }
    return EnhancedTransportResult(sw=field, report=report)


def compute_tvd_water_flux_1d(
    sw: Field3D | ArrayLike,
    flux_x: ArrayLike,
    relperm_params: dict[str, float],
    limiter: str = "minmod",
) -> NDArray[np.float64]:
    """Compute TVD/MUSCL reconstructed water flux for a 1D grid."""
    line = _line(sw)
    nx = line.size
    flux = np.asarray(flux_x, dtype=float)
    if flux.shape == (1, 1, nx + 1):
        total_flux = flux[0, 0, :]
    elif flux.shape == (nx + 1,):
        total_flux = flux
    else:
        raise ValueError(f"flux_x shape {flux.shape} does not match 1D shape")
    if not np.isfinite(total_flux).all():
        raise ValueError("flux_x must be finite")
    params = _params(relperm_params)
    slopes = compute_limited_slopes(line, limiter=limiter)
    lower = params["swi"]
    upper = 1.0 - params["sor"]
    injected_left = float(params.get("injected_sw_left", params.get("injected_sw", upper)))
    injected_right = float(params.get("injected_sw_right", params.get("injected_sw", upper)))
    water_flux = np.zeros(nx + 1, dtype=float)
    for face, total in enumerate(total_flux):
        if total == 0.0:
            continue
        if face == 0:
            reconstructed = injected_left if total > 0.0 else line[0] - 0.5 * slopes[0]
        elif face == nx:
            reconstructed = line[-1] + 0.5 * slopes[-1] if total > 0.0 else injected_right
        elif total > 0.0:
            reconstructed = line[face - 1] + 0.5 * slopes[face - 1]
        else:
            reconstructed = line[face] - 0.5 * slopes[face]
        reconstructed = float(np.clip(reconstructed, lower, upper))
        water_flux[face] = float(fractional_flow_water(reconstructed, **_fractional_kwargs(params))) * float(total)
    return water_flux.reshape(1, 1, nx + 1)


def _params(relperm_params: dict[str, float]) -> dict[str, float]:
    params = dict(DEFAULT_RELPERM_PARAMS)
    params.update(relperm_params)
    validate_saturation_params(float(params["swi"]), float(params["sor"]))
    return params


def _fractional_kwargs(params: dict[str, float]) -> dict[str, float]:
    return {
        "swi": float(params["swi"]),
        "sor": float(params["sor"]),
        "krw0": float(params["krw0"]),
        "kro0": float(params["kro0"]),
        "nw": float(params["nw"]),
        "no": float(params["no"]),
        "mu_w": float(params["mu_w"]),
        "mu_o": float(params["mu_o"]),
    }


def _validate_1d_grid(grid: Grid3D) -> None:
    if grid.ny != 1 or grid.nz != 1:
        raise NotImplementedError("enhanced TVD/MUSCL transport currently supports 1D grids only")


def _field_values(grid: Grid3D, value: Field3D | ArrayLike) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        if value.grid != grid:
            raise ValueError("Field3D grid mismatch")
        array = np.asarray(value.values, dtype=float)
    else:
        array = np.asarray(value, dtype=float)
        if array.shape == ():
            array = np.full(grid.shape, float(array), dtype=float)
    if array.shape != grid.shape:
        raise ValueError(f"field shape {array.shape} does not match grid shape {grid.shape}")
    if not np.isfinite(array).all():
        raise ValueError("field values must be finite")
    return array


def _flux_x(grid: Grid3D, flux_x: ArrayLike) -> NDArray[np.float64]:
    flux = np.asarray(flux_x, dtype=float)
    expected = (grid.nz, grid.ny, grid.nx + 1)
    if flux.shape != expected:
        raise ValueError(f"flux_x shape {flux.shape} does not match {expected}")
    if not np.isfinite(flux).all():
        raise ValueError("flux_x must be finite")
    return flux


def _line(sw: Field3D | ArrayLike) -> NDArray[np.float64]:
    values = sw.values if isinstance(sw, Field3D) else sw
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[0] != 1 or array.shape[1] != 1:
            raise ValueError("3D sw input must represent a 1D line with shape (1, 1, nx)")
        array = array[0, 0, :]
    if array.ndim != 1:
        raise ValueError("sw must be a 1D line or shape (1, 1, nx)")
    if not np.isfinite(array).all():
        raise ValueError("sw must be finite")
    return array
