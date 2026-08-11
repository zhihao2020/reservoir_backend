"""Lightweight IMPES-style sequential pressure-saturation loop.

The implementation composes existing pressure, face-flux, CFL, and explicit
saturation transport routines. It is intentionally sequential and explicit in
saturation; it is not a fully implicit simulator or black-oil model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.cfl import compute_cfl_number, estimate_stable_dt
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d
from reservoir_backend.solver.relperm import (
    corey_relative_permeability,
    fractional_flow_water,
    oil_mobility,
    water_mobility,
)
from reservoir_backend.solver.saturation_solver import DEFAULT_RELPERM_PARAMS, advance_saturation_3d
from reservoir_backend.solver.velocity import FaceFluxes, compute_face_fluxes
from reservoir_backend.simulation.production import build_production_summary, detect_breakthrough_time


@dataclass(frozen=True)
class IMPESConfig:
    """Configuration for the lightweight sequential oil-water loop."""

    grid: Grid3D
    phi: float | ArrayLike | Field3D
    kx: float | ArrayLike | Field3D
    ky: float | ArrayLike | Field3D
    kz: float | ArrayLike | Field3D
    initial_sw: float | ArrayLike | Field3D
    dt: float
    num_steps: int
    pressure_boundaries: dict[str, float]
    relperm_params: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RELPERM_PARAMS))
    max_cfl: float = 0.9
    wells: list[Well] = field(default_factory=list)
    producer_boundary: str = "right"
    breakthrough_water_cut: float = 0.01
    case_id: str = "synthetic_impes_waterflood"


@dataclass(frozen=True)
class IMPESStepResult:
    """Output from one sequential pressure-saturation time step."""

    step_index: int
    time: float
    pressure: Field3D
    face_fluxes: FaceFluxes
    sw: Field3D
    mobility_report: dict[str, float | bool]
    cfl_report: dict[str, object]
    saturation_report: dict[str, object]
    production_summary: dict[str, object]
    mass_balance_error: float
    pressure_report: dict[str, object]


@dataclass(frozen=True)
class IMPESRunResult:
    """Output from a multi-step IMPES-style run."""

    config: IMPESConfig
    steps: list[IMPESStepResult]
    production_curve: list[dict[str, float | int]]
    breakthrough_time: float | None
    summary: dict[str, object]


def run_impes_step(
    *,
    config: IMPESConfig,
    sw: Field3D | ArrayLike,
    step_index: int = 0,
    cumulative_water: float = 0.0,
    cumulative_oil: float = 0.0,
) -> IMPESStepResult:
    """Run one pressure -> flux -> saturation sequential coupling step."""
    grid = config.grid
    sw_values = _field_values(grid, sw, "sw")
    phi_values = _field_values(grid, config.phi, "phi")
    kx = _field_values(grid, config.kx, "kx")
    ky = _field_values(grid, config.ky, "ky")
    kz = _field_values(grid, config.kz, "kz")
    params = _normalized_relperm_params(config.relperm_params)
    mobility = compute_mobility_fields(sw_values, params)

    kx_eff = kx * mobility["lambda_t"]
    ky_eff = ky * mobility["lambda_t"]
    kz_eff = kz * mobility["lambda_t"]
    pressure_result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=kx_eff,
        ky=ky_eff,
        kz=kz_eff,
        mu=1.0,
        dirichlet_boundaries=config.pressure_boundaries,
        wells=config.wells,
    )
    noflow_fluxes = compute_face_fluxes(
        grid=grid,
        pressure=pressure_result.pressure,
        kx=kx_eff,
        ky=ky_eff,
        kz=kz_eff,
        mu=1.0,
    )
    face_fluxes = _add_dirichlet_boundary_fluxes(
        grid=grid,
        pressure=pressure_result.pressure.values,
        kx=kx_eff,
        ky=ky_eff,
        kz=kz_eff,
        fluxes=noflow_fluxes,
        boundaries=config.pressure_boundaries,
    )
    cfl_field, cfl_report = compute_cfl_number(
        grid=grid,
        phi=phi_values,
        flux_x=face_fluxes.flux_x,
        flux_y=face_fluxes.flux_y,
        flux_z=face_fluxes.flux_z,
        dt=config.dt,
    )
    cfl_report["max_cfl_allowed"] = float(config.max_cfl)
    cfl_report["stable"] = bool(cfl_report["max_cfl"] <= config.max_cfl and not cfl_report["has_nan"] and not cfl_report["has_inf"])
    cfl_report["suggested_stable_dt"] = float(
        estimate_stable_dt(
            grid=grid,
            phi=phi_values,
            flux_x=face_fluxes.flux_x,
            flux_y=face_fluxes.flux_y,
            flux_z=face_fluxes.flux_z,
            max_cfl=config.max_cfl,
        )
    )
    if not bool(cfl_report["stable"]):
        raise ValueError(
            f"IMPES step CFL is unstable: max_cfl={cfl_report['max_cfl']} "
            f"allowed={config.max_cfl}"
        )

    saturation_result = advance_saturation_3d(
        grid=grid,
        sw=sw_values,
        phi=phi_values,
        flux_x=face_fluxes.flux_x,
        flux_y=face_fluxes.flux_y,
        flux_z=face_fluxes.flux_z,
        dt=config.dt,
        relperm_params=params,
        max_cfl=config.max_cfl,
    )
    production = build_production_summary(
        grid=grid,
        sw=sw_values,
        flux_x=face_fluxes.flux_x,
        flux_y=face_fluxes.flux_y,
        flux_z=face_fluxes.flux_z,
        relperm_params=params,
        dt=config.dt,
        producer_boundary=config.producer_boundary,
        cumulative_water=cumulative_water,
        cumulative_oil=cumulative_oil,
    )
    mobility_report = _mobility_report(mobility)
    mobility_report["fractional_flow_min"] = float(np.min(mobility["fw"]))
    mobility_report["fractional_flow_max"] = float(np.max(mobility["fw"]))
    return IMPESStepResult(
        step_index=int(step_index),
        time=float((step_index + 1) * config.dt),
        pressure=pressure_result.pressure,
        face_fluxes=face_fluxes,
        sw=saturation_result.sw,
        mobility_report=mobility_report,
        cfl_report=_json_ready(cfl_report),
        saturation_report=_json_ready(saturation_result.report),
        production_summary=_json_ready(production),
        mass_balance_error=float(saturation_result.report["material_balance_error"]),
        pressure_report=_json_ready(pressure_result.report),
    )


def run_impes_simulation(config: IMPESConfig) -> IMPESRunResult:
    """Run a multi-step IMPES-style waterflood case."""
    if config.num_steps <= 0:
        raise ValueError("num_steps must be positive")
    sw = Field3D(grid=config.grid, values=_field_values(config.grid, config.initial_sw, "initial_sw"), name="sw", unit="fraction")
    steps: list[IMPESStepResult] = []
    production_curve: list[dict[str, float | int]] = []
    cumulative_water = 0.0
    cumulative_oil = 0.0
    for step_index in range(config.num_steps):
        step = run_impes_step(
            config=config,
            sw=sw,
            step_index=step_index,
            cumulative_water=cumulative_water,
            cumulative_oil=cumulative_oil,
        )
        steps.append(step)
        production = step.production_summary
        cumulative_water = float(production["cumulative_water"])
        cumulative_oil = float(production["cumulative_oil"])
        production_curve.append(
            {
                "step": int(step_index + 1),
                "time": float(step.time),
                "total_liquid_rate": float(production["total_liquid_rate"]),
                "water_rate": float(production["water_rate"]),
                "oil_rate": float(production["oil_rate"]),
                "water_cut": float(production["water_cut"]),
                "cumulative_water": cumulative_water,
                "cumulative_oil": cumulative_oil,
            }
        )
        sw = step.sw

    breakthrough_time = detect_breakthrough_time(
        [entry["time"] for entry in production_curve],
        [entry["water_cut"] for entry in production_curve],
        threshold=config.breakthrough_water_cut,
    )
    summary = build_impes_summary(config=config, steps=steps, production_curve=production_curve, breakthrough_time=breakthrough_time)
    return IMPESRunResult(
        config=config,
        steps=steps,
        production_curve=production_curve,
        breakthrough_time=breakthrough_time,
        summary=summary,
    )


def compute_mobility_fields(sw: ArrayLike, relperm_params: dict[str, float]) -> dict[str, NDArray[np.float64]]:
    """Compute Sw-dependent relperm, mobility, total mobility, and water fraction."""
    values = np.asarray(sw, dtype=float)
    params = _normalized_relperm_params(relperm_params)
    krw, kro = corey_relative_permeability(
        values,
        params["swi"],
        params["sor"],
        params["krw0"],
        params["kro0"],
        params["nw"],
        params["no"],
    )
    lambda_w = water_mobility(krw, params["mu_w"])
    lambda_o = oil_mobility(kro, params["mu_o"])
    lambda_t = np.asarray(lambda_w, dtype=float) + np.asarray(lambda_o, dtype=float)
    relperm_args = _relperm_function_args(params)
    fw = np.asarray(fractional_flow_water(values, **relperm_args), dtype=float)
    if np.any(~np.isfinite(lambda_t)) or np.any(lambda_t <= 0.0):
        raise ValueError("total mobility must be positive and finite")
    return {
        "krw": np.asarray(krw, dtype=float),
        "kro": np.asarray(kro, dtype=float),
        "lambda_w": np.asarray(lambda_w, dtype=float),
        "lambda_o": np.asarray(lambda_o, dtype=float),
        "lambda_t": lambda_t,
        "fw": fw,
    }


def build_impes_summary(
    *,
    config: IMPESConfig,
    steps: list[IMPESStepResult],
    production_curve: list[dict[str, float | int]],
    breakthrough_time: float | None,
) -> dict[str, object]:
    """Build a JSON-serializable IMPES run summary."""
    pressure_min = min(float(np.min(step.pressure.values)) for step in steps)
    pressure_max = max(float(np.max(step.pressure.values)) for step in steps)
    sw_min = min(float(np.min(step.sw.values)) for step in steps)
    sw_max = max(float(np.max(step.sw.values)) for step in steps)
    max_cfl = max(float(step.cfl_report["max_cfl"]) for step in steps)
    max_mass_balance_error = max(float(step.mass_balance_error) for step in steps)
    max_flux = max(_max_abs_flux(step.face_fluxes) for step in steps)
    has_nan = any(
        bool(np.isnan(step.pressure.values).any())
        or bool(np.isnan(step.sw.values).any())
        or bool(step.saturation_report.get("has_nan", False))
        for step in steps
    )
    has_inf = any(
        bool(np.isinf(step.pressure.values).any())
        or bool(np.isinf(step.sw.values).any())
        or bool(step.saturation_report.get("has_inf", False))
        for step in steps
    )
    return {
        "benchmark_name": "impes_sequential_loop",
        "case_id": config.case_id,
        "success": bool(not has_nan and not has_inf and sw_min >= 0.0 and sw_max <= 1.0 and max_cfl <= config.max_cfl),
        "num_steps": len(steps),
        "grid_shape": list(config.grid.shape),
        "dt": float(config.dt),
        "pressure_min": pressure_min,
        "pressure_max": pressure_max,
        "sw_min": sw_min,
        "sw_max": sw_max,
        "max_cfl": max_cfl,
        "max_flux": max_flux,
        "max_mass_balance_error": max_mass_balance_error,
        "breakthrough_time": breakthrough_time,
        "final_water_cut": float(production_curve[-1]["water_cut"]) if production_curve else 0.0,
        "production_curve": production_curve,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "warnings": [],
        "limitations": [
            "Sequential pressure-saturation coupling only; no fully implicit simulator.",
            "Oil-water incompressible MVP only; no black-oil PVT.",
            "Boundary-production summary only; no complex well-control model.",
        ],
    }


def create_synthetic_waterflood_case() -> IMPESConfig:
    """Return a small deterministic 3D waterflood case for tests and reports."""
    grid = Grid3D(nx=8, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    params = dict(DEFAULT_RELPERM_PARAMS)
    params["injected_sw"] = 1.0 - params["sor"]
    return IMPESConfig(
        grid=grid,
        phi=0.25,
        kx=1.0e-8,
        ky=1.0e-8,
        kz=1.0e-8,
        initial_sw=np.full(grid.shape, params["swi"], dtype=float),
        dt=500.0,
        num_steps=40,
        pressure_boundaries={"left": 100.0, "right": 0.0},
        relperm_params=params,
        max_cfl=0.8,
        producer_boundary="right",
        breakthrough_water_cut=1.0e-6,
        case_id="synthetic_impes_waterflood",
    )


def _add_dirichlet_boundary_fluxes(
    *,
    grid: Grid3D,
    pressure: NDArray[np.float64],
    kx: NDArray[np.float64],
    ky: NDArray[np.float64],
    kz: NDArray[np.float64],
    fluxes: FaceFluxes,
    boundaries: dict[str, float],
) -> FaceFluxes:
    fx = np.asarray(fluxes.flux_x, dtype=float).copy()
    fy = np.asarray(fluxes.flux_y, dtype=float).copy()
    fz = np.asarray(fluxes.flux_z, dtype=float).copy()
    if "left" in boundaries:
        t = 2.0 * kx[:, :, 0] * (grid.spacing_j[None, :] * grid.spacing_k[:, None]) / grid.spacing_i[0]
        fx[:, :, 0] = t * (float(boundaries["left"]) - pressure[:, :, 0])
    if "right" in boundaries:
        t = 2.0 * kx[:, :, -1] * (grid.spacing_j[None, :] * grid.spacing_k[:, None]) / grid.spacing_i[-1]
        fx[:, :, -1] = t * (pressure[:, :, -1] - float(boundaries["right"]))
    if "front" in boundaries:
        t = 2.0 * ky[:, 0, :] * (grid.spacing_i[None, :] * grid.spacing_k[:, None]) / grid.spacing_j[0]
        fy[:, 0, :] = t * (float(boundaries["front"]) - pressure[:, 0, :])
    if "back" in boundaries:
        t = 2.0 * ky[:, -1, :] * (grid.spacing_i[None, :] * grid.spacing_k[:, None]) / grid.spacing_j[-1]
        fy[:, -1, :] = t * (pressure[:, -1, :] - float(boundaries["back"]))
    if "bottom" in boundaries:
        t = 2.0 * kz[0, :, :] * (grid.spacing_i[None, :] * grid.spacing_j[:, None]) / grid.spacing_k[0]
        fz[0, :, :] = t * (float(boundaries["bottom"]) - pressure[0, :, :])
    if "top" in boundaries:
        t = 2.0 * kz[-1, :, :] * (grid.spacing_i[None, :] * grid.spacing_j[:, None]) / grid.spacing_k[-1]
        fz[-1, :, :] = t * (pressure[-1, :, :] - float(boundaries["top"]))
    return FaceFluxes(flux_x=fx, flux_y=fy, flux_z=fz)


def _field_values(grid: Grid3D, value: float | ArrayLike | Field3D, name: str) -> NDArray[np.float64]:
    if isinstance(value, Field3D):
        if value.grid != grid:
            raise ValueError(f"{name} Field3D is defined on a different grid")
        values = value.values.astype(float, copy=False)
    else:
        values = np.asarray(value, dtype=float)
        if values.shape == ():
            values = np.full(grid.shape, float(values), dtype=float)
        elif values.shape != grid.shape:
            raise ValueError(f"{name} shape {values.shape} does not match grid shape {grid.shape}")
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(f"{name} must be finite")
    return values.copy()


def _normalized_relperm_params(params: dict[str, float]) -> dict[str, float]:
    normalized = dict(DEFAULT_RELPERM_PARAMS)
    normalized.update(params)
    return {key: float(value) for key, value in normalized.items()}


def _relperm_function_args(params: dict[str, float]) -> dict[str, float]:
    keys = ("swi", "sor", "krw0", "kro0", "nw", "no", "mu_w", "mu_o")
    return {key: float(params[key]) for key in keys}


def _mobility_report(mobility: dict[str, NDArray[np.float64]]) -> dict[str, float | bool]:
    arrays = [mobility["lambda_w"], mobility["lambda_o"], mobility["lambda_t"], mobility["fw"]]
    return {
        "lambda_w_min": float(np.min(mobility["lambda_w"])),
        "lambda_w_max": float(np.max(mobility["lambda_w"])),
        "lambda_o_min": float(np.min(mobility["lambda_o"])),
        "lambda_o_max": float(np.max(mobility["lambda_o"])),
        "lambda_t_min": float(np.min(mobility["lambda_t"])),
        "lambda_t_max": float(np.max(mobility["lambda_t"])),
        "has_nan": bool(any(np.isnan(array).any() for array in arrays)),
        "has_inf": bool(any(np.isinf(array).any() for array in arrays)),
    }


def _max_abs_flux(face_fluxes: FaceFluxes) -> float:
    return float(
        max(
            np.max(np.abs(face_fluxes.flux_x)),
            np.max(np.abs(face_fluxes.flux_y)),
            np.max(np.abs(face_fluxes.flux_z)),
        )
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value
