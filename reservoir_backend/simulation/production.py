"""Production diagnostics for lightweight sequential simulations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import compute_upwind_water_flux_3d


def build_production_summary(
    *,
    grid: Grid3D,
    sw: ArrayLike,
    flux_x: ArrayLike,
    flux_y: ArrayLike,
    flux_z: ArrayLike,
    relperm_params: dict[str, float],
    dt: float,
    producer_boundary: str = "right",
    cumulative_water: float = 0.0,
    cumulative_oil: float = 0.0,
) -> dict[str, float | str | bool | list[int]]:
    """Return oil-water production rates and water cut for one time step.

    Rates are reported as positive production magnitudes. The current MVP
    supports boundary-production summaries; it does not implement industrial
    well-control accounting.
    """
    sw_values = np.asarray(sw, dtype=float)
    fx = np.asarray(flux_x, dtype=float)
    fy = np.asarray(flux_y, dtype=float)
    fz = np.asarray(flux_z, dtype=float)
    if sw_values.shape != grid.shape:
        raise ValueError(f"sw shape {sw_values.shape} does not match grid shape {grid.shape}")

    water_x, water_y, water_z = compute_upwind_water_flux_3d(sw_values, fx, fy, fz, relperm_params)
    if producer_boundary == "right":
        total_rate = float(np.sum(np.maximum(fx[:, :, -1], 0.0)))
        water_rate = float(np.sum(np.maximum(water_x[:, :, -1], 0.0)))
    elif producer_boundary == "left":
        total_rate = float(np.sum(np.maximum(-fx[:, :, 0], 0.0)))
        water_rate = float(np.sum(np.maximum(-water_x[:, :, 0], 0.0)))
    elif producer_boundary == "back":
        total_rate = float(np.sum(np.maximum(fy[:, -1, :], 0.0)))
        water_rate = float(np.sum(np.maximum(water_y[:, -1, :], 0.0)))
    elif producer_boundary == "front":
        total_rate = float(np.sum(np.maximum(-fy[:, 0, :], 0.0)))
        water_rate = float(np.sum(np.maximum(-water_y[:, 0, :], 0.0)))
    elif producer_boundary == "top":
        total_rate = float(np.sum(np.maximum(fz[-1, :, :], 0.0)))
        water_rate = float(np.sum(np.maximum(water_z[-1, :, :], 0.0)))
    elif producer_boundary == "bottom":
        total_rate = float(np.sum(np.maximum(-fz[0, :, :], 0.0)))
        water_rate = float(np.sum(np.maximum(-water_z[0, :, :], 0.0)))
    else:
        raise ValueError("unsupported producer_boundary")

    oil_rate = max(total_rate - water_rate, 0.0)
    water_cut = 0.0 if total_rate <= 0.0 else float(np.clip(water_rate / total_rate, 0.0, 1.0))
    new_cum_water = float(cumulative_water + water_rate * float(dt))
    new_cum_oil = float(cumulative_oil + oil_rate * float(dt))
    values = np.array([total_rate, water_rate, oil_rate, water_cut, new_cum_water, new_cum_oil], dtype=float)
    return {
        "success": bool(np.isfinite(values).all()),
        "producer_boundary": producer_boundary,
        "total_liquid_rate": total_rate,
        "water_rate": water_rate,
        "oil_rate": oil_rate,
        "water_cut": water_cut,
        "cumulative_water": new_cum_water,
        "cumulative_oil": new_cum_oil,
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
        "flux_shape_x": list(fx.shape),
        "flux_shape_y": list(fy.shape),
        "flux_shape_z": list(fz.shape),
    }


def detect_breakthrough_time(
    times: Sequence[float],
    water_cuts: Sequence[float],
    threshold: float = 0.01,
) -> float | None:
    """Return the first time where water cut reaches `threshold`.

    `None` means breakthrough was not observed within the supplied curve.
    """
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be within [0, 1]")
    for time_value, water_cut in zip(times, water_cuts, strict=False):
        if float(water_cut) >= threshold:
            return float(time_value)
    return None
