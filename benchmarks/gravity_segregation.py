"""Gravity segregation benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import advance_saturation_3d_with_gravity, compute_total_water_flux_3d_with_gravity


def run_benchmark() -> dict:
    grid = Grid3D(nx=3, ny=3, nz=5, dx=1.0, dy=1.0, dz=1.0)
    sw0 = np.full(grid.shape, 0.5)
    zero = _zero_fluxes(grid)
    _, gravity_flux, _, _ = compute_total_water_flux_3d_with_gravity(
        grid, sw0, *zero, _relperm(), _gravity(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    result = advance_saturation_3d_with_gravity(grid, sw0, 0.2, *zero, 1000.0, _relperm(), _gravity(), 1.0e-12, 1.0e-12, 1.0e-12)
    sw = result.sw.values
    gz_internal = gravity_flux[2][1:-1, :, :]
    observed_sign = float(np.sign(np.mean(gz_internal)))
    bottom_change = float(np.mean(sw[0, :, :] - sw0[0, :, :]))
    top_change = float(np.mean(sw[-1, :, :] - sw0[-1, :, :]))
    return {
        "benchmark_name": "gravity_segregation",
        "success": bool(observed_sign < 0.0 and bottom_change > 0.0 and top_change < 0.0),
        "expected_gravity_flux_sign": -1.0,
        "observed_gravity_flux_sign": observed_sign,
        "bottom_sw_change": bottom_change,
        "top_sw_change": top_change,
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "has_nan": bool(np.isnan(sw).any() or np.isnan(gz_internal).any()),
        "has_inf": bool(np.isinf(sw).any() or np.isinf(gz_internal).any()),
        "warnings": [],
    }


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((grid.nz, grid.ny, grid.nx + 1)), np.zeros((grid.nz, grid.ny + 1, grid.nx)), np.zeros((grid.nz + 1, grid.ny, grid.nx)))


def _relperm() -> dict[str, float]:
    return {"swi": 0.2, "sor": 0.2, "krw0": 1.0, "kro0": 1.0, "nw": 2.0, "no": 2.0, "mu_w": 1.0e-3, "mu_o": 5.0e-3}


def _gravity() -> dict[str, float | bool | str]:
    return {"enabled": True, "g": 9.80665, "rho_w": 1000.0, "rho_o": 800.0, "depth_axis": "z", "depth_positive": "down"}


if __name__ == "__main__":
    print(run_benchmark())
