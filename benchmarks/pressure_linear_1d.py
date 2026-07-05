"""1D linear pressure benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_1d
from reservoir_backend.solver.velocity import compute_face_fluxes


def run_benchmark() -> dict:
    grid = Grid3D(nx=20, ny=1, nz=1, dx=5.0, dy=1.0, dz=1.0)
    left_pressure = 10.0e6
    right_pressure = 1.0e6
    k = 100.0e-15
    mu = 1.0e-3
    result = solve_steady_state_pressure_1d(grid, k, mu, left_pressure, right_pressure)
    x = (np.arange(grid.nx) + 0.5) * grid.dx
    expected = left_pressure + (right_pressure - left_pressure) * x / (grid.nx * grid.dx)
    pressure = result.pressure.values[0, 0, :]
    error = np.abs(pressure - expected)
    flux = compute_face_fluxes(grid, result.pressure, k, k, k, mu).flux_x[0, 0, 1:-1]
    max_flux_variation = float(np.max(flux) - np.min(flux)) if flux.size else 0.0
    max_pressure_error = float(np.max(error))
    relative_pressure_error = float(max_pressure_error / max(abs(left_pressure - right_pressure), 1.0))
    report = {
        "benchmark_name": "pressure_linear_1d",
        "success": bool(max_pressure_error < 1.0e-2 and max_flux_variation < 1.0e-12),
        "max_pressure_error": max_pressure_error,
        "relative_pressure_error": relative_pressure_error,
        "max_flux_variation": max_flux_variation,
        "mass_balance_error": float(result.report.get("mass_balance_error", 0.0)),
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
        "warnings": [],
    }
    return report


if __name__ == "__main__":
    print(run_benchmark())
