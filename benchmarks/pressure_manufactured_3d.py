"""3D manufactured linear pressure benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d


def run_benchmark() -> dict:
    grid = Grid3D(nx=6, ny=5, nz=4, dx=2.0, dy=3.0, dz=4.0)
    left_pressure = 12.0e6
    right_pressure = 3.0e6
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=100.0e-15,
        ky=100.0e-15,
        kz=100.0e-15,
        mu=1.0e-3,
        dirichlet_boundaries={"left": left_pressure, "right": right_pressure},
    )
    x = (np.arange(grid.nx) + 0.5) * float(grid.dx[0])
    expected_x = left_pressure + (right_pressure - left_pressure) * x / (grid.nx * float(grid.dx[0]))
    expected = np.broadcast_to(expected_x.reshape(1, 1, grid.nx), grid.shape)
    pressure = result.pressure.values
    diff = pressure - expected
    l2_error = float(np.sqrt(np.mean(diff**2)))
    linf_error = float(np.max(np.abs(diff)))
    relative_l2_error = float(l2_error / max(np.sqrt(np.mean(expected**2)), 1.0))
    return {
        "benchmark_name": "pressure_manufactured_3d",
        "success": bool(relative_l2_error < 1.0e-10 and linf_error < 1.0e-2),
        "l2_error": l2_error,
        "linf_error": linf_error,
        "relative_l2_error": relative_l2_error,
        "has_nan": bool(np.isnan(pressure).any()),
        "has_inf": bool(np.isinf(pressure).any()),
        "warnings": [],
    }


if __name__ == "__main__":
    print(run_benchmark())
