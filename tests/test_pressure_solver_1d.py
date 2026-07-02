from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_1d


def test_1d_linear_pressure_dirichlet_boundary_faces() -> None:
    grid = Grid3D(nx=10, ny=1, nz=1, dx=10.0, dy=1.0, dz=1.0)
    left_pressure = 10.0e6
    right_pressure = 0.0

    result = solve_steady_state_pressure_1d(
        grid=grid,
        kx=100.0e-15,
        mu=1.0e-3,
        left_pressure=left_pressure,
        right_pressure=right_pressure,
    )

    cell_centers = (np.arange(grid.nx) + 0.5) * grid.dx
    domain_length = grid.nx * grid.dx
    expected = left_pressure + (right_pressure - left_pressure) * cell_centers / domain_length

    assert np.allclose(result.pressure.values[0, 0, :], expected, rtol=1e-10, atol=1e-3)
    assert result.report["status"] == "converged"
