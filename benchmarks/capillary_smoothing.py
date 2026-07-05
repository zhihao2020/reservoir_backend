"""Capillary smoothing benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import advance_saturation_3d_with_capillary


def run_benchmark() -> dict:
    grid = Grid3D(nx=8, ny=3, nz=3, dx=1.0, dy=1.0, dz=1.0)
    sw0 = np.full(grid.shape, 0.35)
    sw0[:, :, : grid.nx // 2] = 0.65
    initial_gradient_norm = _gradient_norm(sw0)
    zero = _zero_fluxes(grid)
    result = advance_saturation_3d_with_capillary(
        grid, sw0, 0.2, *zero, 500.0, _relperm(), _capillary(), 1.0e-9, 1.0e-9, 1.0e-9
    )
    sw = result.sw.values
    final_gradient_norm = _gradient_norm(sw)
    return {
        "benchmark_name": "capillary_smoothing",
        "success": bool(final_gradient_norm < initial_gradient_norm and result.report["max_abs_capillary_flux"] > 0.0),
        "initial_gradient_norm": initial_gradient_norm,
        "final_gradient_norm": final_gradient_norm,
        "gradient_reduction": float(initial_gradient_norm - final_gradient_norm),
        "max_abs_capillary_flux": float(result.report["max_abs_capillary_flux"]),
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "has_nan": bool(np.isnan(sw).any()),
        "has_inf": bool(np.isinf(sw).any()),
        "warnings": [],
    }


def _gradient_norm(sw: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(sw, axis=2)))


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((grid.nz, grid.ny, grid.nx + 1)), np.zeros((grid.nz, grid.ny + 1, grid.nx)), np.zeros((grid.nz + 1, grid.ny, grid.nx)))


def _relperm() -> dict[str, float]:
    return {"swi": 0.2, "sor": 0.2, "krw0": 1.0, "kro0": 1.0, "nw": 2.0, "no": 2.0, "mu_w": 1.0e-3, "mu_o": 5.0e-3}


def _capillary() -> dict[str, float | bool | str]:
    return {"enabled": True, "model": "brooks_corey", "swi": 0.2, "sor": 0.2, "entry_pressure_pa": 1000.0, "lambda_pc": 2.0}


if __name__ == "__main__":
    print(run_benchmark())
