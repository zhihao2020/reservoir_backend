"""Combined capillary + gravity stability benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import advance_saturation_3d_with_capillary_and_gravity


def run_benchmark() -> dict:
    grid = Grid3D(nx=6, ny=4, nz=4, dx=1.0, dy=1.0, dz=1.0)
    sw0 = np.full(grid.shape, 0.35)
    sw0[:, :, : grid.nx // 2] = 0.65
    zero = _zero_fluxes(grid)
    result = advance_saturation_3d_with_capillary_and_gravity(
        grid, sw0, 0.2, *zero, 100.0, _relperm(), _capillary(), _gravity(), 1.0e-12, 1.0e-12, 1.0e-12
    )
    sw = result.sw.values
    report = result.report
    return {
        "benchmark_name": "combined_transport_stability",
        "success": bool(
            report["max_capillary_flux"] > 0.0
            and report["max_gravity_flux"] > 0.0
            and report["max_cfl"] <= 1.0
            and abs(report["material_balance_error"]) < 1.0e-8
        ),
        "max_abs_capillary_flux": float(report["max_capillary_flux"]),
        "max_abs_gravity_flux": float(report["max_gravity_flux"]),
        "max_total_water_flux": float(report["max_total_water_flux"]),
        "max_effective_flux": float(report["max_effective_flux"]),
        "material_balance_error": float(report["material_balance_error"]),
        "max_cfl": float(report["max_cfl"]),
        "sw_min": float(np.min(sw)),
        "sw_max": float(np.max(sw)),
        "has_nan": bool(np.isnan(sw).any()),
        "has_inf": bool(np.isinf(sw).any()),
        "warnings": [],
    }


def _zero_fluxes(grid: Grid3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.zeros((grid.nz, grid.ny, grid.nx + 1)), np.zeros((grid.nz, grid.ny + 1, grid.nx)), np.zeros((grid.nz + 1, grid.ny, grid.nx)))


def _relperm() -> dict[str, float]:
    return {"swi": 0.2, "sor": 0.2, "krw0": 1.0, "kro0": 1.0, "nw": 2.0, "no": 2.0, "mu_w": 1.0e-3, "mu_o": 5.0e-3}


def _capillary() -> dict[str, float | bool | str]:
    return {"enabled": True, "model": "brooks_corey", "swi": 0.2, "sor": 0.2, "entry_pressure_pa": 1000.0, "lambda_pc": 2.0}


def _gravity() -> dict[str, float | bool | str]:
    return {"enabled": True, "g": 9.80665, "rho_w": 1000.0, "rho_o": 800.0, "depth_axis": "z", "depth_positive": "down"}


if __name__ == "__main__":
    print(run_benchmark())
