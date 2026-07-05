"""Qualitative Buckley-Leverett-style 1D waterflood benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.saturation_solver import advance_saturation_1d


def run_benchmark() -> dict:
    grid = Grid3D(nx=40, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    params = _relperm()
    sw: Field3D | np.ndarray = Field3D.from_constant(grid, params["swi"], name="sw", unit="fraction")
    flux_x = np.full((1, 1, grid.nx + 1), 1.0e-5)
    report = {}
    for _ in range(40):
        step = advance_saturation_1d(grid, sw, 0.2, flux_x, 200.0, params)
        sw = step.sw
        report = step.report
    values = sw.values
    wet = np.flatnonzero(values[0, 0, :] > params["swi"] + 1.0e-4)
    front_position = int(wet[-1]) if wet.size else 0
    return {
        "benchmark_name": "buckley_leverett_1d",
        "success": bool(front_position > 0 and _bounded(values, params) and report["max_cfl"] <= 1.0),
        "sw_min": float(np.min(values)),
        "sw_max": float(np.max(values)),
        "front_position_estimate": front_position,
        "material_balance_error": float(report["material_balance_error"]),
        "max_cfl": float(report["max_cfl"]),
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
        "warnings": [],
    }


def _relperm() -> dict[str, float]:
    return {"swi": 0.2, "sor": 0.2, "krw0": 1.0, "kro0": 1.0, "nw": 2.0, "no": 2.0, "mu_w": 1.0e-3, "mu_o": 5.0e-3}


def _bounded(sw: np.ndarray, params: dict[str, float]) -> bool:
    return bool(np.min(sw) >= params["swi"] - 1.0e-12 and np.max(sw) <= 1.0 - params["sor"] + 1.0e-12)


if __name__ == "__main__":
    print(run_benchmark())
