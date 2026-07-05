"""Three-phase closure and material-balance benchmark."""

from __future__ import annotations

import numpy as np

from reservoir_backend.solver.three_phase_transport import advance_three_phase_saturation_3d


def run_benchmark() -> dict:
    sw = np.full((3, 3, 4), 0.30)
    sg = np.full_like(sw, 0.10)
    fx = np.zeros((3, 3, 5))
    fy = np.zeros((3, 4, 4))
    fz = np.zeros((4, 3, 4))
    sw_new, sg_new, so_new, report = advance_three_phase_saturation_3d(fx, fy, fz, sw, sg, 0.2, 1.0, 100.0, _params())
    closure_error = float(np.max(np.abs(sw_new + sg_new + so_new - 1.0)))
    return {
        "benchmark_name": "three_phase_closure",
        "success": bool(
            closure_error < 1.0e-12
            and abs(report["water_balance_error"]) < 1.0e-12
            and abs(report["gas_balance_error"]) < 1.0e-12
            and abs(report["oil_balance_error"]) < 1.0e-12
        ),
        "closure_error_max": closure_error,
        "sw_min": float(np.min(sw_new)),
        "sw_max": float(np.max(sw_new)),
        "sg_min": float(np.min(sg_new)),
        "sg_max": float(np.max(sg_new)),
        "so_min": float(np.min(so_new)),
        "so_max": float(np.max(so_new)),
        "water_balance_error": float(report["water_balance_error"]),
        "gas_balance_error": float(report["gas_balance_error"]),
        "oil_balance_error": float(report["oil_balance_error"]),
        "has_nan": bool(np.isnan(sw_new).any() or np.isnan(sg_new).any() or np.isnan(so_new).any()),
        "has_inf": bool(np.isinf(sw_new).any() or np.isinf(sg_new).any() or np.isinf(so_new).any()),
        "warnings": [],
    }


def _params() -> dict[str, float]:
    return {"swi": 0.2, "sor": 0.2, "sgc": 0.05, "krw0": 0.3, "kro0": 0.8, "krg0": 0.6, "nw": 2.0, "no": 2.0, "ng": 2.0, "mu_w": 1.0e-3, "mu_o": 5.0e-3, "mu_g": 1.0e-5}


if __name__ == "__main__":
    print(run_benchmark())
