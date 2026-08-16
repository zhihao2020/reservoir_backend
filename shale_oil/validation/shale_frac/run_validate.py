"""Validate automatic inversion on the shale-oil fracture twin (no CMG)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    build_shale_fracture_twin,
    run_automatic_inversion,
    run_time_series,
)

HERE = Path(__file__).resolve().parent


def _k_ratio(k, mask) -> float:
    return float(np.mean(k[mask]) / max(float(np.mean(k[~mask])), 1.0e-30))


def main() -> int:
    twin = build_shale_fracture_twin(nx=12, ny=10, nz=5, n_fracs=4, n_perf=6, n_times=4)
    prior = 1.0e-16
    pf = run_time_series(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=prior,
        viscosity_pa_s=2.0e-3,
        n_k_iterations=2,
    )
    auto = run_automatic_inversion(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=prior,
        viscosity_pa_s=2.0e-3,
        ne=8,
        n_assimilations=1,
        n_k_iterations=2,
    )
    name = next(n for n, r in twin.mesh.well_role.items() if r == "producer")
    i, j, k = twin.mesh.grid.ijk(twin.mesh.well_cell_id[name])
    p_obs = float(twin.samples[-1].well_pressure[name])
    p_auto = float(auto.history[-1].pressure[k, j, i])
    report = {
        "case": "shale_horizontal_frac_strips",
        "n_cells": twin.mesh.n_cells,
        "n_frac_cells": int(np.sum(twin.true_channel_mask)),
        "n_completions": sum(1 for r in twin.mesh.well_role.values() if r == "producer"),
        "truth_k_contrast": float(np.max(twin.true_k) / np.min(twin.true_k)),
        "well_pressure_abs_err_Pa": abs(p_auto - p_obs),
        "k_ratio_point_first": _k_ratio(pf[-1].permeability, twin.true_channel_mask),
        "k_ratio_automatic": _k_ratio(auto.k_mean, twin.true_channel_mask),
        "stack_weights": auto.member_weights,
        "frac_theta": any("frac θ" in n for n in auto.notes),
        "notes": auto.notes,
    }
    dest = HERE / "validation_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(HERE / "truth_frac_mask.npy", twin.true_channel_mask)
    np.save(HERE / "k_automatic.npy", auto.k_mean)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
