"""Synthetic (and optional IMEX) validation for the 30 cm lab-box twin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    build_lab_layer_twin,
    mask_overlap,
    run_shape_discovery,
    run_time_series,
)

HERE = Path(__file__).resolve().parent


def run_synthetic(n: int = 15) -> dict[str, object]:
    twin = build_lab_layer_twin(nx=n, ny=n, nz=n, n_times=4, include_fault=False)
    history = run_time_series(twin.mesh, twin.samples, permeability_prior_m2=1.0e-12)
    well_p_err: dict[str, float] = {}
    last = history[-1]
    for name in ("INJ", "PROD"):
        c = twin.mesh.well_cell_id[name]
        i, j, k = twin.mesh.grid.ijk(c)
        truth = float(twin.samples[-1].well_pressure[name])
        well_p_err[name] = abs(float(last.pressure[k, j, i]) - truth)
    result = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=1.0e-12,
        refine=False,
        indicator_threshold=0.30,
    )
    metrics = mask_overlap(result.active_mask, twin.true_channel_mask)
    sw = twin.sw_series[-1]
    high_sw = sw >= 0.45
    sw_on_layer = float(np.mean(twin.true_channel_mask[high_sw])) if np.any(high_sw) else 0.0
    report = {
        "mode": "synthetic",
        "n": n,
        "n_cells": twin.mesh.n_cells,
        "n_highk": int(np.sum(twin.true_channel_mask)),
        "well_pressure_abs_err_Pa": well_p_err,
        "discovery": metrics,
        "high_sw_on_highk_frac": sw_on_layer,
        "horizon_relief_m": float(np.ptp(twin.z_horizon)) if twin.z_horizon is not None else None,
    }
    (HERE / "synthetic_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.save(HERE / "truth_highk_mask.npy", twin.true_channel_mask)
    np.save(HERE / "discovered_active_mask.npy", result.active_mask)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", action="store_true", help="run software twin (no CMG)")
    p.add_argument("--n", type=int, default=15)
    args = p.parse_args(argv)
    if not args.synthetic:
        print("Only --synthetic is implemented until IMEX .out exists.", file=sys.stderr)
        print("Build the deck with: python validation/black_oil/lab_box_30cm/build_lab_case.py --n 30", file=sys.stderr)
        return 2
    report = run_synthetic(args.n)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
