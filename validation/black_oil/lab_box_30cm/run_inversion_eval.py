"""Evaluate current sensor inversion on the 30 cm lab-box twin.

Two observation layouts:
  wells  — INJ/PROD only (as the twin is built)
  probes — wells + exclusive virtual p/S probes sampled from truth

Usage:
  python validation/black_oil/lab_box_30cm/run_inversion_eval.py
  python validation/black_oil/lab_box_30cm/run_inversion_eval.py --n 12 --ne 12 --na 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    WellPoint,
    build_lab_layer_twin,
    build_mesh,
    mask_overlap,
    place_uniform_probes,
)
from reservoir_backend.pipeline.inversion import run_sensor_inversion  # noqa: E402
from reservoir_backend.pipeline.lab_horizon import make_horizon_fn  # noqa: E402
from reservoir_backend.pipeline.state import SensorSample  # noqa: E402

HERE = Path(__file__).resolve().parent


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    b = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(b.ravel())) + 1.0e-30
    return float(np.linalg.norm((np.asarray(a, dtype=float) - b).ravel()) / den)


def _logk_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    lp = np.log(np.clip(pred, 1.0e-20, None))
    lt = np.log(np.clip(truth, 1.0e-20, None))
    return float(np.sqrt(np.mean((lp - lt) ** 2)))


def _k_ratio(k: np.ndarray, high: np.ndarray) -> float:
    inside = float(np.mean(k[high]))
    outside = float(np.mean(k[~high]))
    return inside / max(outside, 1.0e-30)


def _well_pressure_err(mesh, history, samples) -> dict[str, float]:
    last = history[-1]
    sample = samples[-1]
    out: dict[str, float] = {}
    for name, p_obs in (sample.well_pressure or {}).items():
        if name not in mesh.well_cell_id:
            continue
        i, j, k = mesh.grid.ijk(mesh.well_cell_id[name])
        out[name] = abs(float(last.pressure[k, j, i]) - float(p_obs))
    return out


def _attach_probes(twin, n_p: int, n_s: int, seed: int = 3):
    """Rebuild mesh with exclusive probes; copy truth readings into samples."""
    probes = place_uniform_probes(twin.mesh, n_p, n_s, seed=seed)
    wells = []
    for name, cid in twin.mesh.well_cell_id.items():
        i, j, k = twin.mesh.grid.ijk(cid)
        wells.append(
            WellPoint(
                name,
                float(twin.mesh.x[cid]),
                float(twin.mesh.y[cid]),
                float(twin.mesh.z[cid]),
                role=twin.mesh.well_role[name],
            )
        )
    for pr in probes:
        wells.append(WellPoint(pr.name, pr.x, pr.y, pr.z, role=pr.role))
    assert twin.mesh.bounds is not None
    b = twin.mesh.bounds
    nx, ny, nz = twin.mesh.grid.nx, twin.mesh.grid.ny, twin.mesh.grid.nz
    mesh = build_mesh(b, b.xmax / nx, b.ymax / ny, b.zmax / nz, wells=wells)
    mesh.horizon_z = twin.mesh.horizon_z or make_horizon_fn()

    samples: list[SensorSample] = []
    for ti, s in enumerate(twin.samples):
        p_map = dict(s.well_pressure)
        sat_map = dict(s.well_saturation)
        p_true = twin.pressure_series[ti]
        sw_true = twin.sw_series[ti]
        for pr in probes:
            i, j, k = mesh.grid.ijk(pr.cell_id)
            if pr.role == "observer_p":
                p_map[pr.name] = float(p_true[k, j, i])
            elif pr.role == "observer_s":
                sw = float(sw_true[k, j, i])
                sat_map[pr.name] = (sw, 1.0 - sw, 0.0)
        samples.append(
            SensorSample(
                time=s.time,
                well_pressure=p_map,
                well_saturation=sat_map,
                boundary=s.boundary,
                well_rate=dict(s.well_rate),
            )
        )
    return mesh, samples, [p.name for p in probes]


def _score(mesh, twin, result, samples, layout: str, n_probes: int, elapsed_s: float) -> dict:
    k_inv = np.asarray(result.k_mean)
    high = twin.true_channel_mask
    k_hat = (k_inv >= np.quantile(k_inv, 0.80))
    last = result.history[-1]
    sw_true = twin.sw_series[-1]
    high_sw_inv = last.sw >= 0.45
    high_sw_true = sw_true >= 0.45
    return {
        "layout": layout,
        "n_probes": n_probes,
        "n_cells": mesh.n_cells,
        "n_highk_truth": int(np.sum(high)),
        "elapsed_s": round(elapsed_s, 1),
        "well_pressure_abs_err_Pa": _well_pressure_err(mesh, result.history, samples),
        "obs_nrmse": [float(x) for x in result.observation_nrmse],
        "theta_mean": [float(x) for x in result.theta_mean],
        "k_ch_over_out_truth": _k_ratio(twin.true_k, high),
        "k_ch_over_out_inv": _k_ratio(k_inv, high),
        "k_mean_in_truth_md": float(np.mean(k_inv[high]) / 9.869233e-16),
        "k_mean_out_truth_md": float(np.mean(k_inv[~high]) / 9.869233e-16),
        "logk_rmse": _logk_rmse(k_inv, twin.true_k),
        "sw_field_rel_l2_last": _rel_l2(last.sw, sw_true),
        "p_field_rel_l2_last": _rel_l2(last.pressure, twin.pressure_series[-1]),
        "highk_mask_from_k_quantile": mask_overlap(k_hat, high),
        "delta_sw_dice_inv_vs_truth": mask_overlap(high_sw_inv, high_sw_true)
        if np.any(high_sw_inv) and np.any(high_sw_true)
        else {"dice": 0.0, "precision": 0.0, "recall": 0.0},
        "high_sw_on_true_highk": float(np.mean(high[high_sw_inv])) if np.any(high_sw_inv) else 0.0,
        "notes": result.notes,
    }


def run_one(
    twin,
    *,
    layout: str,
    n_p: int,
    n_s: int,
    ne: int,
    na: int,
    seed: int,
) -> dict:
    if layout == "probes":
        mesh, samples, names = _attach_probes(twin, n_p, n_s)
        n_probes = len(names)
    else:
        mesh, samples, n_probes = twin.mesh, twin.samples, 0
    t0 = time.perf_counter()
    result = run_sensor_inversion(
        mesh,
        samples,
        permeability_prior_m2=float(np.median(twin.true_k)),
        porosity_prior=float(np.mean(twin.true_phi)),
        ne=ne,
        n_assimilations=na,
        max_times=len(samples),
        n_k_iterations=2,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    report = _score(mesh, twin, result, samples, layout, n_probes, elapsed)
    stem = HERE / f"inversion_{layout}"
    np.save(stem.with_suffix(".k.npy"), result.k_mean)
    if result.history:
        np.save(HERE / f"inversion_{layout}_sw.npy", result.history[-1].sw)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ne", type=int, default=12)
    p.add_argument("--na", type=int, default=3)
    p.add_argument("--n-p", type=int, default=4)
    p.add_argument("--n-s", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--layout",
        choices=("both", "wells", "probes"),
        default="both",
    )
    args = p.parse_args(argv)

    twin = build_lab_layer_twin(
        nx=args.n, ny=args.n, nz=args.n, n_times=4, include_fault=False
    )
    layouts = ["wells", "probes"] if args.layout == "both" else [args.layout]
    reports = []
    for layout in layouts:
        print(f"running inversion layout={layout} n={args.n} ne={args.ne} Na={args.na} ...", flush=True)
        reports.append(
            run_one(
                twin,
                layout=layout,
                n_p=args.n_p,
                n_s=args.n_s,
                ne=args.ne,
                na=args.na,
                seed=args.seed,
            )
        )
        print(json.dumps(reports[-1], indent=2), flush=True)

    out = {
        "model": "lab_box_30cm mountain-drape",
        "n": args.n,
        "ne": args.ne,
        "n_assimilations": args.na,
        "truth_k_contrast": float(np.max(twin.true_k) / np.min(twin.true_k)),
        "horizon_relief_m": float(np.ptp(twin.z_horizon)) if twin.z_horizon is not None else None,
        "results": reports,
    }
    dest = HERE / "inversion_eval_report.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
