"""Cross-scenario inversion eval: same knobs, train/val probes, no case-specific prior.

Default path is point-first. Optional full-grid ES-MDA. Channel-tube is a
comparison column only.

Usage:
  python validation/black_oil/inversion_generality/run_eval.py
  python validation/black_oil/inversion_generality/run_eval.py --methods point_first --n 12
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
    build_channel_twin,
    build_faulted_channel_twin,
    build_lab_layer_twin,
    build_mesh,
    drop_probes_from_samples,
    holdout_probe_rmse,
    mask_overlap,
    place_uniform_probes,
    run_time_series,
    split_exclusive_probes,
)
from reservoir_backend.pipeline.lab_horizon import make_horizon_fn  # noqa: E402
from reservoir_backend.pipeline.state import SensorSample  # noqa: E402

HERE = Path(__file__).resolve().parent

# Frozen knobs — same for every twin.
K_PRIOR_M2 = 1.0e-13
CORR_LEN_CELLS = 3.0
VAL_FRAC = 0.30
SPLIT_SEED = 0
PROBE_SEED = 3


def _rel_l2(a, b) -> float:
    b = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(b.ravel())) + 1.0e-30
    return float(np.linalg.norm((np.asarray(a, dtype=float) - b).ravel()) / den)


def _k_ratio(k, high) -> float:
    if not np.any(high) or not np.any(~high):
        return float("nan")
    return float(np.mean(k[high]) / max(float(np.mean(k[~high])), 1.0e-30))


def _well_p_err(mesh, history, samples) -> dict[str, float]:
    last = history[-1]
    sample = samples[-1]
    out = {}
    for name, role in mesh.well_role.items():
        if role not in ("injector", "producer"):
            continue
        obs = (sample.well_pressure or {}).get(name)
        if obs is None:
            continue
        i, j, k = mesh.grid.ijk(mesh.well_cell_id[name])
        out[name] = abs(float(last.pressure[k, j, i]) - float(obs))
    return out


def attach_probes(twin, n_total: int):
    if n_total <= 0:
        return twin.mesh, list(twin.samples), []
    n_p = n_total // 2
    n_s = n_total - n_p
    probes = place_uniform_probes(twin.mesh, n_p, n_s, seed=PROBE_SEED)
    wells = []
    for name, cid in twin.mesh.well_cell_id.items():
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
    g = twin.mesh.grid
    mesh = build_mesh(b, b.xmax / g.nx, b.ymax / g.ny, b.zmax / g.nz, wells=wells)
    if getattr(twin.mesh, "horizon_z", None) is not None:
        mesh.horizon_z = twin.mesh.horizon_z
    elif twin.z_horizon is not None:
        mesh.horizon_z = make_horizon_fn()
    samples = []
    for ti, s in enumerate(twin.samples):
        p_map = dict(s.well_pressure)
        sat_map = dict(s.well_saturation)
        p_true = twin.pressure_series[ti]
        sw_true = twin.sw_series[ti]
        for pr in probes:
            i, j, k = mesh.grid.ijk(pr.cell_id)
            if pr.role == "observer_p":
                p_map[pr.name] = float(p_true[k, j, i])
            else:
                sw = float(sw_true[k, j, i])
                sat_map[pr.name] = (sw, max(0.0, 1.0 - sw), 0.0)
        samples.append(
            SensorSample(
                time=s.time,
                well_pressure=p_map,
                well_saturation=sat_map,
                boundary=s.boundary,
                well_rate=dict(s.well_rate),
            )
        )
    return mesh, samples, probes


def build_twin(name: str, n: int):
    if name == "channel":
        return build_channel_twin(nx=n, ny=max(n - 2, 6), nz=max(n // 3, 3), n_times=4)
    if name == "fault":
        return build_faulted_channel_twin(nx=n, ny=max(n - 2, 6), nz=max(n // 3, 3), n_times=4)
    if name == "lab_drape":
        return build_lab_layer_twin(nx=n, ny=n, nz=n, n_times=4, include_fault=False)
    raise ValueError(name)


def run_case(twin_name: str, n: int, n_probes: int, method: str, ne: int, na: int) -> dict:
    twin = build_twin(twin_name, n)
    mesh, samples_full, _probes = attach_probes(twin, n_probes)
    split = split_exclusive_probes(mesh, val_frac=VAL_FRAC, seed=SPLIT_SEED)
    train = drop_probes_from_samples(samples_full, split.val_names)
    t0 = time.perf_counter()
    if method == "point_first":
        history = run_time_series(
            mesh, train, permeability_prior_m2=K_PRIOR_M2, n_k_iterations=2
        )
    elif method == "grid_esmda":
        history = run_time_series(
            mesh,
            train,
            permeability_prior_m2=K_PRIOR_M2,
            n_k_iterations=1,
            k_prior="grid_esmda",
            esmda_ne=ne,
            esmda_assimilations=na,
        )
    elif method == "channel_tube":
        history = run_time_series(
            mesh,
            train,
            permeability_prior_m2=K_PRIOR_M2,
            n_k_iterations=1,
            k_prior="channel_tube",
            esmda_ne=ne,
            esmda_assimilations=na,
        )
    else:
        raise ValueError(method)
    elapsed = time.perf_counter() - t0
    last = history[-1]
    hold = holdout_probe_rmse(mesh, history, samples_full, split)
    high = twin.true_channel_mask
    sw_true = twin.sw_series[-1]
    high_sw_inv = last.sw >= 0.45
    high_sw_true = sw_true >= 0.45
    return {
        "twin": twin_name,
        "method": method,
        "n_grid": n,
        "n_probes": n_probes,
        "n_cells": mesh.n_cells,
        "elapsed_s": round(elapsed, 1),
        "val_names": sorted(split.val_names),
        "train_probe_names": sorted(split.train_names),
        "well_pressure_abs_err_Pa": _well_p_err(mesh, history, train),
        "holdout": hold,
        "k_ch_over_out": _k_ratio(last.permeability, high),
        "p_field_rel_l2": _rel_l2(last.pressure, twin.pressure_series[-1]),
        "sw_field_rel_l2": _rel_l2(last.sw, sw_true),
        "delta_sw_dice": mask_overlap(high_sw_inv, high_sw_true)
        if np.any(high_sw_inv) and np.any(high_sw_true)
        else {"dice": 0.0},
        "high_sw_on_true_highk": float(np.mean(high[high_sw_inv])) if np.any(high_sw_inv) else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ne", type=int, default=8)
    p.add_argument("--na", type=int, default=2)
    p.add_argument(
        "--twins",
        default="channel,fault,lab_drape",
        help="comma list: channel,fault,lab_drape",
    )
    p.add_argument("--probes", default="0,8,16")
    p.add_argument(
        "--methods",
        default="point_first,grid_esmda",
        help="point_first,grid_esmda,channel_tube",
    )
    args = p.parse_args(argv)
    twins = [x.strip() for x in args.twins.split(",") if x.strip()]
    ns = [int(x) for x in args.probes.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    rows = []
    for twin_name in twins:
        for n_probes in ns:
            for method in methods:
                if method != "point_first" and n_probes == 0:
                    # wells-only ES-MDA is allowed but slow; still run
                    pass
                print(
                    f"{twin_name} N={n_probes} {method} ...",
                    flush=True,
                )
                row = run_case(twin_name, args.n, n_probes, method, args.ne, args.na)
                rows.append(row)
                hold = row["holdout"]
                print(
                    json.dumps(
                        {
                            "twin": twin_name,
                            "N": n_probes,
                            "method": method,
                            "val_p_rmse": hold.get("val_p_rmse_Pa"),
                            "val_sw_rmse": hold.get("val_sw_rmse"),
                            "well_p": row["well_pressure_abs_err_Pa"],
                            "k_ratio": row["k_ch_over_out"],
                            "s": row["elapsed_s"],
                        }
                    ),
                    flush=True,
                )

    report = {
        "knobs": {
            "k_prior_m2": K_PRIOR_M2,
            "corr_len_cells": CORR_LEN_CELLS,
            "val_frac": VAL_FRAC,
            "split_seed": SPLIT_SEED,
            "n": args.n,
            "ne": args.ne,
            "na": args.na,
        },
        "note": "Validation scores are hold-out probes. Well p error is train-set hard constraint.",
        "results": rows,
    }
    dest = HERE / "generality_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
