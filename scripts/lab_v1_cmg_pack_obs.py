"""Export GEM monthly snapshots to the single software CSV set in lab_cube."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.domain.types import State
from reservoir_backend.twin.cmg_benchmark import (
    fill_hidden_rock,
    load_hidden_truth,
    load_twin_case,
    parse_gem_out_summaries,
    slice_hidden_truth,
    write_grid_csv,
    write_hidden_truth,
)

P3D = ROOT / "examples" / "lab_v1" / "cmg_gem" / "physical_3d"
LAB_CUBE = ROOT / "examples" / "lab_cube"
N_MONTHS = 30 * 12
MONTH_S = 30.0 * 86400.0
MONTH_TIMES_S = tuple(n * MONTH_S for n in range(1, N_MONTHS + 1))

_WELLS = (
    ("INJ", "injector", 0.15, 0.15, 0.20),
    ("PROD1", "producer", 0.05, 0.25, 0.25),
    ("PROD2", "producer", 0.25, 0.25, 0.15),
    ("PROD3", "producer", 0.05, 0.05, 0.15),
    ("PROD4", "producer", 0.25, 0.05, 0.05),
)


def _clip_to_history(twin, truth):
    end = getattr(getattr(twin, "experiment", None), "history_end_s", None)
    if end is None:
        return truth
    times = np.asarray(truth.times_s, dtype=float).ravel()
    mask = times <= float(end) + 1.0e-9
    if not np.any(mask):
        return truth
    return slice_hidden_truth(truth, mask)


def _near(t: float, target: float, atol: float = 1.0) -> bool:
    return abs(float(t) - float(target)) <= atol


def _month_mask(times_s) -> np.ndarray:
    times = np.asarray(times_s, dtype=float).ravel()
    return np.array([any(_near(t, m) for m in MONTH_TIMES_S) for t in times], dtype=bool)


def write_software_csvs(dest: Path, twin, truth) -> None:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    times = np.asarray(truth.times_s, dtype=float).ravel()
    field_idx = np.flatnonzero(_month_mask(times))

    with (dest / "probes.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x_m", "y_m", "z_m", "kinds"])
        for sen in twin.experiment.sensors:
            if sen.kind != "pressure":
                continue
            w.writerow([sen.name, f"{sen.x:.6g}", f"{sen.y:.6g}", f"{sen.z:.6g}", "pressure,sw,so"])

    with (dest / "wells.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "kind", "x_m", "y_m", "z_m"])
        for name, kind, x, y, z in _WELLS:
            w.writerow([name, kind, x, y, z])

    with (dest / "well_series.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "well", "pw_pa", "q_m3s", "qw", "qo", "qg", "fw", "gor"])
        rows = list(truth.wells or [])
        rows.sort(key=lambda r: (float(r.get("time_s", 0.0)), str(r.get("well", ""))))
        for row in rows:
            t = float(row.get("time_s", float("nan")))
            if not any(_near(t, m) for m in MONTH_TIMES_S):
                continue
            q_day = float(row.get("q_m3_day") or 0.0)
            w.writerow(
                [
                    f"{t:.9g}",
                    row.get("well"),
                    f"{float(row.get('pw_pa') or 0.0):.9g}",
                    f"{q_day / 86400.0:.9g}",
                    f"{float(row.get('qw_m3_day') or 0.0) / 86400.0:.9g}",
                    f"{float(row.get('qo_m3_day') or 0.0) / 86400.0:.9g}",
                    f"{float(row.get('qg_m3_day') or 0.0) / 86400.0:.9g}",
                    f"{float(row.get('fw') or 0.0):.9g}",
                    f"{float(row.get('gor') or 0.0):.9g}",
                ]
            )

    sw_all = np.zeros_like(truth.pressure) if truth.sw is None else np.asarray(truth.sw, dtype=float)
    sg_all = np.zeros_like(truth.pressure) if truth.sg is None else np.asarray(truth.sg, dtype=float)
    if truth.so is None:
        so_all = 1.0 - sw_all - sg_all
    else:
        so_all = np.asarray(truth.so, dtype=float)

    with (dest / "observations.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "probe", "quantity", "value"])
        for it in field_idx:
            t = float(times[it])
            p = np.asarray(truth.pressure[it], dtype=float).ravel()
            sw = np.asarray(sw_all[it], dtype=float).ravel()
            so = np.asarray(so_all[it], dtype=float).ravel()
            sg = None if truth.sg is None else np.asarray(sg_all[it], dtype=float).ravel()
            st = State(pressure=p, sw=sw, sg=sg, time_s=t)
            for sen in twin.experiment.sensors:
                if sen.kind != "pressure":
                    continue
                w.writerow([f"{t:.9g}", sen.name, "pressure", f"{float(twin.operator.sample(sen, st)):.9g}"])
                w.writerow([f"{t:.9g}", sen.name, "sw", f"{float(twin.operator.sample_field(sen, sw)):.9g}"])
                w.writerow([f"{t:.9g}", sen.name, "so", f"{float(twin.operator.sample_field(sen, so)):.9g}"])

    n_c = twin.grid.n_cells
    phi = np.zeros((times.size, n_c)) if truth.porosity is None else np.asarray(truth.porosity, dtype=float)
    perm = np.zeros((times.size, n_c)) if truth.permeability is None else np.asarray(truth.permeability, dtype=float)
    with (dest / "grid_series.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "cell", "i", "j", "k", "p_pa", "sw", "sg", "so", "phi", "k_m2"])
        for it in field_idx:
            t = float(times[it])
            for c in range(n_c):
                i, j, k = twin.grid.ijk(c)
                w.writerow(
                    [
                        f"{t:.9g}",
                        c,
                        i,
                        j,
                        k,
                        f"{float(truth.pressure[it, c]):.9g}",
                        f"{float(sw_all[it, c]):.9g}",
                        f"{float(sg_all[it, c]):.9g}",
                        f"{float(so_all[it, c]):.9g}",
                        f"{float(phi[it, c]):.9g}",
                        f"{float(perm[it, c]):.9g}",
                    ]
                )


def _attach_wells(truth, case: Path, hidden: Path):
    out_cand = ROOT / "results" / "lab_v1" / "cmg_gem_physical_3d" / "sanwei_co2.out"
    if not out_cand.is_file():
        out_cand = Path(case).parent / "sanwei_co2.out"
    if not out_cand.is_file():
        return truth
    summary = parse_gem_out_summaries(out_cand.read_text(encoding="utf-8", errors="replace"))
    (Path(hidden) / "gem_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wells = []
    for rep in summary.get("reports") or []:
        wells.extend(rep.get("wells") or [])
    if wells:
        return replace(truth, wells=wells)
    return truth


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_gem_physical_3d" / "hidden")
    p.add_argument("--export", type=Path, default=LAB_CUBE)
    p.add_argument("--case", type=Path, default=P3D / "case.yaml")
    args = p.parse_args(argv)
    twin = load_twin_case(args.case)
    truth = _clip_to_history(twin, load_hidden_truth(args.hidden))
    rock = twin.rock_from_theta(np.zeros(twin.parameterization.n_params))
    truth = fill_hidden_rock(
        truth,
        porosity=float(rock.porosity.mean()),
        permeability_m2=float(rock.permeability.mean()),
    )
    hidden = Path(args.hidden)
    hidden.mkdir(parents=True, exist_ok=True)
    truth = _attach_wells(truth, args.case, hidden)
    write_hidden_truth(hidden, truth)
    write_grid_csv(twin, hidden / "grid.csv")
    dest = Path(args.export)
    write_software_csvs(dest, twin, truth)
    print(f"wrote {dest} n_months={len(MONTH_TIMES_S)} n_gem_times={int(truth.times_s.size)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
