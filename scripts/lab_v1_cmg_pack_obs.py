"""Export-prep: sample H(F_CMG) into observations.csv. Not the invert path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shutil

from reservoir_backend.twin.cmg_benchmark import load_hidden_truth, load_twin_case, sample_observations_from_hidden, write_grid_csv
from reservoir_backend.twin.lab_v1 import spatial_holdout, write_controls_csv, write_observations_csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    p.add_argument("--case", type=Path, default=None, help="YAML case whose sensors define H")
    args = p.parse_args(argv)
    twin = load_twin_case(args.case)
    truth = load_hidden_truth(args.hidden)
    held = spatial_holdout(list(twin.experiment.sensors), seed=3)
    series = sample_observations_from_hidden(twin, truth, holdout=held)
    dest = Path(args.export)
    dest.mkdir(parents=True, exist_ok=True)
    write_observations_csv(dest / "observations.csv", series)
    if args.case is not None and twin.experiment.controls:
        import csv

        with (dest / "controls.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time_s", "port", "kind", "value"])
            for c in twin.experiment.controls:
                for t, v in zip(c.times_s, c.values):
                    w.writerow([float(t), c.port_name, c.kind, float(v)])
    else:
        write_controls_csv(
            dest / "controls.csv",
            t_end=float(np.max(truth.times_s)) if truth.times_s.size else 60.0,
            q_inj=3.0e-4,
            p_prod=1.18e7,
        )
    hidden_dest = dest / "hidden"
    hidden_dest.mkdir(parents=True, exist_ok=True)
    src = Path(args.hidden)
    for name in (
        "pressure.npy",
        "sg.npy",
        "so.npy",
        "sw.npy",
        "pressure_fracture.npy",
        "pressure_matrix.npy",
        "meta.json",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, hidden_dest / name)
    write_grid_csv(twin, hidden_dest / "grid.csv")
    print(
        f"wrote {dest / 'observations.csv'} n_channels={len(series)} n_times={int(truth.times_s.size)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
