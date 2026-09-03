"""Export-prep: sample H(F_CMG) into observations.csv. Not the invert path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import load_hidden_truth, sample_observations_from_hidden, write_grid_csv
from reservoir_backend.twin.lab_v1 import load_lab_v1, spatial_holdout, write_observations_csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    args = p.parse_args(argv)
    twin = load_lab_v1(dev=True)
    truth = load_hidden_truth(args.hidden)
    held = spatial_holdout(list(twin.experiment.sensors), seed=3)
    series = sample_observations_from_hidden(twin, truth, holdout=held)
    dest = Path(args.export)
    dest.mkdir(parents=True, exist_ok=True)
    write_observations_csv(dest / "observations.csv", series)
    write_grid_csv(twin, dest / "hidden" / "grid.csv")
    print(f"wrote {dest / 'observations.csv'} n_channels={len(series)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
