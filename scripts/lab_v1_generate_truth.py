"""Generate 30 cm (or --dev) synthetic truth + Case A/B/C observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.lab_v1 import (
    CF_TRUE_M2,
    case_path,
    generate_truth,
    load_lab_v1,
    write_observations_csv,
    write_truth_bundle,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true", help="use case_dev.yaml (not 30³)")
    p.add_argument("--case", choices=["A", "B", "C"], default="B")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--cf-true", type=float, default=CF_TRUE_M2)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    twin = load_lab_v1(dev=bool(args.dev))
    truth = generate_truth(
        twin,
        cf_true=float(args.cf_true),
        noise=bool(args.noise),
        case=str(args.case),
        seed=int(args.seed),
    )
    dest = args.out or (case_path(dev=bool(args.dev)).parent / "truth")
    write_truth_bundle(dest, twin, truth)
    obs_copy = case_path(dev=bool(args.dev)).parent / "observations_synthetic.csv"
    write_observations_csv(obs_copy, list(twin.experiment.observations))
    meta = {k: v for k, v in truth.items() if k not in {"pressure", "sw", "sg", "history"}}
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
