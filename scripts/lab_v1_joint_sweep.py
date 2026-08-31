"""Four-truth sweep: (Cf/Cref, beta_mf) in {0.5, 2.0}². Uses the tiny face twin."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.synthetic import make_lab_v1_face_twin
from reservoir_backend.twin.lab_v1 import physical_from_theta


CASES = (("T1", 0.5, 0.5), ("T2", 0.5, 2.0), ("T3", 2.0, 0.5), ("T4", 2.0, 2.0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ne", type=int, default=6)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "joint_sweep")
    args = p.parse_args(argv)
    rows = []
    cref = 1.0e-12
    for name, cf_fac, beta in CASES:
        cf = cref * float(cf_fac)
        syn = make_lab_v1_face_twin(
            cf_true=cf,
            ensemble_size=int(args.ne),
            assimilation_steps=2,
            seed=3,
        )
        # override Tmf truth after construction by re-encoding
        syn.theta_true = syn.twin.parameterization.encode(np.array([cf, float(beta)]))
        syn.twin.simulate(parameters=syn.theta_true, t_end=syn.twin.experiment.history_end_s)
        post = syn.twin.calibrate()
        phys = physical_from_theta(syn.twin, post.theta)
        rows.append(
            {
                "case": name,
                "cf_true": cf,
                "tmf_true": float(beta),
                "cf_p50": float(phys["cf_m2"]),
                "tmf_p50": float(phys["tmf_multiplier"]),
                "cf_rel": abs(phys["cf_m2"] - cf) / cf,
                "tmf_rel": abs(phys["tmf_multiplier"] - beta) / beta,
            }
        )
        print(json.dumps(rows[-1]))
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "joint_sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (dest / "joint_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
