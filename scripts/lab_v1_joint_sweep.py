"""Four-truth sweep: (Cf/Cref, beta_mf) in {0.5, 2.0}².

Each truth rebuilds observations from that (Cf, T_mf). Do not mutate theta_true
on a twin whose observations were generated at a different T_mf.
"""

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
from reservoir_backend.twin.lab_v1 import CF_TRUE_M2, generate_truth, load_lab_v1, physical_from_theta
from reservoir_backend.twin.history_match import HistoryMatchWorkflow


CASES = (("T1", 0.5, 0.5), ("T2", 0.5, 2.0), ("T3", 2.0, 0.5), ("T4", 2.0, 2.0))


def _run_tiny(cf: float, beta: float, ne: int, seed: int):
    syn = make_lab_v1_face_twin(
        cf_true=cf,
        tmf_true=float(beta),
        ensemble_size=int(ne),
        assimilation_steps=5,
        seed=int(seed),
    )
    post = syn.twin.calibrate()
    return syn.twin, post


def _run_dev(cf: float, beta: float, ne: int, seed: int):
    twin = load_lab_v1(dev=True)
    generate_truth(twin, cf_true=cf, tmf_true=float(beta), case="B", seed=int(seed))
    prior_cf = cf * 0.3
    prior_tmf = float(beta) * 0.5
    twin.parameterization.prior_mean = twin.parameterization.encode(np.array([prior_cf, prior_tmf], dtype=float))
    twin.inverse.prior_mean = np.asarray(twin.parameterization.prior_mean, dtype=float)
    twin.inverse.ensemble_size = int(ne)
    twin.inverse.seed = int(seed)
    if twin.inverse.n_workers is None:
        twin.inverse.n_workers = min(8, int(ne))
    post = HistoryMatchWorkflow().run(twin)
    return twin, post


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ne", type=int, default=8)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--dev", action="store_true", help="30 cm / 4×4×2 M1b fixture")
    p.add_argument("--tiny", action="store_true", help="diagnostic 4×2×1 twin")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "joint_sweep")
    p.add_argument("--only", nargs="*", default=None, help="subset of T1 T2 T3 T4")
    args = p.parse_args(argv)
    rows = []
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    cref = CF_TRUE_M2
    use_dev = bool(args.dev) or not bool(args.tiny)
    want = {s.upper() for s in args.only} if args.only else {c[0] for c in CASES}
    for name, cf_fac, beta in CASES:
        if name not in want:
            continue
        cf = cref * float(cf_fac)
        if use_dev:
            twin, post = _run_dev(cf, beta, args.ne, args.seed)
        else:
            twin, post = _run_tiny(cf, beta, args.ne, args.seed)
        phys = physical_from_theta(twin, post.theta if post.theta.ndim == 1 else np.mean(post.ensemble.theta_members, axis=0))
        if post.ensemble is not None:
            phys_members = [physical_from_theta(twin, post.ensemble.theta_members[j]) for j in range(post.ensemble.theta_members.shape[0])]
            cf_p50 = float(np.quantile([p["cf_m2"] for p in phys_members], 0.50))
            tmf_p50 = float(np.quantile([p["tmf_multiplier"] for p in phys_members], 0.50))
        else:
            cf_p50 = float(phys["cf_m2"])
            tmf_p50 = float(phys["tmf_multiplier"])
        rows.append(
            {
                "case": name,
                "cf_true": cf,
                "tmf_true": float(beta),
                "cf_p50": cf_p50,
                "tmf_p50": tmf_p50,
                "cf_rel": abs(cf_p50 - cf) / cf,
                "tmf_rel": abs(tmf_p50 - beta) / beta,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        (dest / "partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (dest / "joint_sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (dest / "joint_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = all(r["cf_rel"] < 0.05 and r["tmf_rel"] < 0.10 for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
