"""Seed sweep for M1b Case B/C: fail rate and posterior coverage of truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from lab_v1_offline import run_offline  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["A", "B", "C"], default="B")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 5, 7, 11, 13])
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "seed_sweep")
    args = p.parse_args(argv)
    rows = []
    for seed in args.seeds:
        dest = Path(args.out) / f"seed_{int(seed)}"
        report = run_offline(
            dev=True,
            case=str(args.case),
            noise=bool(args.noise),
            seed=int(seed),
            out=dest,
            skip_detectability=True,
        )
        covered = bool(report["cf_p05"] <= report["cf_true"] <= report["cf_p95"]) and bool(
            report["tmf_p05"] <= report["tmf_true"] <= report["tmf_p95"]
        )
        row = {
            "seed": int(seed),
            "cf_rel": report["gates"]["cf_rel_error"],
            "tmf_rel": report["gates"]["tmf_rel_error"],
            "holdout_ratio": report["holdout_rmse_ratio"],
            "fail_rate": report.get("fail_rate", 0.0),
            "repeated_fail": report.get("repeated_fail", False),
            "covered": covered,
            "pass": report["gates"]["pass"],
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "seed_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mean_fail = float(np.mean([r["fail_rate"] for r in rows])) if rows else 1.0
    ok = all(r["pass"] for r in rows) and mean_fail < 0.05
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
