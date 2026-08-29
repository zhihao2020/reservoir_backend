"""Offline S1–S5 shale inversion vs IMEX PRES (ruler only; no CMG at invert time).

Usage (repo root):
  python validation/shale_oil/cmg_shale_suite/run_suite_inversion.py
  python validation/shale_oil/cmg_shale_suite/run_suite_inversion.py --case S1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
VAL = HERE.parent

from build_shale_suite import CASE_DIR, CASES  # noqa: E402
from reservoir_backend.io.shale_case import invert_shale_case  # noqa: E402


def invert_case(case: str, *, max_iter: int = 12) -> dict:
    case = str(case).upper()
    case_dir = VAL / CASE_DIR[case]
    truth_path = case_dir / f"truth_{case.lower()}.json"
    return invert_shale_case(
        truth_path,
        out_path=case_dir / f"mxshale_{case.lower()}.out",
        n_times=5,
        max_iter=max_iter,
    )


def _passes_gates(rec: dict) -> bool:
    """MVP cross-simulator gates (MIN BHP + aligned rates + 4-D θ)."""
    if not rec.get("ok"):
        return False
    if not rec.get("dp_sign_match"):
        return False
    ratio = rec.get("dp_ratio")
    if ratio is None or float(ratio) < 0.2:
        return False
    n_inv = float(rec.get("inv_n_frac", 0))
    n_true = float(rec.get("truth_n_frac_planes", 0))
    if abs(n_inv - n_true) > 1.0:
        return False
    if float(rec.get("assimilate_nrmse", 99.0)) >= 10.0:
        return False
    if float(rec.get("k_frac_over_matrix", 0.0)) < 100.0:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S1–S5 LM inversion vs IMEX")
    parser.add_argument("--case", default="all", help="S1..S5 or all")
    parser.add_argument("--max-iter", type=int, default=12)
    args = parser.parse_args(argv)
    wanted = CASES if str(args.case).lower() == "all" else (str(args.case).upper(),)
    reports = []
    for case in wanted:
        print(f"=== {case} ===", flush=True)
        rec = invert_case(case, max_iter=int(args.max_iter))
        rec["gates_pass"] = _passes_gates(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "run_report"}, indent=2), flush=True)
        reports.append(rec)
    suite = {
        "algorithm": "LM",
        "schema": "run_report",
        "cases": [{k: v for k, v in r.items() if k != "run_report"} for r in reports],
        "run_reports": [r.get("run_report") for r in reports if r.get("run_report")],
    }
    dest = HERE / (
        "s1_inversion_report.json" if wanted == ("S1",) else "suite_inversion_report.json"
    )
    if wanted == ("S1",):
        dest.write_text(json.dumps(reports[0], indent=2), encoding="utf-8")
    else:
        dest.write_text(json.dumps(suite, indent=2), encoding="utf-8")
        s1 = next((r for r in reports if r.get("case") == "S1"), None)
        if s1:
            (HERE / "s1_inversion_report.json").write_text(
                json.dumps(s1, indent=2), encoding="utf-8"
            )
    print(f"wrote {dest}", flush=True)
    ok = all(r.get("ok") for r in reports)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
