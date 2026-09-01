"""Finish remaining M1b/M1c plan items: T2 retry, Case A, Case C, seed sweep, M1c D_Cf.

Writes a JSON line after every item so an interrupt does not lose progress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lab_v1_offline import run_offline  # noqa: E402
from lab_v1_identifiability import run_identifiability  # noqa: E402

OUT = ROOT / "results" / "lab_v1" / "m1_remaining"
LOG = OUT / "progress.jsonl"


def _log(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(row), flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _done(name: str) -> bool:
    if not LOG.exists():
        return False
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("item") == name and rec.get("finished"):
            return True
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rc = 0

    if not _done("T2"):
        report = run_offline(
            dev=True,
            case="B",
            cf_true=5.0e-13,
            tmf_true=2.0,
            seed=5,
            out=OUT / "T2_seed5",
            skip_detectability=True,
        )
        row = {
            "item": "T2",
            "finished": True,
            "cf_rel": report["gates"]["cf_rel_error"],
            "tmf_rel": report["gates"]["tmf_rel_error"],
            "pass": report["gates"]["pass"],
            "fail_rate": report.get("fail_rate", 0.0),
        }
        _log(row)
        if not row["pass"]:
            rc = 1

    if not _done("A"):
        report = run_offline(dev=True, case="A", seed=3, out=OUT / "case_A", skip_detectability=True)
        _log(
            {
                "item": "A",
                "finished": True,
                "cf_rel": report["gates"]["cf_rel_error"],
                "tmf_rel": report["gates"]["tmf_rel_error"],
                "pass": report["gates"]["pass"],
                "fail_rate": report.get("fail_rate", 0.0),
            }
        )
        if not report["gates"]["pass"]:
            rc = 1

    if not _done("C"):
        report = run_offline(
            dev=True, case="C", noise=True, seed=3, out=OUT / "case_C", skip_detectability=True
        )
        _log(
            {
                "item": "C",
                "finished": True,
                "cf_rel": report["gates"]["cf_rel_error"],
                "tmf_rel": report["gates"]["tmf_rel_error"],
                "pass": report["gates"]["pass"],
                "fail_rate": report.get("fail_rate", 0.0),
            }
        )
        if not report["gates"]["pass"]:
            rc = 1

    seeds = [5, 7, 11, 13]
    seed_rows = []
    for seed in seeds:
        name = f"seed_{seed}"
        if _done(name):
            continue
        report = run_offline(
            dev=True, case="B", seed=int(seed), out=OUT / name, skip_detectability=True
        )
        covered = bool(report["cf_p05"] <= report["cf_true"] <= report["cf_p95"]) and bool(
            report["tmf_p05"] <= report["tmf_true"] <= report["tmf_p95"]
        )
        row = {
            "item": name,
            "finished": True,
            "seed": int(seed),
            "cf_rel": report["gates"]["cf_rel_error"],
            "tmf_rel": report["gates"]["tmf_rel_error"],
            "holdout_ratio": report["holdout_rmse_ratio"],
            "fail_rate": report.get("fail_rate", 0.0),
            "repeated_fail": report.get("repeated_fail", False),
            "covered": covered,
            "pass": report["gates"]["pass"],
        }
        seed_rows.append(row)
        _log(row)
        if not row["pass"]:
            rc = 1
    if seed_rows:
        (OUT / "seed_sweep.json").write_text(json.dumps(seed_rows, indent=2), encoding="utf-8")

    if not _done("M1c_detect"):
        rec = run_identifiability(tiny=False, sigma_p_fracture=2.0e3, q_inj=1.05e-2)
        _log(
            {
                "item": "M1c_detect",
                "finished": True,
                "d_cf": rec["d_cf"],
                "d_cf_ok": rec["d_cf_ok"],
                "joint_ok": rec["joint_ok"],
                "q_inj": 1.05e-2,
                "sigma_p_fracture": 2000.0,
            }
        )
        if rec["d_cf_ok"]:
            report = run_offline(
                dev=True,
                case="B",
                seed=3,
                out=OUT / "m1c_B",
                skip_detectability=True,
                q_inj=1.05e-2,
                sigma_p_fracture=2.0e3,
            )
            _log(
                {
                    "item": "M1c_B",
                    "finished": True,
                    "cf_rel": report["gates"]["cf_rel_error"],
                    "tmf_rel": report["gates"]["tmf_rel_error"],
                    "pass": report["gates"]["pass"],
                    "fail_rate": report.get("fail_rate", 0.0),
                }
            )
            if not report["gates"]["pass"]:
                rc = 1
        else:
            rc = 1
            _log({"item": "M1c_B", "finished": True, "skipped": True, "reason": "D_Cf<2 at q=0.0105"})

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
