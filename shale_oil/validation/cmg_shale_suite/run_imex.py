"""Run IMEX on one or all shale-oil analog rulers."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = HERE.parent
IMEX_EXE = Path(r"D:\Tool\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe")
CASE_DIR = {
    "S1": "cmg_s1_hw5frac",
    "S2": "cmg_s2_hw9frac",
    "S3": "cmg_s3_twohw",
    "S4": "cmg_s4_parent_child",
    "S5": "cmg_s5_shutin",
}


def run_one(case: str) -> bool:
    d = VAL / CASE_DIR[case]
    dats = list(d.glob("mxshale_*.dat"))
    if not dats:
        print(f"{case}: no dat in {d}", file=sys.stderr)
        return False
    dat = dats[0]
    cmd = [str(IMEX_EXE), "-f", dat.name]
    print(" ", " ".join(cmd), "cwd=", d)
    proc = subprocess.run(cmd, cwd=str(d))
    out = dat.with_suffix(".out")
    text = out.read_text(encoding="latin-1", errors="replace") if out.is_file() else ""
    ok = "Normal Termination" in text
    print(f"  {case} rc={proc.returncode} normal={ok}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", help="S1..S5 or all")
    args = ap.parse_args()
    if not IMEX_EXE.is_file():
        print(f"IMEX not found: {IMEX_EXE}", file=sys.stderr)
        return 2
    cases = list(CASE_DIR) if args.case.lower() == "all" else [args.case.upper()]
    ok = True
    for c in cases:
        if c not in CASE_DIR:
            print("unknown", c)
            ok = False
            continue
        ok = run_one(c) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
