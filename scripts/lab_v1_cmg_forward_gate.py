"""M2a forward-equivalence gate: F_ours(theta_true) vs hidden CMG field.

Does not run ES-MDA. Exits 2 when GEM export is missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    check_alignment,
    export_blocked_reason,
    forward_at_theta,
    forward_equivalence_report,
    load_alignment_spec,
    load_hidden_truth,
    load_twin_case,
    theta_true_from_twin,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_forward_gate")
    p.add_argument("--wiring", action="store_true", help="spec vs case_dev only; not M2a PASS")
    p.add_argument("--case", type=Path, default=None, help="YAML case; default M2 case_dev")
    p.add_argument("--cf-m2", type=float, default=None, help="theta_true C_f override")
    p.add_argument("--tmf", type=float, default=None, help="theta_true beta_mf override")
    p.add_argument("--k-m2", type=float, default=None, dest="k_m2")
    args = p.parse_args(argv)
    twin = load_twin_case(args.case)
    spec = None if args.case is not None else load_alignment_spec()
    if spec is not None:
        align = check_alignment(spec, twin)
        print(json.dumps({"alignment": align}, indent=2), flush=True)
        if not align["ok"]:
            return 1
        if args.wiring:
            (Path(args.out)).mkdir(parents=True, exist_ok=True)
            (Path(args.out) / "wiring.json").write_text(json.dumps(align, indent=2), encoding="utf-8")
            return 0
    elif args.wiring:
        rec = {"ok": True, "n_cells": int(twin.grid.n_cells), "case": str(args.case)}
        (Path(args.out)).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "wiring.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps({"alignment": rec}, indent=2), flush=True)
        return 0

    blocked = export_blocked_reason(args.export)
    hidden = Path(args.export) / "hidden"
    if not (hidden / "pressure.npy").is_file():
        print(json.dumps({"blocked": blocked or "missing hidden CMG truth"}, indent=2), flush=True)
        return 2

    truth = load_hidden_truth(hidden)
    theta = theta_true_from_twin(
        twin, spec, cf_m2=args.cf_m2, tmf_multiplier=args.tmf, k_m2=args.k_m2
    )
    ours = forward_at_theta(twin, theta, truth.times_s)
    report = forward_equivalence_report(ours, truth)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "forward_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
