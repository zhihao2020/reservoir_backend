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
    theta_true_from_spec,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_forward_gate")
    p.add_argument("--wiring", action="store_true", help="spec vs case_dev only; not M2a PASS")
    p.add_argument("--cf-m2", type=float, default=None, help="theta_true C_f override")
    p.add_argument("--tmf", type=float, default=None, help="theta_true beta_mf override")
    args = p.parse_args(argv)
    spec = load_alignment_spec()
    twin = load_lab_v1(dev=True)
    align = check_alignment(spec, twin)
    print(json.dumps({"alignment": align}, indent=2), flush=True)
    if not align["ok"]:
        return 1
    if args.wiring:
        (Path(args.out)).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "wiring.json").write_text(json.dumps(align, indent=2), encoding="utf-8")
        return 0

    blocked = export_blocked_reason(args.export)
    hidden = Path(args.export) / "hidden"
    if not (hidden / "pressure.npy").is_file():
        print(json.dumps({"blocked": blocked or "missing hidden CMG truth"}, indent=2), flush=True)
        return 2

    truth = load_hidden_truth(hidden)
    theta = theta_true_from_spec(twin, spec, cf_m2=args.cf_m2, tmf_multiplier=args.tmf)
    ours = forward_at_theta(twin, theta, truth.times_s)
    report = forward_equivalence_report(ours, truth)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "forward_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
