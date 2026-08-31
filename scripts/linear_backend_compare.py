"""Plan §20: compare GMRES / CPR / direct on the same DPDP scale-gate step."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dpdp_scale_gate import run_standard_step


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="cubic n; 30 is the product gate")
    p.add_argument("--backends", nargs="+", default=["gmres", "cpr"])
    p.add_argument("--json-out", type=str, default="docs/bench/linear_backend_compare.json")
    args = p.parse_args(argv)
    rows = []
    for name in args.backends:
        if name == "direct" and int(args.n) >= 20:
            rows.append({"backend": name, "skipped": True, "reason": "direct too large"})
            continue
        os.environ["RESERVOIR_LINEAR"] = str(name)
        rec = run_standard_step(int(args.n), t_end=0.05, threads=1)
        rec["requested_linear"] = name
        rows.append(rec)
        print(json.dumps({"backend": name, "wall_s": rec["wall_s"], "solve_s": rec["solve_s"], "ok": rec["ok"]}))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0 if all(r.get("ok", r.get("skipped")) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
