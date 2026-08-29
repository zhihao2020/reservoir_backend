"""S5 shut-in forecast ruler (slow; requires IMEX .out)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from reservoir_backend.io.shale_case import forecast_shale_case, invert_shale_case, twin_from_shale_truth  # noqa: E402


def main() -> int:
    case_dir = ROOT / "validation" / "shale_oil" / "cmg_s5_shutin"
    truth = case_dir / "truth_s5.json"
    out = case_dir / "mxshale_s5.out"
    if not out.is_file():
        print(json.dumps({"ok": False, "error": "missing IMEX .out"}))
        return 2
    inv = invert_shale_case(truth, out_path=out, n_times=5, max_iter=6, time_limit_s=600.0)
    if not inv.get("ok"):
        print(json.dumps(inv, indent=2))
        return 2
    twin = twin_from_shale_truth(truth, out_path=out, n_times=5, max_iter=6)
    twin.inverse.post_ensemble_enabled = False
    post = twin.calibrate(max_iter=6, time_limit_s=600.0)
    traj, fc_score = forecast_shale_case(twin, post)
    report = {
        "case": "S5",
        "ok": True,
        "assimilate_nrmse": float(post.assimilate_rmse),
        "forecast_rmse": float(fc_score),
        "history_end_s": twin.experiment.history_end_s,
        "forecast_end_s": float(traj.times_s[-1]) if traj.times_s.size else None,
    }
    dest = HERE / "forecast_s5_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
