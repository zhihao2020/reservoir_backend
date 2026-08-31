"""Lab workflow gate: face BCs + sensors + composition. Distinct from dpdp_scale_gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.eos.threads import cap_flash_threads
from reservoir_backend.io.case import load_case
from reservoir_backend.twin.lab_v1 import case_path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return ""


def _notes(reports) -> dict[str, str]:
    out: dict[str, str] = {}
    if not reports:
        return out
    for item in reports[-1].notes:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def run_lab_gate(*, dev: bool = True, t_end: float | None = None, threads: int = 1) -> dict:
    cap_flash_threads(int(threads))
    os.environ["RESERVOIR_FLASH"] = os.environ.get("RESERVOIR_FLASH", "fast")
    twin = load_case(case_path(dev=dev))
    cf = 1.0e-12
    theta = twin.parameterization.encode(np.array([cf], dtype=float))
    t_end = float(twin.physics.dt_init if t_end is None else t_end)
    tracemalloc.start()
    t0 = time.perf_counter()
    traj = twin.simulate(parameters=theta, t_end=t_end, report_times=np.array([t_end]))
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    meta = _notes(traj.reports)
    last = traj.states[-1]
    physical = float(traj.times_s[-1] - traj.times_s[0]) if traj.times_s.size > 1 else float(t_end)
    rec = {
        "gate": "lab_v1_gate",
        "dev": bool(dev),
        "grid": [twin.grid.nx, twin.grid.ny, twin.grid.nz],
        "n_cells": int(twin.grid.n_cells),
        "n_inlet_cells": int(twin.ports[0].cell_ids.size),
        "n_outlet_cells": int(twin.ports[1].cell_ids.size),
        "n_sensors": len(twin.experiment.sensors),
        "wall_s": wall,
        "physical_time_advanced_s": physical,
        "wall_per_physical_s": wall / max(physical, 1.0e-12),
        "accepted_steps": int(float(meta.get("n_accept", len(traj.reports)))),
        "rejected_steps": int(float(meta.get("n_reject", 0))),
        "newton_iterations": [int(r.newton_its or 0) for r in traj.reports],
        "linear_iterations": int(float(meta.get("linear_iterations", 0))),
        "linear_method": meta.get("linear_method", ""),
        "flash_calls": int(float(meta.get("n_flash_main", 0))),
        "mass_error": float(traj.reports[-1].mass.relative_balance_error) if traj.reports else float("nan"),
        "max_dp": float(meta.get("max_dp", 0.0) or 0.0),
        "max_dS": float(meta.get("max_dS", 0.0) or 0.0),
        "peak_memory": peak / (1024 * 1024),
        "jac_s": float(meta.get("sum_jac_s", 0.0)),
        "solve_s": float(meta.get("sum_solve_s", 0.0)),
        "flash_s": float(meta.get("sum_flash_s", 0.0)),
        "linear_setup_s": float(meta.get("linear_setup_s", 0.0) or 0.0),
        "p_mean": float(np.mean(last.pressure)),
        "sw_mean": float(np.mean(last.sw)),
        "threads": int(threads),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "commit": _git_sha(),
        "flash_backend": os.environ.get("RESERVOIR_FLASH", "fast"),
        "linear_backend": os.environ.get("RESERVOIR_LINEAR", "auto"),
        "ok": bool(traj.reports) and (traj.reports[-1].mass.relative_balance_error < 1.0e-4 if traj.reports else False),
    }
    return rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true", default=True)
    p.add_argument("--product", action="store_true", help="30³ product case (slow)")
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--linear", type=str, default=None, help="gmres | cpr | direct")
    p.add_argument("--json-out", type=str, default="docs/bench/lab_v1_gate.json")
    args = p.parse_args(argv)
    if args.product:
        args.dev = False
    if args.linear:
        os.environ["RESERVOIR_LINEAR"] = str(args.linear)
    rec = run_lab_gate(dev=bool(args.dev), t_end=args.t_end, threads=int(args.threads))
    print(json.dumps(rec), flush=True)
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
