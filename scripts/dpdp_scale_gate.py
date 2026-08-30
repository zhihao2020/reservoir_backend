"""Canonical DPDP scale gate. All performance claims must use this workload.

Same initial state, physical time, max_steps, thread cap, and Flash backend
for every grid. One accepted step: t_end = dt_init = dt_max.
"""

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

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.eos.threads import cap_flash_threads
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp


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


def run_standard_step(n: int, *, t_end: float = 0.05, threads: int = 1) -> dict:
    cap_flash_threads(int(threads))
    os.environ["RESERVOIR_FLASH"] = os.environ.get("RESERVOIR_FLASH", "fast")
    dx = 0.01 if n >= 10 else 0.1 / max(n, 1)
    if n == 4:
        nxyz = (4, 3, 2)
        dx = 0.1
    else:
        nxyz = (n, n, n)
    nx, ny, nz = nxyz
    grid = CartesianGrid.uniform((nx * dx, ny * dx, nz * dx), (dx, dx, dx))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(
        grid.n_cells, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02
    )
    transfer = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    ctx = DPDPModelContext.build(grid, spec.nc)
    tracemalloc.start()
    t0 = time.perf_counter()
    traj, end = simulate_dual_comp(
        grid,
        dual,
        spec,
        transfer,
        [],
        [],
        state,
        t_end=float(t_end),
        dt_init=float(t_end),
        dt_max=float(t_end),
        max_steps=1,
        context=ctx,
    )
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n0 = state.total_moles()
    n1 = end.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    meta = _notes(traj.reports)
    rec = {
        "gate": "dpdp_scale_gate",
        "grid": list(nxyz),
        "n_cells": grid.n_cells,
        "n_unknowns": ctx.pattern.n_u,
        "nnz": ctx.pattern.nnz,
        "t_end_s": float(t_end),
        "max_steps": 1,
        "dt_init_s": float(t_end),
        "controls": [],
        "wall_s": wall,
        "jac_s": float(meta.get("sum_jac_s", 0.0)),
        "solve_s": float(meta.get("sum_solve_s", 0.0)),
        "resid_s": float(meta.get("sum_resid_s", 0.0)),
        "flash_s": float(meta.get("sum_flash_s", 0.0)),
        "flash_main_s": float(meta.get("sum_flash_main_s", 0.0)),
        "flash_jacobian_s": float(meta.get("sum_flash_jacobian_s", 0.0)),
        "n_accept": int(float(meta.get("n_accept", len(traj.reports)))),
        "n_reject": int(float(meta.get("n_reject", 0))),
        "newton_its": [int(r.newton_its or 0) for r in traj.reports],
        "mass_rel": rel,
        "peak_mib": peak / (1024 * 1024),
        "threads": int(threads),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "commit": _git_sha(),
        "flash_backend": os.environ.get("RESERVOIR_FLASH", "fast"),
        "linear_backend": os.environ.get("RESERVOIR_LINEAR", "auto"),
        "ok": rel < 1.0e-4 and bool(traj.reports),
    }
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[5, 10])
    p.add_argument("--t-end", type=float, default=0.05)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--json-out", type=str, default="docs/bench/dpdp_scale_gate.json")
    args = p.parse_args()
    rows = []
    for n in args.n:
        rec = run_standard_step(int(n), t_end=float(args.t_end), threads=int(args.threads))
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        if not rec["ok"]:
            return 1
    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
