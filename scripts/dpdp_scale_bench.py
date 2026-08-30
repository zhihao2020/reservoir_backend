"""DPDP forward ladder: 4x3x2, 5^3, 10^3, 20^3, 30^3.

Records n_cells, n_unknowns, nnz(J), Jacobian / linear-solve / residual (flash)
times, Newton iterations, accepted/rejected Δt, peak memory, mass balance.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tracemalloc

import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp


def _notes(reports) -> dict[str, str]:
    out: dict[str, str] = {}
    if not reports:
        return out
    for item in reports[-1].notes:
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def run_level(nxyz: tuple[int, int, int], dx: float, t_end: float, *, max_steps: int = 8) -> dict:
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
        t_end=t_end,
        dt_init=min(0.5, t_end),
        dt_max=t_end,
        max_steps=int(max_steps),
        context=ctx,
    )
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n0 = state.total_moles()
    n1 = end.total_moles()
    rel = float(np.max(np.abs(n1 - n0)) / max(float(np.max(np.abs(n0))), 1.0e-18))
    meta = _notes(traj.reports)
    its = [int(r.newton_its or 0) for r in traj.reports]
    rec = {
        "grid": list(nxyz),
        "n_cells": grid.n_cells,
        "n_unknowns": ctx.pattern.n_u,
        "nnz": ctx.pattern.nnz,
        "n_colors": int(np.max(ctx.colors)) + 1 if ctx.colors.size else 0,
        "wall_s": wall,
        "jac_s": float(meta.get("sum_jac_s", 0.0)),
        "solve_s": float(meta.get("sum_solve_s", 0.0)),
        "resid_s": float(meta.get("sum_resid_s", 0.0)),
        "flash_s": float(meta.get("sum_flash_s", 0.0)),
        "flash_main_s": float(meta.get("sum_flash_main_s", meta.get("sum_flash_s", 0.0))),
        "flash_jacobian_s": float(meta.get("sum_flash_jacobian_s", 0.0)),
        "n_accept": int(float(meta.get("n_accept", len(traj.reports)))),
        "n_reject": int(float(meta.get("n_reject", 0))),
        "newton_its": its,
        "n_flash_main": int(float(meta.get("n_flash_main", 0))),
        "n_flash_thermo_jac": int(float(meta.get("n_flash_thermo_jac", 0))),
        "n_flash_line_search": int(float(meta.get("n_flash_line_search", 0))),
        "n_jac_reuse": int(float(meta.get("n_jac_reuse", 0))),
        "peak_mib": peak / (1024 * 1024),
        "mass_rel": rel,
        "ok": rel < 1.0e-4 and bool(traj.reports),
        "threads": int(os.cpu_count() or 1),
        "commit": _git_sha(),
        "flash_backend": os.environ.get("RESERVOIR_FLASH", "fast"),
    }
    return rec


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-n", type=int, default=30)
    p.add_argument("--t-end", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--json-out", type=str, default="")
    args = p.parse_args()
    ladder = [
        ((4, 3, 2), 0.1),
        ((5, 5, 5), 0.02),
        ((10, 10, 10), 0.01),
        ((20, 20, 20), 0.005),
        ((30, 30, 30), 0.01),
    ]
    rows = []
    for nxyz, dx in ladder:
        edge = max(nxyz)
        if nxyz[0] == nxyz[1] == nxyz[2] and edge > int(args.max_n):
            print(f"skip {nxyz}")
            continue
        rec = run_level(nxyz, dx, float(args.t_end), max_steps=int(args.max_steps))
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        if not rec["ok"]:
            raise SystemExit(f"level {nxyz} failed mass_rel={rec['mass_rel']}")
    if args.json_out:
        from pathlib import Path

        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
