"""Gate 8: frozen-λ fast step wall time on a given n³ grid."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_compressibility, flash_state
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.fi_comp_dual import initialize_dual_state
from reservoir_backend.solver.frozen_pressure import FrozenPressureContext, step_frozen_pressure


def bench(n: int, dx: float = 0.01, n_steps: int = 3) -> dict:
    grid = CartesianGrid.uniform((n * dx, n * dx, n * dx), (dx, dx, dx))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(grid.n_cells, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, spec.nc)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    pf = flash_state(spec, state.fracture.pressure, state.fracture.moles)
    pm = flash_state(spec, state.matrix.pressure, state.matrix.moles)
    lam_f = pf.lam_l + pf.lam_v
    lam_m = pm.lam_l + pm.lam_v
    ct_f = flash_compressibility(spec, state.fracture.pressure, state.fracture.moles, pf)
    ct_m = flash_compressibility(spec, state.matrix.pressure, state.matrix.moles, pm)
    factor = FrozenPressureContext()
    p_f = state.fracture.pressure.copy()
    p_m = state.matrix.pressure.copy()
    times = []
    for _ in range(int(n_steps)):
        t0 = time.perf_counter()
        p_f, p_m = step_frozen_pressure(
            grid, ctx, dual, tr, p_f, p_m, lam_f, lam_m, 1.0,
            ct_fracture=ct_f, ct_matrix=ct_m, factor=factor,
        )
        times.append(time.perf_counter() - t0)
    rec = {
        "n": n,
        "n_cells": grid.n_cells,
        "first_s": times[0],
        "reuse_s": float(np.mean(times[1:])) if len(times) > 1 else times[0],
        "n_factor": factor.n_factor,
        "n_reuse": factor.n_reuse,
        "t_average_s": float(np.mean(times)),
        "ok": bool(np.all(np.isfinite(p_f)) and np.all(np.isfinite(p_m))),
    }
    return rec


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    rec = bench(int(args.n))
    print(json.dumps(rec))
    if int(args.n) >= 30 and rec["reuse_s"] >= 1.0:
        return 1
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
