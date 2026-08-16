"""Forward-only F(K_CMG) vs IMEX maps. Not an invert."""

from __future__ import annotations

import json

import numpy as np

from reservoir_backend.validation.cmg_harness.adapter import build_twin
from reservoir_backend.validation.cmg_harness.catalog import PSI, get_case
from reservoir_backend.validation.cmg_harness.score import field_gap, maps_from_traj


def one(case_id: str) -> dict:
    spec = get_case(case_id)
    twin, extra = build_twin(spec, with_observations=False)
    k = extra["k_true"]
    days = spec.history_days
    t_end = float(days[-1]) * 86400.0
    pvt = twin.physics.pvt
    print(
        f"== {case_id}  n={twin.grid.n_cells}  g={twin.physics.gravity}  "
        f"kz/kx={twin.physics.kz_over_kx:.3f}  faultT={twin.face_mult_x is not None}"
    )
    traj = twin.simulate(twin.rock_from_k(k), t_end=t_end)
    f_maps = maps_from_traj(traj, days, twin.grid)
    p_rmse, p_demean, sw_rmse, sw_c, sw_f = field_gap(f_maps, extra["maps"])
    last = traj.state_at(t_end)
    p_mean = float(np.mean(last.pressure) / PSI)
    cmg_last = extra["maps"][float(days[-1])]
    cmg_p = float(np.mean(np.asarray(cmg_last["p"], dtype=float)))
    row = {
        "case": case_id,
        "p_rmse_psi": p_rmse,
        "p_demean_psi": p_demean,
        "sw_rmse": sw_rmse,
        "sw_cmg": sw_c,
        "sw_f": sw_f,
        "p_mean_f_psi": p_mean,
        "p_mean_cmg_psi": cmg_p,
        "p_init_psi": float(twin.physics.p_init / PSI),
        "n_steps": len(traj.reports),
    }
    print(json.dumps(row, indent=2))
    return row


if __name__ == "__main__":
    out = [one(c) for c in ("lab_layers", "fault", "fivespot")]
    dest = __file__.replace("_gap_blackoil.py", "blackoil_floor.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", dest)
