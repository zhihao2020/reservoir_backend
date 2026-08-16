"""How much do gauges/fields change with channel contrast? Not an invert."""

from __future__ import annotations

import json

import numpy as np

from reservoir_backend.inverse.parameterization import ContrastParameterization
from reservoir_backend.validation.cmg_harness.adapter import build_twin, inflate_model_error
from reservoir_backend.validation.cmg_harness.catalog import MD_TO_M2, PSI, get_case
from reservoir_backend.validation.cmg_harness.score import field_gap, maps_from_traj


def run_case(case_id: str) -> list[dict]:
    spec = get_case(case_id)
    twin, extra = build_twin(spec, with_observations=True)
    inflate_model_error(twin, extra["k_true"], demean_pressure=spec.dt_max_s >= 3600.0)
    param = twin.parameterization
    assert isinstance(param, ContrastParameterization)
    k_true = extra["k_true"]
    hi = k_true >= max(float(np.median(k_true)) * 1.5, float(np.min(k_true)) * 1.01)
    k_lo_true = float(np.mean(k_true[~hi]))
    k_hi_true = float(np.mean(k_true[hi]))
    print(f"== {case_id}  k_lo={k_lo_true/MD_TO_M2:.1f}md  k_hi={k_hi_true/MD_TO_M2:.1f}md  "
          f"truth_contrast={k_hi_true/k_lo_true:.2f}  prior_contrast={np.exp(param.log_contrast_mean):.2f}")
    t_end = float(spec.history_days[-1]) * 86400.0
    days = spec.history_days
    rows = []
    log_k0 = float(np.log(k_lo_true))
    for c in (2.0, 5.0, 8.0, 15.0, 30.0, 50.0, k_hi_true / k_lo_true):
        theta = np.array([log_k0, float(np.log(c))])
        rock = twin.rock_from_theta(theta)
        traj = twin.simulate(rock, t_end=t_end)
        f_maps = maps_from_traj(traj, days, twin.grid)
        p, pd, s, sc, sf = field_gap(f_maps, extra["maps"])
        # whitened gauge nRMSE on assimilating window
        from reservoir_backend.twin.offline import predict_from_trajectory, stack_observations

        assim = [o for o in twin.experiment.observations if not o.holdout]
        d = stack_observations(assim)
        pred = predict_from_trajectory(twin.operator, twin.experiment, traj, assim)
        nrmse = float(np.sqrt(np.mean(((pred - d.values) / d.sigma) ** 2)))
        row = {
            "case": case_id,
            "contrast": float(c),
            "p_rmse": p,
            "p_demean": pd,
            "sw_rmse": s,
            "gauge_nrmse": nrmse,
        }
        rows.append(row)
        print(f"  contrast={c:6.1f}  p={p:6.1f}  p~={pd:6.1f}  sw={s:.4f}  gauge={nrmse:.3f}")
    return rows


if __name__ == "__main__":
    out = []
    for cid in ("fault", "channel", "fivespot"):
        out.extend(run_case(cid))
    path = __file__.replace("_contrast_sens.py", "contrast_sens.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", path)
