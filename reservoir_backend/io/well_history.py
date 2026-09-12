"""Well-history ruler: rates/BHP nRMSE vs a CSV. Not field L2 / cell-K / Dice."""

from __future__ import annotations

from typing import Any

import numpy as np

from reservoir_backend.domain.types import ObservationSeries
from reservoir_backend.solver.trajectory import Trajectory

WELL_KINDS = frozenset({"bhp", "q_oil", "q_gas", "q_water", "q_inj"})


def well_history_series(observations: list[ObservationSeries]) -> list[ObservationSeries]:
    return [o for o in observations if str(o.kind) in WELL_KINDS]


def predict_well_value(traj: Trajectory, t: float, well: str, kind: str) -> float | None:
    """Read F well report at or before ``t``. ``None`` if that key was not exported."""
    rates, bhp = traj.rates_and_bhp_at(float(t))
    if kind == "bhp":
        if well not in bhp:
            return None
        return float(bhp[well])
    key = str(well) + ":" + str(kind)
    if key in rates:
        return float(rates[key])
    return None


def score_well_history(
    observations: list[ObservationSeries],
    traj: Trajectory,
    *,
    holdout: bool = False,
) -> dict[str, Any] | None:
    """nRMSE and MAE keyed by well×kind. Default uses holdout=0 rows only."""
    series = well_history_series(observations)
    chosen = [o for o in series if bool(o.holdout) == bool(holdout)]
    if not chosen:
        return None
    nrmse: dict[str, float] = {}
    mae: dict[str, float] = {}
    n_samples: dict[str, int] = {}
    missing: list[str] = []
    all_resid: list[float] = []
    for obs in chosen:
        well = str(obs.sensor_name)
        kind = str(obs.kind)
        key = well + ":" + kind
        pred: list[float] = []
        obs_v: list[float] = []
        sig_v: list[float] = []
        for t, val, sig in zip(obs.times_s, obs.values, obs.sigma):
            yhat = predict_well_value(traj, float(t), well, kind)
            if yhat is None:
                continue
            pred.append(float(yhat))
            obs_v.append(float(val))
            sig_v.append(float(sig))
        if not pred:
            missing.append(key)
            continue
        err = np.asarray(pred, dtype=float) - np.asarray(obs_v, dtype=float)
        sigma = np.asarray(sig_v, dtype=float)
        nrmse[key] = float(np.sqrt(np.mean((err / sigma) ** 2)))
        mae[key] = float(np.mean(np.abs(err)))
        n_samples[key] = int(err.size)
        all_resid.extend((err / sigma).tolist())
    overall = float(np.sqrt(np.mean(np.square(all_resid)))) if all_resid else None
    return {
        "compare": "well rates/BHP only; not field L2 / cell-K / Dice",
        "holdout": bool(holdout),
        "nrmse": nrmse,
        "mae": mae,
        "n_samples": n_samples,
        "overall_nrmse": overall,
        "missing": missing,
    }
