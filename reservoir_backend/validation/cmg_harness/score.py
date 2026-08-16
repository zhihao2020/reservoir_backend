"""Scores vs CMG gauges and fields. K_CMG is logged, never used to rank."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.validation.cmg_harness.catalog import DAY_S, PSI, DEFAULT_WEIGHTS


@dataclass
class Score:
    hold: float = float("nan")
    forecast: float = float("nan")
    assimilate: float = float("nan")
    p_rmse_psi: float = float("nan")
    p_rmse_demean_psi: float = float("nan")
    sw_rmse: float = float("nan")
    bt_rel: float = float("nan")
    bt_cmg_d: float = float("nan")
    bt_f_d: float = float("nan")
    sw_mean_cmg: float = float("nan")
    sw_mean_f: float = float("nan")
    J: float = float("nan")
    k_contrast_post: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        return d


def rmse(a: NDArray, b: NDArray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def combine_j(score: Score, weights: dict[str, float] | None = None) -> float:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    parts = [
        w["hold"] * _finite(score.hold, 2.0),
        w["forecast"] * _finite(score.forecast, 2.0),
        w["p"] * (_finite(_p_for_j(score), 200.0) / 100.0),
        w["sw"] * (_finite(score.sw_rmse, 0.4) / 0.1),
        w["bt"] * _finite(score.bt_rel, 1.0),
    ]
    return float(sum(parts))


def _p_for_j(score: Score) -> float:
    d = float(score.p_rmse_demean_psi)
    if np.isfinite(d):
        return d
    return float(score.p_rmse_psi)


def _finite(x: float, cap: float) -> float:
    v = float(x)
    if not np.isfinite(v):
        return cap
    return min(max(v, 0.0), cap * 4.0)


def breakthrough_time_days(
    times_s: NDArray[np.float64],
    sw_at_producer: NDArray[np.float64],
    *,
    threshold: float = 0.35,
) -> float:
    """First time producer-cell mean Sw crosses ``threshold``. inf if never."""
    t = np.asarray(times_s, dtype=float).ravel()
    s = np.asarray(sw_at_producer, dtype=float).ravel()
    if t.size == 0 or s.size != t.size:
        return float("inf")
    hit = np.nonzero(s >= float(threshold))[0]
    if hit.size == 0:
        return float("inf")
    return float(t[int(hit[0])] / DAY_S)


def breakthrough_rel(t_cmg: float, t_f: float) -> float:
    if not np.isfinite(t_cmg) and not np.isfinite(t_f):
        return 0.0
    if not np.isfinite(t_cmg) or not np.isfinite(t_f):
        return 1.0
    return float(abs(t_f - t_cmg) / max(t_cmg, 0.05))


def field_gap(
    f_maps: dict[float, dict[str, NDArray]],
    cmg_maps: dict[float, dict[str, NDArray]],
) -> tuple[float, float, float, float, float]:
    """Last common day: raw p RMSE, demeaned p RMSE, Sw RMSE, Sw means."""
    days = sorted(set(f_maps) & set(cmg_maps))
    if not days:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    d = days[-1]
    fp = np.asarray(f_maps[d]["p"], dtype=float)
    cp = np.asarray(cmg_maps[d]["p"], dtype=float)
    p_rmse = rmse(fp, cp)
    p_demean = rmse(fp - float(np.mean(fp)), cp - float(np.mean(cp)))
    sw_rmse = rmse(f_maps[d]["sw"], cmg_maps[d]["sw"])
    return p_rmse, p_demean, sw_rmse, float(np.mean(cmg_maps[d]["sw"])), float(np.mean(f_maps[d]["sw"]))


def maps_from_traj(traj, days: tuple[float, ...], grid) -> dict[float, dict[str, NDArray]]:
    out = {}
    for d in days:
        st = traj.state_at(float(d) * DAY_S)
        out[float(d)] = {
            "p": grid.reshape_ijk(st.pressure) / PSI,
            "sw": grid.reshape_ijk(st.sw),
        }
    return out


def producer_sw_series(traj, cells: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if cells.size == 0:
        return traj.times_s, np.full(traj.times_s.size, np.nan)
    means = []
    for st in traj.states:
        means.append(float(np.mean(st.sw[cells])))
    return np.asarray(traj.times_s, dtype=float), np.asarray(means, dtype=float)
