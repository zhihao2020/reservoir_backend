"""Cheap single-forward probes. Fail here → do not run the ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reservoir_backend.exceptions import LinearSolveFailure, TimeStepUnderflow
from reservoir_backend.physics.rock import Rock
from reservoir_backend.validation.cmg_harness.catalog import DAY_S, CaseSpec
from reservoir_backend.validation.cmg_harness.score import breakthrough_rel, breakthrough_time_days, producer_sw_series


@dataclass
class ProbeResult:
    ok: bool
    reason: str
    sw_max: float = 0.0
    p_std_psi: float = 0.0
    bt_rel: float = float("nan")


def classify_probe(
    *,
    ran: bool,
    sw0: float,
    sw_max: float,
    p_std_psi: float,
    bt_rel: float | None = None,
    min_dsw: float = 0.02,
    min_p_std: float = 15.0,
) -> ProbeResult:
    if not ran:
        return ProbeResult(False, "prune:underflow", sw_max=sw_max, p_std_psi=p_std_psi)
    if sw_max - sw0 < min_dsw:
        return ProbeResult(False, "prune:no_flood", sw_max=sw_max, p_std_psi=p_std_psi)
    if p_std_psi < min_p_std:
        return ProbeResult(False, "prune:no_dp", sw_max=sw_max, p_std_psi=p_std_psi)
    if bt_rel is not None and np.isfinite(bt_rel) and bt_rel >= 10.0:
        return ProbeResult(False, "prune:bt", sw_max=sw_max, p_std_psi=p_std_psi, bt_rel=bt_rel)
    return ProbeResult(True, "ok", sw_max=sw_max, p_std_psi=p_std_psi, bt_rel=float(bt_rel or 0.0))


def run_probe(spec: CaseSpec, twin, extra: dict) -> ProbeResult:
    from reservoir_backend.validation.cmg_harness.catalog import PSI

    # Lab windows are < 1 d; field IMEX reports start at ~1 d and only move later.
    idx = 0 if spec.history_days[-1] <= 2.0 else min(1, len(spec.history_days) - 1)
    t_end = float(spec.history_days[idx]) * DAY_S
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    k = extra.get("k_true")
    if k is None:
        k = np.full(twin.grid.n_cells, spec.prior_k_md * 9.869233e-16)
    sw0 = float(extra.get("sw_init", 0.20))
    try:
        traj = twin.simulate(twin.rock_from_k(k), t_end=t_end)
    except (TimeStepUnderflow, LinearSolveFailure):
        return ProbeResult(False, "prune:underflow")
    last = traj.states[-1]
    sw_max = float(np.max(last.sw))
    p_std = float(np.std(last.pressure) / PSI)
    bt_rel = None
    cells = extra.get("producer_cells")
    maps = extra.get("maps") or {}
    if cells is not None and getattr(cells, "size", 0) and maps:
        t_f, sw_f = producer_sw_series(traj, cells)
        bt_f = breakthrough_time_days(t_f, sw_f)
        cmg_days = sorted(maps)
        cmg_sw = []
        cmg_t = []
        for d in cmg_days:
            sw = np.asarray(maps[d]["sw"], dtype=float).ravel()
            cmg_sw.append(float(np.mean(sw[cells])))
            cmg_t.append(d * DAY_S)
        bt_c = breakthrough_time_days(np.asarray(cmg_t), np.asarray(cmg_sw))
        bt_rel = breakthrough_rel(bt_c, bt_f)
    return classify_probe(ran=True, sw0=sw0, sw_max=sw_max, p_std_psi=p_std, bt_rel=bt_rel)
