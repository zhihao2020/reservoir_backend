"""One case: probe → invert → score → journal."""

from __future__ import annotations

from typing import Any

import numpy as np

from reservoir_backend.physics.rock import log_permeability
from reservoir_backend.validation.cmg_harness.adapter import build_twin, inflate_model_error
from reservoir_backend.validation.cmg_harness.catalog import DAY_S, MD_TO_M2, CaseSpec, get_case
from reservoir_backend.validation.cmg_harness.journal import Attempt, Journal
from reservoir_backend.validation.cmg_harness.probes import run_probe
from reservoir_backend.validation.cmg_harness.score import (
    Score,
    breakthrough_rel,
    breakthrough_time_days,
    combine_j,
    field_gap,
    maps_from_traj,
    producer_sw_series,
)


DEFAULT_KNOBS = {
    "algorithm": "esmda",
    "n_ensemble": 12,
    "n_assimilations": 3,
    "prior_std": 0.35,
    "inflation": 1.02,
    "seed": 6,
    "n_workers": 4,
}


def run_case(
    spec: CaseSpec | str,
    *,
    knobs: dict[str, Any] | None = None,
    invert: bool = True,
    journal: Journal | None = None,
    parent: str | None = None,
    wave: int = 0,
) -> dict[str, Any]:
    if isinstance(spec, str):
        spec = get_case(spec)
    cfg = dict(DEFAULT_KNOBS)
    if knobs:
        cfg.update(knobs)
    record: dict[str, Any] = {"case": spec.id, "knobs": cfg, "status": spec.status}

    if spec.status != "ready":
        record["probe"] = f"skip:{spec.status}"
        record["score"] = Score(notes=[spec.note or spec.status]).as_dict()
        _maybe_journal(journal, spec, cfg, parent, wave, record, "prune", spec.status)
        return record
    if not spec.out_path.is_file():
        record["probe"] = "skip:need_imex"
        record["score"] = Score(notes=["missing .out"]).as_dict()
        _maybe_journal(journal, spec, cfg, parent, wave, record, "prune", "need_imex")
        return record

    twin, extra = build_twin(spec, knobs=cfg, with_observations=invert)
    probe = run_probe(spec, twin, extra)
    record["probe"] = probe.reason
    record["probe_detail"] = {"sw_max": probe.sw_max, "p_std_psi": probe.p_std_psi, "bt_rel": probe.bt_rel}
    if not probe.ok:
        sc = Score(notes=[probe.reason], bt_rel=probe.bt_rel)
        sc.J = combine_j(sc, spec.weights)
        record["score"] = sc.as_dict()
        _maybe_journal(journal, spec, cfg, parent, wave, record, "prune", probe.reason)
        return record

    if not invert:
        sc = Score(notes=["probe_only"], bt_rel=probe.bt_rel)
        sc.J = combine_j(sc, spec.weights)
        record["score"] = sc.as_dict()
        _maybe_journal(journal, spec, cfg, parent, wave, record, "keep", "probe_only")
        return record

    inflate_model_error(
        twin,
        extra.get("k_true"),
        demean_pressure=spec.dt_max_s >= 3600.0,
        skip_pressure=False,
    )
    post = twin.calibrate()
    fc = twin.forecast(post)
    days = spec.history_days
    f_maps = maps_from_traj(post.history, days, twin.grid)
    p_rmse, p_demean, sw_rmse, sw_c, sw_f = field_gap(f_maps, extra["maps"])

    cells = extra["producer_cells"]
    t_f, sw_prod = producer_sw_series(post.history, cells)
    bt_f = breakthrough_time_days(t_f, sw_prod)
    cmg_t, cmg_sw = [], []
    for d in sorted(extra["maps"]):
        sw = np.asarray(extra["maps"][d]["sw"], dtype=float).ravel()
        cmg_sw.append(float(np.mean(sw[cells])) if cells.size else float("nan"))
        cmg_t.append(d * DAY_S)
    bt_c = breakthrough_time_days(np.asarray(cmg_t), np.asarray(cmg_sw))

    sc = Score(
        hold=float(post.holdout_rmse),
        forecast=float(twin.score_forecast(fc)),
        assimilate=float(post.assimilate_rmse),
        p_rmse_psi=p_rmse,
        p_rmse_demean_psi=p_demean,
        sw_rmse=sw_rmse,
        bt_rel=breakthrough_rel(bt_c, bt_f),
        bt_cmg_d=bt_c if np.isfinite(bt_c) else float("nan"),
        bt_f_d=bt_f if np.isfinite(bt_f) else float("nan"),
        sw_mean_cmg=sw_c,
        sw_mean_f=sw_f,
        notes=list(post.notes),
    )
    k_true = extra.get("k_true")
    if k_true is not None:
        k_post = post.esmda.k_mean
        sc.notes.append(
            f"logk_rmse={float(np.sqrt(np.mean((log_permeability(k_post) - log_permeability(k_true)) ** 2))):.3f}"
        )
        hi_m = k_true >= max(float(np.median(k_true)) * 1.5, float(np.min(k_true)) * 1.01)
        if np.any(hi_m) and np.any(~hi_m):
            sc.k_contrast_post = float(np.mean(k_post[hi_m]) / max(float(np.mean(k_post[~hi_m])), 1.0e-30))
    sc.J = combine_j(sc, spec.weights)
    record["score"] = sc.as_dict()
    record["theta"] = post.esmda.theta_mean.tolist()
    parent_row = journal.by_id(parent) if journal is not None and parent else None
    decision, reason = (journal.decide(sc.J, parent_row) if journal is not None else ("keep", "no journal"))
    _maybe_journal(journal, spec, cfg, parent, wave, record, decision, reason)
    return record


def _maybe_journal(journal, spec, cfg, parent, wave, record, decision, reason) -> None:
    record["decision"] = decision
    record["reason"] = reason
    if journal is None:
        return
    sc = record.get("score") or {}
    attempt = Attempt(
        id=journal.next_id(),
        parent=parent,
        case=spec.id,
        knobs=dict(cfg),
        probe=str(record.get("probe") or ""),
        scores={
            k: sc[k]
            for k in ("hold", "forecast", "p_rmse_psi", "p_rmse_demean_psi", "sw_rmse", "bt_rel")
            if k in sc
        },
        J=float(sc.get("J", float("nan"))),
        decision=decision,
        reason=str(reason),
        wave=int(wave),
    )
    journal.append(attempt)
    record["attempt_id"] = attempt.id


def run_suite(
    case_ids: list[str] | None = None,
    *,
    knobs: dict[str, Any] | None = None,
    invert: bool = True,
    fast: bool = False,
    journal: Journal | None = None,
) -> dict[str, Any]:
    from reservoir_backend.validation.cmg_harness.catalog import list_cases

    specs = [get_case(c) for c in case_ids] if case_ids else list_cases()
    rows = []
    for spec in specs:
        do_inv = bool(invert) and (not fast or spec.invert_in_fast)
        rows.append(run_case(spec, knobs=knobs, invert=do_inv, journal=journal))
    return {"cases": rows}
