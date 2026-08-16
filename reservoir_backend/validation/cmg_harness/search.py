"""Beam search + backtrack over assimilator knobs. Never searches K."""

from __future__ import annotations

import time
from typing import Any

from reservoir_backend.inverse.hpo import SEARCH_SPACE
from reservoir_backend.validation.cmg_harness.catalog import get_case
from reservoir_backend.validation.cmg_harness.journal import Journal
from reservoir_backend.validation.cmg_harness.run_one import DEFAULT_KNOBS, run_case


def neighbors(knobs: dict[str, Any], family: str) -> list[dict[str, Any]]:
    """One-axis children. ``family`` is algorithm | n_ensemble | n_assimilations | prior_std | inflation."""
    cfg = dict(knobs)
    algo = str(cfg.get("algorithm", "esmda"))
    space = SEARCH_SPACE.get(algo, SEARCH_SPACE["esmda"])
    out: list[dict[str, Any]] = []
    if family == "algorithm":
        for a in SEARCH_SPACE:
            if a == algo:
                continue
            child = dict(DEFAULT_KNOBS)
            child.update(space_defaults(a))
            child["algorithm"] = a
            out.append(child)
        return out[:2]
    if family not in space:
        return []
    choices = list(space[family])
    cur = cfg.get(family)
    if cur in choices:
        i = choices.index(cur)
        for j in (i - 1, i + 1):
            if 0 <= j < len(choices):
                child = dict(cfg)
                child[family] = choices[j]
                out.append(child)
        return out
    nearest = min(choices, key=lambda x: abs(float(x) - float(cur)))
    child = dict(cfg)
    child[family] = nearest
    out.append(child)
    return out


def space_defaults(algo: str) -> dict[str, Any]:
    space = SEARCH_SPACE[algo]
    cfg = {"algorithm": algo, "n_assimilations": 1 if algo == "es" else 3}
    for key, choices in space.items():
        cfg[key] = choices[min(1, len(choices) - 1)]
    return cfg


FAMILIES = ("n_ensemble", "n_assimilations", "prior_std", "inflation", "algorithm")


def run_search(
    case_id: str = "lab_layers",
    *,
    time_limit_s: float = 300.0,
    beam: int = 3,
    max_waves: int = 3,
    journal: Journal | None = None,
) -> dict[str, Any]:
    spec = get_case(case_id)
    journal = journal or Journal()
    deadline = time.perf_counter() + float(time_limit_s)
    seed = dict(DEFAULT_KNOBS)
    first = run_case(spec, knobs=seed, invert=True, journal=journal, parent=None, wave=0)
    if first.get("probe", "").startswith("prune") or first.get("probe", "").startswith("skip"):
        return {"stopped": "seed_pruned", "seed": first, "attempts": 1}

    def attempt_j(rec: dict) -> float:
        return float((rec.get("score") or {}).get("J", 9.0))

    beam_rows: list[dict] = [first]
    last_best = attempt_j(first)
    stall = 0
    attempts = 1
    fam_i = 0
    for wave in range(1, max_waves + 1):
        if time.perf_counter() >= deadline:
            break
        family = FAMILIES[fam_i % len(FAMILIES)]
        fam_i += 1
        expanded: list[dict] = []
        pruned_family = False
        for rec in beam_rows:
            parent_id = rec.get("attempt_id")
            parent_knobs = rec.get("knobs") or seed
            for child_k in neighbors(parent_knobs, family):
                if time.perf_counter() >= deadline:
                    break
                child = run_case(
                    spec, knobs=child_k, invert=True, journal=journal, parent=parent_id, wave=wave
                )
                attempts += 1
                if str(child.get("decision")) == "prune":
                    pruned_family = True
                    continue
                if str(child.get("decision")) == "backtrack":
                    continue
                expanded.append(child)
            if pruned_family:
                break
        if not expanded:
            stall += 1
            continue
        beam_rows = sorted(beam_rows + expanded, key=attempt_j)[: max(int(beam), 1)]
        best = attempt_j(beam_rows[0])
        if last_best - best < 0.01:
            stall += 1
        else:
            stall = 0
        last_best = min(last_best, best)
        if stall >= 2:
            break
    return {
        "case": case_id,
        "attempts": attempts,
        "best": beam_rows[0] if beam_rows else first,
        "beam": [{"id": r.get("attempt_id"), "J": attempt_j(r), "knobs": r.get("knobs")} for r in beam_rows],
    }
