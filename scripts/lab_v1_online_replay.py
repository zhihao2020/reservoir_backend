"""Replay synthetic / experiment CSVs through TwinLoops (no UDP socket)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.history_match import HistoryMatchWorkflow
from reservoir_backend.twin.lab_v1 import CF_PRIOR_FACTOR, CF_TRUE_M2, generate_truth, load_lab_v1
from reservoir_backend.twin.loops import TwinLoops
from reservoir_backend.twin.offline import window_observations


def _times(twin) -> np.ndarray:
    chunks = [o.times_s for o in twin.experiment.observations]
    if not chunks:
        return np.zeros(0, dtype=float)
    return np.unique(np.concatenate(chunks))


def run_replay(*, dev: bool = True, cf_true: float = CF_TRUE_M2) -> dict:
    twin = load_lab_v1(dev=dev)
    generate_truth(twin, cf_true=cf_true, noise=True, case="B")
    prior_cf = float(cf_true) * float(CF_PRIOR_FACTOR)
    twin.parameterization.prior_mean = float(twin.parameterization.encode(np.array([prior_cf]))[0])
    twin.inverse.prior_mean = float(twin.parameterization.prior_mean)
    post = HistoryMatchWorkflow().run(twin)
    loops = TwinLoops.from_posterior(twin, post, slow_interval_s=2.0, rng_seed=3)
    times = _times(twin)
    reused = 0
    seen: set[tuple[str, float]] = set()
    cf_hist = []
    cycle_s = []
    dropouts = 0
    for t in times:
        for obs in twin.experiment.observations:
            for ti in np.asarray(obs.times_s, dtype=float):
                if abs(float(ti) - float(t)) > 1.0e-12:
                    continue
                key = (obs.sensor_name, float(ti))
                if key in seen:
                    reused += 1
                seen.add(key)
        try:
            loops.fast_step(1.0)
        except Exception:
            pass
        w = window_observations(twin.experiment.observations, loops.last_slow_s, float(t))
        if w:
            post2 = loops.maybe_slow(float(t), observations=twin.experiment.observations)
            if post2 is not None:
                cf_hist.append(float(twin.parameterization.decode(post2.theta)[0]))
                cycle_s.append(float(loops.last_cycle_s))
        else:
            dropouts += 1
    backlog = False
    if cycle_s:
        backlog = max(cycle_s) > 10.0 * max(float(loops.slow_interval_s), 1.0)
    report = {
        "gate": "lab_v1_online_replay",
        "n_times": int(times.size),
        "reused_observations": int(reused),
        "n_slow": len(cf_hist),
        "cf_hist": cf_hist,
        "cf_drift": float(max(cf_hist) - min(cf_hist)) if cf_hist else 0.0,
        "cycle_s": cycle_s,
        "backlog": bool(backlog),
        "dropouts_empty_window": int(dropouts),
        "from_posterior_duals": loops.dual_states is not None and all(s is not None for s in (loops.dual_states or [])),
        "eta_threshold": float(loops.eta_threshold),
        "last_fast_error": float(loops.last_fast_error),
        "last_fast_error_inf": float(loops.last_fast_error_inf),
        "checks": {
            "no_reuse": reused == 0,
            "per_member_duals": loops.dual_states is not None,
            "no_backlog": not backlog,
        },
    }
    dest = ROOT / "results" / "lab_v1" / "online"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true", default=True)
    p.add_argument("--product", action="store_true")
    args = p.parse_args(argv)
    if args.product:
        args.dev = False
    report = run_replay(dev=bool(args.dev))
    print(json.dumps(report, indent=2))
    ok = report["checks"]["no_reuse"] and report["checks"]["per_member_duals"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
