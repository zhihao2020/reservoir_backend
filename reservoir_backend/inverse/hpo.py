"""Time-limited random search over assimilator knobs.

AutoGluon lesson: search *model* hyperparameters on a validation split, with a
wall-clock budget. Here the split is hold-out sensors. We never search K, and
we never tune F to look like CMG.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from reservoir_backend.inverse.algorithms import ALGORITHMS
from reservoir_backend.inverse.portfolio import LeaderboardRow, greedy_holdout_blend, sort_rows
from reservoir_backend.physics.rock import Rock

# Discrete spaces — same spirit as AutoGluon hyperparameter configs.
SEARCH_SPACE: dict[str, dict[str, list]] = {
    "es": {
        "n_ensemble": [8, 12, 16, 24],
        "prior_std": [0.5, 0.8, 1.2],
        "inflation": [1.0, 1.02],
    },
    "esmda": {
        "n_ensemble": [8, 12, 16, 24],
        "n_assimilations": [2, 4, 6],
        "prior_std": [0.5, 0.8, 1.2],
        "inflation": [1.0, 1.02, 1.04],
    },
    "esmda_geo": {
        "n_ensemble": [8, 12, 16],
        "n_assimilations": [3, 4, 6],
        "prior_std": [0.5, 0.8, 1.2],
        "inflation": [1.0, 1.02],
    },
    "esmda_rs": {
        "n_ensemble": [8, 12, 16],
        "n_assimilations": [4, 6],
        "prior_std": [0.6, 1.0],
        "inflation": [1.02],
    },
    "ies": {
        "n_ensemble": [8, 12, 16],
        "n_assimilations": [3, 5],
        "prior_std": [0.5, 0.8, 1.2],
        "inflation": [1.0, 1.02],
    },
}


def sample_trial(rng: np.random.Generator, *, algorithms: tuple[str, ...] | None = None) -> dict[str, Any]:
    algos = tuple(algorithms) if algorithms is not None else ALGORITHMS
    algo = str(rng.choice(np.asarray(algos, dtype=object)))
    space = SEARCH_SPACE[algo]
    cfg: dict[str, Any] = {"algorithm": algo, "n_assimilations": 1 if algo == "es" else 4}
    for key, choices in space.items():
        cfg[key] = choices[int(rng.integers(0, len(choices)))]
    cfg["seed"] = int(rng.integers(1, 10_000))
    return cfg


def run_hpo(
    twin,
    *,
    time_limit_s: float | None = 45.0,
    n_trials: int | None = None,
    blend: bool = True,
    seed: int | None = None,
    algorithms: tuple[str, ...] | None = None,
):
    """Random search. Rank and optionally blend on hold-out."""
    rng = np.random.default_rng(int(twin.inverse.seed if seed is None else seed) + 17)
    deadline = None if time_limit_s is None else time.perf_counter() + float(time_limit_s)
    cap = 32 if n_trials is None else max(int(n_trials), 1)
    rows: list[LeaderboardRow] = []
    posts: list[object] = []
    notes = ["HPO searches assimilator knobs on hold-out; K is not a hyperparameter"]
    for trial in range(cap):
        remaining = None if deadline is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0.75 and posts:
            notes.append(f"stop HPO after {trial} trials (time_limit)")
            break
        cfg = sample_trial(rng, algorithms=algorithms)
        t0 = time.perf_counter()
        post = twin._calibrate_candidate(
            n_ensemble=int(cfg["n_ensemble"]),
            n_assimilations=int(cfg.get("n_assimilations", 4)),
            prior_std=float(cfg["prior_std"]),
            seed=int(cfg["seed"]),
            inflation=float(cfg["inflation"]),
            algorithm=str(cfg["algorithm"]),
            time_limit_s=remaining,
        )
        elapsed = time.perf_counter() - t0
        name = f"{cfg['algorithm']}_t{trial}"
        rows.append(
            LeaderboardRow(
                name=name,
                holdout_rmse=float(post.holdout_rmse),
                assimilate_rmse=float(post.assimilate_rmse),
                n_ensemble=int(cfg["n_ensemble"]),
                n_assimilations=len(post.esmda.diagnostics.alpha_schedule),
                prior_std=float(cfg["prior_std"]),
                seed=int(cfg["seed"]),
                elapsed_s=elapsed,
                algorithm=str(cfg["algorithm"]),
            )
        )
        posts.append(post)
        notes.append(
            f"{name} hold={post.holdout_rmse:.4g} assim={post.assimilate_rmse:.4g} "
            f"ne={cfg['n_ensemble']} Na={cfg.get('n_assimilations')} t={elapsed:.2f}s"
        )

    if not posts:
        raise ValueError("HPO produced no invert")
    order = sort_rows(rows)
    best = next(p for p, r in zip(posts, rows) if r.name == order[0].name)
    for r in rows:
        r.selected = r.name == order[0].name

    if blend and len(posts) >= 2:
        hist_end = twin.experiment.history_end_s
        t_hist = float(hist_end) if hist_end is not None else float(best.history.times_s[-1])

        def _score_theta(theta_try):
            k_try = twin.parameterization.expand(theta_try)
            hist = twin.simulate(twin.rock_from_k(k_try), t_end=t_hist, report_times=best.history.times_s)
            from reservoir_backend.inverse.portfolio import _holdout_of

            return _holdout_of(twin, hist)

        pairs = [(p.esmda.theta_mean, float(p.holdout_rmse)) for p in posts]
        picked, theta_blend, blend_score = greedy_holdout_blend(pairs, _score_theta)
        names = [rows[i].name for i in picked]
        if theta_blend is not None and np.isfinite(blend_score) and blend_score < float(best.holdout_rmse):
            from dataclasses import replace
            from reservoir_backend.twin.offline import Posterior as Post

            k_blend = twin.parameterization.expand(theta_blend)
            hist = twin.simulate(twin.rock_from_k(k_blend), t_end=t_hist, report_times=best.history.times_s)
            best = Post(
                esmda=replace(best.esmda, theta_mean=theta_blend, k_mean=k_blend),
                assimilate_rmse=float(best.assimilate_rmse),
                holdout_rmse=float(blend_score),
                forecast_rmse=None,
                identifiability=best.identifiability,
                history=hist,
                notes=list(best.notes) + [f"HPO hold-out blend of {names} hold={blend_score:.4g}"],
            )
            for r in rows:
                r.selected = False
            rows.append(
                LeaderboardRow(
                    name="holdout_blend",
                    holdout_rmse=float(blend_score),
                    assimilate_rmse=float(best.assimilate_rmse),
                    n_ensemble=int(best.esmda.theta_ensemble.shape[0]),
                    n_assimilations=len(best.esmda.diagnostics.alpha_schedule),
                    prior_std=float("nan"),
                    seed=-1,
                    elapsed_s=0.0,
                    algorithm="+".join(names),
                    selected=True,
                    blended=True,
                )
            )
            notes.append(f"HPO blend wins {blend_score:.4g} from {names}")
        else:
            notes.append(f"HPO blend discarded; kept {order[0].name}")
    return best, sort_rows(rows), notes
