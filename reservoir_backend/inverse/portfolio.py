"""Hold-out leaderboard over a small invert portfolio.

Borrowed from AutoGluon: try a few strong defaults, rank on validation
(here: hold-out sensors), optionally blend survivors. Not HPO over K,
and not a stack of tabular models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.inverse.presets import portfolio_candidates
from reservoir_backend.physics.rock import Rock


@dataclass
class LeaderboardRow:
    name: str
    holdout_rmse: float
    assimilate_rmse: float
    n_ensemble: int
    n_assimilations: int
    prior_std: float
    seed: int
    elapsed_s: float
    algorithm: str = "esmda"
    selected: bool = False
    blended: bool = False

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "holdout_rmse": self.holdout_rmse,
            "assimilate_rmse": self.assimilate_rmse,
            "n_ensemble": self.n_ensemble,
            "n_assimilations": self.n_assimilations,
            "prior_std": self.prior_std,
            "seed": self.seed,
            "elapsed_s": self.elapsed_s,
            "selected": self.selected,
            "blended": self.blended,
        }


def rank_key(holdout_rmse: float, assimilate_rmse: float) -> tuple[float, float]:
    """Hold-out first (AutoGluon val). No hold-out sorts last."""
    h = float(holdout_rmse)
    a = float(assimilate_rmse)
    if not np.isfinite(h):
        h = 1.0e9
    if not np.isfinite(a):
        a = 1.0e9
    return (h, a)


def sort_rows(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(rows, key=lambda r: rank_key(r.holdout_rmse, r.assimilate_rmse))


def blend_k_means(k_list: list[NDArray[np.float64]], scores: list[float]) -> NDArray[np.float64]:
    """Score-weighted blend of generic, equally shaped vectors."""
    weights = []
    for s in scores:
        v = float(s)
        if not np.isfinite(v) or v <= 0.0:
            v = 1.0e3
        weights.append(1.0 / (v * v))
    w = np.asarray(weights, dtype=float)
    w = w / float(np.sum(w))
    acc = np.zeros_like(k_list[0], dtype=float)
    for wi, k in zip(w, k_list):
        acc = acc + float(wi) * np.asarray(k, dtype=float)
    return acc


def greedy_holdout_blend(
    members: list[tuple[NDArray[np.float64], float]],
    score_fn,
) -> tuple[list[int], NDArray[np.float64] | None, float]:
    """Add members by hold-out. Keep a candidate only if the blend improves.

    ``members`` is ``(vector, holdout_rmse)``. ``score_fn(vector)`` re-evaluates hold-out.
    Returns selected indices, the blended vector (or None), and the winning score.
    """
    if not members:
        raise ValueError("no members to blend")
    order = sorted(range(len(members)), key=lambda i: rank_key(members[i][1], 0.0))
    selected = [order[0]]
    best_k = np.asarray(members[order[0]][0], dtype=float)
    best_score = float(members[order[0]][1])
    for idx in order[1:]:
        trial_idx = selected + [idx]
        k_try = blend_k_means([members[i][0] for i in trial_idx], [members[i][1] for i in trial_idx])
        score = float(score_fn(k_try))
        if np.isfinite(score) and score < best_score:
            selected = trial_idx
            best_k = k_try
            best_score = score
    if len(selected) < 2:
        return selected, None, best_score
    return selected, best_k, best_score


def run_portfolio(twin, *, time_limit_s: float | None = None, blend: bool = True):
    """Run a few invert designs, pick / blend by hold-out.

    ``twin`` is a ``DigitalTwin``. Imported lazily to keep this module free of
    the orchestrator at import time.
    """
    from reservoir_backend.twin.offline import Posterior

    deadline = None if time_limit_s is None else time.perf_counter() + float(time_limit_s)
    candidates = portfolio_candidates(twin.inverse.seed, time_limit_s=time_limit_s)
    rows: list[LeaderboardRow] = []
    posts: list[object] = []
    notes = ["portfolio ranks on hold-out sensors, not assimilate misfit"]
    for name, knobs in candidates:
        remaining = None if deadline is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0.5 and posts:
            notes.append(f"skip {name}: time_limit reached")
            break
        t0 = time.perf_counter()
        post = twin._calibrate_candidate(
            n_ensemble=int(knobs["n_ensemble"]),
            n_assimilations=int(knobs["n_assimilations"]),
            prior_std=float(knobs["prior_std"]),
            seed=int(knobs["seed"]),
            time_limit_s=remaining,
            algorithm=str(knobs.get("algorithm", "esmda")),
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            LeaderboardRow(
                name=name,
                holdout_rmse=float(post.holdout_rmse),
                assimilate_rmse=float(post.assimilate_rmse),
                n_ensemble=int(knobs["n_ensemble"]),
                n_assimilations=len(post.esmda.diagnostics.alpha_schedule),
                prior_std=float(knobs["prior_std"]),
                seed=int(knobs["seed"]),
                elapsed_s=elapsed,
                algorithm=str(knobs.get("algorithm", "esmda")),
            )
        )
        posts.append(post)
        notes.append(f"{name} hold={post.holdout_rmse:.4g} assim={post.assimilate_rmse:.4g} t={elapsed:.2f}s")

    if not posts:
        raise ValueError("portfolio produced no invert")
    order = sort_rows(rows)
    best_name = order[0].name
    best = next(p for p, r in zip(posts, rows) if r.name == best_name)
    for r in rows:
        r.selected = r.name == best_name

    if blend and len(posts) >= 2:
        hist_end = twin.experiment.history_end_s
        t_hist = float(hist_end) if hist_end is not None else float(best.history.times_s[-1])

        def _score_theta(theta_try: NDArray[np.float64]) -> float:
            k_try = twin.parameterization.expand(theta_try)
            hist = twin.simulate(twin.rock_from_k(k_try), t_end=t_hist, report_times=best.history.times_s)
            return _holdout_of(twin, hist)

        pairs = [(p.esmda.theta_mean, float(p.holdout_rmse)) for p in posts]
        picked, theta_blend, blend_score = greedy_holdout_blend(pairs, _score_theta)
        names = [rows[i].name for i in picked]
        if theta_blend is not None and np.isfinite(blend_score) and blend_score < float(best.holdout_rmse):
            k_blend = twin.parameterization.expand(theta_blend)
            hist = twin.simulate(twin.rock_from_k(k_blend), t_end=t_hist, report_times=best.history.times_s)
            esmda = replace(best.esmda, theta_mean=theta_blend, k_mean=k_blend)
            best = Posterior(
                esmda=esmda,
                assimilate_rmse=float(best.assimilate_rmse),
                holdout_rmse=float(blend_score),
                forecast_rmse=None,
                identifiability=best.identifiability,
                history=hist,
                notes=list(best.notes) + [f"hold-out blend of {names} hold={blend_score:.4g}"],
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
            notes.append(f"blend wins hold-out {blend_score:.4g} from {names}")
        else:
            notes.append(f"blend discarded; kept {order[0].name}")

    return best, sort_rows(rows), notes


def _holdout_of(twin, hist) -> float:
    from reservoir_backend.twin.offline import predict_from_trajectory, stack_observations

    hold = [o for o in twin.experiment.observations if o.holdout]
    if not hold:
        return float("nan")
    history_end = twin.experiment.history_end_s
    trimmed = []
    for obs in hold:
        mask = np.ones(obs.times_s.size, dtype=bool)
        if history_end is not None:
            mask = obs.times_s <= float(history_end) + 1.0e-12
        if not np.any(mask):
            continue
        trimmed.append(type(obs)(obs.sensor_name, obs.kind, obs.times_s[mask], obs.values[mask], obs.sigma[mask], True))
    if not trimmed:
        return float("nan")
    d = stack_observations(trimmed)
    pred = predict_from_trajectory(twin.operator, twin.experiment, hist, trimmed)
    return float(np.sqrt(np.mean(((pred - d.values) / d.sigma) ** 2)))
