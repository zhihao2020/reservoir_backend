"""Ensemble prior sampling and failed-member replacement."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import AssimilationError


def sample_log_prior(
    mean: NDArray[np.float64] | float,
    std: NDArray[np.float64] | float,
    n_params: int,
    n_ensemble: int,
    rng: np.random.Generator,
    *,
    log_min: float | None = None,
    log_max: float | None = None,
) -> NDArray[np.float64]:
    """Return (n_params, n_ensemble) draws of ``m = log C_f`` (or log k)."""
    mu = np.broadcast_to(np.asarray(mean, dtype=float).ravel(), (n_params,))
    sd = np.broadcast_to(np.asarray(std, dtype=float).ravel(), (n_params,))
    if n_ensemble < 2:
        raise AssimilationError("ensemble size must be >= 2")
    x = mu[:, None] + sd[:, None] * rng.standard_normal((n_params, int(n_ensemble)))
    if log_min is not None or log_max is not None:
        lo = -np.inf if log_min is None else float(log_min)
        hi = np.inf if log_max is None else float(log_max)
        x = np.clip(x, lo, hi)
    return x


def replace_failed_members(
    members: NDArray[np.float64],
    failed: NDArray[np.bool_],
    rng: np.random.Generator,
    prior_std: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Replace failed columns from a valid neighbor plus small prior noise."""
    x = np.asarray(members, dtype=float).copy()
    mask = np.asarray(failed, dtype=bool).ravel()
    if mask.size != x.shape[1]:
        raise AssimilationError("failed mask length != ensemble size")
    valid = np.flatnonzero(~mask)
    if valid.size == 0:
        raise AssimilationError("all ensemble members failed")
    sd = np.broadcast_to(np.asarray(prior_std, dtype=float).ravel(), (x.shape[0],))
    for j in np.flatnonzero(mask):
        src = int(rng.choice(valid))
        x[:, j] = x[:, src] + 0.25 * sd * rng.standard_normal(x.shape[0])
    return x


def replace_failed_member_bundle(
    members: NDArray[np.float64],
    failed: NDArray[np.bool_],
    rng: np.random.Generator,
    prior_std: NDArray[np.float64] | float,
    dual_states: list | None = None,
    flash_caches: list | None = None,
) -> tuple[NDArray[np.float64], list | None, list | None]:
    """Replace failed θ and the matching DualState / flash cache from the same donor."""
    x = np.asarray(members, dtype=float).copy()
    mask = np.asarray(failed, dtype=bool).ravel()
    if mask.size != x.shape[1]:
        raise AssimilationError("failed mask length != ensemble size")
    valid = np.flatnonzero(~mask)
    if valid.size == 0:
        raise AssimilationError("all ensemble members failed")
    sd = np.broadcast_to(np.asarray(prior_std, dtype=float).ravel(), (x.shape[0],))
    duals = None if dual_states is None else list(dual_states)
    caches = None if flash_caches is None else list(flash_caches)
    for j in np.flatnonzero(mask):
        src = int(rng.choice(valid))
        x[:, j] = x[:, src] + 0.25 * sd * rng.standard_normal(x.shape[0])
        if duals is not None and src < len(duals):
            donor = duals[src]
            duals[int(j)] = None if donor is None else donor.copy()
        if caches is not None and src < len(caches):
            donor_c = caches[src]
            caches[int(j)] = None if donor_c is None else donor_c.copy()
    return x, duals, caches
