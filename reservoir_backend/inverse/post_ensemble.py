"""Local Gaussian ensemble around LM posterior (Ne small, not ES-MDA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.inverse.lm import LMResult, Parameterization


@dataclass
class PosteriorEnsemble:
    theta_members: NDArray[np.float64]
    k_members: NDArray[np.float64]
    k_mean: NDArray[np.float64]
    k_std: NDArray[np.float64]
    theta_mean: NDArray[np.float64]
    theta_std: NDArray[np.float64]
    dual_states: list | None = None
    flash_caches: list | None = None


def sample_posterior_ensemble(
    parameterization: Parameterization,
    lm_result: LMResult,
    *,
    ne: int = 8,
    seed: int = 0,
) -> PosteriorEnsemble:
    """Sample θ ~ N(θ_lm, diag(θ_std²)), project, expand to K."""
    ne = max(int(ne), 1)
    th_mean = np.asarray(lm_result.theta, dtype=float).ravel()
    th_std = np.maximum(np.asarray(lm_result.theta_std, dtype=float).ravel(), 1.0e-8)
    rng = np.random.default_rng(int(seed))
    n = th_mean.size
    draws = th_mean + th_std * rng.standard_normal((ne, n))
    theta_members = np.zeros((ne, n), dtype=float)
    k_list: list[NDArray[np.float64]] = []
    project = getattr(parameterization, "project", None)
    for i in range(ne):
        th = np.asarray(draws[i], dtype=float)
        if callable(project):
            th = np.asarray(project(th), dtype=float)
        theta_members[i] = th
        k_list.append(np.asarray(parameterization.expand(th), dtype=float).ravel())
    k_members = np.stack(k_list, axis=0)
    theta_mean = np.mean(theta_members, axis=0)
    theta_std = np.std(theta_members, axis=0, ddof=1) if ne > 1 else th_std.copy()
    k_mean = np.mean(k_members, axis=0)
    k_std = np.std(k_members, axis=0, ddof=1) if ne > 1 else np.zeros_like(k_mean)
    return PosteriorEnsemble(
        theta_members=theta_members,
        k_members=k_members,
        k_mean=k_mean,
        k_std=k_std,
        theta_mean=theta_mean,
        theta_std=theta_std,
    )
