"""Posterior local ensemble around LM result."""

from __future__ import annotations

import numpy as np

from reservoir_backend.inverse.lm import LMResult
from reservoir_backend.inverse.post_ensemble import sample_posterior_ensemble


class _LinParam:
    n_params = 2

    def __init__(self, h: np.ndarray) -> None:
        self.h = h

    def expand(self, theta):
        return np.asarray(theta, dtype=float)

    def project(self, theta):
        return np.asarray(theta, dtype=float)


def test_posterior_ensemble_spreads_k() -> None:
    h = np.eye(2)
    param = _LinParam(h)
    lm = LMResult(
        theta=np.array([0.0, 1.0]),
        k=np.array([0.0, 1.0]),
        theta_std=np.array([0.1, 0.2]),
        theta_cov=np.eye(2) * 0.04,
        misfit=[1.0],
        n_forward=3,
    )
    ens = sample_posterior_ensemble(param, lm, ne=8, seed=0)
    assert ens.k_members.shape == (8, 2)
    assert float(np.max(ens.k_std)) > 0.0
    assert ens.theta_members.shape == (8, 2)
