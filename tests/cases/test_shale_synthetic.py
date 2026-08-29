"""Synthetic shale depletion LM smoke test (no CMG)."""

from __future__ import annotations

import numpy as np

from reservoir_backend.inverse.frac import decode_frac_theta
from reservoir_backend.synthetic import make_shale_depletion


def test_shale_synthetic_lm_reduces_misfit() -> None:
    case = make_shale_depletion(n_times=3, max_iter=6, t_end=60.0 * 86400.0)
    twin = case.twin
    param = twin.parameterization
    assert param.n_params == 4
    assert twin.physics.fully_implicit is False
    assert any(o.holdout for o in twin.experiment.observations)
    assert twin.ports[0].min_bhp_Pa is not None
    post = twin.calibrate(max_iter=6)
    eng = decode_frac_theta(param, post.theta)
    assert post.assimilate_rmse < 2.5
    assert np.isfinite(post.holdout_rmse)
    assert eng["n_frac"] == 4.0
    assert float(np.std(post.theta)) > 0.0
