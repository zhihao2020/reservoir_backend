"""Smoke: 3×1×1, Ne=4, Na=1. Default suite (not slow)."""

import numpy as np
import pytest

from reservoir_backend.synthetic import make_scalar_cf_twin

pytestmark = pytest.mark.assimilation


def test_smoke_scalar_cf_esmda() -> None:
    case = make_scalar_cf_twin(
        n=(3, 1, 1),
        t_end=2.0,
        n_times=1,
        noise_p=0.0,
        ensemble_size=4,
        assimilation_steps=1,
        seed=1,
    )
    post = case.twin.calibrate()
    post_m = float(post.theta[0])
    assert np.isfinite(post_m)
    assert post.n_forward >= 4
