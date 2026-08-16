from reservoir_backend.inverse.algorithms import ALGORITHMS
from reservoir_backend.inverse.hpo import SEARCH_SPACE, sample_trial
import numpy as np


def test_search_space_covers_all_algorithms() -> None:
    assert set(SEARCH_SPACE) == set(ALGORITHMS)


def test_sample_trial_is_valid() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        cfg = sample_trial(rng)
        assert cfg["algorithm"] in ALGORITHMS
        assert cfg["n_ensemble"] >= 8
        assert cfg["prior_std"] > 0.0
        assert cfg["inflation"] >= 1.0
