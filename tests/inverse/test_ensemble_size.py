import numpy as np

from reservoir_backend.inverse.ensemble_size import CANDIDATE_NE, candidate_sizes, posterior_spread, recommend_ne


def test_candidate_sizes_match_plan() -> None:
    assert candidate_sizes() == (8, 12, 16, 24, 32)
    assert CANDIDATE_NE[0] == 8


def test_recommend_ne_picks_smallest_stable() -> None:
    rows = [
        {"ne": 8, "mean": 1.0, "std": 0.40},
        {"ne": 12, "mean": 0.51, "std": 0.21},
        {"ne": 16, "mean": 0.50, "std": 0.20},
        {"ne": 24, "mean": 0.50, "std": 0.20},
        {"ne": 32, "mean": 0.50, "std": 0.20},
    ]
    assert recommend_ne(rows) == 12


def test_posterior_spread_quantiles() -> None:
    members = np.array([[0.0], [1.0], [2.0], [3.0]])
    s = posterior_spread(members)
    assert s["p05"] <= s["p50"] <= s["p95"]
    assert s["mean"] == np.mean(members[:, 0])
