import numpy as np

from reservoir_backend.inverse.esmda import run_esmda
from reservoir_backend.inverse.portfolio import (
    LeaderboardRow,
    blend_k_means,
    greedy_holdout_blend,
    rank_key,
    sort_rows,
)
from reservoir_backend.inverse.presets import knobs_for, portfolio_candidates, preset_names
from reservoir_backend.validation.synthetic import make_two_layer_waterflood


def test_presets_are_named_quality_levels() -> None:
    assert set(preset_names()) == {"fast", "balanced", "strict"}
    fast = knobs_for("fast")
    strict = knobs_for("strict")
    assert int(fast["n_ensemble"]) < int(strict["n_ensemble"])
    assert int(fast["n_assimilations"]) < int(strict["n_assimilations"])


def test_rank_holdout_beats_assimilate() -> None:
    rows = [
        LeaderboardRow("overfit", 3.0, 0.1, 8, 2, 0.8, 1, 1.0),
        LeaderboardRow("honest", 0.8, 1.2, 8, 2, 0.8, 2, 1.0),
        LeaderboardRow("no_hold", float("nan"), 0.05, 8, 2, 0.8, 3, 1.0),
    ]
    ranked = sort_rows(rows)
    assert ranked[0].name == "honest"
    assert ranked[-1].name == "no_hold"
    assert rank_key(float("nan"), 0.1) > rank_key(1.0, 9.0)


def test_blend_weights_better_holdout() -> None:
    a = np.array([1.0, 3.0])
    b = np.array([2.0, 4.0])
    out = blend_k_means([a, b], [0.5, 2.0])
    assert out[0] < 1.4


def test_greedy_blend_keeps_only_if_better() -> None:
    k0 = np.array([1.0, 1.0])
    k1 = np.array([5.0, 5.0])
    truth = np.array([1.0, 1.0])

    def score(k):
        return float(np.sqrt(np.mean((k - truth) ** 2)))

    picked, blended, _ = greedy_holdout_blend([(k0, 1.0), (k1, 1.05)], score)
    assert picked == [0]
    assert blended is None

    k2 = np.array([1.2, 1.2])
    picked2, blended2, s2 = greedy_holdout_blend([(k0, 0.20), (k2, 0.20)], score)
    assert set(picked2) == {0, 1}
    assert blended2 is not None
    assert s2 < 0.15


def test_esmda_time_limit_keeps_one_step() -> None:
    h = np.eye(2)
    obs = np.array([0.2, -0.1])

    class _P:
        n_params = 2

        def expand(self, theta):
            return np.asarray(theta, dtype=float).ravel()

        def sample_prior(self, n_ensemble, mean, std, seed):
            rng = np.random.default_rng(seed)
            return rng.normal(0.0, 1.0, size=(n_ensemble, 2))

    result = run_esmda(
        _P(),
        lambda th: h @ np.asarray(th, dtype=float).ravel(),
        obs,
        np.array([0.04, 0.04]),
        n_ensemble=12,
        n_assimilations=8,
        prior_mean=0.0,
        prior_std=1.0,
        seed=2,
        inflation=1.0,
        time_limit_s=0.0,
    )
    assert len(result.diagnostics.data_mismatch) == 1
    assert any("time_limit" in n for n in result.diagnostics.notes)


def test_calibrate_auto_leaderboard_on_synthetic() -> None:
    case = make_two_layer_waterflood(n_times=3, t_end=120.0, seed=3, history_frac=0.65)
    post = case.twin.calibrate_auto(time_limit_s=20.0, blend=False, n_trials=3)
    board = case.twin.last_leaderboard
    assert board
    assert any(r["selected"] for r in board)
    assert np.isfinite(post.holdout_rmse)
    algos = {r.get("algorithm") for r in board}
    assert algos & {"es", "esmda", "esmda_geo", "esmda_rs", "ies"}


def test_portfolio_skips_rs_when_budget_tiny() -> None:
    names = [n for n, _ in portfolio_candidates(1, time_limit_s=5.0)]
    assert "esmda_rs" not in names
    assert "es" in names
