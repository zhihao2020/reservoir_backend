import numpy as np

from reservoir_backend.synthetic import (
    evaluate_synthetic,
    make_channel_waterflood,
    make_two_layer_waterflood,
)


def test_synthetic_observations_come_from_forward() -> None:
    case = make_two_layer_waterflood(n_times=3, t_end=180.0, seed=1)
    assert case.twin.experiment.observations
    assert any(o.holdout for o in case.twin.experiment.observations)
    assert any(not o.holdout for o in case.twin.experiment.observations)
    kinds = {c.kind for c in case.twin.experiment.controls if c.port_name == "INJ"}
    assert "rate" in kinds
    assert "pressure" not in kinds
    names = {s.name for s in case.twin.experiment.sensors}
    assert "Pin_bot" in names and "Pin_top" in names


def test_lm_recovers_layer_permeability() -> None:
    case = make_two_layer_waterflood(n_times=6, t_end=700.0, seed=2, history_frac=0.85)
    post = case.twin.calibrate()
    metrics = evaluate_synthetic(case, post)
    assert metrics["posterior_data_nrmse"] < metrics["prior_data_nrmse"]
    assert metrics["posterior_logk_rmse"] < 0.35
    assert metrics["posterior_logk_rmse"] < 0.45 * metrics["prior_logk_rmse"]
    assert 6.0 <= metrics["contrast_post"] <= 16.0
    assert post.assimilate_rmse < 2.0
    assert metrics["forward_match_nrmse"] < 0.3
    assert np.allclose(post.k, case.twin.parameterization.expand(post.theta))
    assert post.history.reports[-1].mass.relative_balance_error < 0.08


def test_lm_recovers_known_channel() -> None:
    case = make_channel_waterflood(n_times=6, t_end=700.0, seed=4)
    post = case.twin.calibrate()
    metrics = evaluate_synthetic(case, post)
    assert metrics["posterior_data_nrmse"] < metrics["prior_data_nrmse"]
    assert metrics["posterior_logk_rmse"] < 0.40
    assert 6.0 <= metrics["contrast_post"] <= 16.0
    assert metrics["forward_match_nrmse"] < 0.40
    assert np.allclose(post.k, case.twin.parameterization.expand(post.theta))
    assert int(np.max(case.twin.parameterization.region_id)) == 1
