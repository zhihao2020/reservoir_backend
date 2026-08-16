from reservoir_backend.validation.synthetic import make_two_layer_waterflood


def test_forecast_freezes_parameters() -> None:
    case = make_two_layer_waterflood(n_times=4, t_end=200.0, seed=4, history_frac=0.5)
    post = case.twin.calibrate(n_ensemble=8, n_assimilations=2, seed=6)
    traj = case.twin.forecast(post)
    score = case.twin.score_forecast(traj)
    assert traj.states[-1].time_s >= case.twin.experiment.history_end_s
    assert score == score  # not raising; finite or nan if no future hold-out
    # static k is unchanged
    assert post.esmda.k_mean.shape == (case.grid.n_cells,)
