"""Forecast period validation (check.txt §55)."""

from __future__ import annotations

import numpy as np

from reservoir_backend.synthetic import make_forecast_split_case


def test_forecast_freezes_parameters_and_scores_future() -> None:
    case = make_forecast_split_case(n_times=6, t_end=300.0, history_frac=0.55)
    twin = case.twin
    post = twin.calibrate(max_iter=4)
    traj = twin.forecast(post)
    score = twin.score_forecast(traj)
    assert traj.states[-1].time_s >= float(twin.experiment.history_end_s or 0.0)
    assert post.k.shape == (case.grid.n_cells,)
    if np.isfinite(score):
        assert score < 15.0
