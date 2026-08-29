"""Shale S5 forecast smoke (slow)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reservoir_backend.io.shale_case import forecast_shale_case, twin_from_shale_truth

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_shale_s5_forecast_smoke() -> None:
    truth = ROOT / "validation" / "shale_oil" / "cmg_s5_shutin" / "truth_s5.json"
    out = ROOT / "validation" / "shale_oil" / "cmg_s5_shutin" / "mxshale_s5.out"
    if not out.is_file():
        pytest.skip("missing IMEX .out")
    twin = twin_from_shale_truth(truth, out_path=out, n_times=3, max_iter=3)
    twin.inverse.post_ensemble_enabled = False
    post = twin.calibrate(max_iter=3, time_limit_s=300.0)
    traj, score = forecast_shale_case(twin, post)
    assert traj.times_s.size >= 1
    assert score == score
