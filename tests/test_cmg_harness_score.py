import numpy as np

from reservoir_backend.validation.cmg_harness.catalog import DAY_S
from reservoir_backend.validation.cmg_harness.probes import classify_probe
from reservoir_backend.validation.cmg_harness.score import (
    Score,
    breakthrough_rel,
    breakthrough_time_days,
    combine_j,
    field_gap,
    rmse,
)


def test_rmse_and_j() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert abs(rmse(a, a) ) < 1e-12
    assert rmse(a, a + 1.0) == 1.0
    s = Score(hold=0.6, forecast=0.5, p_rmse_psi=20.0, sw_rmse=0.08, bt_rel=0.1)
    j = combine_j(s)
    assert 1.0 < j < 3.0


def test_breakthrough_time() -> None:
    t = np.array([0.0, 0.25, 0.5, 1.0]) * DAY_S
    sw = np.array([0.20, 0.22, 0.40, 0.55])
    bt = breakthrough_time_days(t, sw, threshold=0.35)
    assert abs(bt - 0.5) < 1e-12
    assert breakthrough_time_days(t, np.full(4, 0.2)) == float("inf")
    assert abs(breakthrough_rel(0.5, 0.5)) < 1e-12
    assert breakthrough_rel(0.5, float("inf")) == 1.0


def test_field_gap_last_day() -> None:
    cmg = {0.25: {"p": np.zeros((2, 2, 2)), "sw": np.full((2, 2, 2), 0.2)}}
    cmg[1.0] = {"p": np.full((2, 2, 2), 10.0), "sw": np.full((2, 2, 2), 0.5)}
    f = {0.25: {"p": np.zeros((2, 2, 2)), "sw": np.full((2, 2, 2), 0.2)}}
    f[1.0] = {"p": np.full((2, 2, 2), 13.0), "sw": np.full((2, 2, 2), 0.6)}
    p_rmse, p_demean, s_rmse, sc, sf = field_gap(f, cmg)
    assert abs(p_rmse - 3.0) < 1e-12
    assert abs(p_demean) < 1e-12
    assert abs(s_rmse - 0.1) < 1e-12
    assert abs(sc - 0.5) < 1e-12 and abs(sf - 0.6) < 1e-12


def test_classify_probe_prunes_stuck_sw() -> None:
    stuck = classify_probe(ran=True, sw0=0.20, sw_max=0.201, p_std_psi=40.0)
    assert stuck.ok is False and stuck.reason == "prune:no_flood"
    dead = classify_probe(ran=False, sw0=0.2, sw_max=0.2, p_std_psi=0.0)
    assert dead.reason == "prune:underflow"
    flatp = classify_probe(ran=True, sw0=0.2, sw_max=0.5, p_std_psi=2.0)
    assert flatp.reason == "prune:no_dp"
    ok = classify_probe(ran=True, sw0=0.2, sw_max=0.45, p_std_psi=50.0, bt_rel=0.2)
    assert ok.ok and ok.reason == "ok"
