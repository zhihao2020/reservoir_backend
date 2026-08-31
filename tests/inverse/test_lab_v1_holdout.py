from reservoir_backend.io.case import load_case
from reservoir_backend.twin.lab_v1 import generate_truth, spatial_holdout, zone_of_x


def test_case_abc_share_truth_and_spatial_holdout() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    held = spatial_holdout(list(twin.experiment.sensors), seed=3)
    zones = {zone_of_x(s.x) for s in twin.experiment.sensors}
    assert zones == {"inlet", "middle", "outlet"}
    # Case A keeps only pressure channels; B keeps P+S. No forward here.
    n_p = sum(1 for s in twin.experiment.sensors if s.kind == "pressure")
    n_s = sum(1 for s in twin.experiment.sensors if s.kind == "saturation")
    assert n_p >= 3 and n_s >= 3
    assert held
    assert not held >= {s.name for s in twin.experiment.sensors}


def test_offline_gates_noiseless_threshold() -> None:
    from reservoir_backend.twin.lab_v1 import NOISELESS_CF_TOL, NOISY_CF_TOL, offline_gates

    ok = offline_gates({"cf_true": 1.0e-12, "cf_p50": 1.02e-12, "noise": False, "holdout_rmse_ratio": 1.0})
    assert ok["pass"] is True
    assert NOISELESS_CF_TOL == 0.05
    bad = offline_gates({"cf_true": 1.0e-12, "cf_p50": 1.2e-12, "noise": False, "holdout_rmse_ratio": 0.5})
    assert bad["pass"] is False
    noisy = offline_gates(
        {"cf_true": 1.0e-12, "cf_p50": 1.10e-12, "noise": True, "holdout_rmse_ratio": 0.5}
    )
    assert noisy["cf_ok"] is True
    assert NOISY_CF_TOL == 0.15
