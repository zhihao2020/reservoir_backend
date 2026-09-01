from reservoir_backend.twin.lab_v1 import D_CF_MIN, offline_gates


def test_detectability_gate_in_offline_gates() -> None:
    report = {
        "cf_true": 1.0e-12,
        "cf_p50": 1.02e-12,
        "tmf_true": 2.0,
        "tmf_p50": 2.05,
        "noise": False,
        "holdout_rmse_ratio": 0.5,
        "d_cf": 0.5,
    }
    g = offline_gates(report)
    assert g["cf_ok"] is True
    assert g["detect_ok"] is False
    assert g["pass"] is False
    report["d_cf"] = D_CF_MIN + 0.1
    assert offline_gates(report)["pass"] is True


def test_fail_rate_gate() -> None:
    report = {
        "cf_true": 1.0e-12,
        "cf_p50": 1.01e-12,
        "tmf_true": 2.0,
        "tmf_p50": 2.05,
        "noise": False,
        "holdout_rmse_ratio": 0.5,
        "fail_rate": 0.2,
        "repeated_fail": False,
    }
    assert offline_gates(report)["pass"] is False
    report["fail_rate"] = 0.01
    assert offline_gates(report)["pass"] is True
