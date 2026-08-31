from reservoir_backend.twin.lab_v1 import offline_gates


def test_noiseless_cf_ok_but_holdout_worse_fails() -> None:
    report = {
        "cf_true": 1.0e-12,
        "cf_p50": 1.01e-12,
        "tmf_true": 2.0,
        "tmf_p50": 2.05,
        "noise": False,
        "holdout_rmse_ratio": 1.2,
    }
    g = offline_gates(report)
    assert g["cf_ok"] is True
    assert g["tmf_ok"] is True
    assert g["rmse_ok"] is False
    assert g["pass"] is False


def test_noisy_requires_holdout_ratio() -> None:
    report = {
        "cf_true": 1.0e-12,
        "cf_p50": 1.10e-12,
        "tmf_true": 2.0,
        "tmf_p50": 2.2,
        "noise": True,
        "holdout_rmse_ratio": 0.9,
    }
    g = offline_gates(report)
    assert g["cf_ok"] is True
    assert g["pass"] is False
