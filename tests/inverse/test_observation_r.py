import numpy as np
import pytest

from reservoir_backend.inverse.observation_r import fisher_from_sensitivity, mahalanobis_d, observation_covariance


def test_diagonal_r_matches_sum_of_squares() -> None:
    names = ["P", "P", "S"]
    times = np.array([1.0, 2.0, 1.0])
    sig = np.array([2.0, 2.0, 0.5])
    kinds = ["pressure", "pressure", "gas_saturation"]
    r = observation_covariance(names, times, sig, kinds, rho_bias=0.0, tau_s=None)
    assert np.allclose(r, np.diag(sig * sig))
    dy = np.array([4.0, 0.0, 1.0])
    d = mahalanobis_d(dy, r)
    assert d == pytest.approx(np.sqrt((4.0 / 2.0) ** 2 + (1.0 / 0.5) ** 2), rel=1e-9)


def test_time_correlation_reduces_repeated_samples() -> None:
    names = ["P", "P"]
    times = np.array([1.0, 1.01])
    sig = np.array([2.0e3, 2.0e3])
    kinds = ["pressure", "pressure"]
    dy = np.array([2.0e3, 2.0e3])
    r_ind = observation_covariance(names, times, sig, kinds, rho_bias=0.0, tau_s=None)
    r_cor = observation_covariance(names, times, sig, kinds, rho_bias=0.0, tau_s=5.0)
    d_ind = mahalanobis_d(dy, r_ind)
    d_cor = mahalanobis_d(dy, r_cor)
    assert d_ind == pytest.approx(np.sqrt(2.0), rel=1e-9)
    assert d_cor < d_ind
    assert d_cor < 1.2


def test_fisher_is_whitened_gram() -> None:
    s = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=float)
    r = np.diag([1.0, 4.0, 1.0])
    fish = fisher_from_sensitivity(s, r)
    expected = s.T @ np.linalg.solve(r, s)
    assert np.allclose(fish, expected, rtol=1e-9, atol=1e-9)


def test_pressure_bias_correlates_distinct_gauges() -> None:
    names = ["P_in", "P_out"]
    times = np.array([1.0, 1.0])
    sig = np.array([2.0e3, 2.0e3])
    kinds = ["pressure", "pressure"]
    r = observation_covariance(names, times, sig, kinds, rho_bias=0.30, tau_s=None)
    assert r[0, 1] == pytest.approx(0.30 * 2.0e3 * 2.0e3, rel=1e-12)
    assert r[1, 0] == r[0, 1]


def test_two_absolute_gauges_do_not_improve_delta_p_sigma() -> None:
    from reservoir_backend.twin.experiment_design import two_gauge_delta_sigma

    sig = 2.0e3
    r = observation_covariance(
        ["P1", "P2"], np.array([0.0, 0.0]), np.array([sig, sig]), ["pressure", "pressure"], rho_bias=0.0, tau_s=None
    )
    contrast = np.array([1.0, -1.0])
    sigma_dp = float(np.sqrt(contrast @ r @ contrast))
    assert sigma_dp == pytest.approx(two_gauge_delta_sigma(sig), rel=1e-12)
    assert sigma_dp == pytest.approx(sig * np.sqrt(2.0), rel=1e-12)
    assert sigma_dp > sig


def test_legacy_m1b_rate_is_pv_infeasible() -> None:
    from reservoir_backend.twin.experiment_design import (
        Design,
        Instrument,
        LabEnvelope,
        Stage,
        injected_volume_m3,
        pore_volume_m3,
    )
    from reservoir_backend.twin.lab_v1 import load_lab_v1

    stages = [Stage(60.0, 3.0e-4)]
    twin = load_lab_v1(dev=True)
    n_pv = injected_volume_m3(stages) / pore_volume_m3(twin)
    env = LabEnvelope()
    assert n_pv > env.pv_max
    d = Design("legacy", stages, Instrument(h="tapped_channel"))
    assert injected_volume_m3(d.stages) > 0.0
