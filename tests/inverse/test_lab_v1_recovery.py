"""Plan boxed task: face BCs + P/S + holdout + ES-MDA recovers scalar C_f."""

import numpy as np
import pytest

from reservoir_backend.synthetic import make_lab_v1_face_twin
from reservoir_backend.twin.lab_v1 import NOISELESS_CF_TOL, offline_gates
from reservoir_backend.twin.offline import predict_from_trajectory, split_history_observations, stack_observations


def test_lab_v1_face_twin_has_face_ports_and_bulk_s() -> None:
    case = make_lab_v1_face_twin(ensemble_size=2, assimilation_steps=1, t_end=1.0, n_times=1)
    assert case.twin.ports[0].cell_ids.size == case.grid.ny * case.grid.nz
    assert case.twin.ports[1].cell_ids.size == case.grid.ny * case.grid.nz
    kinds = {s.kind for s in case.twin.experiment.sensors}
    assert "pressure" in kinds
    assert "gas_saturation" in kinds or "saturation" in kinds
    media = {s.medium for s in case.twin.experiment.sensors}
    assert "fracture" in media and "matrix" in media
    p_f = [s for s in case.twin.experiment.sensors if s.kind == "pressure" and s.medium == "fracture"]
    p_m = [s for s in case.twin.experiment.sensors if s.kind == "pressure" and s.medium == "matrix"]
    assert p_f and p_m
    assert p_f[0].sigma < p_m[0].sigma
    assert any(s.medium == "bulk" for s in case.twin.experiment.sensors)
    assert any(o.holdout for o in case.twin.experiment.observations)
    sat = [o for o in case.twin.experiment.observations if o.kind == "gas_saturation"]
    assert sat, "compositional EXAMPLE has no water; observe Sg not Sw"
    assert float(np.max(np.abs(sat[0].values))) > 1.0e-3


@pytest.mark.slow
@pytest.mark.assimilation
def test_lab_v1_face_esmda_recovers_cf_and_holdout() -> None:
    """Noiseless Case B: |Cf P50-true|/true < 5% and holdout RMSE drops."""
    case = make_lab_v1_face_twin(
        ensemble_size=8,
        assimilation_steps=5,
        seed=3,
        with_saturation=True,
        noise_p=0.0,
        noise_s=0.0,
    )
    twin = case.twin
    from reservoir_backend.twin.lab_v1 import physical_from_theta

    cf_true = float(physical_from_theta(twin, case.theta_true)["cf_m2"])
    cf_prior = float(physical_from_theta(twin, np.asarray(twin.parameterization.prior_mean, dtype=float))["cf_m2"])
    tmf_true = float(physical_from_theta(twin, case.theta_true)["tmf_multiplier"])
    post = twin.calibrate()
    members = post.ensemble.theta_members
    cf = np.array([physical_from_theta(twin, members[j])["cf_m2"] for j in range(members.shape[0])])
    tmf = np.array([physical_from_theta(twin, members[j])["tmf_multiplier"] for j in range(members.shape[0])])
    cf_p50 = float(np.quantile(cf, 0.50))
    rel = abs(cf_p50 - cf_true) / cf_true
    assim, hold = split_history_observations(twin.experiment.observations, twin.experiment.history_end_s)
    t_end = float(twin.experiment.history_end_s)
    times = stack_observations(assim).times
    hist_prior = twin.simulate(
        parameters=np.asarray(twin.parameterization.prior_mean, dtype=float).ravel(),
        t_end=t_end,
        report_times=times,
    )
    d_h = stack_observations(hold)
    prior_h = predict_from_trajectory(twin.operator, twin.experiment, hist_prior, hold)
    post_h = predict_from_trajectory(twin.operator, twin.experiment, post.history, hold)
    rmse_prior = float(np.sqrt(np.mean(((prior_h - d_h.values) / np.maximum(d_h.sigma, 1.0e-12)) ** 2)))
    rmse_post = float(np.sqrt(np.mean(((post_h - d_h.values) / np.maximum(d_h.sigma, 1.0e-12)) ** 2)))
    report = {
        "cf_true": cf_true,
        "cf_prior": cf_prior,
        "cf_p50": cf_p50,
        "tmf_true": tmf_true,
        "tmf_p50": float(np.quantile(tmf, 0.50)),
        "noise": False,
        "holdout_rmse_ratio": rmse_post / max(rmse_prior, 1.0e-12),
    }
    gates = offline_gates(report)
    assert rel < abs(cf_prior - cf_true) / cf_true
    assert gates["cf_ok"], f"Cf rel error {rel:.3%} exceeds {NOISELESS_CF_TOL:.0%}"
    assert gates["tmf_ok"], f"Tmf rel error {gates['tmf_rel_error']:.3%} exceeds 10%"
    assert rmse_post < rmse_prior
    assert gates["pass"]
    assert post.ensemble.dual_states is not None
    assert all(s is not None for s in post.ensemble.dual_states)
