import numpy as np

from reservoir_backend.synthetic import make_scalar_cf_twin


def test_esmda_scalar_cf_moves_toward_truth() -> None:
    """Plan Case A: noiseless H(F(C_f^true)); posterior C_f closer than the prior."""
    case = make_scalar_cf_twin(
        n=(5, 4, 2),
        t_end=60.0,
        n_times=3,
        noise_p=0.0,
        ensemble_size=8,
        assimilation_steps=4,
        seed=5,
    )
    prior_m = float(np.asarray(case.twin.parameterization.prior_mean).ravel()[0])
    true_m = float(case.theta_true[0])
    post = case.twin.calibrate()
    post_m = float(post.theta[0])
    assert abs(post_m - true_m) < abs(prior_m - true_m)
    assert post.ensemble is not None
    assert post.ensemble.theta_members.shape[0] == 8
    q05, q50, q95 = np.quantile(post.ensemble.theta_members[:, 0], [0.05, 0.50, 0.95])
    assert q05 <= q50 <= q95
    assert post.misfit[-1] <= post.misfit[0] * 1.05
    assert post.history.reports[-1].mass.relative_balance_error < 0.08
    cf_post = float(case.twin.parameterization.decode(post.theta)[0])
    assert cf_post > 0.0


def test_yaml_log_conductivity_selects_esmda(tmp_path) -> None:
    from pathlib import Path

    import yaml

    from reservoir_backend.grid.cartesian import CartesianGrid
    from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
    from reservoir_backend.io.case import inverse_spec_from_cfg
    from reservoir_backend.io.parameterization_cfg import parameterization_from_cfg

    grid = CartesianGrid.uniform((0.2, 0.12, 0.08), (0.1, 0.12, 0.08))
    param = parameterization_from_cfg(
        grid, {"inverse": {"parameterization": "log_conductivity"}, "rock": {"porosity": 0.2}}, tmp_path
    )
    assert isinstance(param, LogConductivityParameterization)
    assert param.n_params == 1
    spec = inverse_spec_from_cfg({"parameterization": "log_conductivity", "ensemble_size": 12})
    assert spec.algorithm == "esmda"
    assert spec.ensemble_size == 12
    _ = yaml
    _ = Path
