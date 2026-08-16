import numpy as np

from reservoir_backend.inverse.esmda import run_esmda
from reservoir_backend.inverse.parameterization import ContrastParameterization, RegionParameterization


class _LinearMap:
    def __init__(self, h: np.ndarray) -> None:
        self.h = np.asarray(h, dtype=float)
        self.n_params = self.h.shape[1]
        self.region_id = np.arange(self.n_params, dtype=np.int64)

    def expand(self, theta):
        return np.asarray(theta, dtype=float).ravel()

    def sample_prior(self, n_ensemble, mean, std, seed):
        rng = np.random.default_rng(seed)
        mu = np.broadcast_to(np.asarray(mean, dtype=float), (self.n_params,))
        sig = np.broadcast_to(np.asarray(std, dtype=float), (self.n_params,))
        return rng.normal(mu[None, :], sig[None, :], size=(int(n_ensemble), self.n_params))


def test_linear_gaussian_esmda_moves_toward_truth() -> None:
    rng = np.random.default_rng(0)
    h = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.3, 1.0, 0.0],
            [0.0, 0.4, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    m_true = np.array([0.8, -0.4, 0.3])
    r_diag = np.full(4, 0.04)
    d_obs = h @ m_true + rng.normal(0.0, 0.2, size=4)
    param = _LinearMap(h)

    def fwd(theta):
        return h @ np.asarray(theta, dtype=float).ravel()

    result = run_esmda(
        param,
        fwd,
        d_obs,
        r_diag,
        n_ensemble=80,
        n_assimilations=4,
        prior_mean=0.0,
        prior_std=1.0,
        seed=4,
        inflation=1.0,
    )
    prior_err = float(np.linalg.norm(np.zeros(3) - m_true))
    post_err = float(np.linalg.norm(result.theta_mean - m_true))
    assert post_err < prior_err
    assert result.diagnostics.data_mismatch[-1] < result.diagnostics.data_mismatch[0] * 1.05
    assert result.theta_ensemble.shape == (80, 3)


def test_failed_member_is_pulled_not_cloned() -> None:
    h = np.eye(2)
    obs = np.array([0.4, -0.2])
    calls: list[float] = []

    class _P:
        n_params = 2

        def expand(self, theta):
            return np.asarray(theta, dtype=float).ravel()

        def sample_prior(self, n_ensemble, mean, std, seed):
            rng = np.random.default_rng(seed)
            return rng.normal(0.0, 1.4, size=(n_ensemble, 2))

    def fwd(theta):
        th = np.asarray(theta, dtype=float).ravel()
        calls.append(float(np.max(np.abs(th))))
        if np.max(np.abs(th)) > 2.2:
            raise RuntimeError("stiff member")
        return h @ th

    result = run_esmda(
        _P(),
        fwd,
        obs,
        np.array([0.04, 0.04]),
        n_ensemble=16,
        n_assimilations=3,
        prior_mean=0.0,
        prior_std=1.4,
        seed=3,
        inflation=1.0,
    )
    assert result.theta_ensemble.shape[1] == 2
    assert any("pulled" in n or "resampled" in n or "dropped" in n for n in result.diagnostics.notes)
    assert result.diagnostics.data_mismatch[-1] <= result.diagnostics.data_mismatch[0] * 1.2


def test_es_and_esmda_rs_run() -> None:
    h = np.eye(2)
    obs = np.array([0.3, -0.2])

    class _P:
        n_params = 2

        def expand(self, theta):
            return np.asarray(theta, dtype=float).ravel()

        def sample_prior(self, n_ensemble, mean, std, seed):
            rng = np.random.default_rng(seed)
            return rng.normal(0.0, 1.0, size=(n_ensemble, 2))

    def fwd(theta):
        return h @ np.asarray(theta, dtype=float).ravel()

    es = run_esmda(_P(), fwd, obs, np.array([0.04, 0.04]), n_ensemble=20, algorithm="es", seed=1, inflation=1.0)
    rs = run_esmda(
        _P(), fwd, obs, np.array([0.04, 0.04]), n_ensemble=20, n_assimilations=5, algorithm="esmda_rs", seed=2, inflation=1.0
    )
    assert len(es.diagnostics.alpha_schedule) == 1
    assert abs(es.diagnostics.alpha_schedule[0] - 1.0) < 1.0e-12
    assert 1 <= len(rs.diagnostics.alpha_schedule) <= 5
    assert rs.diagnostics.data_mismatch[-1] <= rs.diagnostics.data_mismatch[0] * 1.05


def test_geo_and_ies_schedules() -> None:
    from reservoir_backend.inverse.algorithms import geometric_alphas, plan_alphas

    geo = geometric_alphas(4)
    assert geo.size == 4
    assert abs(float(np.sum(1.0 / geo)) - 1.0) < 1.0e-12
    assert geo[0] > geo[-1]
    ies = plan_alphas("ies", 3)
    assert ies is not None and ies.size == 3
    assert np.allclose(ies, 2.0)


def test_region_expand_count() -> None:
    ids = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    p = RegionParameterization(ids)
    k = p.expand(np.log(np.array([1e-13, 1e-12])))
    assert k.shape == (5,)
    assert np.allclose(k[:2], 1e-13)
    assert np.allclose(k[2:], 1e-12)


def test_contrast_keeps_body_high() -> None:
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    p = ContrastParameterization(ids)
    k = p.expand(np.array([np.log(1e-13), -2.0]))
    assert float(np.min(k[2:])) >= float(np.max(k[:2])) * (1.0 - 1.0e-12)
    ens = p.sample_prior(16, np.log(1e-12), 0.5, seed=4)
    assert ens.shape == (16, 2)
    assert np.all(ens[:, 1] >= 0.0)
    for row in ens:
        kk = p.expand(row)
        assert float(np.min(kk[2:])) >= float(np.max(kk[:2])) * (1.0 - 1.0e-12)


def test_theta_is_rock_only() -> None:
    """PVT is known fluid, not a component of θ."""
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    p = RegionParameterization(ids)
    assert p.n_params == 2
    theta = np.array([np.log(1e-13), np.log(1e-12)])
    k = p.expand(theta)
    assert k.shape == (4,)
    assert np.allclose(k[:2], 1e-13)
    ens = p.sample_prior(8, np.log(1e-12), 0.5, seed=3)
    assert ens.shape == (8, 2)
