import numpy as np

from reservoir_backend.inverse.lm import prior_theta, run_lm
from reservoir_backend.inverse.parameterization import ContrastParameterization, RegionParameterization


class _LinearMap:
    def __init__(self, h: np.ndarray) -> None:
        self.h = np.asarray(h, dtype=float)
        self.n_params = self.h.shape[1]

    def expand(self, theta):
        return np.asarray(theta, dtype=float).ravel()


def test_linear_gaussian_lm_moves_toward_truth() -> None:
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
    d_obs = h @ m_true + rng.normal(0.0, 0.05, size=4)
    param = _LinearMap(h)

    def fwd(theta):
        return h @ np.asarray(theta, dtype=float).ravel()

    result = run_lm(
        param,
        fwd,
        d_obs,
        np.full(4, 0.2),
        prior_mean=0.0,
        prior_std=1.0,
        max_iter=12,
        fd_rel=0.05,
    )
    prior_err = float(np.linalg.norm(np.zeros(3) - m_true))
    post_err = float(np.linalg.norm(result.theta - m_true))
    assert post_err < prior_err
    assert result.misfit[-1] < result.misfit[0] * 1.05
    assert result.n_forward >= 1
    assert np.allclose(result.k, param.expand(result.theta))


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
    th0 = prior_theta(p, np.log(1e-12))
    assert th0.shape == (2,)
    assert th0[1] >= 0.0
    kk = p.expand(th0)
    assert float(np.min(kk[2:])) >= float(np.max(kk[:2])) * (1.0 - 1.0e-12)


def test_theta_is_rock_only() -> None:
    ids = np.array([0, 0, 1, 1], dtype=np.int64)
    p = RegionParameterization(ids)
    assert p.n_params == 2
    theta = np.array([np.log(1e-13), np.log(1e-12)])
    k = p.expand(theta)
    assert k.shape == (4,)
    assert np.allclose(k[:2], 1e-13)
