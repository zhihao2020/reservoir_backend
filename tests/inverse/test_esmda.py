import numpy as np
import pytest

pytestmark = pytest.mark.assimilation

from reservoir_backend.inverse.ensemble import replace_failed_members, sample_log_prior
from reservoir_backend.inverse.esmda import esmda_update, inflation_schedule
from reservoir_backend.exceptions import AssimilationError


def test_inflation_sums_to_one() -> None:
    a = inflation_schedule(4)
    assert a.size == 4
    assert a == pytest.approx(np.full(4, 4.0))
    assert float(np.sum(1.0 / a)) == pytest.approx(1.0)
    custom = inflation_schedule(3, np.array([2.0, 4.0, 4.0]))
    assert float(np.sum(1.0 / custom)) == pytest.approx(1.0)


def test_esmda_linear_recovers_scalar() -> None:
    """y = H m with H ones. Posterior mean of log-parameter moves toward truth."""
    rng = np.random.default_rng(1)
    m_true = np.array([[-28.0]])
    h = np.array([[1.0], [0.5], [1.5], [0.8]])
    d = (h @ m_true).ravel()
    sigma = np.full(4, 0.15)
    n_ens = 24
    prior = np.full((1, n_ens), -30.0) + 0.8 * rng.standard_normal((1, n_ens))
    x = prior.copy()
    alphas = inflation_schedule(4)
    y = h @ x
    for a in alphas:
        y = h @ x
        x = esmda_update(x, y, d, sigma, float(a), rng)
    post = float(np.mean(x))
    prior_err = abs(-30.0 - float(m_true[0, 0]))
    post_err = abs(post - float(m_true[0, 0]))
    assert post_err < prior_err
    assert post_err < 0.6


def test_esmda_does_not_call_linalg_inv() -> None:
    src = open("reservoir_backend/inverse/esmda.py", encoding="utf-8").read()
    assert "np.linalg.inv(" not in src
    assert "numpy.linalg.inv(" not in src


def test_failed_member_replacement() -> None:
    rng = np.random.default_rng(2)
    x = sample_log_prior(-28.0, 0.5, 1, 6, rng)
    failed = np.array([False, True, False, False, True, False])
    out = replace_failed_members(x, failed, rng, 0.5)
    assert np.all(np.isfinite(out))
    assert not np.allclose(out[:, 1], x[:, 1])


def test_all_failed_raises() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros((1, 3))
    with pytest.raises(AssimilationError, match="all ensemble members failed"):
        replace_failed_members(x, np.array([True, True, True]), rng, 0.5)
