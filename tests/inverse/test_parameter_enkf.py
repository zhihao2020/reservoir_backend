import numpy as np

from reservoir_backend.inverse.esmda import inflation_schedule
from reservoir_backend.inverse.parameter_enkf import parameter_enkf_update


def test_parameter_enkf_is_single_es_step() -> None:
    rng = np.random.default_rng(4)
    h = np.array([[1.0], [0.7]])
    m_true = -27.0
    d = np.array([m_true, 0.7 * m_true])
    x = np.full((1, 20), -29.0) + 0.4 * rng.standard_normal((1, 20))
    y = h @ x
    xa = parameter_enkf_update(x, y, d, np.full(2, 0.2), rng, q_std=0.01)
    assert abs(float(np.mean(xa)) - m_true) < abs(float(np.mean(x)) - m_true)
    assert inflation_schedule(1)[0] == 1.0
