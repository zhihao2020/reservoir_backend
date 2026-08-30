import numpy as np

from reservoir_backend.inverse.esmda import inflation_schedule
from reservoir_backend.inverse.parameter_enkf import analysis_parameters, forecast_parameters


def test_parameter_enkf_forecast_then_analysis() -> None:
    rng = np.random.default_rng(4)
    h = np.array([[1.0], [0.7]])
    m_true = -27.0
    d = np.array([m_true, 0.7 * m_true])
    x = np.full((1, 20), -29.0) + 0.4 * rng.standard_normal((1, 20))
    xf = forecast_parameters(x, 0.01, rng)
    y = h @ xf
    xa = analysis_parameters(xf, y, d, np.full(2, 0.2), rng)
    assert abs(float(np.mean(xa)) - m_true) < abs(float(np.mean(xf)) - m_true)
    assert inflation_schedule(1)[0] == 1.0

