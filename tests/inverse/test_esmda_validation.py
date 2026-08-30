"""Scalar C_f ES-MDA checks: noise, dropout, outlier, coverage. Linear and QC, not 30³."""

import numpy as np
import pytest

pytestmark = pytest.mark.assimilation

from reservoir_backend.inverse.esmda import esmda_update
from reservoir_backend.observation.qc import ObservationStatus, classify_observations


def _linear_truth(n_ens: int = 32, seed: int = 1):
    rng = np.random.default_rng(seed)
    m_true = 0.4
    h = np.array([[1.0], [0.5]])
    x = m_true + 0.8 * rng.standard_normal((1, n_ens))
    y = h @ x
    d = h.ravel() * m_true
    return x, y, d, m_true, rng


def test_noiseless_linear_coverage() -> None:
    x, y, d, m_true, rng = _linear_truth()
    xa = esmda_update(x, y, d, np.array([0.05, 0.05]), alpha=1.0, rng=rng)
    q05, q95 = np.quantile(xa[0], [0.05, 0.95])
    assert q05 <= m_true <= q95
    assert abs(float(np.mean(xa)) - m_true) < abs(float(np.mean(x)) - m_true)


def test_pressure_noise_still_moves_toward_truth() -> None:
    from reservoir_backend.inverse.esmda import inflation_schedule

    rng = np.random.default_rng(2)
    m_true = 0.4
    h = np.array([[1.0], [0.5]])
    x = np.full((1, 40), -1.0) + 0.4 * rng.standard_normal((1, 40))
    d = h.ravel() * m_true + rng.normal(0.0, 0.04, size=2)
    xa = x.copy()
    for a in inflation_schedule(4):
        y = h @ xa
        xa = esmda_update(xa, y, d, np.array([0.08, 0.08]), alpha=float(a), rng=rng)
    assert abs(float(np.mean(xa)) - m_true) < abs(float(np.mean(x)) - m_true)


def test_sensor_dropout_qc_keeps_active_rows() -> None:
    y = np.array([[1.0, 1.1, 0.9], [np.nan, np.nan, np.nan]])
    d = np.array([1.0, 2.0])
    sig = np.array([0.1, 0.1])
    st = classify_observations(y, d, sig)
    assert st[0] == ObservationStatus.ACTIVE.value
    assert st[1] == ObservationStatus.MISSING_RESPONSE.value
    active = st == ObservationStatus.ACTIVE.value
    rng = np.random.default_rng(0)
    x = np.array([[0.0, 0.2, -0.1]])
    xa = esmda_update(x, y[active], d[active], sig[active], alpha=1.0, rng=rng)
    assert xa.shape == x.shape


def test_outlier_qc_drops_row() -> None:
    y = np.array([[1.0, 1.05, 0.95], [0.0, 0.1, -0.1]])
    d = np.array([1.0, 50.0])
    sig = np.array([0.1, 0.1])
    st = classify_observations(y, d, sig)
    assert st[1] == ObservationStatus.OUTLIER.value
    assert st[0] == ObservationStatus.ACTIVE.value
