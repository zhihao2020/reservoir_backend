import numpy as np

from reservoir_backend.observation.qc import ObservationStatus, classify_observations


def test_classify_active_and_missing() -> None:
    y = np.array([[1.0, 1.1, 0.9], [np.nan, np.nan, np.nan]])
    d = np.array([1.0, 2.0])
    sig = np.array([0.2, 0.2])
    st = classify_observations(y, d, sig)
    assert st[0] == ObservationStatus.ACTIVE.value
    assert st[1] == ObservationStatus.MISSING_RESPONSE.value


def test_classify_low_spread_and_outlier() -> None:
    y_low = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    d = np.array([1.0, 50.0])
    sig = np.array([0.2, 0.2])
    st = classify_observations(y_low, d, sig)
    assert st[0] == ObservationStatus.LOW_ENSEMBLE_SPREAD.value
    assert st[1] == ObservationStatus.OUTLIER.value
