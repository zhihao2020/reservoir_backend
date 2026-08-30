import numpy as np

from reservoir_backend.observation.signal import LinearSaturationMap, observations_from_saturation


def test_linear_map_carries_sigma() -> None:
    model = LinearSaturationMap(a=0.5, b=0.1, sigma=0.03)
    raw = np.array([0.2, 0.8])
    samples = model.invert(raw, x=0.1, y=0.1, z=0.05, times_s=np.array([1.0, 2.0]), name="S1")
    assert len(samples) == 2
    assert samples[0].sigma == 0.03
    assert 0.0 <= samples[0].value <= 1.0
    series = observations_from_saturation(samples)
    assert series.kind in {"saturation", "sw"}
    assert series.sensor_name == "S1"
    assert float(series.sigma[0]) == 0.03
