from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.inversion.acoustic import AcousticInverter


def test_acoustic_linear_scalar() -> None:
    sw = AcousticInverter().invert(3000.0, {"model": "linear", "a": 1.0e-4, "b": 0.1})
    assert sw == pytest.approx(0.4)


def test_acoustic_linear_array() -> None:
    velocity = np.array([2000.0, 3000.0])
    sw = AcousticInverter().invert(velocity, {"model": "linear", "a": 1.0e-4, "b": 0.1})
    assert isinstance(sw, np.ndarray)
    assert sw.shape == velocity.shape


def test_acoustic_linear_field3d() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    velocity = Field3D(grid, np.array([[[2000.0, 3000.0]]]), name="vp")
    sw = AcousticInverter().invert(velocity, {"model": "linear", "a": 1.0e-4, "b": 0.1})
    assert isinstance(sw, Field3D)
    assert sw.values.shape == grid.shape
    assert sw.confidence is not None


def test_acoustic_polynomial_array() -> None:
    velocity = np.array([1.0, 2.0, 3.0])
    sw = AcousticInverter().invert(
        velocity,
        {"model": "polynomial", "coefficients": [0.1, 0.1, 0.05]},
    )
    assert np.allclose(sw, 0.1 + 0.1 * velocity + 0.05 * velocity**2)


def test_acoustic_calibrate_linear() -> None:
    params = AcousticInverter().calibrate_linear([1000.0, 2000.0, 3000.0], [0.2, 0.3, 0.4])
    assert params["model"] == "linear"
    assert params["a"] == pytest.approx(1.0e-4)


def test_acoustic_calibrate_polynomial() -> None:
    inverter = AcousticInverter()
    velocity = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    sw = 0.1 + 1.0e-4 * velocity + 1.0e-8 * velocity**2
    params = inverter.calibrate_polynomial(velocity, sw, degree=2)
    assert np.allclose(inverter.invert(velocity, params), sw)


def test_acoustic_saturation_clipped() -> None:
    sw = AcousticInverter().invert(
        np.array([1.0, 10000.0]),
        {"model": "linear", "a": 1.0, "b": 0.0, "swi": 0.2, "sor": 0.3},
    )
    assert np.allclose(sw, [0.7, 0.7])


def test_acoustic_confidence_in_range() -> None:
    confidence = AcousticInverter().compute_confidence(np.array([2000.0, 3000.0]), {"calibration_range": [1000.0, 4000.0]})
    assert np.allclose(confidence, 1.0)


def test_acoustic_confidence_out_of_range() -> None:
    confidence = AcousticInverter().compute_confidence(np.array([2000.0, 8000.0]), {"calibration_range": [1000.0, 4000.0]})
    assert confidence[1] < confidence[0]


def test_acoustic_invalid_velocity_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        AcousticInverter().invert(0.0, {"model": "linear", "a": 1.0e-4, "b": 0.1})
    with pytest.raises(InvalidPhysicalValueError):
        AcousticInverter().invert(-1.0, {"model": "linear", "a": 1.0e-4, "b": 0.1})


def test_acoustic_low_confidence_policy() -> None:
    sw, confidence = AcousticInverter().invert_with_confidence(
        np.array([2000.0, -1.0]),
        {"model": "linear", "a": 1.0e-4, "b": 0.1},
        invalid_policy="low_confidence",
    )
    assert confidence[1] == pytest.approx(0.0)
    assert np.isnan(sw[1])


def test_acoustic_gassmann_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        AcousticInverter().invert_gassmann()
    with pytest.raises(NotImplementedError):
        AcousticInverter().predict_saturation(2000.0, {"model": "gassmann"})
