from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.inversion.electromagnetic import ElectromagneticInverter


def test_em_linear_scalar() -> None:
    inverter = ElectromagneticInverter()
    sw = inverter.invert(2.0, {"model": "linear", "a": 0.2, "b": 0.1})
    assert sw == pytest.approx(0.5)


def test_em_linear_array() -> None:
    inverter = ElectromagneticInverter()
    signal = np.array([1.0, 2.0, 3.0])
    sw = inverter.invert(signal, {"model": "linear", "a": 0.1, "b": 0.2})
    assert isinstance(sw, np.ndarray)
    assert sw.shape == signal.shape


def test_em_linear_field3d() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    signal = Field3D(grid, np.array([[[1.0, 2.0]]]), name="em")
    sw = ElectromagneticInverter().invert(signal, {"model": "linear", "a": 0.1, "b": 0.2})
    assert isinstance(sw, Field3D)
    assert sw.values.shape == grid.shape
    assert sw.confidence is not None


def test_em_polynomial_array() -> None:
    signal = np.array([0.0, 1.0, 2.0])
    sw = ElectromagneticInverter().invert(
        signal,
        {"model": "polynomial", "coefficients": [0.1, 0.1, 0.05]},
    )
    assert np.allclose(sw, 0.1 + 0.1 * signal + 0.05 * signal**2)


def test_em_calibrate_linear() -> None:
    inverter = ElectromagneticInverter()
    params = inverter.calibrate_linear([0.0, 1.0, 2.0], [0.1, 0.3, 0.5])
    assert params["model"] == "linear"
    assert params["a"] == pytest.approx(0.2)
    assert params["b"] == pytest.approx(0.1)


def test_em_calibrate_polynomial() -> None:
    inverter = ElectromagneticInverter()
    signal = np.array([0.0, 1.0, 2.0, 3.0])
    sw = 0.1 + 0.1 * signal + 0.02 * signal**2
    params = inverter.calibrate_polynomial(signal, sw, degree=2)
    predicted = inverter.invert(signal, params)
    assert np.allclose(predicted, sw)


def test_em_saturation_clipped() -> None:
    sw = ElectromagneticInverter().invert(
        np.array([-10.0, 10.0]),
        {"model": "linear", "a": 1.0, "b": 0.0, "swi": 0.2, "sor": 0.3},
    )
    assert np.allclose(sw, [0.2, 0.7])


def test_em_confidence_in_range() -> None:
    confidence = ElectromagneticInverter().compute_confidence(
        np.array([0.5, 1.0]),
        {"calibration_range": [0.0, 2.0]},
    )
    assert np.allclose(confidence, 1.0)


def test_em_confidence_out_of_range() -> None:
    confidence = ElectromagneticInverter().compute_confidence(
        np.array([1.0, 4.0]),
        {"calibration_range": [0.0, 2.0]},
    )
    assert confidence[1] < confidence[0]


def test_em_invalid_signal_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        ElectromagneticInverter().invert(np.array([1.0, np.nan]), {"model": "linear", "a": 1.0, "b": 0.0})
    with pytest.raises(InvalidPhysicalValueError):
        ElectromagneticInverter().invert(np.array([1.0, np.inf]), {"model": "linear", "a": 1.0, "b": 0.0})


def test_em_low_confidence_policy() -> None:
    sw, confidence = ElectromagneticInverter().invert_with_confidence(
        np.array([1.0, np.nan]),
        {"model": "linear", "a": 0.1, "b": 0.2},
        invalid_policy="low_confidence",
    )
    assert confidence[1] == pytest.approx(0.0)
    assert np.isnan(sw[1])


def test_em_complex_physics_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ElectromagneticInverter().invert_complex_physics()
    with pytest.raises(NotImplementedError):
        ElectromagneticInverter().predict_saturation(1.0, {"model": "maxwell"})
