from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.solver.relperm import fractional_flow_water, validate_viscosity


def test_fractional_flow_range() -> None:
    sw = np.linspace(0.0, 1.0, 101)
    fw = fractional_flow_water(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    assert np.all(fw >= 0.0)
    assert np.all(fw <= 1.0)


def test_fractional_flow_low_sw() -> None:
    fw = fractional_flow_water(0.2, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    assert fw == pytest.approx(0.0)


def test_fractional_flow_high_sw() -> None:
    fw = fractional_flow_water(0.8, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    assert fw == pytest.approx(1.0)


def test_fractional_flow_monotonic() -> None:
    sw = np.linspace(0.2, 0.8, 51)
    fw = fractional_flow_water(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    assert np.all(np.diff(fw) >= -1.0e-15)


def test_fractional_flow_viscosity_effect() -> None:
    sw = 0.5
    base = fractional_flow_water(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    viscous_water = fractional_flow_water(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 5.0e-3, 5.0e-3)
    assert viscous_water < base


def test_fractional_flow_array_shape() -> None:
    sw = np.array([[0.2, 0.5, 0.8]])
    fw = fractional_flow_water(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, 5.0e-3)
    assert fw.shape == sw.shape


def test_invalid_viscosity_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_viscosity(0.0, 1.0e-3)
    with pytest.raises(InvalidPhysicalValueError):
        fractional_flow_water(0.5, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0, 1.0e-3, -1.0)
