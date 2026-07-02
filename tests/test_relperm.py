from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.relperm import (
    corey_relative_permeability,
    effective_saturation,
    oil_mobility,
    validate_saturation_params,
    water_mobility,
)


def test_effective_saturation_at_swi() -> None:
    assert effective_saturation(0.2, swi=0.2, sor=0.2) == pytest.approx(0.0)


def test_effective_saturation_at_one_minus_sor() -> None:
    assert effective_saturation(0.8, swi=0.2, sor=0.2) == pytest.approx(1.0)


def test_effective_saturation_clipped_low() -> None:
    assert effective_saturation(0.1, swi=0.2, sor=0.2) == pytest.approx(0.0)


def test_effective_saturation_clipped_high() -> None:
    assert effective_saturation(0.9, swi=0.2, sor=0.2) == pytest.approx(1.0)


def test_corey_krw_at_swi() -> None:
    krw, _ = corey_relative_permeability(0.2, 0.2, 0.2, 0.7, 0.9, 2.0, 2.0)
    assert krw == pytest.approx(0.0)


def test_corey_kro_at_swi() -> None:
    _, kro = corey_relative_permeability(0.2, 0.2, 0.2, 0.7, 0.9, 2.0, 2.0)
    assert kro == pytest.approx(0.9)


def test_corey_krw_at_one_minus_sor() -> None:
    krw, _ = corey_relative_permeability(0.8, 0.2, 0.2, 0.7, 0.9, 2.0, 2.0)
    assert krw == pytest.approx(0.7)


def test_corey_kro_at_one_minus_sor() -> None:
    _, kro = corey_relative_permeability(0.8, 0.2, 0.2, 0.7, 0.9, 2.0, 2.0)
    assert kro == pytest.approx(0.0)


def test_relperm_array_input() -> None:
    sw = np.array([0.2, 0.5, 0.8])
    krw, kro = corey_relative_permeability(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0)
    assert isinstance(krw, np.ndarray)
    assert isinstance(kro, np.ndarray)
    assert krw.shape == sw.shape
    assert kro.shape == sw.shape


def test_relperm_field3d_input_uses_values() -> None:
    grid = Grid3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    sw = Field3D(grid, np.array([[[0.2, 0.8]]]), name="sw", unit="fraction")
    krw, kro = corey_relative_permeability(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0)
    assert np.allclose(krw, [[[0.0, 1.0]]])
    assert np.allclose(kro, [[[1.0, 0.0]]])


def test_relperm_monotonic_krw() -> None:
    sw = np.linspace(0.0, 1.0, 21)
    krw, _ = corey_relative_permeability(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0)
    assert np.all(np.diff(krw) >= 0.0)


def test_relperm_monotonic_kro() -> None:
    sw = np.linspace(0.0, 1.0, 21)
    _, kro = corey_relative_permeability(sw, 0.2, 0.2, 1.0, 1.0, 2.0, 2.0)
    assert np.all(np.diff(kro) <= 0.0)


def test_water_and_oil_mobility() -> None:
    assert water_mobility(0.5, 1.0e-3) == pytest.approx(500.0)
    assert oil_mobility(0.25, 5.0e-3) == pytest.approx(50.0)


def test_invalid_residual_saturation_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        validate_saturation_params(0.6, 0.4)
    with pytest.raises(InvalidPhysicalValueError):
        effective_saturation(0.5, swi=-0.1, sor=0.2)


def test_invalid_corey_exponent_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        corey_relative_permeability(0.5, 0.2, 0.2, 1.0, 1.0, 0.0, 2.0)
    with pytest.raises(InvalidPhysicalValueError):
        corey_relative_permeability(0.5, 0.2, 0.2, 1.0, 1.0, 2.0, -1.0)


def test_invalid_endpoint_relperm_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        corey_relative_permeability(0.5, 0.2, 0.2, -1.0, 1.0, 2.0, 2.0)
    with pytest.raises(InvalidPhysicalValueError):
        corey_relative_permeability(0.5, 0.2, 0.2, 1.0, -1.0, 2.0, 2.0)
