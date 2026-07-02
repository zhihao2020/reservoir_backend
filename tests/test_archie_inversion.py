from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.inversion.resistivity_archie import ArchieInverter


def test_archie_scalar() -> None:
    inverter = ArchieInverter(a=1.0, m=2.0, n=2.0)
    sw = inverter.invert(rt=25.0, rw=0.25, phi=0.2)
    expected = ((1.0 * 0.25) / ((0.2**2.0) * 25.0)) ** 0.5
    assert sw == pytest.approx(expected)


def test_archie_array() -> None:
    inverter = ArchieInverter(a=1.0, m=2.0, n=2.0)
    rt = np.array([[25.0, 100.0], [400.0, 6.25]])
    sw = inverter.invert(rt=rt, rw=0.25, phi=0.2)
    assert isinstance(sw, np.ndarray)
    assert sw.shape == rt.shape
    assert np.all((sw >= 0.0) & (sw <= 1.0))


def test_archie_field3d() -> None:
    grid = Grid3D(nx=2, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    rt = Field3D.from_constant(grid, 25.0, name="Rt", unit="ohm.m")
    phi = Field3D.from_constant(grid, 0.2, name="phi", unit="fraction")
    inverter = ArchieInverter(a=1.0, m=2.0, n=2.0)

    sw = inverter.invert(rt=rt, rw=0.25, phi=phi)

    assert isinstance(sw, Field3D)
    assert sw.grid == grid
    assert sw.values.shape == grid.shape
    assert sw.confidence is not None
    assert np.allclose(sw.confidence, 1.0)


def test_archie_forward_inverse_closure() -> None:
    inverter = ArchieInverter(a=0.8, m=1.9, n=2.1, swi=0.1, sor=0.15)
    sw_true = np.array([0.2, 0.35, 0.5, 0.75])
    phi = np.array([0.18, 0.22, 0.27, 0.31])
    rw = 0.12
    rt = inverter.forward_resistivity(sw_true, rw=rw, phi=phi)

    sw = inverter.invert(rt=rt, rw=rw, phi=phi)

    assert np.max(np.abs(sw - sw_true)) < 1e-10


def test_archie_clip_saturation() -> None:
    inverter = ArchieInverter(a=1.0, m=2.0, n=2.0, swi=0.2, sor=0.25)
    high_sw = inverter.invert(rt=1.0, rw=0.25, phi=0.2)
    low_sw = inverter.invert(rt=1.0e9, rw=0.25, phi=0.2)
    assert high_sw == pytest.approx(0.75)
    assert low_sw == pytest.approx(0.2)


def test_archie_invalid_rt() -> None:
    inverter = ArchieInverter()
    with pytest.raises(InvalidPhysicalValueError):
        inverter.invert(rt=0.0, rw=0.25, phi=0.2)
    with pytest.raises(InvalidPhysicalValueError):
        inverter.invert(rt=-1.0, rw=0.25, phi=0.2)


def test_archie_invalid_phi() -> None:
    inverter = ArchieInverter()
    with pytest.raises(InvalidPhysicalValueError):
        inverter.invert(rt=25.0, rw=0.25, phi=0.0)
    with pytest.raises(InvalidPhysicalValueError):
        inverter.invert(rt=25.0, rw=0.25, phi=-0.1)


def test_archie_noise_sensitivity() -> None:
    rng = np.random.default_rng(42)
    inverter = ArchieInverter(a=1.0, m=2.0, n=2.0, swi=0.05, sor=0.05)
    sw_true = np.linspace(0.2, 0.8, 20)
    phi = np.full_like(sw_true, 0.25)
    rt = inverter.forward_resistivity(sw_true, rw=0.2, phi=phi)
    noisy_rt = rt * (1.0 + rng.normal(0.0, 0.05, size=rt.shape))

    sw = inverter.invert(rt=noisy_rt, rw=0.2, phi=phi)

    assert np.mean(np.abs(sw - sw_true)) < 0.03


def test_archie_confidence() -> None:
    grid = Grid3D(nx=3, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    rt = Field3D(grid, np.array([[[25.0, -1.0, 1.0e9]]]))
    phi = Field3D.from_constant(grid, 0.2)
    inverter = ArchieInverter(swi=0.2, sor=0.2, invalid_policy="low_confidence")

    sw = inverter.invert(rt=rt, rw=0.25, phi=phi)

    assert isinstance(sw, Field3D)
    assert sw.confidence is not None
    assert sw.confidence[0, 0, 0] == pytest.approx(1.0)
    assert sw.confidence[0, 0, 1] == pytest.approx(0.0)
    assert sw.confidence[0, 0, 2] < sw.confidence[0, 0, 0]
    assert np.isnan(sw.values[0, 0, 1])
