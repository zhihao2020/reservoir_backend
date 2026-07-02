from __future__ import annotations

import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError, WellControlError
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import ControlType, Well, WellType


def test_injection_well_by_cell_index() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    cell = grid.index(1, 1, 0)
    well = Well("I1", WellType.INJECTION, grid, cell_index=cell, rate=100.0)
    assert well.location == (1, 1, 0)
    assert well.signed_rate == pytest.approx(100.0)


def test_production_well_by_ijk() -> None:
    grid = Grid3D(nx=3, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0)
    well = Well("P1", "production", grid, i=2, j=1, k=1, rate=75.0)
    assert well.cell_index == grid.index(2, 1, 1)
    assert well.signed_rate == pytest.approx(-75.0)


def test_rate_control_requires_rate() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(WellControlError):
        Well("I1", "injection", grid, i=0, j=0, k=0)


def test_rate_must_be_positive() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(InvalidPhysicalValueError):
        Well("I1", "injection", grid, i=0, j=0, k=0, rate=0.0)


def test_bhp_control_reserved() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(NotImplementedError):
        Well(
            "P1",
            "production",
            grid,
            i=0,
            j=0,
            k=0,
            control=ControlType.BHP,
            bhp=1.0e7,
        )


def test_well_requires_single_location_mode() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(WellControlError):
        Well("I1", "injection", grid, cell_index=0, i=0, j=0, k=0, rate=10.0)
    with pytest.raises(WellControlError):
        Well("I2", "injection", grid, rate=10.0)


def test_partial_ijk_location_raises() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(WellControlError):
        Well("I1", "injection", grid, i=0, j=0, rate=10.0)


def test_invalid_well_type_raises() -> None:
    grid = Grid3D(nx=3, ny=3, nz=1, dx=1.0, dy=1.0, dz=1.0)
    with pytest.raises(ValueError):
        Well("X1", "observer", grid, i=0, j=0, k=0, rate=10.0)
