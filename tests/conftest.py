"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from reservoir_backend.core.grid import Grid3D


@pytest.fixture
def small_grid() -> Grid3D:
    """Return a small 3D grid used by core tests."""
    return Grid3D(nx=4, ny=3, nz=2, dx=2.0, dy=3.0, dz=4.0)
