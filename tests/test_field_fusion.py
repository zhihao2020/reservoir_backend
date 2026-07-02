from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.core.exceptions import GridMismatchError
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.fusion.confidence import combine_confidence, normalize_confidence
from reservoir_backend.fusion.field_fusion import (
    fuse_saturation_fields,
    update_simulated_with_observed,
    weighted_average_fields,
)


def test_weighted_average_two_fields() -> None:
    grid = _grid()
    a = Field3D.from_constant(grid, 1.0)
    b = Field3D.from_constant(grid, 3.0)
    fused, _ = weighted_average_fields([a, b], weights=[1.0, 3.0])
    assert np.allclose(fused.values, 2.5)


def test_weighted_average_three_fields() -> None:
    grid = _grid()
    fields = [Field3D.from_constant(grid, value) for value in (1.0, 2.0, 7.0)]
    fused, _ = weighted_average_fields(fields, weights=[1.0, 1.0, 2.0])
    assert np.allclose(fused.values, 4.25)


def test_weighted_average_equal_weights() -> None:
    grid = _grid()
    fields = [Field3D.from_constant(grid, value) for value in (1.0, 3.0)]
    fused, _ = weighted_average_fields(fields)
    assert np.allclose(fused.values, 2.0)


def test_confidence_weighting() -> None:
    grid = _grid()
    low = Field3D.from_constant(grid, 0.2, confidence=0.1)
    high = Field3D.from_constant(grid, 0.8, confidence=0.9)
    fused, _ = weighted_average_fields([low, high])
    assert np.all(fused.values > 0.5)


def test_nan_ignored_in_fusion() -> None:
    grid = _grid()
    a = Field3D(grid, np.array([[[np.nan, 1.0], [np.nan, 4.0]]]))
    b = Field3D.from_constant(grid, 2.0)
    fused, _ = weighted_average_fields([a, b])
    assert fused.values[0, 0, 0] == pytest.approx(2.0)
    assert not np.isnan(fused.values[0, 0, 0])


def test_all_nan_cell_reported() -> None:
    grid = _grid()
    a = Field3D(grid, np.full(grid.shape, np.nan))
    b = Field3D(grid, np.full(grid.shape, np.nan))
    fused, report = weighted_average_fields([a, b])
    assert np.isnan(fused.values).all()
    assert report["nan_cells_count"] == grid.total_cells


def test_zero_total_weight_raises_or_reports() -> None:
    grid = _grid()
    a = Field3D.from_constant(grid, 1.0, confidence=0.0)
    b = Field3D.from_constant(grid, 2.0, confidence=0.0)
    fused, report = weighted_average_fields([a, b])
    assert np.isnan(fused.values).all()
    assert report["zero_weight_cells"] == grid.total_cells


def test_fused_saturation_bounds() -> None:
    grid = _grid()
    low = Field3D.from_constant(grid, 0.0)
    high = Field3D.from_constant(grid, 1.0)
    fused, _ = fuse_saturation_fields([low, high], swi=0.2, sor=0.3)
    assert fused.values.min() >= 0.2
    assert fused.values.max() <= 0.7


def test_fusion_clipped_cells_report() -> None:
    grid = _grid()
    high = Field3D.from_constant(grid, 1.0)
    fused, report = fuse_saturation_fields([high], swi=0.2, sor=0.3)
    assert np.allclose(fused.values, 0.7)
    assert report["clipped_cells"] == grid.total_cells


def test_update_sim_with_obs_alpha_zero() -> None:
    grid = _grid()
    sim = Field3D.from_constant(grid, 0.3)
    obs = Field3D.from_constant(grid, 0.7)
    updated, _ = update_simulated_with_observed(sim, obs, alpha=0.0)
    assert np.allclose(updated.values, obs.values)


def test_update_sim_with_obs_alpha_one() -> None:
    grid = _grid()
    sim = Field3D.from_constant(grid, 0.3)
    obs = Field3D.from_constant(grid, 0.7)
    updated, _ = update_simulated_with_observed(sim, obs, alpha=1.0)
    assert np.allclose(updated.values, sim.values)


def test_update_sim_with_obs_alpha_half() -> None:
    grid = _grid()
    sim = Field3D.from_constant(grid, 0.3)
    obs = Field3D.from_constant(grid, 0.7)
    updated, _ = update_simulated_with_observed(sim, obs, alpha=0.5)
    assert np.allclose(updated.values, 0.5)


def test_invalid_alpha_raises() -> None:
    grid = _grid()
    sim = Field3D.from_constant(grid, 0.3)
    obs = Field3D.from_constant(grid, 0.7)
    with pytest.raises(ValueError):
        update_simulated_with_observed(sim, obs, alpha=-0.1)
    with pytest.raises(ValueError):
        update_simulated_with_observed(sim, obs, alpha=1.1)


def test_different_grid_raises() -> None:
    a = Field3D.from_constant(_grid(), 1.0)
    b = Field3D.from_constant(Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0), 2.0)
    with pytest.raises(GridMismatchError):
        weighted_average_fields([a, b])


def test_invalid_weight_raises() -> None:
    grid = _grid()
    fields = [Field3D.from_constant(grid, 1.0), Field3D.from_constant(grid, 2.0)]
    with pytest.raises(ValueError):
        weighted_average_fields(fields, weights=[1.0, -1.0])


def test_confidence_normalization() -> None:
    normalized = normalize_confidence(np.array([0.0, 5.0, 10.0]))
    assert np.allclose(normalized, [0.0, 0.5, 1.0])


def test_combine_confidence_shape() -> None:
    grid = _grid()
    c1 = Field3D.from_constant(grid, 0.2)
    c2 = Field3D.from_constant(grid, 0.8)
    combined = combine_confidence([c1, c2])
    assert combined.shape == grid.shape


def test_fusion_report_keys() -> None:
    grid = _grid()
    fused, report = weighted_average_fields([Field3D.from_constant(grid, 1.0)])
    assert fused.values.shape == grid.shape
    keys = {"field_count", "used_weights", "nan_cells_count", "clipped_cells", "fused_min", "fused_max"}
    assert keys.issubset(report)


def _grid() -> Grid3D:
    return Grid3D(nx=2, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
