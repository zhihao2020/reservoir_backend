"""Unit tests for auto IDW / ordinary kriging / stack spatial interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.spatial_interp import (
    N_MIN_KRIGING,
    auto_interpolate_to_grid,
    idw_points_to_grid,
    leave_one_out_rmse,
    ordinary_kriging_to_grid,
    points_geometry_ok,
)
from reservoir_backend.pipeline.state import AxisAlignedBounds


def _small_mesh():
    bounds = AxisAlignedBounds(0.0, 100.0, 0.0, 100.0, 0.0, 20.0)
    return build_mesh(bounds, dx=20.0, dy=20.0, dz=20.0, wells=[])


def _scattered_points(n: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack(
        [
            rng.uniform(5.0, 95.0, n),
            rng.uniform(5.0, 95.0, n),
            rng.uniform(2.0, 18.0, n),
        ]
    )
    # smooth-ish field with mild noise → kriging can compete with IDW
    vals = (
        1.0e-13
        * np.exp(-((pts[:, 0] - 50.0) ** 2 + (pts[:, 1] - 50.0) ** 2) / 40.0**2)
        * (1.0 + 0.05 * rng.standard_normal(n))
    )
    vals = np.clip(vals, 1.0e-15, None)
    return pts, vals


def test_idw_exact_at_sample() -> None:
    mesh = _small_mesh()
    # place samples exactly on cell centers so IDW exact-hit path triggers
    i0, i1 = 0, mesh.n_cells - 1
    pts = np.array(
        [
            [mesh.x[i0], mesh.y[i0], mesh.z[i0]],
            [mesh.x[i1], mesh.y[i1], mesh.z[i1]],
        ],
        dtype=float,
    )
    vals = np.array([2.0, 8.0], dtype=float)
    field = idw_points_to_grid(mesh, pts, vals)
    assert abs(field.flat[i0] - 2.0) < 1e-12
    assert abs(field.flat[i1] - 8.0) < 1e-12


def test_few_points_force_idw() -> None:
    mesh = _small_mesh()
    pts, vals = _scattered_points(n=4)
    res = auto_interpolate_to_grid(mesh, pts, vals, log_transform=True)
    assert res.method == "idw"
    assert res.n_points == 4
    assert any("n=4" in n for n in res.notes)
    assert np.all(np.isfinite(res.values))
    assert np.all(res.values > 0.0)


def test_geometry_ok_rejects_collinear() -> None:
    pts = np.column_stack(
        [np.linspace(0, 100, 10), np.zeros(10), np.zeros(10)]
    )
    assert not points_geometry_ok(pts)
    pts2, _ = _scattered_points(n=N_MIN_KRIGING)
    assert points_geometry_ok(pts2)


def test_auto_with_enough_points_runs_loo() -> None:
    mesh = _small_mesh()
    pts, vals = _scattered_points(n=12)
    res = auto_interpolate_to_grid(
        mesh, pts, vals, log_transform=True, clip=(1e-18, 1e-10)
    )
    assert res.method in ("idw", "kriging", "stack")
    assert res.loo_rmse_idw is not None
    assert res.n_points == 12
    assert np.all(np.isfinite(res.values))
    assert np.all(res.values > 0.0)
    assert np.all(res.values <= 1e-10 + 1e-30)
    assert any("auto-interp" in n for n in res.notes)


def test_ordinary_kriging_finite() -> None:
    mesh = _small_mesh()
    pts, vals = _scattered_points(n=10)
    logv = np.log(vals)
    field = ordinary_kriging_to_grid(mesh, pts, logv)
    assert field.shape == mesh.grid.shape
    assert np.all(np.isfinite(field))


def test_loo_rmse_positive() -> None:
    pts, vals = _scattered_points(n=9)
    rmse_i = leave_one_out_rmse(pts, np.log(vals), method="idw")
    rmse_k = leave_one_out_rmse(pts, np.log(vals), method="kriging")
    assert rmse_i >= 0.0
    assert np.isfinite(rmse_k)


def test_empty_points_fill() -> None:
    mesh = _small_mesh()
    res = auto_interpolate_to_grid(
        mesh,
        np.zeros((0, 3)),
        np.zeros(0),
        fill=0.2,
    )
    assert res.method == "idw"
    assert np.allclose(res.values, 0.2)


def test_empty_points_without_fill_raises() -> None:
    mesh = _small_mesh()
    with pytest.raises(ValueError, match="no points"):
        auto_interpolate_to_grid(mesh, np.zeros((0, 3)), np.zeros(0))
