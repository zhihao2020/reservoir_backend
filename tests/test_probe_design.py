"""Tests for uniform and adaptive probe design."""

from __future__ import annotations

import numpy as np

from reservoir_backend.pipeline import (
    AxisAlignedBounds,
    WellPoint,
    build_mesh,
    place_uniform_probes,
    recommend_probes,
    split_n_probes,
)
from reservoir_backend.pipeline.probe_design import field_variance_over_time


def _mesh():
    bounds = AxisAlignedBounds(0.0, 100.0, 0.0, 80.0, 0.0, 30.0)
    wells = [
        WellPoint("INJ", 10.0, 40.0, 15.0, role="injector"),
        WellPoint("PROD", 90.0, 40.0, 15.0, role="producer"),
    ]
    return build_mesh(bounds, dx=10.0, dy=10.0, dz=10.0, wells=wells)


def test_split_n_probes() -> None:
    assert split_n_probes(0) == (0, 0)
    assert split_n_probes(4) == (2, 2)
    assert split_n_probes(5) == (3, 2)


def test_uniform_counts_and_roles() -> None:
    mesh = _mesh()
    specs = place_uniform_probes(mesh, n_p=3, n_s=2)
    assert len(specs) == 5
    roles = [s.role for s in specs]
    assert roles.count("observer_p") == 3
    assert roles.count("observer_s") == 2
    # no overlap with well cells
    well_cells = set(mesh.well_cell_id.values())
    assert all(s.cell_id not in well_cells for s in specs)
    # exclusive cells
    assert len({s.cell_id for s in specs}) == 5


def test_maximin_increases_spread() -> None:
    mesh = _mesh()
    s1 = recommend_probes(mesh, n_p=1, n_s=0, mode="maximin")
    s4 = recommend_probes(mesh, n_p=2, n_s=2, mode="maximin")
    assert len(s1) == 1
    assert len(s4) == 4
    # pairwise min distance among 4 should be > 0
    pts = np.array([[s.x, s.y, s.z] for s in s4])
    dmin = 1e30
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dmin = min(dmin, float(np.linalg.norm(pts[i] - pts[j])))
    assert dmin > 1.0


def test_hybrid_runs_with_ensemble() -> None:
    mesh = _mesh()
    specs = recommend_probes(
        mesh, n_p=2, n_s=2, mode="hybrid", seed=1, k_mean=1.0e-13
    )
    assert len(specs) == 4
    assert {s.role for s in specs} == {"observer_p", "observer_s"}
    assert all(s.score is not None for s in specs)


def test_variance_mode_prefers_high_var_cells() -> None:
    mesh = _mesh()
    # well cells + face neighbors are excluded by design
    blocked = set(mesh.well_cell_id.values())
    for cid in list(mesh.well_cell_id.values()):
        i0, j0, k0 = int(mesh.i[cid]), int(mesh.j[cid]), int(mesh.k[cid])
        for di, dj, dk in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            i, j, k = i0 + di, j0 + dj, k0 + dk
            if 0 <= i < mesh.grid.nx and 0 <= j < mesh.grid.ny and 0 <= k < mesh.grid.nz:
                blocked.add(int(mesh.grid.index(i, j, k)))
    target = next(c for c in range(mesh.n_cells) if c not in blocked)
    var = np.zeros(mesh.grid.shape, dtype=float)
    var.flat[target] = 10.0
    specs = recommend_probes(
        mesh,
        n_p=1,
        n_s=0,
        mode="variance",
        prior_var_p=var,
        prior_var_s=np.ones(mesh.grid.shape),
    )
    assert len(specs) == 1
    assert specs[0].cell_id == target


def test_field_variance_over_time() -> None:
    a = np.ones((2, 2, 2))
    b = np.ones((2, 2, 2)) * 3.0
    v = field_variance_over_time([(0.0, a), (1.0, b)])
    assert np.allclose(v, 1.0)  # var of {1,3} = 1.0
