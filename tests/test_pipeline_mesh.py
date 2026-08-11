from __future__ import annotations

import numpy as np

from reservoir_backend.pipeline import AxisAlignedBounds, WellPoint, build_mesh


def test_build_mesh_counts_and_well_map() -> None:
    bounds = AxisAlignedBounds(0.0, 100.0, 0.0, 50.0, 0.0, 20.0)
    wells = [WellPoint("A", 5.0, 5.0, 5.0), WellPoint("B", 95.0, 45.0, 15.0)]
    mesh = build_mesh(bounds, dx=10.0, dy=10.0, dz=10.0, wells=wells)
    assert mesh.grid.nx == 10
    assert mesh.grid.ny == 5
    assert mesh.grid.nz == 2
    assert mesh.n_cells == 100
    assert set(mesh.well_cell_id) == {"A", "B"}
    assert mesh.well_cell_id["A"] != mesh.well_cell_id["B"]
    assert np.all(np.isfinite(mesh.x))
    # centers should lie inside bounds
    assert mesh.x.min() >= bounds.xmin
    assert mesh.x.max() <= bounds.xmax
