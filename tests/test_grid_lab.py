from reservoir_backend.grid.cartesian import CartesianGrid


def test_lab_30cm_10mm_is_30_cubed() -> None:
    grid = CartesianGrid.uniform((0.30, 0.30, 0.30), 0.01)
    assert grid.nx == 30
    assert grid.ny == 30
    assert grid.nz == 30
    assert grid.n_cells == 27_000
    vol = grid.total_volume()
    assert abs(vol - 0.3**3) / (0.3**3) < 1.0e-12


def test_fields_are_flat() -> None:
    grid = CartesianGrid.uniform((0.12, 0.08, 0.04), 0.02)
    centers = grid.cell_centers()
    assert centers.shape == (grid.n_cells, 3)
    assert grid.cell_volumes().shape == (grid.n_cells,)
    assert len(grid.neighbors(0)) == 3
