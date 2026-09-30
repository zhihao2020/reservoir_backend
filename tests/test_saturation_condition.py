"""Probe residual correction on top of a transport saturation prior."""

import numpy as np

from src.core.cartesian import CartesianGrid
from src.programs.interpolate import ordinary_kriging, residual_variogram
from src.programs.mesh import PointMap
from src.programs.pipeline import _condition_saturations
from src.programs.pressure import interpolate_pressure_wells


def _bottom_case(grid: CartesianGrid):
    # 3x3 probes on the bottom three layers, matching the shale-oil layout.
    xyz = []
    for z in (0.01, 0.03, 0.05):
        for x in (0.05, 0.15, 0.25):
            for y in (0.05, 0.15, 0.25):
                xyz.append((x, y, z))
    xyz = np.asarray(xyz, dtype=float)
    cells = []
    for x, y, z in xyz:
        i = int(np.clip(np.floor(x / 0.02), 0, grid.nx - 1))
        j = int(np.clip(np.floor(y / 0.02), 0, grid.ny - 1))
        k = int(np.clip(np.floor(z / 0.02), 0, grid.nz - 1))
        cells.append(k * grid.ny * grid.nx + j * grid.nx + i)
    cells = np.asarray(cells, dtype=np.int64)
    n = cells.size
    obs = 0.44
    mesh_probes = PointMap(
        ids=tuple(f"p{i}" for i in range(n)),
        xyz=xyz,
        i=np.zeros(n, dtype=np.int64),
        j=np.zeros(n, dtype=np.int64),
        k=np.zeros(n, dtype=np.int64),
        cell=cells,
    )

    class _Case:
        sw = np.zeros((1, n))
        so = np.full((1, n), 1.0 - obs)
        sg = np.full((1, n), obs)

    mesh = type("Mesh", (), {})()
    mesh.probes = mesh_probes
    mesh.grid = grid
    return _Case(), mesh, cells


def test_bottom_residual_raises_bottom_and_leaves_the_top():
    grid = CartesianGrid(nx=15, ny=15, nz=15, dx=0.02, dy=0.02, dz=0.02)
    case, mesh, cells = _bottom_case(grid)
    n_c = grid.n_cells
    prior = 0.054
    sw = np.zeros((1, n_c))
    so = np.full((1, n_c), 1.0 - prior)
    sg = np.full((1, n_c), prior)
    sw, so, sg = _condition_saturations(case, mesh, sw, so, sg)
    layers = sg[0].reshape(grid.nz, grid.ny, grid.nx)
    bottom = float(layers[:3].mean())
    top = float(layers[-3:].mean())
    assert float(np.mean(np.abs(sg[0, cells] - 0.44))) < 0.05
    assert bottom > 0.15
    assert top < bottom
    assert int((sg[0] > 0.1).sum()) < 2000


def test_unobserved_phase_is_not_corrected():
    grid = CartesianGrid(nx=8, ny=8, nz=8, dx=0.02, dy=0.02, dz=0.02)
    case, mesh, _cells = _bottom_case(grid)
    case.sw = np.full_like(case.sg, np.nan)
    n_c = grid.n_cells
    sw = np.full((1, n_c), 0.2)
    so = np.full((1, n_c), 0.5)
    sg = np.full((1, n_c), 0.3)
    sw2, _so2, _sg2 = _condition_saturations(case, mesh, sw.copy(), so.copy(), sg.copy())
    # Water was unobserved, so its change comes only from the simplex projection.
    assert np.nanmax(np.abs(sw2 - sw)) < 0.5


def test_vertical_range_stays_short_for_bottom_probes():
    grid = CartesianGrid(nx=15, ny=15, nz=15, dx=0.02, dy=0.02, dz=0.02)
    _case, mesh, _cells = _bottom_case(grid)
    model = residual_variogram(mesh.probes.xyz, np.full(mesh.probes.xyz.shape[0], 0.4))
    assert model.range_v < 0.08
    assert model.range_h > model.range_v


def test_simple_kriging_residual_decays_to_zero():
    points = np.array([[0.1, 0.1, 0.01], [0.2, 0.1, 0.01], [0.1, 0.2, 0.03]])
    values = np.full(3, 0.4)
    targets = np.array([[0.15, 0.15, 0.01], [0.15, 0.15, 0.28]])
    model = residual_variogram(points, values)
    out, _var = ordinary_kriging(points, values, targets, model=model, known_mean=0.0)
    assert out[0] > 0.2
    assert abs(out[1]) < 0.05


def test_pressure_skips_probes_without_a_reading():
    grid = CartesianGrid(nx=5, ny=5, nz=5, dx=0.02, dy=0.02, dz=0.02)
    xyz = np.array([[0.05, 0.05, 0.01], [0.15, 0.05, 0.01]])
    pressure = np.array([np.nan, 19.0e6])
    cells = (np.array([0]),)
    field = interpolate_pressure_wells(grid, xyz, pressure, cells, np.array([19.2e6]))
    assert np.isfinite(field).all()
    assert float(field.mean()) > 18.0e6
