"""Corner-point COORD/ZCORN: volumes + two-point T; Cartesian recovery."""

import numpy as np
import pytest
import yaml

from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.exceptions import GridError
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.grid.corner_point import (
    CornerPointGrid,
    coord_zcorn_from_cartesian,
)
from reservoir_backend.io.grid_cfg import grid_from_cfg

VOL_RTOL = 1.0e-12
VOL_ATOL = 1.0e-16
T_RTOL = 1.0e-10
T_ATOL = 1.0e-14


def _uniform():
    return CartesianGrid.uniform((0.20, 0.12, 0.08), (0.05, 0.04, 0.04), origin=(0.0, 0.0, 0.0))


def test_orthogonal_cpg_recovers_cartesian_volumes_and_t():
    cart = _uniform()
    cpg = CornerPointGrid.from_cartesian(cart)
    assert (cpg.nx, cpg.ny, cpg.nz) == (cart.nx, cart.ny, cart.nz)
    assert cpg.n_cells == cart.n_cells
    assert np.all(cpg.active)
    assert np.allclose(cpg.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)
    assert np.allclose(cpg.cell_centers(), cart.cell_centers(), rtol=VOL_RTOL, atol=VOL_ATOL)
    k = np.full(cart.n_cells, 1.2e-12)
    kz = np.full(cart.n_cells, 3.4e-13)
    tx_c, ty_c, tz_c = geometric_transmissibility(cart, k, kz=kz)
    tx_p, ty_p, tz_p = geometric_transmissibility(cpg, k, kz=kz)
    assert np.allclose(tx_p, tx_c, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(ty_p, ty_c, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(tz_p, tz_c, rtol=T_RTOL, atol=T_ATOL)


def test_variable_dz_orthogonal_recovers_volumes():
    cart = CartesianGrid(nx=3, ny=2, nz=3, dx=0.1, dy=0.2, dz=[0.05, 0.10, 0.15])
    cpg = CornerPointGrid.from_cartesian(cart)
    assert np.allclose(cpg.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)
    k = np.linspace(1.0e-13, 5.0e-13, cart.n_cells)
    tx_c, ty_c, tz_c = geometric_transmissibility(cart, k)
    tx_p, ty_p, tz_p = geometric_transmissibility(cpg, k)
    assert np.allclose(tx_p, tx_c, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(ty_p, ty_c, rtol=T_RTOL, atol=T_ATOL)
    assert tz_p.shape == tz_c.shape
    assert np.all(np.isfinite(tz_p))
    assert np.all(tz_p > 0.0)


def test_yaml_type_corner_point_and_cpg_alias():
    cart = CartesianGrid(nx=2, ny=2, nz=1, dx=0.1, dy=0.1, dz=0.2)
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    for gtype in ("corner_point", "cpg"):
        grid = grid_from_cfg(
            {
                "grid": {
                    "type": gtype,
                    "nx": 2,
                    "ny": 2,
                    "nz": 1,
                    "coord": coord.tolist(),
                    "zcorn": zcorn.tolist(),
                }
            }
        )
        assert isinstance(grid, CornerPointGrid)
        assert np.allclose(grid.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)


def test_sidecar_yaml_and_grdecl_snippet(tmp_path):
    cart = CartesianGrid(nx=2, ny=1, nz=2, dx=0.3, dy=0.4, dz=[0.1, 0.2])
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    yml = tmp_path / "pillars.yaml"
    yml.write_text(
        yaml.safe_dump({"coord": coord.tolist(), "zcorn": zcorn.tolist()}),
        encoding="utf-8",
    )
    from_yaml = grid_from_cfg(
        {"grid": {"type": "corner_point", "nx": 2, "ny": 1, "nz": 2, "file": "pillars.yaml"}},
        cfg_dir=tmp_path,
    )
    assert np.allclose(from_yaml.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)

    decl = tmp_path / "box.grdecl"
    decl.write_text(
        "COORD\n"
        + " ".join(str(float(v)) for v in coord)
        + "\n/\nZCORN\n"
        + " ".join(str(float(v)) for v in zcorn)
        + "\n/\n",
        encoding="utf-8",
    )
    from_grdecl = grid_from_cfg(
        {"grid": {"type": "cpg", "nx": 2, "ny": 1, "nz": 2, "file": "box.grdecl"}},
        cfg_dir=tmp_path,
    )
    assert np.allclose(from_grdecl.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)


def test_degenerate_cell_is_inactive_not_crash():
    cart = CartesianGrid(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    zc = zcorn.reshape(2, 2, 4)
    zc[:, :, 2:4] = 0.0
    grid = CornerPointGrid.from_coord_zcorn(2, 1, 1, coord, zc.ravel())
    assert grid.active[0]
    assert not grid.active[1]
    assert grid.cell_volumes()[1] == 0.0
    k = np.ones(2)
    tx, ty, tz = geometric_transmissibility(grid, k)
    assert np.isfinite(tx).all()
    assert tx.shape == (1, 1, 1)
    assert tx[0, 0, 0] == 0.0


def test_cartesian_sidecar_star_coord_needs_zcorn(tmp_path):
    bad = tmp_path / "deck.inc"
    bad.write_text("*COORD\n0 0 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(GridError, match="zcorn"):
        grid_from_cfg({"grid": {"file": "deck.inc", "nx": 1, "ny": 1, "nz": 1}}, cfg_dir=tmp_path)


def test_missing_coord_raises():
    with pytest.raises(GridError, match="coord"):
        grid_from_cfg({"grid": {"type": "corner_point", "nx": 1, "ny": 1, "nz": 1}})
