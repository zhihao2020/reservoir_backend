"""CMG/Eclipse *GRID file loader: Cartesian and orthogonal CPG."""

from pathlib import Path

import numpy as np
import pytest

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


def _yaml_cart_cfg():
    return {
        "grid": {
            "nx": 4,
            "ny": 3,
            "nz": 2,
            "dx": 0.05,
            "dy": 0.04,
            "dz": 0.04,
        }
    }


def test_cmg_cart_con_matches_yaml_and_uniform(tmp_path: Path) -> None:
    decl = tmp_path / "box.grdecl"
    decl.write_text(
        "\n".join(
            [
                "*GRID *CART 4 3 2",
                "*DI *CON",
                "0.05",
                "*DJ *CON",
                "0.04",
                "*DK *CON",
                "0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    from_file = grid_from_cfg({"grid": {"file": "box.grdecl"}}, cfg_dir=tmp_path)
    from_yaml = grid_from_cfg(_yaml_cart_cfg())
    ref = _uniform()
    assert isinstance(from_file, CartesianGrid)
    assert (from_file.nx, from_file.ny, from_file.nz) == (ref.nx, ref.ny, ref.nz)
    assert np.allclose(from_file.dx, from_yaml.dx)
    assert np.allclose(from_file.dy, from_yaml.dy)
    assert np.allclose(from_file.dz, from_yaml.dz)
    assert np.allclose(from_file.cell_volumes(), ref.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)
    assert np.allclose(from_file.cell_centers(), ref.cell_centers(), rtol=VOL_RTOL, atol=VOL_ATOL)
    k = np.full(ref.n_cells, 1.2e-12)
    kz = np.full(ref.n_cells, 3.4e-13)
    tx_f, ty_f, tz_f = geometric_transmissibility(from_file, k, kz=kz)
    tx_r, ty_r, tz_r = geometric_transmissibility(ref, k, kz=kz)
    assert np.allclose(tx_f, tx_r, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(ty_f, ty_r, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(tz_f, tz_r, rtol=T_RTOL, atol=T_ATOL)


def test_eclipse_dx_repeat_and_nostar_keywords(tmp_path: Path) -> None:
    decl = tmp_path / "ecl.grdecl"
    decl.write_text(
        "\n".join(
            [
                "CARTESIAN",
                "NX 4",
                "NY 3",
                "NZ 2",
                "DX",
                "4*0.05 /",
                "DY",
                "3*0.04 /",
                "DZ",
                "2*0.04 /",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    grid = grid_from_cfg({"grid": {"file": "ecl.grdecl"}}, cfg_dir=tmp_path)
    ref = _uniform()
    assert np.allclose(grid.cell_volumes(), ref.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)


def test_kvar_dz_matches_yaml_list(tmp_path: Path) -> None:
    decl = tmp_path / "layers.dat"
    decl.write_text(
        "\n".join(
            [
                "*GRID *CART 3 2 3",
                "*DI *CON 0.1",
                "*DJ *CON 0.2",
                "*DK *KVAR",
                "0.05 0.10 0.15",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    from_file = grid_from_cfg({"grid": {"file": "layers.dat"}}, cfg_dir=tmp_path)
    from_yaml = grid_from_cfg(
        {"grid": {"nx": 3, "ny": 2, "nz": 3, "dx": 0.1, "dy": 0.2, "dz": [0.05, 0.10, 0.15]}}
    )
    assert np.allclose(from_file.dz, from_yaml.dz)
    assert np.allclose(from_file.cell_volumes(), from_yaml.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)


def test_orthogonal_cpg_grdecl_matches_yaml_and_cartesian(tmp_path: Path) -> None:
    cart = _uniform()
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    decl = tmp_path / "cpg.grdecl"
    decl.write_text(
        "CORNER-POINT\nNX 4\nNY 3\nNZ 2\nCOORD\n"
        + " ".join(str(float(v)) for v in coord)
        + "\n/\nZCORN\n"
        + " ".join(str(float(v)) for v in zcorn)
        + "\n/\n",
        encoding="utf-8",
    )
    from_file = grid_from_cfg({"grid": {"file": "cpg.grdecl"}}, cfg_dir=tmp_path)
    from_yaml = grid_from_cfg(
        {
            "grid": {
                "type": "corner_point",
                "nx": 4,
                "ny": 3,
                "nz": 2,
                "coord": coord.tolist(),
                "zcorn": zcorn.tolist(),
            }
        }
    )
    assert isinstance(from_file, CornerPointGrid)
    assert isinstance(from_yaml, CornerPointGrid)
    assert np.allclose(from_file.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)
    assert np.allclose(from_file.cell_volumes(), from_yaml.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)
    assert np.allclose(from_file.cell_centers(), cart.cell_centers(), rtol=VOL_RTOL, atol=VOL_ATOL)
    k = np.full(cart.n_cells, 1.2e-12)
    kz = np.full(cart.n_cells, 3.4e-13)
    tx_f, ty_f, tz_f = geometric_transmissibility(from_file, k, kz=kz)
    tx_c, ty_c, tz_c = geometric_transmissibility(cart, k, kz=kz)
    assert np.allclose(tx_f, tx_c, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(ty_f, ty_c, rtol=T_RTOL, atol=T_ATOL)
    assert np.allclose(tz_f, tz_c, rtol=T_RTOL, atol=T_ATOL)


def test_star_coord_file_without_type_is_cpg(tmp_path: Path) -> None:
    cart = CartesianGrid(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    decl = tmp_path / "deck.inc"
    decl.write_text(
        "*COORD\n"
        + " ".join(str(float(v)) for v in coord)
        + "\n/\n*ZCORN\n"
        + " ".join(str(float(v)) for v in zcorn)
        + "\n/\n",
        encoding="utf-8",
    )
    grid = grid_from_cfg(
        {"grid": {"file": "deck.inc", "nx": 2, "ny": 1, "nz": 1}},
        cfg_dir=tmp_path,
    )
    assert isinstance(grid, CornerPointGrid)
    assert np.allclose(grid.cell_volumes(), cart.cell_volumes(), rtol=VOL_RTOL, atol=VOL_ATOL)


def test_coord_without_zcorn_raises(tmp_path: Path) -> None:
    bad = tmp_path / "deck.inc"
    bad.write_text("*COORD\n0 0 0 0 0 1\n", encoding="utf-8")
    with pytest.raises(GridError, match="zcorn"):
        grid_from_cfg(
            {"grid": {"file": "deck.inc", "nx": 1, "ny": 1, "nz": 1}},
            cfg_dir=tmp_path,
        )


def test_cartesian_actnum_zeros_volume_and_t(tmp_path: Path) -> None:
    decl = tmp_path / "act.grdecl"
    decl.write_text(
        "\n".join(
            [
                "*GRID *CART 2 1 1",
                "*DI *CON 1.0",
                "*DJ *CON 1.0",
                "*DK *CON 1.0",
                "ACTNUM",
                "1 0 /",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    grid = grid_from_cfg({"grid": {"file": "act.grdecl"}}, cfg_dir=tmp_path)
    assert isinstance(grid, CartesianGrid)
    assert grid.active is not None
    assert bool(grid.active[0])
    assert not bool(grid.active[1])
    vol = grid.cell_volumes()
    assert vol[0] == pytest.approx(1.0)
    assert vol[1] == 0.0
    tx, _ty, _tz = geometric_transmissibility(grid, np.ones(2))
    assert tx.shape == (1, 1, 1)
    assert tx[0, 0, 0] == 0.0


def test_cpg_actnum_marks_inactive_like_zero_volume(tmp_path: Path) -> None:
    cart = CartesianGrid(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=1.0)
    coord, zcorn = coord_zcorn_from_cartesian(cart)
    decl = tmp_path / "cpg_act.grdecl"
    decl.write_text(
        "NX 2\nNY 1\nNZ 1\nCOORD\n"
        + " ".join(str(float(v)) for v in coord)
        + "\n/\nZCORN\n"
        + " ".join(str(float(v)) for v in zcorn)
        + "\n/\nACTNUM\n1 0 /\n",
        encoding="utf-8",
    )
    grid = grid_from_cfg({"grid": {"file": "cpg_act.grdecl"}}, cfg_dir=tmp_path)
    assert isinstance(grid, CornerPointGrid)
    assert grid.active[0]
    assert not grid.active[1]
    assert grid.cell_volumes()[1] == 0.0
    tx, _ty, _tz = geometric_transmissibility(grid, np.ones(2))
    assert tx[0, 0, 0] == 0.0
