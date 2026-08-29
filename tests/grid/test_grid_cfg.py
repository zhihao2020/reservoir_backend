"""Variable Cartesian DX/DY/DZ YAML input; uniform spacing_m still works."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.exceptions import GridError
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.io.grid_cfg import grid_from_cfg


def test_uniform_spacing_fallback_matches_lab_30cm() -> None:
    grid = grid_from_cfg(
        {"geometry": {"size_m": [0.3, 0.3, 0.3]}, "grid": {"type": "cartesian", "spacing_m": 0.01}}
    )
    ref = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.01)
    assert grid.nx == 30
    assert grid.ny == 30
    assert grid.nz == 30
    assert grid.n_cells == ref.n_cells
    assert np.allclose(grid.dx, ref.dx)
    assert np.allclose(grid.dy, ref.dy)
    assert np.allclose(grid.dz, ref.dz)
    assert grid.size_m() == pytest.approx((0.3, 0.3, 0.3))


def test_lab_yaml_defaults_unchanged() -> None:
    lab = Path("examples/lab/lab_30cm.yaml")
    channel = Path("examples/lab/lab_channel.yaml")
    g30 = grid_from_cfg(yaml.safe_load(lab.read_text(encoding="utf-8")), cfg_dir=lab.parent)
    gch = grid_from_cfg(yaml.safe_load(channel.read_text(encoding="utf-8")), cfg_dir=channel.parent)
    assert (g30.nx, g30.ny, g30.nz) == (30, 30, 30)
    assert g30.size_m() == pytest.approx((0.3, 0.3, 0.3))
    assert (gch.nx, gch.ny, gch.nz) == (12, 12, 12)
    assert gch.size_m() == pytest.approx((0.3, 0.3, 0.3))


def test_variable_dz_layers() -> None:
    grid = grid_from_cfg(
        {
            "geometry": {"size_m": [0.2, 0.2, 0.3]},
            "grid": {
                "type": "cartesian",
                "nx": 4,
                "ny": 4,
                "dx": 0.05,
                "dy": 0.05,
                "dz": [0.05, 0.10, 0.15],
            },
        }
    )
    assert grid.nx == 4
    assert grid.ny == 4
    assert grid.nz == 3
    assert np.allclose(grid.dz, [0.05, 0.10, 0.15])
    assert grid.size_m() == pytest.approx((0.2, 0.2, 0.3))
    vol = grid.reshape_ijk(grid.cell_volumes())
    assert vol[0, 0, 0] == pytest.approx(0.05 * 0.05 * 0.05)
    assert vol[2, 0, 0] == pytest.approx(0.05 * 0.05 * 0.15)


def test_infer_counts_from_array_lengths() -> None:
    grid = grid_from_cfg(
        {
            "grid": {
                "dx": [0.1, 0.2, 0.1],
                "dy": [0.25, 0.25],
                "dz": [0.4],
            }
        }
    )
    assert (grid.nx, grid.ny, grid.nz) == (3, 2, 1)
    assert grid.size_m() == pytest.approx((0.4, 0.5, 0.4))


def test_cmg_DX_DY_DZ_aliases() -> None:
    grid = grid_from_cfg(
        {
            "grid": {
                "nx": 2,
                "ny": 2,
                "nz": 3,
                "DX": 0.1,
                "DY": [0.2, 0.3],
                "DZ": [0.05, 0.05, 0.10],
            }
        }
    )
    assert np.allclose(grid.dx, [0.1, 0.1])
    assert np.allclose(grid.dy, [0.2, 0.3])
    assert np.allclose(grid.dz, [0.05, 0.05, 0.10])
    assert grid.size_m() == pytest.approx((0.2, 0.5, 0.20))


def test_size_m_must_match_axis_sums() -> None:
    with pytest.raises(GridError, match="size_m"):
        grid_from_cfg(
            {
                "geometry": {"size_m": [0.2, 0.2, 1.0]},
                "grid": {"nx": 2, "ny": 2, "dx": 0.1, "dy": 0.1, "dz": [0.05, 0.10]},
            }
        )


def test_sidecar_file_yaml_and_json(tmp_path: Path) -> None:
    yml = tmp_path / "layers.yaml"
    yml.write_text("dz: [0.05, 0.10, 0.15]\ndx: 0.1\ndy: 0.1\n", encoding="utf-8")
    from_yaml = grid_from_cfg(
        {
            "geometry": {"size_m": [0.4, 0.4, 0.3]},
            "grid": {"file": "layers.yaml", "nx": 4, "ny": 4},
        },
        cfg_dir=tmp_path,
    )
    assert from_yaml.nz == 3
    assert np.allclose(from_yaml.dz, [0.05, 0.10, 0.15])
    assert from_yaml.size_m() == pytest.approx((0.4, 0.4, 0.3))

    js = tmp_path / "layers.json"
    js.write_text('{"grid": {"DX": [0.2, 0.2], "DY": [0.2, 0.2], "DZ": [0.3]}}', encoding="utf-8")
    from_json = grid_from_cfg({"grid": {"file": "layers.json"}}, cfg_dir=tmp_path)
    assert (from_json.nx, from_json.ny, from_json.nz) == (2, 2, 1)
    assert from_json.size_m() == pytest.approx((0.4, 0.4, 0.3))


def test_inline_overrides_sidecar(tmp_path: Path) -> None:
    side = tmp_path / "base.yaml"
    side.write_text("nx: 2\nny: 2\nnz: 2\ndx: 0.1\ndy: 0.1\ndz: [0.1, 0.1]\n", encoding="utf-8")
    grid = grid_from_cfg(
        {"grid": {"file": "base.yaml", "dz": [0.05, 0.15]}},
        cfg_dir=tmp_path,
    )
    assert np.allclose(grid.dz, [0.05, 0.15])
    assert grid.size_m()[2] == pytest.approx(0.20)


def test_cartesian_constructor_variable_dz() -> None:
    grid = CartesianGrid(nx=2, ny=2, nz=3, dx=0.1, dy=0.1, dz=[0.05, 0.10, 0.15])
    assert grid.nz == 3
    assert grid.size_m()[2] == pytest.approx(0.30)
    assert abs(grid.center_distance_z()[0] - 0.075) < 1.0e-15
