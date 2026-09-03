import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.ports.flow import FlowPort, face_half_cell_wi, half_cell_wi, make_face_port


def test_face_port_xmin_xmax_counts() -> None:
    grid = CartesianGrid.uniform((0.30, 0.30, 0.30), 0.01)
    inj = make_face_port(grid, "INJ", "injector", "rate", "xmin")
    prod = make_face_port(grid, "PROD", "producer", "pressure", "xmax")
    assert inj.cell_ids.size == 30 * 30
    assert prod.cell_ids.size == 30 * 30
    assert np.array_equal(inj.cell_ids, grid.face_cells("left"))
    assert set(inj.cell_ids).isdisjoint(set(prod.cell_ids))


def test_yaml_face_ports_match_helper() -> None:
    from reservoir_backend.grid.cartesian import CartesianGrid
    from reservoir_backend.io.well_load import ports_from_cfg

    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    cfg = {
        "ports": [
            {"name": "INJ", "role": "injector", "control": "rate", "perforation": "face", "face": "xmin"},
            {"name": "PROD", "role": "producer", "control": "pressure", "face": "xmax"},
        ]
    }
    ports = ports_from_cfg(cfg, grid)
    ref = [
        FlowPort.face(grid, "INJ", "injector", "rate", "xmin"),
        FlowPort.face(grid, "PROD", "producer", "pressure", "xmax"),
    ]
    assert len(ports) == 2
    assert np.array_equal(ports[0].cell_ids, ref[0].cell_ids)
    assert np.array_equal(ports[1].cell_ids, ref[1].cell_ids)
    assert ports[0].cell_ids.size == grid.ny * grid.nz
    assert ports[0].face == "xmin"
    assert ports[1].face == "xmax"


def test_face_half_cell_wi_is_one_direction() -> None:
    grid = CartesianGrid.uniform((0.30, 0.30, 0.30), 0.075)
    k = 1.0e-12
    cell = 0
    tx_ty = half_cell_wi(grid, cell, k)
    tx = face_half_cell_wi(grid, cell, k, "xmin")
    assert tx == pytest.approx(tx_ty / 2.0)
    assert make_face_port(grid, "INJ", "injector", "rate", "xmin").face == "xmin"
