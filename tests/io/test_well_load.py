"""YAML ports stay the same; a tiny *WELL snippet builds the same FlowPort."""

from pathlib import Path

import numpy as np
import yaml

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.io.case import load_case
from reservoir_backend.io.well_load import parse_well_deck, ports_from_cfg, ports_from_well_file
from reservoir_backend.ports.flow import FlowPort


def _grid() -> CartesianGrid:
    return CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)


def _case_body() -> dict:
    return {
        "geometry": {"size_m": [0.16, 0.08, 0.08]},
        "grid": {"spacing_m": 0.04},
        "physics": {"model": "two_phase_immiscible", "capillary": "none"},
        "sensors": [
            {"name": "P1", "kind": "pressure", "x": 0.06, "y": 0.04, "z": 0.04, "sigma": 2000},
        ],
        "inverse": {
            "parameterization": "region",
            "n_regions": 2,
            "max_iter": 4,
        },
        "experiment": {
            "controls": [
                {"port": "INJ", "kind": "rate", "times": [0, 10], "values": [3.0e-8, 3.0e-8]},
                {"port": "PROD", "kind": "pressure", "times": [0, 10], "values": [1.0e5, 1.0e5]},
            ],
            "history_end_s": 10,
        },
    }


def _same_well(a: FlowPort, b: FlowPort) -> None:
    assert a.name == b.name
    assert a.role == b.role
    assert a.control == b.control
    assert np.array_equal(a.cell_ids, b.cell_ids)
    assert a.sw_inj == b.sw_inj
    assert a.use_productivity == b.use_productivity
    assert a.rw_m == b.rw_m
    assert a.skin == b.skin
    assert a.geofac == b.geofac
    assert a.wi_multiplier == b.wi_multiplier


def test_yaml_xyz_ports_match_flowport_at_point() -> None:
    grid = _grid()
    cfg = {
        "ports": [
            {"name": "INJ", "role": "injector", "control": "rate", "x": 0.02, "y": 0.04, "z": 0.04},
            {"name": "PROD", "role": "producer", "control": "pressure", "x": 0.14, "y": 0.04, "z": 0.04},
        ]
    }
    ports = ports_from_cfg(cfg, grid)
    ref = [
        FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04)),
        FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04)),
    ]
    assert len(ports) == 2
    _same_well(ports[0], ref[0])
    _same_well(ports[1], ref[1])


def test_yaml_column_ports_match_flowport_column() -> None:
    grid = _grid()
    cfg = {
        "ports": [
            {
                "name": "INJ",
                "role": "injector",
                "control": "rate",
                "perforation": "column",
                "x": 0.02,
                "y": 0.04,
                "sw_inj": 0.85,
            }
        ]
    }
    ports = ports_from_cfg(cfg, grid)
    ref = FlowPort.column(grid, "INJ", "injector", "rate", 0.02, 0.04, sw_inj=0.85)
    assert len(ports) == 1
    _same_well(ports[0], ref)


def test_gem_perf_k_range_matches_yaml_k_perf() -> None:
    grid = CartesianGrid.uniform((0.30, 0.30, 0.30), 0.02)
    assert grid.nx == 15 and grid.nz == 15
    yaml_cfg = {
        "ports": [
            {
                "name": "INJ",
                "role": "injector",
                "control": "rate",
                "ijk": [8, 8],
                "k_perf": [1, 11],
                "use_productivity": True,
            }
        ]
    }
    from_yaml = ports_from_cfg(yaml_cfg, grid)
    snippet = "\n".join(
        [
            "*WELL 1 'INJ'",
            "*INJECTOR 1",
            "*OPERATE *MAX *BHF 0.0072",
            "*GEOMETRY *K 0.003 0.34 1.0 0.0",
            "*PERF *GEO 1",
            "  8  8  1:11  1.0",
        ]
    )
    from_deck = parse_well_deck(snippet, grid)
    assert from_yaml[0].cell_ids.size == 11
    assert from_deck[0].cell_ids.size == 11
    assert int(from_yaml[0].cell_ids[0]) == grid.index(7, 7, 0)
    assert int(from_yaml[0].cell_ids[-1]) == grid.index(7, 7, 10)
    assert np.array_equal(from_yaml[0].cell_ids, from_deck[0].cell_ids)


def test_cmg_well_snippet_matches_yaml_ijk(tmp_path: Path) -> None:
    grid = _grid()
    yaml_cfg = {
        "wells": [
            {"name": "INJ", "role": "injector", "control": "pressure", "ijk": [1, 1, 1], "use_productivity": True},
            {"name": "PROD", "role": "producer", "control": "pressure", "ijk": [4, 2, 1], "use_productivity": True},
        ]
    }
    from_yaml = ports_from_cfg(yaml_cfg, grid)
    snippet = "\n".join(
        [
            "*WELL 1 'INJ'",
            "*INJECTOR 1",
            "*OPERATE *MAX *BHP 2.0e5",
            "*PERF *GEO 1",
            "  1 1 1 1.0",
            "*WELL 2 'PROD'",
            "*PRODUCER 2",
            "*OPERATE *MIN *BHP 1.0e5",
            "*PERF *GEO 2",
            "  4 2 1 1.0",
        ]
    )
    inc = tmp_path / "wells.inc"
    inc.write_text(snippet + "\n", encoding="utf-8")
    from_file = ports_from_well_file(inc, grid)
    assert len(from_yaml) == 2
    assert len(from_file) == 2
    assert from_file[0].cell_ids[0] == grid.index(0, 0, 0)
    assert from_file[1].cell_ids[0] == grid.index(3, 1, 0)
    _same_well(from_file[0], from_yaml[0])
    _same_well(from_file[1], from_yaml[1])
    direct = parse_well_deck(snippet, grid)
    _same_well(direct[0], from_yaml[0])


def test_case_wells_file_builds_twin_ports(tmp_path: Path) -> None:
    snippet = "\n".join(
        [
            "*WELL 1 INJ",
            "*INJECTOR 1",
            "*OPERATE *MAX *STW 3.0e-8",
            "*PERF 1",
            "  1 1 1",
            "*WELL 2 PROD",
            "*PRODUCER 2",
            "*OPERATE *MIN *BHP 1.0e5",
            "*PERF 2",
            "  4 2 1",
        ]
    )
    (tmp_path / "wells.inc").write_text(snippet + "\n", encoding="utf-8")
    body = _case_body()
    body["wells"] = {"file": "wells.inc"}
    yml = tmp_path / "case.yaml"
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    twin = load_case(yml)
    assert [p.name for p in twin.ports] == ["INJ", "PROD"]
    assert twin.ports[0].role == "injector"
    assert twin.ports[0].control == "rate"
    assert twin.ports[1].role == "producer"
    assert twin.ports[1].control == "pressure"
    assert int(twin.ports[0].cell_ids[0]) == twin.grid.index(0, 0, 0)
    assert int(twin.ports[1].cell_ids[0]) == twin.grid.index(3, 1, 0)


def test_load_case_yaml_ports_still_work(tmp_path: Path) -> None:
    body = _case_body()
    body["ports"] = [
        {"name": "INJ", "role": "injector", "control": "rate", "x": 0.02, "y": 0.04, "z": 0.04},
        {"name": "PROD", "role": "producer", "control": "pressure", "x": 0.14, "y": 0.04, "z": 0.04},
    ]
    yml = tmp_path / "case.yaml"
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    twin = load_case(yml)
    grid = twin.grid
    ref_inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04))
    ref_prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04))
    _same_well(twin.ports[0], ref_inj)
    _same_well(twin.ports[1], ref_prod)


def test_geometry_maps_existing_wi_fields() -> None:
    grid = _grid()
    snippet = "\n".join(
        [
            "*WELL 1 'INJ'",
            "*INJECTOR 1",
            "*OPERATE *MAX *BHP 2.0e5",
            "*GEOMETRY *K 0.02 0.34 1.0 0.5",
            "*PERF *GEO 1",
            "  1 1 1 1.0",
        ]
    )
    port = parse_well_deck(snippet, grid)[0]
    assert port.use_productivity is True
    assert port.rw_m == 0.02
    assert port.geofac == 0.34
    assert port.skin == 0.5
    assert port.wi_multiplier == 1.0


def test_first_operate_is_control_kind() -> None:
    grid = _grid()
    rate_first = parse_well_deck(
        "\n".join(
            [
                "*WELL 1 INJ",
                "*INJECTOR 1",
                "*OPERATE *MAX *STW 5.0",
                "*OPERATE *MAX *BHP 9.0e3",
                "*PERF 1",
                "  1 1 1",
            ]
        ),
        grid,
    )[0]
    bhp_only = parse_well_deck(
        "\n".join(
            [
                "*WELL 2 PROD",
                "*PRODUCER 2",
                "*OPERATE *MIN *BHP 1.0e5",
                "*PERF 2",
                "  2 1 1",
            ]
        ),
        grid,
    )[0]
    assert rate_first.control == "rate"
    assert bhp_only.control == "pressure"


def test_skip_unsupported_keywords_still_builds() -> None:
    grid = _grid()
    snippet = "\n".join(
        [
            "*WELL 1 INJ",
            "*INJECTOR 1",
            "*INCOMP *WATER",
            "*WELLHYD 1",
            "*OPERATE *MAX *BHP 2.0e5",
            "*PERF *GEO 1",
            "  1 1 1 1.0",
            "*TIME 1.0",
            "*DATE 1988 01 01",
            "*GROUP 'G1'",
        ]
    )
    ports = parse_well_deck(snippet, grid)
    assert len(ports) == 1
    assert ports[0].name == "INJ"
    assert ports[0].role == "injector"
    assert int(ports[0].cell_ids[0]) == grid.index(0, 0, 0)


def test_multi_perf_ijk() -> None:
    grid = CartesianGrid.uniform((0.16, 0.08, 0.12), 0.04)
    snippet = "\n".join(
        [
            "*WELL 1 INJ",
            "*INJECTOR 1",
            "*OPERATE *MAX *BHP 2.0e5",
            "*PERF *GEO 1",
            "  1 1 1 1.0",
            "  1 1 2 1.0",
            "  1 1 3 1.0",
        ]
    )
    port = parse_well_deck(snippet, grid)[0]
    expect = np.array(
        [grid.index(0, 0, 0), grid.index(0, 0, 1), grid.index(0, 0, 2)],
        dtype=np.int64,
    )
    assert np.array_equal(port.cell_ids, expect)
    yaml_port = ports_from_cfg(
        {
            "ports": [
                {
                    "name": "INJ",
                    "role": "injector",
                    "control": "pressure",
                    "ijk": [[1, 1, 1], [1, 1, 2], [1, 1, 3]],
                    "use_productivity": True,
                }
            ]
        },
        grid,
    )[0]
    _same_well(port, yaml_port)
