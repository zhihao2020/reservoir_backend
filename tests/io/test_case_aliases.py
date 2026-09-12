"""CMG-habit aliases: schedule, eos_yaml, top-level fluid. No new schema."""

from pathlib import Path

import yaml

from reservoir_backend.io.case import load_case


def _tiny_body() -> dict:
    return {
        "geometry": {"size_m": [0.16, 0.08, 0.08]},
        "grid": {"spacing_m": 0.08},
        "physics": {
            "model": "compositional_dpdp",
            "capillary": "none",
            "p_init": 1.0e6,
            "dt_init": 0.5,
            "dt_max": 1.0,
            "max_steps": 4,
        },
        "rock": {"porosity": 0.08, "k_matrix_m2": 1.0e-15},
        "ports": [
            {"name": "INJ", "role": "injector", "control": "rate", "x": 0.04, "y": 0.04, "z": 0.04},
            {"name": "PROD", "role": "producer", "control": "pressure", "x": 0.12, "y": 0.04, "z": 0.04},
        ],
        "sensors": [
            {"name": "P1", "kind": "pressure", "x": 0.08, "y": 0.04, "z": 0.04, "sigma": 2000},
        ],
    }


def test_schedule_alias_loads_like_controls(tmp_path: Path) -> None:
    body = _tiny_body()
    body["fluid"] = "example"
    body["schedule"] = [
        {"port": "INJ", "kind": "rate", "times": [0, 2], "values": [1.0e-8, 1.0e-8]},
        {"port": "PROD", "kind": "pressure", "times": [0, 2], "values": [1.0e5, 1.0e5]},
    ]
    yml = tmp_path / "case.yaml"
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    twin = load_case(yml)
    kinds = {(c.port_name, c.kind) for c in twin.experiment.controls}
    assert ("INJ", "rate") in kinds
    assert ("PROD", "pressure") in kinds


def test_schedule_file_mapping(tmp_path: Path) -> None:
    csv = tmp_path / "sched.csv"
    csv.write_text(
        "time_s,port,kind,value\n"
        "0,INJ,rate,1e-8\n"
        "2,INJ,rate,1e-8\n"
        "0,PROD,pressure,1.0e5\n"
        "2,PROD,pressure,1.0e5\n",
        encoding="utf-8",
    )
    body = _tiny_body()
    body["physics"]["fluid"] = "example"
    body["experiment"] = {"schedule": {"file": "sched.csv"}, "history_end_s": 2}
    yml = tmp_path / "case.yaml"
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    twin = load_case(yml)
    assert len(twin.experiment.controls) == 2


def test_eos_yaml_alias_and_top_level_fluid(tmp_path: Path) -> None:
    body = _tiny_body()
    pvt = Path("examples/lab_v1/pvt.yaml").resolve()
    body["fluid"] = {"eos_yaml": str(pvt)}
    body["schedule"] = [
        {"port": "INJ", "kind": "rate", "times": [0, 1], "values": [1.0e-8, 1.0e-8]},
        {"port": "PROD", "kind": "pressure", "times": [0, 1], "values": [1.0e5, 1.0e5]},
    ]
    yml = tmp_path / "case.yaml"
    yml.write_text(yaml.safe_dump(body), encoding="utf-8")
    twin = load_case(yml)
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.eos.names == ("C1", "nC10")


def test_examples_run_case_uses_well_snippet() -> None:
    twin = load_case("examples/run/case.yaml")
    assert [p.name for p in twin.ports] == ["INJ", "PROD"]
    assert twin.ports[0].role == "injector"
    assert twin.ports[0].control == "rate"
    assert twin.ports[1].role == "producer"
    assert twin.ports[1].control == "pressure"
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.eos.names[0] == "C1"
    assert any(c.port_name == "INJ" for c in twin.experiment.controls)
