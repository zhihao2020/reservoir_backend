"""Well-history CSV load + nRMSE, plus EXAMPLE monthly export. Not field L2."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from reservoir_backend.cli.main import main
from reservoir_backend.domain.types import ObservationSeries, State
from reservoir_backend.io.case import load_case
from reservoir_backend.io.well_history import (
    WELL_HISTORY_COLUMNS,
    export_case_well_history,
    predict_well_value,
    read_well_history_csv,
    score_well_history,
)
from reservoir_backend.solver.trajectory import Trajectory

CASE = Path("examples/compositional/comp_example_1inj4prod_monthly.yaml")


def _traj(*, q_inj=2.0e-4, q_oil=1.0e-8, q_gas=1.0e-8, q_water=0.0, p_inj=1.2e7, p_prod=1.18e7):
    times = np.array([0.0, 2.0])
    dummy = State(pressure=np.ones(1), sw=np.full(1, 0.2))
    rates = {
        "INJ:q_inj": q_inj,
        "PROD:q_oil": q_oil,
        "PROD:q_gas": q_gas,
        "PROD:q_water": q_water,
    }
    bhp = {"INJ": p_inj, "PROD": p_prod}
    return Trajectory(
        times_s=times,
        states=[dummy, dummy],
        reports=[],
        port_rates=[dict(rates), dict(rates)],
        port_bhp=[dict(bhp), dict(bhp)],
    )


def test_csv_well_column_loads_kinds() -> None:
    twin = load_case("config/fixtures/jiyang_hnp_well_history.yaml")
    kinds = {(o.sensor_name, o.kind, o.holdout) for o in twin.experiment.observations}
    assert ("INJ", "bhp", False) in kinds
    assert ("INJ", "q_inj", False) in kinds
    assert ("PROD", "q_oil", False) in kinds
    assert ("PROD", "q_gas", True) in kinds
    assert ("PROD", "q_water", False) in kinds
    names = {s.name for s in twin.experiment.sensors}
    assert "INJ" in names and "PROD" in names
    inj = next(s for s in twin.experiment.sensors if s.name == "INJ")
    assert inj.port_name == "INJ"


def test_score_nrmse_perfect_match() -> None:
    twin = load_case("config/fixtures/jiyang_hnp_well_history.yaml")
    traj = _traj()
    rec = score_well_history(twin.experiment.observations, traj)
    assert rec is not None
    assert rec["holdout"] is False
    assert "PROD:q_gas" not in rec["nrmse"]
    assert rec["nrmse"]["INJ:bhp"] == 0.0
    assert rec["nrmse"]["INJ:q_inj"] == 0.0
    assert rec["nrmse"]["PROD:q_water"] == 0.0
    assert rec["mae"]["PROD:bhp"] == 0.0
    assert rec["overall_nrmse"] == 0.0
    assert rec["missing"] == []


def test_score_nrmse_and_mae_from_error() -> None:
    series = [
        ObservationSeries("INJ", "bhp", np.array([0.0, 2.0]), np.array([1.0e7, 1.0e7]), 1.0e5),
    ]
    traj = _traj(p_inj=1.2e7)
    rec = score_well_history(series, traj)
    assert rec is not None
    assert rec["mae"]["INJ:bhp"] == 2.0e6
    assert rec["nrmse"]["INJ:bhp"] == 20.0
    assert rec["n_samples"]["INJ:bhp"] == 2


def test_missing_rate_key_is_reported() -> None:
    series = [ObservationSeries("X1", "q_oil", np.array([0.0]), np.array([1.0]), 1.0)]
    traj = _traj()
    rec = score_well_history(series, traj)
    assert rec is not None
    assert rec["missing"] == ["X1:q_oil"]
    assert rec["nrmse"] == {}


def test_predict_well_value_reads_traj_keys() -> None:
    traj = _traj(q_oil=3.5e-5)
    assert predict_well_value(traj, 2.0, "PROD", "q_oil") == 3.5e-5
    assert predict_well_value(traj, 2.0, "INJ", "bhp") == 1.2e7
    assert predict_well_value(traj, 2.0, "NOPE", "bhp") is None


def test_well_history_file_mapping(tmp_path: Path) -> None:
    csv = tmp_path / "wh.csv"
    csv.write_text(
        "time_s,well,kind,value,sigma,holdout\n0,INJ,bhp,1.0e7,1.0e5,0\n",
        encoding="utf-8",
    )
    yml = tmp_path / "case.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "geometry": {"size_m": [0.16, 0.08, 0.08]},
                "grid": {"spacing_m": 0.08},
                "physics": {"model": "compositional_dpdp", "fluid": "example", "capillary": "none"},
                "ports": [
                    {"name": "INJ", "role": "injector", "control": "rate", "x": 0.04, "y": 0.04, "z": 0.04},
                    {"name": "PROD", "role": "producer", "control": "pressure", "x": 0.12, "y": 0.04, "z": 0.04},
                ],
                "schedule": [
                    {"port": "INJ", "kind": "rate", "times": [0, 1], "values": [1.0e-8, 1.0e-8]},
                    {"port": "PROD", "kind": "pressure", "times": [0, 1], "values": [1.0e5, 1.0e5]},
                ],
                "well_history": {"file": "wh.csv"},
            }
        ),
        encoding="utf-8",
    )
    twin = load_case(yml)
    assert len(twin.experiment.observations) == 1
    assert twin.experiment.observations[0].kind == "bhp"
    assert twin.experiment.observations[0].sensor_name == "INJ"


def test_monthly_1inj4prod_yaml_is_example_not_jiyang() -> None:
    text = CASE.read_text(encoding="utf-8")
    assert "EXAMPLE" in text
    assert "monthly" in text.lower()
    assert "not a Jiyang" in text
    assert "gem_deck" not in text or "refuse" in text.lower()
    twin = load_case(CASE)
    assert twin.physics.model == "compositional"
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.has_water
    assert [p.name for p in twin.ports] == ["INJ", "PROD1", "PROD2", "PROD3", "PROD4"]
    assert twin.ports[0].role == "injector"
    assert all(p.role == "producer" for p in twin.ports[1:])
    assert all(p.use_productivity for p in twin.ports)
    t_end = max(float(c.times_s[-1]) for c in twin.experiment.controls)
    assert t_end == pytest.approx(30.0 * 86400.0)
    assert twin.physics.dt_init == pytest.approx(86400.0)
    assert twin.physics.dt_max == pytest.approx(10.0 * 86400.0)


def test_missing_real_gem_card_still_refuses(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        CASE.read_text(encoding="utf-8").replace(
            "fluid: example",
            "fluid: {gem_deck: missing_jiyang.gem}",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not found|refuse invented"):
        load_case(p)


def test_export_well_history_csv_schema_and_wells(tmp_path: Path) -> None:
    out = tmp_path / "well_history.csv"
    path = export_case_well_history(CASE, out, t_end=8.0, dt_init=2.0, dt_max=8.0)
    assert path.is_file()
    rows = read_well_history_csv(path)
    assert rows
    assert list(rows[0].keys()) == list(WELL_HISTORY_COLUMNS)
    wells = {r["well"] for r in rows}
    assert wells == {"INJ", "PROD1", "PROD2", "PROD3", "PROD4"}
    inj = [r for r in rows if r["well"] == "INJ"]
    assert inj[0]["role"] == "injector"
    assert inj[0]["control"] == "rate"
    assert float(inj[-1]["q_mol_s"]) > 0.0
    assert float(inj[-1]["bhp_pa"]) > 0.0
    for key in ("gem_q_oil_m3_s", "gem_q_gas_m3_s", "gem_q_water_m3_s", "gem_bhp_pa"):
        assert inj[-1][key] == ""
    prod = [r for r in rows if r["well"] == "PROD1"]
    assert prod[0]["control"] == "pressure"
    assert float(prod[-1]["bhp_pa"]) == pytest.approx(1.1e7)


def test_simulate_cli_writes_well_history(tmp_path: Path) -> None:
    """Existing short EXAMPLE case: ``simulate --output`` emits the same schema."""
    code = main(
        ["simulate", "examples/compositional/comp_example_water.yaml", "--output", str(tmp_path)]
    )
    assert code == 0
    csv_path = tmp_path / "well_history.csv"
    assert csv_path.is_file()
    rows = read_well_history_csv(csv_path)
    assert list(rows[0].keys()) == list(WELL_HISTORY_COLUMNS)
    assert {r["well"] for r in rows} >= {"INJ", "PROD"}


def test_product_modules_do_not_import_references() -> None:
    touched = [
        Path("reservoir_backend/eos/flash.py"),
        Path("reservoir_backend/eos/flash_batch.py"),
        Path("reservoir_backend/comp/wells.py"),
        Path("reservoir_backend/io/well_history.py"),
        Path("reservoir_backend/cli/main.py"),
    ]
    for path in touched:
        text = path.read_text(encoding="utf-8")
        assert "import references" not in text
        assert "from references" not in text
