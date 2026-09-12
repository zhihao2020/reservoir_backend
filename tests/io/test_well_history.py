"""Well-history CSV load + nRMSE report. Not field L2."""

from pathlib import Path

import numpy as np
import yaml

from reservoir_backend.domain.types import ObservationSeries, State
from reservoir_backend.io.case import load_case
from reservoir_backend.io.well_history import predict_well_value, score_well_history
from reservoir_backend.solver.trajectory import Trajectory


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
