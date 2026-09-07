from pathlib import Path

import pytest
import yaml

from reservoir_backend.twin.apply import (
    P_FIELD_NRMSE_MAX,
    SW_FIELD_NRMSE_MAX,
    accept_demo,
    attach_cf_demo,
    demo_field_gate,
    demo_sample_times,
)
from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case


def test_lab_cf_is_v1_dpdp_apply_path() -> None:
    twin = load_case("examples/lab/lab_cf.yaml")
    assert twin.uses_dpdp()
    assert twin.inverse.algorithm == "auto"
    assert twin.parameterization.n_params == 2
    k_true = attach_cf_demo(twin, holdout=["P_out"])
    assert twin.experiment.observations
    assert k_true.size == twin.grid.n_cells


def test_lab_v1_dev_is_lab_ready() -> None:
    twin = load_case("examples/lab_v1/case_dev.yaml")
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "LogCfTmfParameterization"
    assert twin.experiment.history_end_s == 60.0
    assert all(abs(s.probe_diameter_m - 0.006) < 1e-12 for s in twin.experiment.sensors)
    assert not twin.experiment.observations


def test_demo_sample_times_span_history_and_forecast() -> None:
    twin = load_case("examples/lab/lab_cf.yaml")
    times = demo_sample_times(twin)
    assert times[0] > 0.0
    hist = float(twin.experiment.history_end_s or 0.0)
    assert times.max() >= hist - 1.0e-12
    assert (times <= hist + 1.0e-12).sum() >= 1


def test_demo_field_gate_is_field_nrmse_not_contrast() -> None:
    assert demo_field_gate(0.0, 0.0) is True
    assert demo_field_gate(SW_FIELD_NRMSE_MAX * 0.5, P_FIELD_NRMSE_MAX * 0.5) is True
    assert demo_field_gate(SW_FIELD_NRMSE_MAX, 0.0) is False
    assert demo_field_gate(0.0, P_FIELD_NRMSE_MAX) is False
    assert demo_field_gate(float("nan"), 0.0) is False
    assert demo_field_gate(0.0, float("inf")) is False


def _fake_posterior(twin, theta, k):
    from types import SimpleNamespace

    return SimpleNamespace(
        k=k,
        theta=theta,
        holdout_rmse=0.1,
        forecast_rmse=0.1,
    )


def _stub_obs_and_forwards(twin, monkeypatch, *, sw_post, sw_true, p_post, p_true):
    import numpy as np
    from types import SimpleNamespace

    from reservoir_backend.domain.types import ObservationSeries

    times = np.array([10.0, 20.0])
    twin.experiment.observations = [
        ObservationSeries(
            s.name,
            s.kind,
            times,
            np.zeros(2),
            np.full(2, max(float(s.sigma), 1.0)),
            False,
        )
        for s in twin.experiment.sensors
    ]
    st_true = SimpleNamespace(sw=sw_true, pressure=p_true)
    st_post = SimpleNamespace(sw=sw_post, pressure=p_post)
    calls = {"n": 0}

    def fake_sim(*_a, **_k):
        calls["n"] += 1
        st = st_true if calls["n"] == 1 else st_post
        return SimpleNamespace(states=[st], times_s=times, port_rates=[{}, {}])

    monkeypatch.setattr(twin, "simulate", fake_sim)
    monkeypatch.setattr(
        "reservoir_backend.twin.offline.predict_from_trajectory",
        lambda *_a, **_k: np.zeros(len(twin.experiment.sensors) * times.size),
    )


def test_accept_demo_reports_field_nrmse(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    from reservoir_backend.twin.offline import encode_physical_theta

    yml = _small_lab_yaml(tmp_path)
    twin = load_case(yml)
    n = twin.grid.n_cells
    theta = encode_physical_theta(twin.parameterization, cf_m2=1.0e-12, tmf_multiplier=1.0)
    k_true = twin.parameterization.expand(theta)
    k_post = k_true.copy()
    sw = np.full(n, 0.30)
    p = np.full(n, 1.2e7)
    _stub_obs_and_forwards(twin, monkeypatch, sw_post=sw, sw_true=sw, p_post=p, p_true=p)
    acc = accept_demo(twin, _fake_posterior(twin, theta, k_post), k_true)
    assert acc["pass"] is True
    assert acc["p_field_nrmse"] < 1.0e-12
    assert "similarity" not in acc
    assert "posterior_logk_rmse" in acc
    assert "comparison-not-CMG" in acc["gate"]


def test_accept_demo_fails_on_field_nrmse(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    from reservoir_backend.twin.offline import encode_physical_theta

    yml = _small_lab_yaml(tmp_path)
    twin = load_case(yml)
    n = twin.grid.n_cells
    theta = encode_physical_theta(twin.parameterization, cf_m2=1.0e-12, tmf_multiplier=1.0)
    k_true = twin.parameterization.expand(theta)
    k_post = k_true.copy()
    sw_t = np.full(n, 0.30)
    sw_p = np.full(n, 0.90)
    p_t = np.full(n, 1.2e7)
    p_p = np.full(n, 2.0e7)
    _stub_obs_and_forwards(twin, monkeypatch, sw_post=sw_p, sw_true=sw_t, p_post=p_p, p_true=p_t)
    acc = accept_demo(twin, _fake_posterior(twin, theta, k_post), k_true)
    assert acc["pass"] is False


def _small_lab_yaml(tmp_path: Path, observations: str | None = None) -> Path:
    cfg = {
        "geometry": {"size_m": [0.30, 0.10, 0.10]},
        "grid": {"nx": 3, "ny": 1, "nz": 1},
        "physics": {
            "model": "compositional_dpdp",
            "fluid": "example",
            "capillary": "none",
            "p_init": 1.2e7,
            "dt_init": 0.5,
            "dt_max": 2.0,
            "max_steps": 40,
            "shape_factor": 40.0,
            "phi_fracture": 0.02,
            "k_matrix_m2": 1.0e-15,
        },
        "rock": {"porosity": 0.08, "k_matrix_m2": 1.0e-15},
        "ports": [
            {"name": "INJ", "role": "injector", "control": "rate", "perforation": "column", "x": 0.05, "y": 0.05},
            {"name": "PROD", "role": "producer", "control": "pressure", "perforation": "column", "x": 0.25, "y": 0.05},
        ],
        "sensors_defaults": {"probe_diameter_m": 0.006},
        "sensors": [
            {"name": "P_in", "kind": "pressure", "x": 0.10, "y": 0.05, "z": 0.05, "sigma": 2.0e4},
            {"name": "P_out", "kind": "pressure", "x": 0.20, "y": 0.05, "z": 0.05, "sigma": 2.0e4},
        ],
        "inverse": {
            "parameterization": "log_cf_tmf",
            "algorithm": "auto",
            "prior_mean": [0.0, 0.0],
            "prior_std": [0.8, 0.5],
            "max_iter": 3,
            "k_matrix_m2": 1.0e-15,
            "phi_fracture": 0.02,
        },
        "experiment": {
            "history_end_s": 6,
            "holdout_sensors": ["P_out"],
            "controls": [
                {"port": "INJ", "kind": "rate", "times": [0, 6], "values": [2.0e-4, 2.0e-4]},
                {"port": "INJ", "kind": "composition", "times": [0, 6], "values": [0.95, 0.95]},
                {"port": "PROD", "kind": "pressure", "times": [0, 6], "values": [1.18e7, 1.18e7]},
            ],
        },
    }
    if observations:
        cfg["experiment"]["observations"] = observations
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_observation_csv_lab_units(tmp_path: Path) -> None:
    (tmp_path / "obs.csv").write_text(
        "time,time_unit,sensor,kind,value,unit,sigma,holdout\n"
        "2,min,P_in,pressure,12000,kPa,20,0\n"
        "2,min,P_out,pressure,11800,kPa,20,1\n",
        encoding="utf-8",
    )
    yml = _small_lab_yaml(tmp_path, observations="obs.csv")
    twin = load_case(yml)
    by_name = {o.sensor_name: o for o in twin.experiment.observations}
    assert abs(float(by_name["P_in"].times_s[0]) - 120.0) < 1.0e-12
    assert abs(float(by_name["P_in"].values[0]) - 1.2e7) < 1.0e-6
    assert abs(float(by_name["P_in"].sigma[0]) - 2.0e4) < 1.0e-6
    assert by_name["P_out"].holdout


def test_unknown_sensor_in_csv_errors(tmp_path: Path) -> None:
    (tmp_path / "obs.csv").write_text(
        "time_s,sensor,kind,value,sigma,holdout\n"
        "10,P_ghost,pressure,1.1e5,2000,0\n",
        encoding="utf-8",
    )
    yml = _small_lab_yaml(tmp_path, observations="obs.csv")
    with pytest.raises(ValueError, match="P_ghost"):
        load_case(yml)


def test_empty_observation_csv_errors(tmp_path: Path) -> None:
    (tmp_path / "obs.csv").write_text("time_s,sensor,kind,value,sigma,holdout\n", encoding="utf-8")
    yml = _small_lab_yaml(tmp_path, observations="obs.csv")
    with pytest.raises(ValueError, match="empty"):
        load_case(yml)
