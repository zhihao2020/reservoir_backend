import json
from pathlib import Path

import pytest
import yaml

from reservoir_backend.twin.apply import (
    P_FIELD_NRMSE_MAX,
    SW_FIELD_NRMSE_MAX,
    accept_demo,
    attach_two_layer_demo,
    demo_field_gate,
    demo_sample_times,
    write_observation_csv,
)
from reservoir_backend.cli.main import main
from reservoir_backend.io.case import load_case


def test_lab_apply_case_is_lab_ready() -> None:
    twin = load_case("examples/lab/lab_apply.yaml")
    assert twin.parameterization.n_params == 2
    assert type(twin.parameterization).__name__ == "RegionParameterization"
    assert twin.experiment.history_end_s == 500.0
    assert all(abs(s.probe_diameter_m - 0.006) < 1e-12 for s in twin.experiment.sensors)
    assert all(p.cell_ids.size == twin.grid.nz for p in twin.ports)
    assert twin.physics.implicit_transport is True
    assert not twin.experiment.observations


def test_demo_sample_times_span_history_and_forecast() -> None:
    twin = load_case("examples/lab/lab_apply.yaml")
    times = demo_sample_times(twin)
    assert times[0] > 0.0
    assert times.max() == 700.0
    assert (times <= 500.0).sum() >= 4
    assert (times > 500.0).sum() >= 1


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

    yml = _small_lab_yaml(tmp_path)
    twin = load_case(yml)
    n = twin.grid.n_cells
    rid = np.asarray(twin.parameterization.region_id, dtype=np.int64).ravel()
    k_true = np.where(rid == 1, 2.0e-12, 2.0e-13)
    theta = np.log(np.array([2.0e-13, 2.0e-12], dtype=float))
    k_post = twin.parameterization.expand(theta)
    sw = np.full(n, 0.30)
    p = np.full(n, 1.5e5)
    _stub_obs_and_forwards(twin, monkeypatch, sw_post=sw, sw_true=sw, p_post=p, p_true=p)
    acc = accept_demo(twin, _fake_posterior(twin, theta, k_post), k_true)
    assert acc["pass"] is True
    assert acc["sw_field_nrmse"] < 1.0e-12
    assert acc["p_field_nrmse"] < 1.0e-12
    assert "similarity" not in acc
    assert "contrast_post" in acc
    assert "posterior_logk_rmse" in acc
    assert "comparison-not-CMG" in acc["gate"]


def test_accept_demo_fails_on_field_nrmse(tmp_path: Path, monkeypatch) -> None:
    import numpy as np

    yml = _small_lab_yaml(tmp_path)
    twin = load_case(yml)
    n = twin.grid.n_cells
    rid = np.asarray(twin.parameterization.region_id, dtype=np.int64).ravel()
    k_true = np.where(rid == 1, 2.0e-12, 2.0e-13)
    theta = np.full(twin.parameterization.n_params, np.log(1.0e-12), dtype=float)
    k_post = twin.parameterization.expand(theta)
    sw_t = np.full(n, 0.30)
    sw_p = np.full(n, 0.90)
    p = np.full(n, 1.5e5)
    _stub_obs_and_forwards(twin, monkeypatch, sw_post=sw_p, sw_true=sw_t, p_post=p, p_true=p)
    acc = accept_demo(twin, _fake_posterior(twin, theta, k_post), k_true)
    assert acc["sw_field_nrmse"] >= SW_FIELD_NRMSE_MAX
    assert acc["pass"] is False
    assert "contrast_post" in acc


@pytest.mark.slow
def test_apply_demo_writes_fields(tmp_path: Path) -> None:
    out = tmp_path / "lab"
    code = main(["apply", "examples/lab/lab_apply.yaml", "--demo", "--output", str(out)])
    assert code == 0
    assert (out / "apply.json").is_file()
    assert (out / "k.npy").is_file()
    assert (out / "observations.csv").is_file()
    assert (out / "forecast_pressure.npy").is_file()
    assert (out / "forecast_sw.npy").is_file()
    assert (out / "figures" / "posterior_fields_xz.png").is_file()
    report = json.loads((out / "apply.json").read_text(encoding="utf-8"))
    assert report["n_theta"] == 2
    assert report["parameterization"] == "RegionParameterization"
    assert all(abs(d - 0.006) < 1e-12 for d in report["probe_diameter_m"])
    assert report["forecast_rmse"] == report["forecast_rmse"]
    assert report["holdout_rmse"] == report["holdout_rmse"]
    acc = report["acceptance"]
    assert acc["pass"] is True
    assert 6.0 <= acc["contrast_post"] <= 16.0
    assert acc["posterior_logk_rmse"] < 0.5
    assert acc["k_vs_expand_max"] < 1.0e-12
    assert acc["sw_field_nrmse"] < SW_FIELD_NRMSE_MAX
    assert acc["p_field_nrmse"] < P_FIELD_NRMSE_MAX
    assert "similarity" not in acc
    header = (out / "observations.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "time_s,sensor,kind,value,sigma,holdout"


def _small_lab_yaml(tmp_path: Path, observations: str | None = None) -> Path:
    cfg = {
        "geometry": {"size_m": [0.20, 0.12, 0.08]},
        "grid": {"spacing_m": 0.04},
        "physics": {
            "model": "two_phase_immiscible",
            "capillary": "brooks_corey",
            "sw_init": 0.20,
            "p_init": 1.5e5,
            "dt_init": 2.0,
            "dt_max": 10.0,
            "fully_implicit": False,
        },
        "rock": {"porosity": 0.20},
        "ports": [
            {"name": "INJ", "role": "injector", "control": "rate", "perforation": "column", "x": 0.02, "y": 0.06, "sw_inj": 0.85},
            {"name": "PROD", "role": "producer", "control": "pressure", "perforation": "column", "x": 0.18, "y": 0.06},
        ],
        "sensors_defaults": {"probe_diameter_m": 0.006},
        "sensors": [
            {"name": "P_in_bot", "kind": "pressure", "x": 0.06, "y": 0.06, "z": 0.02, "sigma": 2.0e3},
            {"name": "P_in_top", "kind": "pressure", "x": 0.06, "y": 0.06, "z": 0.06, "sigma": 2.0e3},
            {"name": "P_out_top", "kind": "pressure", "x": 0.14, "y": 0.06, "z": 0.06, "sigma": 2.0e3},
            {"name": "S_mid_bot", "kind": "saturation", "x": 0.10, "y": 0.06, "z": 0.02, "sigma": 0.04},
        ],
        "inverse": {
            "parameterization": "region",
            "region_axis": "z",
            "n_regions": 2,
            "prior_mean": -28.2,
            "prior_std": 1.0,
            "max_iter": 3,
        },
        "experiment": {
            "history_end_s": 80,
            "holdout_sensors": ["P_out_top"],
            "controls": [
                {"port": "INJ", "kind": "rate", "times": [0, 80, 120], "values": [4.0e-7, 4.0e-7, 4.0e-7]},
                {"port": "INJ", "kind": "composition", "times": [0, 120], "values": [0.85, 0.85]},
                {"port": "PROD", "kind": "pressure", "times": [0, 120], "values": [1.0e5, 1.0e5]},
            ],
        },
    }
    if observations:
        cfg["experiment"]["observations"] = observations
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_apply_without_demo_reads_csv(tmp_path: Path) -> None:
    blank = _small_lab_yaml(tmp_path)
    twin = load_case(blank)
    attach_two_layer_demo(twin, holdout=["P_out_top"])
    write_observation_csv(tmp_path / "observations.csv", twin)
    yml = _small_lab_yaml(tmp_path, observations="observations.csv")
    out = tmp_path / "from_csv"
    code = main(["apply", str(yml), "--output", str(out)])
    assert code == 0
    report = json.loads((out / "apply.json").read_text(encoding="utf-8"))
    assert report["demo"] is False
    assert "acceptance" not in report
    assert report["n_theta"] == 2
    assert report["forecast_rmse"] == report["forecast_rmse"]
    assert report["holdout_rmse"] == report["holdout_rmse"]
    assert (out / "forecast_pressure.npy").is_file()
    loaded = load_case(yml)
    assert loaded.experiment.history_end_s == 80.0
    assert any(o.holdout for o in loaded.experiment.observations)
    assert max(float(o.times_s.max()) for o in loaded.experiment.observations) > 80.0


def test_observation_csv_lab_units(tmp_path: Path) -> None:
    (tmp_path / "obs.csv").write_text(
        "time,time_unit,sensor,kind,value,unit,sigma,holdout\n"
        "2,min,P_in_bot,pressure,150,kPa,2,0\n"
        "2,min,S_mid_bot,saturation,0.25,,0.04,1\n",
        encoding="utf-8",
    )
    yml = _small_lab_yaml(tmp_path, observations="obs.csv")
    twin = load_case(yml)
    by_name = {o.sensor_name: o for o in twin.experiment.observations}
    assert abs(float(by_name["P_in_bot"].times_s[0]) - 120.0) < 1.0e-12
    assert abs(float(by_name["P_in_bot"].values[0]) - 1.5e5) < 1.0e-6
    assert abs(float(by_name["P_in_bot"].sigma[0]) - 2.0e3) < 1.0e-6
    assert by_name["S_mid_bot"].holdout
    assert abs(float(by_name["S_mid_bot"].values[0]) - 0.25) < 1.0e-12


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
