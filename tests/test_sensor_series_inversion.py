"""Multi-time inversion from injectors, producers, and exclusive probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reservoir_backend.pipeline import build_mesh, run_time_series
from reservoir_backend.pipeline.run import load_sensor_config, mesh_from_config, main
from reservoir_backend.pipeline.sensor_io import load_sensor_series


def test_load_series_exclusive_probes() -> None:
    samples, roles, locs = load_sensor_series(
        "config/sensor_series_wells.csv",
        "config/sensor_series_boundary.csv",
    )
    assert len(samples) == 4
    assert roles["OBS_P1"] == "observer_p"
    assert roles["OBS_S1"] == "observer_s"
    assert "OBS_P1" in samples[0].well_pressure
    assert "OBS_P1" not in samples[0].well_saturation
    assert "OBS_S1" in samples[0].well_saturation
    assert "OBS_S1" not in samples[0].well_pressure
    assert samples[0].well_rate["INJ"] > 0
    assert samples[0].well_rate["PROD"] < 0
    assert "OBS_P1" in locs


def test_time_series_point_first_inversion() -> None:
    cfg = load_sensor_config("config/sensor_series_case.yaml")
    mesh = mesh_from_config(cfg)
    samples, _, _ = load_sensor_series(
        "config/sensor_series_wells.csv",
        "config/sensor_series_boundary.csv",
    )
    history = run_time_series(mesh, samples, n_k_iterations=1, mode="point_first")
    assert len(history) == 4
    assert history[0].time < history[-1].time
    # k carried / updated in time (not frozen nan)
    assert np.all(history[-1].permeability > 0)
    assert any("time-series inversion" in n for n in history[-1].notes)
    assert any("point-first" in n for n in history[-1].notes)
    # observer_p hard pressure match at last time
    c = mesh.well_cell_id["OBS_P1"]
    i, j, k = mesh.grid.ijk(c)
    assert abs(history[-1].pressure[k, j, i] - samples[-1].well_pressure["OBS_P1"]) < 1.0
    # observer_s hard saturation match
    c2 = mesh.well_cell_id["OBS_S1"]
    i2, j2, k2 = mesh.grid.ijk(c2)
    assert abs(history[-1].sw[k2, j2, i2] - samples[-1].well_saturation["OBS_S1"][0]) < 1e-6


def test_cli_series_mode(tmp_path: Path) -> None:
    out = tmp_path / "series"
    code = main(
        [
            "--config",
            "config/sensor_series_case.yaml",
            "--mode",
            "series",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "t_0000" / "summary.json").is_file()
    assert (out / "t_0003" / "summary.json").is_file()
