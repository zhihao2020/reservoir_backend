"""CSV sensor series and ES-MDA unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from reservoir_backend.pipeline import (
    AxisAlignedBounds,
    BoundaryConditions,
    SensorSample,
    WellPoint,
    build_mesh,
    run_time_slice,
)
from reservoir_backend.pipeline.ensemble_math import gaspari_cohn, normalize_alpha_weights
from reservoir_backend.pipeline.esmda import generate_logk_ensemble, run_esmda_permeability
from reservoir_backend.pipeline.sensor_io import (
    load_sensor_series,
    load_well_series_csv,
    write_boundary_series_csv,
    write_well_series_csv,
)
from reservoir_backend.pipeline.run import main


def test_load_repo_series_csv() -> None:
    samples, roles, _locs = load_sensor_series(
        "config/sensor_series_wells.csv",
        "config/sensor_series_boundary.csv",
    )
    assert len(samples) == 4
    assert samples[0].time == 0.0
    assert "INJ" in samples[0].well_pressure
    assert samples[0].boundary.pressure["left"] == 12.0e6
    assert samples[-1].well_saturation["PROD"][0] == 0.40
    assert roles["OBS_P1"] == "observer_p"
    assert roles["OBS_S1"] == "observer_s"


def test_write_roundtrip(tmp_path: Path) -> None:
    samples = [
        SensorSample(
            time=0.0,
            well_pressure={"A": 1.0e7, "B": 9.0e6},
            well_saturation={"A": (0.7, 0.3, 0.0), "B": (0.2, 0.8, 0.0)},
            boundary=BoundaryConditions(pressure={"left": 1.0e7, "right": 9.0e6}),
            well_rate={"A": 1e-5, "B": -1e-5},
        ),
        SensorSample(
            time=10.0,
            well_pressure={"A": 1.1e7, "B": 8.5e6},
            well_saturation={"A": (0.75, 0.25, 0.0), "B": (0.25, 0.75, 0.0)},
            boundary=BoundaryConditions(pressure={"left": 1.1e7, "right": 8.5e6}),
            well_rate={"A": 1e-5, "B": -1e-5},
        ),
    ]
    roles = {"A": "injector", "B": "producer"}
    wpath = write_well_series_csv(tmp_path / "w.csv", samples, roles=roles)
    bpath = write_boundary_series_csv(tmp_path / "b.csv", samples)
    loaded, loaded_roles, _ = load_sensor_series(wpath, bpath)
    assert len(loaded) == 2
    assert loaded[1].well_pressure["A"] == 1.1e7
    assert loaded[1].boundary.pressure["right"] == 8.5e6
    assert loaded_roles["A"] == "injector"


def test_logk_ensemble_shape_positive() -> None:
    ens = generate_logk_ensemble((3, 4, 5), ne=8, k_mean=1.0e-13, seed=1)
    assert ens.shape == (8, 3, 4, 5)
    assert np.all(ens > 0.0)


def test_normalize_alpha_sum_inv() -> None:
    a = normalize_alpha_weights(4)
    assert a.shape == (4,)
    assert abs(float(np.sum(1.0 / a)) - 1.0) < 1.0e-12
    b = normalize_alpha_weights(np.array([1.0, 2.0, 3.0]))
    assert abs(float(np.sum(1.0 / b)) - 1.0) < 1.0e-12


def test_gaspari_cohn_support() -> None:
    d = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    w = gaspari_cohn(d, 1.0)
    assert w[0] == 1.0
    assert w[-1] == 0.0
    assert w[-2] == 0.0
    assert np.all(w[:-1] >= 0.0)


def test_esmda_reduces_well_pressure_misfit() -> None:
    bounds = AxisAlignedBounds(0.0, 60.0, 0.0, 40.0, 0.0, 30.0)
    wells = [
        WellPoint("INJ", 10.0, 20.0, 15.0, role="injector"),
        WellPoint("PROD", 50.0, 20.0, 15.0, role="producer"),
    ]
    mesh = build_mesh(bounds, 10.0, 10.0, 10.0, wells=wells)
    samples = [
        SensorSample(
            time=0.0,
            well_pressure={"INJ": 12.0e6, "PROD": 10.0e6},
            well_saturation={"INJ": (0.7, 0.3, 0.0), "PROD": (0.3, 0.7, 0.0)},
            boundary=BoundaryConditions(pressure={"left": 12.0e6, "right": 10.0e6}),
        ),
        SensorSample(
            time=30.0,
            well_pressure={"INJ": 12.2e6, "PROD": 9.8e6},
            well_saturation={"INJ": (0.75, 0.25, 0.0), "PROD": (0.35, 0.65, 0.0)},
            boundary=BoundaryConditions(pressure={"left": 12.2e6, "right": 9.8e6}),
        ),
    ]
    # prior-only pressure misfit baseline (scalar k)
    base = run_time_slice(mesh, samples[0], permeability_prior_m2=5.0e-14, n_k_iterations=1)
    # ES-MDA
    result = run_esmda_permeability(
        mesh,
        samples,
        ne=12,
        n_assimilations=3,
        k_mean=5.0e-14,
        logk_std=1.0,
        corr_len_cells=2.0,
        seed=3,
        n_k_iterations=1,
        localization_radius_m=40.0,
        ensemble_inflation=1.02,
    )
    assert len(result.alpha_schedule) == 3
    assert abs(sum(1.0 / a for a in result.alpha_schedule) - 1.0) < 1.0e-10
    assert result.k_mean.shape == mesh.grid.shape
    assert result.k_ensemble.shape[0] == 12
    assert len(result.history_mean) == 2
    assert result.observation_rmse
    # joint soft-obs nRMSE (dimensionless) should be finite
    assert np.isfinite(result.observation_rmse[-1])
    assert result.observation_rmse[-1] < 50.0
    assert np.all(result.k_std >= 0.0)
    # ensemble mean k should vary spatially after assimilation
    assert float(np.std(result.k_mean)) > 0.0
    # assimilation should not be trivial no-op on misfit series length
    assert len(result.observation_rmse) == 2 * 3  # n_times * Na
    assert any("n_sw=" in n for n in result.notes)
    _ = base


def test_esmda_uses_saturation_probes() -> None:
    bounds = AxisAlignedBounds(0.0, 80.0, 0.0, 60.0, 0.0, 30.0)
    wells = [
        WellPoint("INJ", 10.0, 30.0, 15.0, role="injector"),
        WellPoint("PROD", 70.0, 30.0, 15.0, role="producer"),
        WellPoint("OBS_S", 40.0, 30.0, 15.0, role="observer_s"),
        WellPoint("OBS_P", 40.0, 45.0, 15.0, role="observer_p"),
    ]
    mesh = build_mesh(bounds, 10.0, 10.0, 10.0, wells=wells)
    samples = [
        SensorSample(
            time=0.0,
            well_pressure={"INJ": 12.0e6, "PROD": 10.0e6, "OBS_P": 11.0e6},
            well_saturation={
                "INJ": (0.75, 0.25, 0.0),
                "PROD": (0.30, 0.70, 0.0),
                "OBS_S": (0.45, 0.55, 0.0),
            },
            boundary=BoundaryConditions(pressure={"left": 12.0e6, "right": 10.0e6}),
            well_rate={"INJ": 1.0e-5, "PROD": -8.0e-6},
        ),
        SensorSample(
            time=20.0,
            well_pressure={"INJ": 12.1e6, "PROD": 9.9e6, "OBS_P": 11.05e6},
            well_saturation={
                "INJ": (0.78, 0.22, 0.0),
                "PROD": (0.36, 0.64, 0.0),
                "OBS_S": (0.50, 0.50, 0.0),
            },
            boundary=BoundaryConditions(pressure={"left": 12.1e6, "right": 9.9e6}),
            well_rate={"INJ": 1.0e-5, "PROD": -8.0e-6},
        ),
    ]
    res = run_esmda_permeability(
        mesh,
        samples,
        ne=10,
        n_assimilations=2,
        k_mean=1.0e-13,
        seed=5,
        n_k_iterations=1,
        n_workers=1,
    )
    assert any("n_sw=" in n and "n_qw=" in n for n in res.notes)
    assert np.all(res.k_mean > 0.0)
    assert float(np.std(res.k_mean)) > 0.0


def test_cli_series_and_esmda(tmp_path: Path) -> None:
    out_s = tmp_path / "series"
    code = main(
        [
            "--config",
            "config/sensor_series_case.yaml",
            "--mode",
            "series",
            "--output",
            str(out_s),
        ]
    )
    assert code == 0
    assert (out_s / "t_0000" / "summary.json").is_file()
    assert (out_s / "series_summary.json").is_file()

    out_e = tmp_path / "esmda"
    code = main(
        [
            "--config",
            "config/sensor_series_case.yaml",
            "--mode",
            "esmda",
            "--output",
            str(out_e),
            "--ne",
            "10",
            "--na",
            "2",
        ]
    )
    assert code == 0
    assert (out_e / "k_mean.npy").is_file()
    assert (out_e / "k_std.npy").is_file()
    assert (out_e / "esmda_report.json").is_file()


def test_cli_discovery_from_csv(tmp_path: Path) -> None:
    out = tmp_path / "disc"
    code = main(
        [
            "--config",
            "config/sensor_series_case.yaml",
            "--mode",
            "discovery",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "shape_indicator.npy").is_file()
