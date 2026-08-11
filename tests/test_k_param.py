"""Low-dimensional k parameterization tests."""

from __future__ import annotations

import numpy as np

from reservoir_backend.pipeline import AxisAlignedBounds, WellPoint, build_mesh
from reservoir_backend.pipeline.inversion import run_sensor_inversion
from reservoir_backend.pipeline.k_param import (
    N_K_PARAMS,
    default_k_param_prior,
    expand_k_from_params,
    project_k_to_params,
    sample_k_param_ensemble,
)
from reservoir_backend.pipeline.state import BoundaryConditions, SensorSample


def _mesh():
    bounds = AxisAlignedBounds(0.0, 100.0, 0.0, 60.0, 0.0, 30.0)
    wells = [
        WellPoint("INJ", 15.0, 30.0, 15.0, role="injector"),
        WellPoint("PROD", 85.0, 30.0, 15.0, role="producer"),
        WellPoint("OBS_P", 50.0, 40.0, 15.0, role="observer_p"),
        WellPoint("OBS_S", 50.0, 20.0, 15.0, role="observer_s"),
    ]
    return build_mesh(bounds, 10.0, 10.0, 10.0, wells=wells)


def test_expand_channel_higher_than_background() -> None:
    mesh = _mesh()
    theta = np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0])
    k = expand_k_from_params(mesh, theta)
    mid = k[:, :, k.shape[2] // 2]
    edge = k[:, :, 0]
    assert float(np.mean(mid)) > float(np.mean(edge))


def test_ensemble_shape() -> None:
    prior = default_k_param_prior(1e-13)
    ens = sample_k_param_ensemble(prior, ne=10, seed=1)
    assert ens.shape == (10, N_K_PARAMS)


def test_project_roundtrip_reasonable() -> None:
    mesh = _mesh()
    theta = np.array([np.log(2e-14), np.log(8e-13), 0.2, 0.1])
    k = expand_k_from_params(mesh, theta)
    th2 = project_k_to_params(mesh, k)
    assert th2[1] > th2[0]


def test_param_inversion_runs() -> None:
    mesh = _mesh()
    samples = [
        SensorSample(
            time=0.0,
            well_pressure={"INJ": 12e6, "PROD": 10e6, "OBS_P": 11e6},
            well_saturation={
                "INJ": (0.8, 0.2, 0.0),
                "PROD": (0.3, 0.7, 0.0),
                "OBS_S": (0.45, 0.55, 0.0),
            },
            boundary=BoundaryConditions(pressure={"left": 12e6, "right": 10e6}),
            well_rate={"INJ": 1e-5, "PROD": -8e-6},
        ),
        SensorSample(
            time=30.0,
            well_pressure={"INJ": 12.2e6, "PROD": 9.8e6, "OBS_P": 11.1e6},
            well_saturation={
                "INJ": (0.82, 0.18, 0.0),
                "PROD": (0.38, 0.62, 0.0),
                "OBS_S": (0.52, 0.48, 0.0),
            },
            boundary=BoundaryConditions(pressure={"left": 12.2e6, "right": 9.8e6}),
            well_rate={"INJ": 1e-5, "PROD": -8e-6},
        ),
    ]
    res = run_sensor_inversion(
        mesh, samples, ne=16, n_assimilations=3, max_times=4, n_k_iterations=1, seed=2
    )
    assert len(res.history) == 2
    assert res.k_mean.shape == mesh.grid.shape
    assert res.theta_mean.size == N_K_PARAMS
    assert res.theta_mean[1] >= res.theta_mean[0] - 1e-9
    assert any("param joint ES-MDA" in n for n in res.notes)
    assert np.all(res.k_mean > 0.0)
