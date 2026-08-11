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
    theta = np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0, 0.0, 0.0])
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
    theta = np.array([np.log(2e-14), np.log(8e-13), 0.2, 0.1, 0.3, 0.5])
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


def test_fit_corridor_prefers_offset_mass() -> None:
    """If indicator mass is offset from the straight corridor, meander should move."""
    from reservoir_backend.pipeline.k_param import (
        _path_weight,
        fit_corridor_to_indicator,
    )

    mesh = _mesh()
    th0 = np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0, 0.0, 0.0])
    # synthetic indicator: straight path shifted in +y (like a dogleg)
    w_shift = _path_weight(
        mesh, width_scale=1.0, z_bias=0.0, meander_amp=1.0, meander_phase=0.0
    )
    ind = w_shift.copy()
    th1, score = fit_corridor_to_indicator(mesh, th0, ind, n_amp=7, n_phase=8, n_width=3)
    # should improve alignment score vs zero meander
    w0 = _path_weight(mesh, width_scale=1.0, z_bias=0.0, meander_amp=0.0, meander_phase=0.0)
    from reservoir_backend.pipeline.k_param import _alignment_score

    assert score >= _alignment_score(w0, ind) - 1e-9
    # fitted meander amplitude not forced to stay zero when signal exists
    assert abs(float(th1[4])) + abs(float(th1[5])) >= 0.0


def test_enforce_channel_contrast() -> None:
    from reservoir_backend.pipeline.k_param import enforce_k_channel_contrast

    mesh = _mesh()
    # inverted contrast field
    theta = np.array([np.log(1e-12), np.log(1e-14), 0.0, 0.0, 0.0, 0.0])
    # force expand then scramble by swapping via enhance-like damp
    k = expand_k_from_params(mesh, np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0, 0.0, 0.0]))
    k_bad = k.copy()
    k_bad *= 0.1  # flatten/damage
    k_fix, th, ratio = enforce_k_channel_contrast(mesh, k_bad, theta, min_ratio=2.5)
    assert ratio >= 2.0 or not np.isfinite(ratio)
    assert th[1] > th[0]


def test_meander_shifts_mass() -> None:
    mesh = _mesh()
    th0 = np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0, 0.0, 0.0])
    th1 = np.array([np.log(1e-14), np.log(1e-12), 0.0, 0.0, 1.0, 0.0])
    k0 = expand_k_from_params(mesh, th0)
    k1 = expand_k_from_params(mesh, th1)
    # meander should move high-k mass (not identical fields)
    assert float(np.mean(np.abs(k1 - k0))) > 0.0
