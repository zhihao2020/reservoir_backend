from __future__ import annotations

import numpy as np
import pytest

from reservoir_backend.pipeline import (
    AxisAlignedBounds,
    BoundaryConditions,
    SensorSample,
    WellPoint,
    build_mesh,
    invert_rock_properties,
    reconstruct_pressure,
    reconstruct_saturation,
    run_time_slice,
)
from reservoir_backend.pipeline.point_workflow import (
    validate_exclusive_observers,
    run_point_first_slice,
)


def _mesh_and_sample():
    bounds = AxisAlignedBounds(0.0, 60.0, 0.0, 40.0, 0.0, 30.0)
    wells = [
        WellPoint("INJ", 10.0, 20.0, 15.0, role="injector"),
        WellPoint("PROD", 50.0, 20.0, 15.0, role="producer"),
        WellPoint("OBS_P", 30.0, 20.0, 15.0, role="observer_p"),
        WellPoint("OBS_S", 40.0, 25.0, 15.0, role="observer_s"),
    ]
    mesh = build_mesh(bounds, 10.0, 10.0, 10.0, wells=wells)
    sample = SensorSample(
        time=0.0,
        well_pressure={"INJ": 12.0e6, "PROD": 10.0e6, "OBS_P": 11.0e6},
        well_saturation={
            "INJ": (0.7, 0.3, 0.0),
            "PROD": (0.3, 0.7, 0.0),
            "OBS_S": (0.55, 0.45, 0.0),
        },
        boundary=BoundaryConditions(pressure={"left": 12.0e6, "right": 10.0e6}),
        well_rate={"INJ": 2.0e-5, "PROD": -2.0e-5},
    )
    return mesh, sample


def test_pressure_matches_pressure_sensors() -> None:
    mesh, sample = _mesh_and_sample()
    p, notes = reconstruct_pressure(mesh, sample)
    assert p.shape == mesh.grid.shape
    for name, value in sample.well_pressure.items():
        cell = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(cell)
        assert abs(p[k, j, i] - value) < 1.0e-6
    assert mesh.well_role["OBS_P"] == "observer_p"
    assert mesh.well_role["OBS_S"] == "observer_s"


def test_saturation_matches_sat_sensors() -> None:
    mesh, sample = _mesh_and_sample()
    sw, so, sg, notes = reconstruct_saturation(mesh, sample)
    assert np.allclose(sw + so + sg, 1.0, atol=1e-8)
    for name, phases in sample.well_saturation.items():
        cell = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(cell)
        assert abs(sw[k, j, i] - phases[0]) < 1e-8
    assert notes


def test_reject_probe_with_both_p_and_s() -> None:
    mesh, sample = _mesh_and_sample()
    bad = SensorSample(
        time=0.0,
        well_pressure={"INJ": 12e6, "OBS_P": 11e6},
        well_saturation={"INJ": (0.7, 0.3, 0.0), "OBS_P": (0.5, 0.5, 0.0)},
        boundary=BoundaryConditions(pressure={"left": 12e6, "right": 10e6}),
    )
    with pytest.raises(ValueError, match="pressure-only"):
        validate_exclusive_observers(mesh, bad)


def test_point_first_assigns_complementary_values() -> None:
    """OBS_P has no measured S but gets S from field; OBS_S gets p from field."""
    mesh, sample = _mesh_and_sample()
    fields = run_point_first_slice(mesh, sample, n_k_iterations=1, use_transport=False)
    assert any("point-first" in n for n in fields.notes)

    c_p = mesh.well_cell_id["OBS_P"]
    ip, jp, kp = mesh.grid.ijk(c_p)
    # measured p
    assert abs(fields.pressure[kp, jp, ip] - 11.0e6) < 1.0e-3
    # assigned S (not nan, in [0,1])
    assert 0.0 <= fields.sw[kp, jp, ip] <= 1.0

    c_s = mesh.well_cell_id["OBS_S"]
    is_, js, ks = mesh.grid.ijk(c_s)
    assert abs(fields.sw[ks, js, is_] - 0.55) < 1e-8
    assert fields.pressure[ks, js, is_] > 0.0

    # rock fields finite and positive
    assert np.all(fields.permeability > 0.0)
    assert np.all((fields.porosity > 0.0) & (fields.porosity < 1.0))
    assert any("spatial IDW" in n for n in fields.notes)


def test_property_inversion_positive() -> None:
    mesh, sample = _mesh_and_sample()
    p, _ = reconstruct_pressure(mesh, sample)
    sw, so, sg, _ = reconstruct_saturation(mesh, sample)
    k, phi, notes, fluxes = invert_rock_properties(mesh, p, sw, so, sg)
    assert k.shape == mesh.grid.shape
    assert np.all(k > 0.0)
    assert "flux_x" in fluxes


def test_run_time_slice_e2e() -> None:
    mesh, sample = _mesh_and_sample()
    fields = run_time_slice(mesh, sample)
    assert fields.pressure.shape == mesh.grid.shape
    assert np.allclose(fields.sw + fields.so + fields.sg, 1.0, atol=1e-8)
    assert any("point-first" in n for n in fields.notes)


def test_transport_between_times() -> None:
    mesh, sample0 = _mesh_and_sample()
    f0 = run_time_slice(mesh, sample0, n_k_iterations=1)
    sample1 = SensorSample(
        time=30.0,
        well_pressure={"INJ": 12.2e6, "PROD": 9.8e6, "OBS_P": 11.1e6},
        well_saturation={
            "INJ": (0.8, 0.2, 0.0),
            "PROD": (0.35, 0.65, 0.0),
            "OBS_S": (0.58, 0.42, 0.0),
        },
        boundary=BoundaryConditions(pressure={"left": 12.2e6, "right": 9.8e6}),
        well_rate={"INJ": 2.0e-5, "PROD": -2.0e-5},
    )
    f1 = run_time_slice(mesh, sample1, previous=f0, dt=30.0, n_k_iterations=1)
    assert any("fractional flow" in n or "transport" in n for n in f1.notes)
    assert np.allclose(f1.sw + f1.so + f1.sg, 1.0, atol=1e-8)


def test_fractional_flow_bounds() -> None:
    from reservoir_backend.pipeline.fractional_flow import water_fractional_flow

    assert water_fractional_flow(0.2) < 0.05
    assert water_fractional_flow(0.9) > 0.8
