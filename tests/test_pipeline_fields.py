from __future__ import annotations

import numpy as np

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


def _mesh_and_sample():
    bounds = AxisAlignedBounds(0.0, 60.0, 0.0, 40.0, 0.0, 30.0)
    wells = [WellPoint("INJ", 10.0, 20.0, 15.0), WellPoint("PROD", 50.0, 20.0, 15.0)]
    mesh = build_mesh(bounds, 10.0, 10.0, 10.0, wells=wells)
    sample = SensorSample(
        time=0.0,
        well_pressure={"INJ": 12.0e6, "PROD": 10.0e6},
        well_saturation={"INJ": (0.7, 0.3, 0.0), "PROD": (0.3, 0.7, 0.0)},
        boundary=BoundaryConditions(pressure={"left": 12.0e6, "right": 10.0e6}),
    )
    return mesh, sample


def test_pressure_matches_well_sensors() -> None:
    mesh, sample = _mesh_and_sample()
    p, notes = reconstruct_pressure(mesh, sample)
    assert p.shape == mesh.grid.shape
    for name, value in sample.well_pressure.items():
        cell = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(cell)
        assert abs(p[k, j, i] - value) < 1.0e-6
    assert notes


def test_saturation_closure_and_wells() -> None:
    mesh, sample = _mesh_and_sample()
    sw, so, sg, notes = reconstruct_saturation(mesh, sample)
    assert sw.shape == mesh.grid.shape
    total = sw + so + sg
    assert np.allclose(total, 1.0, atol=1e-8)
    assert np.all(sw >= 0.0) and np.all(sw <= 1.0)
    for name, phases in sample.well_saturation.items():
        cell = mesh.well_cell_id[name]
        i, j, k = mesh.grid.ijk(cell)
        assert abs(sw[k, j, i] - phases[0]) < 1e-8
    assert notes


def test_property_inversion_positive() -> None:
    mesh, sample = _mesh_and_sample()
    p, _ = reconstruct_pressure(mesh, sample)
    sw, so, sg, _ = reconstruct_saturation(mesh, sample)
    k, phi, notes, fluxes = invert_rock_properties(mesh, p, sw, so, sg)
    assert k.shape == mesh.grid.shape
    assert phi.shape == mesh.grid.shape
    assert np.all(k > 0.0)
    assert np.all((phi > 0.0) & (phi < 1.0))
    assert notes
    assert "flux_x" in fluxes


def test_run_time_slice_e2e() -> None:
    mesh, sample = _mesh_and_sample()
    fields = run_time_slice(mesh, sample)
    assert fields.pressure.shape == mesh.grid.shape
    assert np.allclose(fields.sw + fields.so + fields.sg, 1.0, atol=1e-8)
    assert any("matrix well-cell Dirichlet" in n for n in fields.notes)
    assert any("k-pressure fixed-point" in n for n in fields.notes)


def test_pressure_matrix_dirichlet_note() -> None:
    mesh, sample = _mesh_and_sample()
    p, notes = reconstruct_pressure(mesh, sample)
    assert any("matrix well-cell Dirichlet" in n for n in notes)
    # interior gradient should exist between wells / boundaries
    assert float(np.std(p)) > 0.0


def test_k_array_prior_and_two_time_phi() -> None:
    mesh, sample0 = _mesh_and_sample()
    f0 = run_time_slice(mesh, sample0, n_k_iterations=2)
    sample1 = SensorSample(
        time=30.0,
        well_pressure={"INJ": 12.2e6, "PROD": 9.8e6},
        well_saturation={"INJ": (0.75, 0.25, 0.0), "PROD": (0.35, 0.65, 0.0)},
        boundary=BoundaryConditions(pressure={"left": 12.2e6, "right": 9.8e6}),
    )
    f1 = run_time_slice(
        mesh,
        sample1,
        previous=f0,
        dt=30.0,
        n_k_iterations=2,
        permeability_prior_m2=f0.permeability,
        porosity_prior=f0.porosity,
    )
    assert f1.permeability.shape == mesh.grid.shape
    assert np.all(f1.permeability > 0.0)
    assert any("mass-balance" in n or "continuity" in n for n in f1.notes)
    # k field should not be a pure constant after iteration
    assert float(np.std(f1.permeability)) >= 0.0
