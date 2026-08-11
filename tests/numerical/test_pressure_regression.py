from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.solver.pressure_solver import (
    compute_hydrostatic_pressure,
    solve_steady_state_pressure_1d,
    solve_steady_state_pressure_2d_no_flow_y,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "regression" / "references"


def test_pressure_1d_linear_dirichlet() -> None:
    with (REFERENCE_DIR / "pressure_1d_linear_dirichlet.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    reference = np.load(REFERENCE_DIR / "pressure_1d_linear_dirichlet.npz")
    grid = Grid3D(**meta["grid"])

    result = solve_steady_state_pressure_1d(
        grid=grid,
        kx=meta["permeability_m2"],
        mu=meta["viscosity_pa_s"],
        left_pressure=meta["left_pressure_pa"],
        right_pressure=meta["right_pressure_pa"],
    )

    pressure = result.pressure.values
    assert pressure.shape == tuple(reference["pressure"].shape)
    assert result.pressure.unit == "Pa"
    assert np.allclose(pressure, reference["pressure"], rtol=1e-11, atol=1e-3)
    assert np.all(np.diff(pressure[0, 0, :]) < 0.0)

    transmissibility = (
        2.0
        * meta["permeability_m2"]
        * float(grid.dy[0])
        * float(grid.dz[0])
        / (meta["viscosity_pa_s"] * float(grid.dx[0]))
    )
    left_flux = transmissibility * (meta["left_pressure_pa"] - pressure[0, 0, 0])
    right_flux = transmissibility * (pressure[0, 0, -1] - meta["right_pressure_pa"])
    assert abs(left_flux - right_flux) / max(abs(left_flux), 1.0) < 1e-12


def test_pressure_2d_no_flow_boundaries() -> None:
    with (REFERENCE_DIR / "pressure_2d_no_flow_boundaries.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    reference = np.load(REFERENCE_DIR / "pressure_2d_no_flow_boundaries.npz")
    grid = Grid3D(**meta["grid"])

    result = solve_steady_state_pressure_2d_no_flow_y(
        grid=grid,
        kx=meta["kx_m2"],
        ky=meta["ky_m2"],
        mu=meta["viscosity_pa_s"],
        left_pressure=meta["left_pressure_pa"],
        right_pressure=meta["right_pressure_pa"],
    )

    pressure = result.pressure.values
    assert pressure.shape == tuple(reference["pressure"].shape)
    assert result.pressure.unit == "Pa"
    assert result.report["boundary_y"] == "no_flow"
    assert np.allclose(pressure, reference["pressure"], rtol=1e-11, atol=1e-3)
    assert np.all(np.diff(pressure, axis=2) < 0.0)
    assert np.allclose(np.diff(pressure, axis=1), 0.0, atol=1e-6)

    transmissibility = (
        2.0 * meta["kx_m2"] * float(grid.dy[0]) * float(grid.dz[0]) / (meta["viscosity_pa_s"] * float(grid.dx[0]))
    )
    left_flux = transmissibility * (meta["left_pressure_pa"] - pressure[0, :, 0])
    right_flux = transmissibility * (pressure[0, :, -1] - meta["right_pressure_pa"])
    assert np.allclose(left_flux, right_flux, rtol=1e-12, atol=1e-18)


def test_hydrostatic_pressure_with_gravity() -> None:
    with (REFERENCE_DIR / "hydrostatic_pressure_with_gravity.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    reference = np.load(REFERENCE_DIR / "hydrostatic_pressure_with_gravity.npz")
    grid = Grid3D(**meta["grid"])

    pressure = compute_hydrostatic_pressure(
        grid=grid,
        datum_pressure=meta["datum_pressure_pa"],
        density=meta["density_kg_m3"],
        gravity=meta["gravity_m_s2"],
        datum_depth=meta["datum_depth_m"],
    )

    assert pressure.values.shape == tuple(reference["pressure"].shape)
    assert pressure.unit == "Pa"
    assert np.allclose(pressure.values, reference["pressure"], rtol=0.0, atol=1e-8)
    assert np.all(np.diff(pressure.values[:, 0, 0]) > 0.0)
    expected_step = meta["density_kg_m3"] * meta["gravity_m_s2"] * float(grid.dz[0])
    assert np.allclose(np.diff(pressure.values[:, 0, 0]), expected_step)
