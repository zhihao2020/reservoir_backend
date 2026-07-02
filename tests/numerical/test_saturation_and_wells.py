from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.wells import Well
from reservoir_backend.solver.material_balance import compute_constant_rate_well_balance
from reservoir_backend.solver.saturation_solver import advance_buckley_leverett_1d

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "regression" / "references"


def test_buckley_leverett_saturation_bounds() -> None:
    with (REFERENCE_DIR / "buckley_leverett_saturation_bounds.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    reference = np.load(REFERENCE_DIR / "buckley_leverett_saturation_bounds.npz")
    grid = Grid3D(**meta["grid"])

    result = advance_buckley_leverett_1d(
        grid=grid,
        sw=meta["initial_sw"],
        velocity_x=meta["velocity_m_s"],
        phi=meta["phi"],
        dt=meta["dt_s"],
        injected_sw=meta["injected_sw"],
        swi=meta["swi"],
        sor=meta["sor"],
    )

    sw = result.sw.values
    assert sw.shape == tuple(reference["sw"].shape)
    assert result.sw.unit == "fraction"
    assert np.allclose(sw, reference["sw"], rtol=1e-12, atol=1e-12)
    assert np.nanmin(sw) >= meta["swi"]
    assert np.nanmax(sw) <= 1.0 - meta["sor"]
    assert sw[0, 0, 0] > meta["initial_sw"]
    assert result.report["storage_change"] > 0.0
    assert result.report["relative_balance_error"] < 1e-12


def test_well_constant_rate_mass_balance() -> None:
    with (REFERENCE_DIR / "well_constant_rate_mass_balance.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    grid = Grid3D(**meta["grid"])
    wells = [
        Well("I1", "injection", grid, i=0, j=0, k=0, rate=meta["injection_rate_m3_s"]),
        Well("P1", "production", grid, i=grid.nx - 1, j=0, k=0, rate=meta["production_rate_m3_s"]),
    ]

    report = compute_constant_rate_well_balance(wells, dt=meta["dt_s"])

    assert report["injected_volume"] == meta["expected_injected_volume_m3"]
    assert report["produced_volume"] == meta["expected_produced_volume_m3"]
    assert report["net_volume"] == meta["expected_net_volume_m3"]
    assert report["relative_error"] <= meta["max_relative_error"]
    assert sum(well.signed_rate for well in wells) == 0.0
