"""Specified-BHP (Dirichlet p_wf). Rate is not a Newton unknown. Not FIM."""

import numpy as np

from reservoir_backend.comp import (
    K_STREAK_M2,
    WELL_BHP_CONSTRAINT,
    accumulate_system,
    example_co2_rich_stream,
    example_horizontal_well,
    example_horizontal_well_bhp,
    example_two_region_k,
    implicit_newton_step_bhp,
    run_horizontal_huff_and_puff_bhp_spec,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid

# 1 Pa offset from the 5 MPa initial pressure: day-scale Peaceman EXAMPLE rates.
INJECT_BHP = 5.0e6 + 1.0
PRODUCE_BHP = 5.0e6 - 1.0


def _hz_setup_spec():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    inj, prod = example_horizontal_well_bhp(
        grid, streak, K_STREAK_M2, mix,
        inject_bhp=INJECT_BHP, produce_bhp=PRODUCE_BHP,
        z_stream=example_co2_rich_stream(mix),
    )
    return mix, grid, streak, k, p, vp, fields, inj, prod


def test_specified_bhp_constraint_is_documented() -> None:
    assert "Dirichlet" in WELL_BHP_CONSTRAINT
    assert "not a Newton unknown" in WELL_BHP_CONSTRAINT
    assert "q_PI" in WELL_BHP_CONSTRAINT or "Peaceman" in WELL_BHP_CONSTRAINT


def test_specified_bhp_pwf_is_not_a_newton_unknown() -> None:
    """Inject: p_wf is Dirichlet; unknowns are (n_i, p) only; ||R|| drops."""
    mix, grid, streak, k, p0, vp, fields, inj, _ = _hz_setup_spec()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    report = implicit_newton_step_bhp(
        fields, 250.0, p0, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, injectors=inj
    )
    assert report.has_pressure_unknown
    assert report.has_bhp_unknown is False
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1)
    assert report.bhp == INJECT_BHP
    assert report.newton_converged
    assert report.n_newton >= 1
    assert len(report.residual_hist) >= 2
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0


def test_rate_control_still_has_pwf_unknown() -> None:
    """Existing rate-control path: p_wf stays a Newton unknown."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    inj, _ = example_horizontal_well(
        grid, streak, K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5,
        z_stream=example_co2_rich_stream(mix),
    )
    report = implicit_newton_step_bhp(
        fields, 250.0, p, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, injectors=inj
    )
    assert report.has_bhp_unknown
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1) + 1


def test_specified_bhp_hz_cycle() -> None:
    """2×4 streak/matrix, 4-perf HZ well, specified-BHP 2/2/3 cycle."""
    mix, grid, streak, k, p0, vp, fields, inj, prod = _hz_setup_spec()
    n0 = fields.n.copy()
    fields, cycle = run_horizontal_huff_and_puff_bhp_spec(
        fields, 250.0, p0, mix, grid, k, inj, prod, vp
    )
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert cycle.accepted_steps == len(dts)
    assert cycle.accepted_steps >= 1
    assert cycle.underflow is False
    assert min(dts) >= DT_MIN
    assert cycle.n_newton >= 1
    hists = cycle.residual_hists or []
    driven = [h for h in hists if len(h) >= 2 and h[-1] < h[0] / 10.0]
    assert driven
    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-6,
        atol=1e-5,
    )
