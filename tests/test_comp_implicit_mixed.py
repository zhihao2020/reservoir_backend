"""Mixed-control HZ cycle: rate inject, shut soak, specified-BHP produce."""

import numpy as np

from reservoir_backend.comp import (
    K_STREAK_M2,
    MIXED_CONTROL,
    accumulate_system,
    example_co2_rich_stream,
    example_horizontal_well_mixed,
    example_two_region_k,
    implicit_newton_step_bhp,
    run_horizontal_huff_and_puff_mixed,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid

PRODUCE_BHP = 5.0e6 - 1.0


def _hz_setup_mixed():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    inj, prod = example_horizontal_well_mixed(
        grid, streak, K_STREAK_M2, mix,
        inject_rate=1.0e-4, produce_bhp=PRODUCE_BHP,
        z_stream=example_co2_rich_stream(mix),
    )
    return mix, grid, streak, k, p, vp, fields, inj, prod


def test_mixed_control_is_documented() -> None:
    assert "rate control" in MIXED_CONTROL
    assert "specified-BHP" in MIXED_CONTROL
    assert "drops p_wf" in MIXED_CONTROL


def test_mixed_inject_has_pwf_unknown_and_drops_residual() -> None:
    """Inject: p_wf is a Newton unknown; ||R|| drops by decades."""
    mix, grid, streak, k, p0, vp, fields, inj, _ = _hz_setup_mixed()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    report = implicit_newton_step_bhp(
        fields, 250.0, p0, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, injectors=inj
    )
    assert report.has_bhp_unknown is True
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1) + 1
    assert report.newton_converged
    assert report.n_newton >= 1
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0


def test_mixed_produce_pwf_is_dirichlet_and_drops_residual() -> None:
    """Produce: p_wf is Dirichlet; unknowns are (n_i, p); ||R|| drops."""
    mix, grid, streak, k, p0, vp, fields, _, prod = _hz_setup_mixed()
    report = implicit_newton_step_bhp(
        fields, 250.0, p0, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, producers=prod
    )
    assert report.has_bhp_unknown is False
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1)
    assert report.bhp == PRODUCE_BHP
    assert report.newton_converged
    assert report.n_newton >= 1
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0


def test_mixed_control_hz_cycle() -> None:
    """2×4 streak/matrix, 4-perf HZ well, rate inject / BHP produce 2/2/3."""
    mix, grid, streak, k, p0, vp, fields, inj, prod = _hz_setup_mixed()
    n0 = fields.n.copy()
    fields, cycle = run_horizontal_huff_and_puff_mixed(
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
        atol=1e-4,
    )
