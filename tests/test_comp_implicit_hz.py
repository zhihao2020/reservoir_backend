"""Implicit Newton HnP on the HZ streak well. Not FIM, not 1-inject-4-produce."""

import numpy as np

from reservoir_backend.comp import (
    K_STREAK_M2,
    accumulate_system,
    example_co2_rich_stream,
    example_horizontal_well,
    example_two_region_k,
    implicit_newton_step,
    run_horizontal_huff_and_puff_implicit,
)
from reservoir_backend.comp.cycle import INJECT_DAYS, PRODUCE_DAYS, SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_newton_drives_residual_down_on_hz_inject() -> None:
    """One implicit step from n_old: ||R|| drops (not an explicit predictor)."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    injectors, _ = example_horizontal_well(
        grid, streak, K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5
    )
    report = implicit_newton_step(
        fields, 250.0, p, mix, grid, k, dt=0.25 * SECONDS_PER_DAY, injectors=injectors
    )
    assert report.newton_converged
    assert len(report.residual_hist) >= 2
    assert report.residual_hist[0] > report.residual_hist[-1]
    assert report.residual_hist[-1] < 1.0e-6 * max(1.0, float(np.max(np.abs(fields.n))))
    assert report.n_newton >= 1


def test_implicit_hz_streak_cycle() -> None:
    """2×4 streak/matrix, 4-perf HZ well, implicit 2/2/3 cycle.

    Same connections inject then produce. Δn = injected − produced.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    assert len(streak) == 4
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    z_stream = example_co2_rich_stream(mix)
    injectors, producers = example_horizontal_well(
        grid, streak, K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5, z_stream=z_stream
    )
    assert [w.cell for w in injectors] == streak
    assert [w.cell for w in producers] == streak

    fields, cycle = run_horizontal_huff_and_puff_implicit(
        fields, 250.0, p, mix, grid, k, injectors, producers
    )
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert cycle.accepted_steps == len(dts)
    assert cycle.accepted_steps >= 1
    assert cycle.underflow is False
    assert min(dts) >= DT_MIN
    grew_or_held = any(dts[i + 1] + 1.0e-12 >= dts[i] for i in range(len(dts) - 1))
    assert grew_or_held

    hists = cycle.residual_hists or []
    driven = [h for h in hists if len(h) >= 2 and h[0] > h[-1]]
    assert driven, "Newton must reduce ||R|| on at least one accepted step"
    assert cycle.n_newton >= 1

    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-8,
        atol=1e-7,
    )
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.injected.sum(), expect_inj, atol=1e-6)
    assert np.isclose(cycle.produced.sum(), expect_prod, atol=1e-6)
