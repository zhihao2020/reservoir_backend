"""Implicit Newton mole step for single-well EXAMPLE HnP. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    accumulate_system,
    example_co2_rich_stream,
    example_huff_n_puff_well,
    implicit_newton_step,
    run_huff_and_puff_implicit,
)
from reservoir_backend.comp.cycle import INJECT_DAYS, PRODUCE_DAYS, SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_implicit_newton_equal_p_conserves_closed() -> None:
    """Equal-p two-cell, no wells: implicit step holds Σ n_i and grows nothing."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    report = implicit_newton_step(fields, 250.0, p, mix, grid, 1.0e-12, dt=0.25 * SECONDS_PER_DAY)
    assert report.newton_converged
    assert report.underflow is False
    assert np.allclose(report.fields.n.sum(axis=0), n0.sum(axis=0), atol=1e-9)


def test_implicit_hnp_cycle_mass_and_dt_grows() -> None:
    """One implicit 2/2/3 single-well HnP. dt grows or holds; Δn = inj − prod.

    Lagged-p Newton on n_i. EXAMPLE CO2-rich stream. Not 1-inject-4-produce.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    p_produce = np.array([5.0e6 - 0.5, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    k = 1.0e-12
    z_stream = example_co2_rich_stream(mix)
    inj, prod = example_huff_n_puff_well(
        grid, 0, k, mix, inject_rate=1.0e-4, produce_rate=5.0e-5, z_stream=z_stream
    )
    assert inj.cell == prod.cell

    fields, cycle = run_huff_and_puff_implicit(
        fields, 250.0, p, mix, grid, k, inj, prod, pressure_produce=p_produce
    )
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert cycle.accepted_steps == len(dts)
    assert cycle.accepted_steps >= 1
    assert cycle.underflow is False
    assert min(dts) >= DT_MIN
    # At least one accepted step grew or held dt (not only chops).
    grew_or_held = any(dts[i + 1] + 1.0e-12 >= dts[i] for i in range(len(dts) - 1))
    assert grew_or_held
    assert max(dts) > min(dts)

    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-8,
        atol=1e-6,
    )
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.injected.sum(), expect_inj, atol=1e-6)
    assert np.isclose(cycle.produced.sum(), expect_prod, atol=1e-6)
    assert cycle.produce.produced.sum() > 0.0
    assert np.allclose(cycle.soak.injected, 0.0, atol=1e-12)
    assert np.allclose(cycle.soak.produced, 0.0, atol=1e-12)
