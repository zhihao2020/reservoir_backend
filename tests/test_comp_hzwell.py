"""Horizontal EXAMPLE well along the streak. Not 1-inject-4-produce, not FIM."""

import numpy as np
import pytest

from reservoir_backend.comp import (
    K_STREAK_M2,
    accumulate_system,
    example_co2_rich_stream,
    example_horizontal_well,
    example_two_region_k,
    run_horizontal_huff_and_puff,
)
from reservoir_backend.comp.cycle import (
    HZ_WELLHEAD_Z_DEFINITION,
    INJECT_DAYS,
    PRODUCE_DAYS,
    SECONDS_PER_DAY,
    STEP_DAYS,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_horizontal_well_rejects_single_perforation() -> None:
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    with pytest.raises(ValueError, match="two perforations"):
        example_horizontal_well(grid, [0], K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5)


def test_horizontal_well_along_streak_one_cycle() -> None:
    """One HZ well: all streak cells perforated. 2/2/3 inject–soak–produce.

    Same connections inject then produce (not 1-inject-4-produce).
    Δn_i = injected_i − produced_i. dt is not chopped to zero.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    matrix = [grid.index(i, 1, 0) for i in range(grid.nx)]
    assert len(streak) > 1
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
    inj_cells = [w.cell for w in injectors]
    prod_cells = [w.cell for w in producers]
    assert len(injectors) > 1
    assert inj_cells == streak
    assert prod_cells == streak
    assert set(inj_cells).isdisjoint(matrix)
    assert len(set(inj_cells)) == len(inj_cells)
    assert all(w.marker == EXAMPLE_LIBRARY_MARKER for w in injectors)

    fields, cycle = run_horizontal_huff_and_puff(
        fields, 250.0, p, mix, grid, k, injectors, producers
    )
    i_co2 = mix.names.index("CO2")

    assert cycle.underflow is False
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert min(dts) == STEP_DAYS * SECONDS_PER_DAY
    assert cycle.wellhead_z_definition == HZ_WELLHEAD_Z_DEFINITION

    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-10,
        atol=1e-9,
    )
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.injected.sum(), expect_inj, atol=1e-9)
    assert np.isclose(cycle.produced.sum(), expect_prod, atol=1e-9)
    assert cycle.produce.produced[i_co2] > 0.0
    assert np.allclose(cycle.soak.injected, 0.0, atol=1e-15)
    assert np.allclose(cycle.soak.produced, 0.0, atol=1e-15)

    # Every perforation was a connection (equal-p: no TPFA), not a single-cell well.
    net = fields.n.sum(axis=1) - n0.sum(axis=1)
    assert np.all(net[streak] > 0.0)
    assert np.allclose(net[matrix], 0.0, atol=1e-12)
