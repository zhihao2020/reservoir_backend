"""Two-region EXAMPLE k: high-k streak vs low-k matrix. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    K_MATRIX_M2,
    K_STREAK_M2,
    accumulate_system,
    added_moles_per_pv,
    example_co2_rich_stream,
    example_drive_pressure,
    example_huff_n_puff_well,
    example_two_region_k,
    run_huff_and_puff,
)
from reservoir_backend.comp.cycle import INJECT_DAYS, SECONDS_PER_DAY, STEP_DAYS
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_example_two_region_k_is_documented_contrast() -> None:
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = [grid.index(i, 0, 0) for i in range(grid.nx)]
    k = example_two_region_k(grid, streak)
    assert K_MATRIX_M2 == 1.0e-18
    assert K_STREAK_M2 == 1.0e-12
    assert k[streak[0]] == K_STREAK_M2
    matrix = [c for c in range(grid.n_cells) if c not in streak]
    assert np.all(k[matrix] == K_MATRIX_M2)


def test_streak_takes_more_co2_per_pv_than_matrix() -> None:
    """2×4 closed grid, single-well inject+soak. Metric: mean added n_CO2 / Vp.

    Streak (j=0): k = 1e-12 m². Matrix (j=1): k = 1e-18 m². Well in the
    streak at (0,0). Prescribed Δp away from the well so both regions have
    a driving force; harmonic T sends CO2 along the streak, not the matrix.
    No produce: produced = 0 and Δn = injected.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    assert grid.nx == 4 and grid.ny == 2 and grid.n_cells == 8
    streak = np.array([grid.index(i, 0, 0) for i in range(grid.nx)], dtype=int)
    matrix = np.array([grid.index(i, 1, 0) for i in range(grid.nx)], dtype=int)
    well = int(streak[0])
    assert well not in set(matrix.tolist())

    k = example_two_region_k(grid, streak)
    p = example_drive_pressure(grid, well, p0=5.0e6, drop_pa=0.5)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()

    z_stream = example_co2_rich_stream(mix)
    assert z_stream[mix.names.index("CO2")] >= 0.8
    inj, prod = example_huff_n_puff_well(
        grid, well, K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5, z_stream=z_stream
    )
    assert inj.cell == prod.cell == well

    fields, cycle = run_huff_and_puff(
        fields, 250.0, p, mix, grid, k, inj, prod, produce_days=0.0
    )
    i_co2 = mix.names.index("CO2")

    assert cycle.underflow is False
    dts = cycle.inject.dt_used + cycle.soak.dt_used
    assert dts and min(dts) == STEP_DAYS * SECONDS_PER_DAY
    assert cycle.produce.dt_used == []
    assert np.allclose(cycle.produced, 0.0, atol=1e-15)

    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-10,
        atol=1e-9,
    )
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.injected.sum(), expect_inj, atol=1e-9)
    assert np.isclose(fields.n[:, i_co2].sum() - n0[:, i_co2].sum(), cycle.injected[i_co2], atol=1e-9)

    streak_dn_pv = added_moles_per_pv(fields.n, n0, vp, i_co2, streak)
    matrix_dn_pv = added_moles_per_pv(fields.n, n0, vp, i_co2, matrix)
    assert streak_dn_pv > matrix_dn_pv

    # Transport, not only the well-cell source: a downstream streak cell gains CO2.
    other_streak = streak[streak != well]
    other_dn = fields.n[other_streak, i_co2] - n0[other_streak, i_co2]
    assert other_dn.max() > 0.0
    other_mean = added_moles_per_pv(fields.n, n0, vp, i_co2, other_streak)
    assert other_mean > matrix_dn_pv
