"""Two day-scale single-well HnP cycles on streak/matrix k. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    K_STREAK_M2,
    accumulate_system,
    added_moles_per_pv,
    example_co2_rich_stream,
    example_drive_pressure,
    example_huff_n_puff_well,
    example_two_region_k,
    run_huff_and_puff_cycles,
)
from reservoir_backend.comp.cycle import (
    HNP_WELLHEAD_Z_DEFINITION,
    INJECT_DAYS,
    PRODUCE_DAYS,
    SECONDS_PER_DAY,
    STEP_DAYS,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_two_hnp_cycles_z_and_mass() -> None:
    """2× (2 d inject / 2 d soak / 3 d produce) on one well.

    Wellhead metric: well-cell overall z_CO2. Each cycle: rises after
    inject, falls from that cycle's peak after produce. Per-cycle and
    cumulative Δn_i = injected_i − produced_i.
    Inject/soak at equal p so the well-cell z signal is clean; produce
    uses a small well-sink Δp so leaner neighbor fluid enters.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((4.0, 2.0, 1.0), 1.0)
    streak = np.array([grid.index(i, 0, 0) for i in range(grid.nx)], dtype=int)
    matrix = np.array([grid.index(i, 1, 0) for i in range(grid.nx)], dtype=int)
    well = int(streak[0])
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    p_produce = example_drive_pressure(grid, well, p0=5.0e6, drop_pa=-0.5)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()

    z_stream = example_co2_rich_stream(mix)
    inj, prod = example_huff_n_puff_well(
        grid, well, K_STREAK_M2, mix, inject_rate=1.0e-4, produce_rate=5.0e-5, z_stream=z_stream
    )
    assert inj.cell == prod.cell == well

    fields, multi = run_huff_and_puff_cycles(
        fields, 250.0, p, mix, grid, k, inj, prod, n_cycles=2, pressure_produce=p_produce
    )
    i_co2 = mix.names.index("CO2")
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY

    assert len(multi.cycles) == 2
    assert multi.underflow is False

    for rec in multi.cycles:
        led = rec.ledger
        assert led.underflow is False
        dts = led.inject.dt_used + led.soak.dt_used + led.produce.dt_used
        assert min(dts) == STEP_DAYS * SECONDS_PER_DAY
        assert led.wellhead_z_definition == HNP_WELLHEAD_Z_DEFINITION
        assert led.z_co2_well_cell_after_inject > led.z_co2_well_cell_initial
        assert led.z_co2_well_cell_after_produce < led.z_co2_well_cell_after_inject
        np.testing.assert_allclose(rec.delta_n, led.injected - led.produced, rtol=1e-10, atol=1e-9)
        assert np.isclose(led.injected.sum(), expect_inj, atol=1e-9)
        assert np.isclose(led.produced.sum(), expect_prod, atol=1e-9)
        assert np.isfinite(led.z_co2_produced_stream)
        assert led.produce.produced[i_co2] > 0.0

    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        multi.injected - multi.produced,
        rtol=1e-10,
        atol=1e-9,
    )

    # Cheap streak preference: well sits in the streak, so added CO2/Vp is larger there.
    streak_dn_pv = added_moles_per_pv(fields.n, n0, vp, i_co2, streak)
    matrix_dn_pv = added_moles_per_pv(fields.n, n0, vp, i_co2, matrix)
    assert streak_dn_pv > matrix_dn_pv
