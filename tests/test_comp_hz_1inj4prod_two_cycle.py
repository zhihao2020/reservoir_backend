"""Two day-scale HZ 1+4 cycles on two-region EXAMPLE k. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    accumulate_system,
    example_co2_rich_stream,
    example_hz_1inj4prod_layout,
    example_hz_1inj4prod_wells,
    example_two_region_k,
    run_hz_1inj4prod_cycles,
)
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid

PRODUCE_BHP = 5.0e6 - 1.0


def _hz_1inj4prod_streak_setup():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((3.0, 5.0, 1.0), 1.0)
    grid, inj_cells, prod_laterals, streak = example_hz_1inj4prod_layout(
        grid, n_perf=2, streak="injector"
    )
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    inj, prod = example_hz_1inj4prod_wells(
        grid, inj_cells, prod_laterals, k, mix,
        inject_rate=1.0e-4, produce_bhp=PRODUCE_BHP,
        z_stream=example_co2_rich_stream(mix),
    )
    return mix, grid, k, p, vp, fields, inj, prod


def test_hz_1inj4prod_two_cycles_opposite_shut() -> None:
    """2× short inject–soak–produce. 1 HZ inj + 4 HZ prod. Opposite wells shut."""
    mix, grid, k, p0, vp, fields, inj, prod = _hz_1inj4prod_streak_setup()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    n0 = fields.n.copy()
    fields, multi = run_hz_1inj4prod_cycles(
        fields, 250.0, p0, mix, grid, k, inj, prod, vp, n_cycles=2
    )
    assert len(multi.cycles) == 2
    assert multi.underflow is False

    total_steps = 0
    all_dts: list[float] = []
    for rec in multi.cycles:
        led = rec.ledger
        dts = led.inject.dt_used + led.soak.dt_used + led.produce.dt_used
        all_dts.extend(dts)
        assert led.accepted_steps == len(dts)
        assert led.accepted_steps >= 1
        assert led.underflow is False
        assert led.inject_n_accepted >= 1
        assert led.produce_n_accepted >= 1
        np.testing.assert_allclose(led.inject.produced, 0.0, atol=1e-12)
        np.testing.assert_allclose(led.produce.injected, 0.0, atol=1e-12)
        np.testing.assert_allclose(led.soak.injected, 0.0, atol=1e-12)
        np.testing.assert_allclose(led.soak.produced, 0.0, atol=1e-12)
        r_inj0 = led.inject_residual_hists[0][0]
        r_inj1 = led.inject_residual_hists[0][-1]
        r_prod0 = led.produce_residual_hists[0][0]
        r_prod1 = led.produce_residual_hists[0][-1]
        assert r_inj1 < r_inj0 / 100.0
        assert r_prod1 < r_prod0 / 100.0
        np.testing.assert_allclose(
            rec.delta_n, led.injected - led.produced, rtol=1e-6, atol=1e-4
        )
        total_steps += led.accepted_steps

    assert min(all_dts) >= DT_MIN
    assert total_steps == sum(rec.ledger.accepted_steps for rec in multi.cycles)
    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        multi.injected - multi.produced,
        rtol=1e-6,
        atol=1e-4,
    )
