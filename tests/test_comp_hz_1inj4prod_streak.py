"""HZ 1+4 with two-region EXAMPLE k. CO2 prefers the streak. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    HZ_1INJ4PROD_CONTROL,
    K_MATRIX_M2,
    K_STREAK_M2,
    accumulate_system,
    added_moles_per_pv,
    example_co2_rich_stream,
    example_hz_1inj4prod_layout,
    example_hz_1inj4prod_wells,
    example_two_region_k,
    moles_per_pv,
    run_hz_1inj4prod_huff_and_puff,
)
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid

PRODUCE_BHP = 5.0e6 - 1.0


def _hz_1inj4prod_streak_setup():
    """3×5: 2 perfs per lateral; injector row is the fracture streak."""
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((3.0, 5.0, 1.0), 1.0)
    grid, inj_cells, prod_laterals, streak = example_hz_1inj4prod_layout(
        grid, n_perf=2, streak="injector"
    )
    matrix = [c for c in range(grid.n_cells) if c not in set(streak)]
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
    return mix, grid, inj_cells, prod_laterals, streak, matrix, k, p, vp, fields, inj, prod


def test_hz_1inj4prod_two_region_k_is_documented() -> None:
    assert "1e-18" in HZ_1INJ4PROD_CONTROL
    assert "1e-12" in HZ_1INJ4PROD_CONTROL
    assert K_MATRIX_M2 == 1.0e-18
    assert K_STREAK_M2 == 1.0e-12
    mix, grid, inj_cells, prod_laterals, streak, matrix, k, p, vp, fields, inj, prod = (
        _hz_1inj4prod_streak_setup()
    )
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert grid.nx == 3 and grid.ny == 5
    assert len(inj) >= 2 and len(prod_laterals) == 4
    assert len(streak) > len(inj_cells)
    assert matrix
    assert set(inj_cells).isdisjoint(matrix)
    np.testing.assert_array_equal(k[np.asarray(streak)], K_STREAK_M2)
    np.testing.assert_array_equal(k[np.asarray(matrix)], K_MATRIX_M2)


def test_hz_1inj4prod_streak_takes_more_co2_than_matrix() -> None:
    """After inject–soak–produce: added n_CO2/Vp on the streak ≫ matrix."""
    mix, grid, inj_cells, prod_laterals, streak, matrix, k, p0, vp, fields, inj, prod = (
        _hz_1inj4prod_streak_setup()
    )
    n0 = fields.n.copy()
    i_co2 = mix.names.index("CO2")
    fields, cycle = run_hz_1inj4prod_huff_and_puff(
        fields, 250.0, p0, mix, grid, k, inj, prod, vp
    )
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert cycle.accepted_steps == len(dts)
    assert cycle.accepted_steps >= 1
    assert cycle.underflow is False
    assert min(dts) >= DT_MIN
    np.testing.assert_allclose(cycle.inject.produced, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.produce.injected, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.soak.injected, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.soak.produced, 0.0, atol=1e-12)

    r_inj0 = cycle.inject_residual_hists[0][0]
    r_inj1 = cycle.inject_residual_hists[0][-1]
    r_prod0 = cycle.produce_residual_hists[0][0]
    r_prod1 = cycle.produce_residual_hists[0][-1]
    assert r_inj1 < r_inj0 / 100.0
    assert r_prod1 < r_prod0 / 100.0

    streak_n_vp = moles_per_pv(fields.n, vp, i_co2, streak)
    matrix_n_vp = moles_per_pv(fields.n, vp, i_co2, matrix)
    streak_dn_vp = added_moles_per_pv(fields.n, n0, vp, i_co2, streak)
    matrix_dn_vp = added_moles_per_pv(fields.n, n0, vp, i_co2, matrix)
    assert streak_n_vp > matrix_n_vp
    assert streak_dn_vp > matrix_dn_vp
    assert streak_dn_vp > 10.0 * max(matrix_dn_vp, 0.0)

    toe = [c for c in streak if c not in set(inj_cells)]
    assert toe
    assert float(np.max(fields.n[np.asarray(toe), i_co2] - n0[np.asarray(toe), i_co2])) > 0.0
