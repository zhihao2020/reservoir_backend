"""EXAMPLE 1 injector + 4 producers. Opposite wells shut. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    FIVE_SPOT_CONTROL,
    accumulate_system,
    example_co2_rich_stream,
    example_five_spot_layout,
    example_five_spot_wells,
    example_two_region_k,
    implicit_newton_step_bhp,
    run_five_spot_huff_and_puff,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture

PRODUCE_BHP = 5.0e6 - 1.0


def _five_spot_setup():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid, inj_cell, prod_cells, streak = example_five_spot_layout()
    k = example_two_region_k(grid, streak)
    p = np.full(grid.n_cells, 5.0e6)
    z = np.tile(np.array([0.40, 0.60]), (grid.n_cells, 1))
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    inj, prod = example_five_spot_wells(
        grid, inj_cell, prod_cells, k, mix,
        inject_rate=1.0e-4, produce_bhp=PRODUCE_BHP,
        z_stream=example_co2_rich_stream(mix),
    )
    return mix, grid, inj_cell, prod_cells, k, p, vp, fields, inj, prod


def test_five_spot_control_is_documented() -> None:
    assert "1 injector + 4 producers" in FIVE_SPOT_CONTROL
    assert "producers shut" in FIVE_SPOT_CONTROL
    assert "injector shut" in FIVE_SPOT_CONTROL


def test_five_spot_layout_is_1_plus_4() -> None:
    mix, grid, inj_cell, prod_cells, k, p, vp, fields, inj, prod = _five_spot_setup()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert len(inj) == 1
    assert len(prod) == 4
    assert inj[0].cell == inj_cell
    assert inj[0].cell not in {w.cell for w in prod}
    assert inj[0].bhp is None
    assert all(w.bhp == PRODUCE_BHP and w.molar_rate is None for w in prod)
    assert len(set(w.cell for w in prod)) == 4


def test_five_spot_inject_pwf_unknown_and_drops_residual() -> None:
    """Inject: producers shut; p_wf is a Newton unknown; ||R|| drops."""
    mix, grid, inj_cell, prod_cells, k, p0, vp, fields, inj, prod = _five_spot_setup()
    report = implicit_newton_step_bhp(
        fields, 250.0, p0, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, injectors=inj
    )
    assert report.has_bhp_unknown is True
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1) + 1
    assert report.newton_converged
    assert float(report.produced.sum()) == 0.0
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0


def test_five_spot_produce_dirichlet_and_drops_residual() -> None:
    """Produce: injector shut; 4 BHP producers; p_wf not unknown; ||R|| drops."""
    mix, grid, inj_cell, prod_cells, k, p0, vp, fields, inj, prod = _five_spot_setup()
    report = implicit_newton_step_bhp(
        fields, 250.0, p0, mix, grid, k, 0.25 * SECONDS_PER_DAY, vp, producers=prod
    )
    assert report.has_bhp_unknown is False
    assert report.n_unknowns == grid.n_cells * (mix.n_components + 1)
    assert report.bhp == PRODUCE_BHP
    assert report.newton_converged
    assert float(report.injected.sum()) == 0.0
    r0, r1 = report.residual_hist[0], report.residual_hist[-1]
    assert r0 > 0.0
    assert r1 < r0 / 100.0


def test_five_spot_cycle_opposite_wells_shut() -> None:
    """2/2/3: inject with prods shut, soak all shut, produce with inj shut."""
    mix, grid, inj_cell, prod_cells, k, p0, vp, fields, inj, prod = _five_spot_setup()
    assert len(inj) == 1 and len(prod) == 4
    n0 = fields.n.copy()
    fields, cycle = run_five_spot_huff_and_puff(
        fields, 250.0, p0, mix, grid, k, inj, prod, vp
    )
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert cycle.accepted_steps == len(dts)
    assert cycle.accepted_steps >= 1
    assert cycle.underflow is False
    assert min(dts) >= DT_MIN
    # Opposite wells shut: no production while injecting, no injection while producing.
    np.testing.assert_allclose(cycle.inject.produced, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.produce.injected, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.soak.injected, 0.0, atol=1e-12)
    np.testing.assert_allclose(cycle.soak.produced, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-6,
        atol=1e-4,
    )
    assert cycle.inject_n_accepted >= 1
    assert cycle.produce_n_accepted >= 1
    assert cycle.inject_residual_hists and cycle.produce_residual_hists
    r_inj0 = cycle.inject_residual_hists[0][0]
    r_inj1 = cycle.inject_residual_hists[0][-1]
    r_prod0 = cycle.produce_residual_hists[0][0]
    r_prod1 = cycle.produce_residual_hists[0][-1]
    assert r_inj1 < r_inj0 / 100.0
    assert r_prod1 < r_prod0 / 100.0
