"""EXAMPLE 1 HZ injector + 4 HZ producers. Opposite wells shut. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    HZ_1INJ4PROD_CONTROL,
    accumulate_system,
    example_co2_rich_stream,
    example_hz_1inj4prod_layout,
    example_hz_1inj4prod_wells,
    example_two_region_k,
    implicit_newton_step_bhp,
    run_hz_1inj4prod_huff_and_puff,
)
from reservoir_backend.comp.cycle import SECONDS_PER_DAY
from reservoir_backend.comp.step import DT_MIN
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture

PRODUCE_BHP = 5.0e6 - 1.0


def _hz_1inj4prod_setup():
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid, inj_cells, prod_laterals, streak = example_hz_1inj4prod_layout()
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
    return mix, grid, inj_cells, prod_laterals, k, p, vp, fields, inj, prod


def test_hz_1inj4prod_control_is_documented() -> None:
    assert "1 horizontal injector + 4 horizontal producers" in HZ_1INJ4PROD_CONTROL
    assert "producers shut" in HZ_1INJ4PROD_CONTROL
    assert "injector shut" in HZ_1INJ4PROD_CONTROL
    assert "lateral" in HZ_1INJ4PROD_CONTROL
    assert "not field-validated" in HZ_1INJ4PROD_CONTROL


def test_hz_1inj4prod_layout_is_five_laterals() -> None:
    mix, grid, inj_cells, prod_laterals, k, p, vp, fields, inj, prod = _hz_1inj4prod_setup()
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert grid.nx == 2 and grid.ny == 5 and grid.nz == 1
    assert len(inj_cells) >= 2
    assert len(prod_laterals) == 4
    assert all(len(lat) >= 2 for lat in prod_laterals)
    assert len(inj) >= 2
    assert len(prod) == sum(len(lat) for lat in prod_laterals)
    inj_set = {w.cell for w in inj}
    prod_set = {w.cell for w in prod}
    assert inj_set == set(inj_cells)
    assert inj_set.isdisjoint(prod_set)
    assert all(w.bhp is None for w in inj)
    assert all(w.bhp == PRODUCE_BHP and w.molar_rate is None for w in prod)
    inj_rows = {grid.ijk(c)[1] for c in inj_cells}
    prod_rows = {grid.ijk(c)[1] for lat in prod_laterals for c in lat}
    assert len(inj_rows) == 1
    assert len(prod_rows) == 4
    assert inj_rows.isdisjoint(prod_rows)


def test_hz_1inj4prod_inject_pwf_unknown_and_drops_residual() -> None:
    """Inject: 4 HZ producers shut; p_wf is a Newton unknown; ||R|| drops."""
    mix, grid, inj_cells, prod_laterals, k, p0, vp, fields, inj, prod = _hz_1inj4prod_setup()
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


def test_hz_1inj4prod_produce_dirichlet_and_drops_residual() -> None:
    """Produce: HZ injector shut; 4 BHP laterals; p_wf not unknown; ||R|| drops."""
    mix, grid, inj_cells, prod_laterals, k, p0, vp, fields, inj, prod = _hz_1inj4prod_setup()
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


def test_hz_1inj4prod_cycle_opposite_wells_shut() -> None:
    """Short cycle: inject with prods shut, soak all shut, produce with inj shut."""
    mix, grid, inj_cells, prod_laterals, k, p0, vp, fields, inj, prod = _hz_1inj4prod_setup()
    assert len(inj) >= 2 and len(prod_laterals) == 4
    n0 = fields.n.copy()
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
