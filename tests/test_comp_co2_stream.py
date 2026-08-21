"""CO2-rich EXAMPLE injectate on the day-scale 1+1 cycle. Not FIM, not GEM."""

import numpy as np

from reservoir_backend.comp import (
    accumulate_system,
    example_co2_rich_stream,
    example_producer,
    example_rate_injector,
    well_cell_molar_z,
)
from reservoir_backend.comp.cycle import (
    INJECT_DAYS,
    PRODUCE_DAYS,
    SECONDS_PER_DAY,
    STEP_DAYS,
    WELLHEAD_Z_DEFINITION,
    run_inject_soak_produce,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_example_co2_rich_stream_is_labeled_and_rich() -> None:
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    z = example_co2_rich_stream(mix)
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    assert "EXAMPLE" in mix.marker
    assert z[mix.names.index("CO2")] >= 0.8
    assert np.isclose(z.sum(), 1.0, atol=1e-15)
    assert np.all(z >= 0.0)


def test_co2_rich_cycle_mass_and_wellhead_z() -> None:
    """2/2/3 cycle: CO2-rich EXAMPLE stream, per-component Δn, z_CO2 up then down.

    Wellhead metric is injector well-cell overall z (not produced-stream z).
    Produced-stream z_CO2 is defined on the produce ledger.
    A small produce-period Δp lets leaner neighbor fluid enter the injector
    cell so well-cell z_CO2 falls from the post-inject peak.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    # ~0.5 Pa: enough TPFA moles to dilute cell 0 without emptying cell 1.
    p_produce = np.array([5.0e6 - 0.5, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    k = 1.0e-12
    z_stream = example_co2_rich_stream(mix)
    assert z_stream[mix.names.index("CO2")] >= 0.8
    inj = example_rate_injector(grid, 0, k, mix, rate=1.0e-4, z_stream=z_stream)
    prod = example_producer(grid, 1, k, mix, molar_rate=5.0e-5)
    assert inj.cell != prod.cell
    assert "EXAMPLE" in inj.marker
    assert np.allclose(inj.z_inj, z_stream)

    fields, cycle = run_inject_soak_produce(
        fields, 250.0, p, mix, grid, k, inj, prod, pressure_produce=p_produce
    )
    i_co2 = mix.names.index("CO2")

    assert cycle.underflow is False
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert min(dts) == STEP_DAYS * SECONDS_PER_DAY
    assert cycle.wellhead_z_definition == WELLHEAD_Z_DEFINITION

    # Per-component mass conservation over the full cycle.
    np.testing.assert_allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        rtol=1e-10,
        atol=1e-9,
    )

    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.inject.injected.sum(), expect_inj, atol=1e-9)
    assert np.isclose(cycle.inject.injected[i_co2], expect_inj * z_stream[i_co2], atol=1e-9)
    assert np.allclose(cycle.soak.injected, 0.0, atol=1e-15)
    assert np.allclose(cycle.soak.produced, 0.0, atol=1e-15)

    # Produced-stream z is defined during produce (cycle-integrated).
    assert cycle.produce.produced.sum() > 0.0
    assert np.isfinite(cycle.z_co2_produced_stream)
    assert 0.0 <= cycle.z_co2_produced_stream <= 1.0
    z_prod_stream = cycle.produce.produced / cycle.produce.produced.sum()
    assert np.isclose(cycle.z_co2_produced_stream, z_prod_stream[i_co2], atol=1e-15)
    z_cell_prod = well_cell_molar_z(fields.cells[prod.cell])
    assert z_cell_prod[i_co2] >= 0.0

    # Injector well-cell overall z_CO2: rise after inject, fall after produce.
    assert cycle.z_co2_well_cell_after_inject > cycle.z_co2_well_cell_initial
    assert np.isclose(
        cycle.z_co2_well_cell_after_soak, cycle.z_co2_well_cell_after_inject, atol=1e-12
    )
    assert cycle.z_co2_well_cell_after_produce < cycle.z_co2_well_cell_after_inject
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.produce.produced.sum(), expect_prod, atol=1e-9)
