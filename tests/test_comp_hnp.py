"""Single-well EXAMPLE huff-and-puff. Not a 1-inj + 1-prod pair, not FIM."""

import numpy as np
import pytest

from reservoir_backend.comp import (
    accumulate_system,
    example_co2_rich_stream,
    example_huff_n_puff_well,
    example_producer,
    example_rate_injector,
    run_huff_and_puff,
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


def test_huff_and_puff_rejects_two_wells() -> None:
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    k = 1.0e-12
    inj = example_rate_injector(grid, 0, k, mix, rate=1.0e-4, stream="CO2")
    prod = example_producer(grid, 1, k, mix, molar_rate=5.0e-5)
    with pytest.raises(ValueError, match="one well"):
        run_huff_and_puff(fields, 250.0, p, mix, grid, k, inj, prod)


def test_single_well_huff_and_puff_mass_and_z() -> None:
    """2/2/3 huff-n-puff on one well: CO2-rich EXAMPLE stream.

    Wellhead metric is the single well-cell overall z. After inject it
    rises; after produce it falls from that peak (small produce Δp lets
    leaner neighbor fluid enter). Δn_i = injected_i − produced_i.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    # Neighbor (no well) stays leaner; Δp drives it into the HnP cell.
    p_produce = np.array([5.0e6 - 0.5, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    k = 1.0e-12
    z_stream = example_co2_rich_stream(mix)
    assert z_stream[mix.names.index("CO2")] >= 0.8
    inj, prod = example_huff_n_puff_well(
        grid, 0, k, mix, inject_rate=1.0e-4, produce_rate=5.0e-5, z_stream=z_stream
    )
    assert inj.cell == prod.cell == 0
    assert inj.well_index == prod.well_index
    assert "EXAMPLE" in inj.marker

    fields, cycle = run_huff_and_puff(
        fields, 250.0, p, mix, grid, k, inj, prod, pressure_produce=p_produce
    )
    i_co2 = mix.names.index("CO2")

    assert cycle.underflow is False
    dts = cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used
    assert min(dts) == STEP_DAYS * SECONDS_PER_DAY
    assert cycle.wellhead_z_definition == HNP_WELLHEAD_Z_DEFINITION

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
    assert np.allclose(cycle.produce.injected, 0.0, atol=1e-15)

    assert cycle.produce.produced.sum() > 0.0
    assert np.isfinite(cycle.z_co2_produced_stream)
    assert 0.0 <= cycle.z_co2_produced_stream <= 1.0
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.produce.produced.sum(), expect_prod, atol=1e-9)

    assert cycle.z_co2_well_cell_after_inject > cycle.z_co2_well_cell_initial
    assert np.isclose(
        cycle.z_co2_well_cell_after_soak, cycle.z_co2_well_cell_after_inject, atol=1e-12
    )
    assert cycle.z_co2_well_cell_after_produce < cycle.z_co2_well_cell_after_inject
