"""Day-scale inject–soak–produce (1 inj + 1 prod). Not FIM, not industrial-grade."""

import numpy as np

from reservoir_backend.comp import accumulate_system, example_producer, example_rate_injector
from reservoir_backend.comp.cycle import (
    INJECT_DAYS,
    PRODUCE_DAYS,
    SECONDS_PER_DAY,
    SOAK_DAYS,
    STEP_DAYS,
    run_inject_soak_produce,
)
from reservoir_backend.eos import EXAMPLE_LIBRARY_MARKER, example_eight_component_mixture
from reservoir_backend.grid.cartesian import CartesianGrid


def test_inject_soak_produce_day_scale_inventory() -> None:
    """2 d inject / 2 d soak / 3 d produce. EXAMPLE CO2–C1, one pair of wells.

    Soak does not inject. Produced CO2 > 0 after produce. Δn = injected − produced.
    Short Picard steps; dt is not chopped to zero.
    """
    mix = example_eight_component_mixture().subset(["C1", "CO2"])
    assert mix.marker == EXAMPLE_LIBRARY_MARKER
    grid = CartesianGrid.uniform((2.0, 1.0, 1.0), 1.0)
    z = np.array([[0.40, 0.60], [0.40, 0.60]])
    p = np.array([5.0e6, 5.0e6])
    vp = 0.2 * grid.cell_volumes()
    fields = accumulate_system(z, 250.0, p, mix, vp)
    n0 = fields.n.copy()
    k = 1.0e-12
    # Day-scale rates stay a fraction of cell inventory (~10² mol).
    inj = example_rate_injector(grid, 0, k, mix, rate=1.0e-4, stream="CO2")
    prod = example_producer(grid, 1, k, mix, molar_rate=5.0e-5)
    assert inj.cell != prod.cell

    fields, cycle = run_inject_soak_produce(fields, 250.0, p, mix, grid, k, inj, prod)
    i_co2 = mix.names.index("CO2")

    assert cycle.underflow is False
    assert min(cycle.inject.dt_used + cycle.soak.dt_used + cycle.produce.dt_used) == STEP_DAYS * SECONDS_PER_DAY

    # Soak: wells shut in — no injection, no production.
    assert np.allclose(cycle.soak.injected, 0.0, atol=1e-15)
    assert np.allclose(cycle.soak.produced, 0.0, atol=1e-15)

    # Inject period only adds the specified CO2 moles.
    expect_inj = 1.0e-4 * INJECT_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.inject.injected[i_co2], expect_inj, atol=1e-9)
    assert np.allclose(cycle.produce.injected, 0.0, atol=1e-15)

    # Produce period yields positive CO2; injector is off.
    assert cycle.produce.produced[i_co2] > 0.0
    expect_prod = 5.0e-5 * PRODUCE_DAYS * SECONDS_PER_DAY
    assert np.isclose(cycle.produce.produced.sum(), expect_prod, atol=1e-9)

    assert np.allclose(
        fields.n.sum(axis=0) - n0.sum(axis=0),
        cycle.injected - cycle.produced,
        atol=1e-9,
    )
    assert SOAK_DAYS == 2.0 and INJECT_DAYS == 2.0 and PRODUCE_DAYS == 3.0
