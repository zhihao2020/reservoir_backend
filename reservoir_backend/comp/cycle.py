"""Day-scale inject–soak–produce for one EXAMPLE injector and one producer.

Documented first-cut schedule (not years, not 30-year, not 1-inject-4-produce):

    inject  2 days   rate-controlled EXAMPLE stream (pure CO2 or CO2-rich mix),
                     producer shut in
    soak    2 days   both wells shut in; flash + TPFA only
    produce 3 days   producer on, injector shut in

``dt`` defaults to 0.5 day. Production is capped to available moles so
``dt`` is not chopped to zero. Standalone; not wired into FIM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.step import CompFields, WellLedger, run_steps
from reservoir_backend.comp.well import RateInjector, RateProducer
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

SECONDS_PER_DAY = 86400.0
INJECT_DAYS = 2.0
SOAK_DAYS = 2.0
PRODUCE_DAYS = 3.0
STEP_DAYS = 0.5

# Wellhead z used by this cycle: injector well-cell overall composition.
# Produced-stream z is defined only on the produce ledger.
WELLHEAD_Z_DEFINITION = (
    "injector well-cell overall z (n_i / sum n_i in the injector cell); "
    "produced-stream z = produce.produced / produce.produced.sum() "
    "(well-cell molar phase mix of the producer, cycle-integrated)"
)


@dataclass
class CycleLedger:
    """Well ledger split by inject / soak / produce.

    ``z_co2_well_cell_*`` is the injector well-cell overall CO2 mole
    fraction. That is the wellhead-z definition for this standalone
    cycle. ``z_co2_produced_stream`` is the produce-period composition
    ``produced_CO2 / sum produced_i`` and is defined when production
    moles are positive.
    """

    inject: WellLedger
    soak: WellLedger
    produce: WellLedger
    underflow: bool
    z_co2_well_cell_initial: float
    z_co2_well_cell_after_inject: float
    z_co2_well_cell_after_soak: float
    z_co2_well_cell_after_produce: float
    z_co2_produced_stream: float
    wellhead_z_definition: str = WELLHEAD_Z_DEFINITION

    @property
    def injected(self) -> NDArray[np.float64]:
        return self.inject.injected + self.soak.injected + self.produce.injected

    @property
    def produced(self) -> NDArray[np.float64]:
        return self.inject.produced + self.soak.produced + self.produce.produced


def _n_steps(days: float, step_days: float) -> tuple[int, float]:
    dt = float(step_days) * SECONDS_PER_DAY
    n = max(1, int(round(float(days) / float(step_days))))
    return n, dt


def injector_well_cell_z_co2(
    fields: CompFields,
    mixture: EosMixture,
    cell: int,
) -> float:
    """Overall CO2 mole fraction in one cell (injector well-cell metric)."""
    n = np.asarray(fields.n[int(cell)], dtype=float)
    total = float(n.sum())
    if total <= 0.0:
        return float("nan")
    return float(n[list(mixture.names).index("CO2")] / total)


def produced_stream_z_co2(ledger: WellLedger, mixture: EosMixture) -> float:
    """Cycle-integrated produced-stream z_CO2; nan if nothing was produced."""
    total = float(ledger.produced.sum())
    if total <= 0.0:
        return float("nan")
    return float(ledger.produced[list(mixture.names).index("CO2")] / total)


def run_inject_soak_produce(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injector: RateInjector,
    producer: RateProducer,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    step_days: float = STEP_DAYS,
    picard: bool = True,
    gravity: float = 0.0,
    pressure_produce: NDArray[np.float64] | float | None = None,
) -> tuple[CompFields, CycleLedger]:
    """Run the documented 2 d / 2 d / 3 d EXAMPLE cycle (1 inj + 1 prod).

    ``pressure_produce`` is used only during produce. A small Δp can drive
    TPFA so the injector well-cell overall z can fall from the post-inject
    peak. Inject and soak keep ``pressure``.
    """
    z0 = injector_well_cell_z_co2(fields, mixture, injector.cell)
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        picard=picard,
        gravity=gravity,
    )
    n_inj, dt = _n_steps(inject_days, step_days)
    fields, led_inj = run_steps(
        fields,
        pressure=pressure,
        **common,
        dt=dt,
        n_steps=n_inj,
        injectors=(injector,),
        producers=None,
    )
    z_after_inject = injector_well_cell_z_co2(fields, mixture, injector.cell)
    n_soak, dt = _n_steps(soak_days, step_days)
    fields, led_soak = run_steps(
        fields,
        pressure=pressure,
        **common,
        dt=dt,
        n_steps=n_soak,
        injectors=None,
        producers=None,
    )
    z_after_soak = injector_well_cell_z_co2(fields, mixture, injector.cell)
    n_prod, dt = _n_steps(produce_days, step_days)
    p_prod = pressure if pressure_produce is None else pressure_produce
    fields, led_prod = run_steps(
        fields,
        pressure=p_prod,
        **common,
        dt=dt,
        n_steps=n_prod,
        injectors=None,
        producers=(producer,),
    )
    z_after_produce = injector_well_cell_z_co2(fields, mixture, injector.cell)
    underflow = led_inj.underflow or led_soak.underflow or led_prod.underflow
    return fields, CycleLedger(
        inject=led_inj,
        soak=led_soak,
        produce=led_prod,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=z_after_produce,
        z_co2_produced_stream=produced_stream_z_co2(led_prod, mixture),
    )
