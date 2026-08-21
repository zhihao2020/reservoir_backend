"""Day-scale inject–soak–produce for one EXAMPLE injector and one producer.

Documented first-cut schedule (not years, not 30-year, not 1-inject-4-produce):

    inject  2 days   rate-controlled EXAMPLE CO2, producer shut in
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


@dataclass
class CycleLedger:
    """Well ledger split by inject / soak / produce."""

    inject: WellLedger
    soak: WellLedger
    produce: WellLedger
    underflow: bool

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
) -> tuple[CompFields, CycleLedger]:
    """Run the documented 2 d / 2 d / 3 d EXAMPLE cycle (1 inj + 1 prod)."""
    common = dict(
        T=T,
        pressure=pressure,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        picard=picard,
        gravity=gravity,
    )
    n_inj, dt = _n_steps(inject_days, step_days)
    fields, led_inj = run_steps(
        fields, **common, dt=dt, n_steps=n_inj, injectors=(injector,), producers=None
    )
    n_soak, dt = _n_steps(soak_days, step_days)
    fields, led_soak = run_steps(
        fields, **common, dt=dt, n_steps=n_soak, injectors=None, producers=None
    )
    n_prod, dt = _n_steps(produce_days, step_days)
    fields, led_prod = run_steps(
        fields, **common, dt=dt, n_steps=n_prod, injectors=None, producers=(producer,)
    )
    underflow = led_inj.underflow or led_soak.underflow or led_prod.underflow
    return fields, CycleLedger(inject=led_inj, soak=led_soak, produce=led_prod, underflow=underflow)
