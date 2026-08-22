"""Day-scale inject–soak–produce for EXAMPLE wells.

Documented first-cut schedule (not years, not 30-year):

    inject  2 days   rate-controlled EXAMPLE stream (pure CO2 or CO2-rich mix)
    soak    2 days   well(s) shut in; flash + TPFA only
    produce 3 days   producer(s) on

Documented well patterns (do not mix them):

    1+1     one injector cell and a different producer cell
    HnP     one well: same cell injects, soaks, then produces
    HZ      one horizontal well: several streak perforations, same cells
            inject then produce
    1+4     one injector + four producers (five-spot EXAMPLE); opposite
            wells shut (producers off while injecting, injector off
            while producing)

``dt`` defaults to 0.5 day. Production is capped to available moles so
``dt`` is not chopped to zero. Standalone; not wired into FIM.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.implicit import run_implicit_period
from reservoir_backend.comp.implicit_bhp import run_implicit_period_bhp
from reservoir_backend.comp.implicit_p import run_implicit_period_np
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

HNP_WELLHEAD_Z_DEFINITION = (
    "single-well well-cell overall z (n_i / sum n_i in the huff-n-puff cell); "
    "produced-stream z = produce.produced / produce.produced.sum() "
    "(same well-cell molar phase mix, cycle-integrated)"
)

HZ_WELLHEAD_Z_DEFINITION = (
    "horizontal-well perforated-cell overall z "
    "(sum n_i over perforations / sum n); "
    "produced-stream z = produce.produced / produce.produced.sum()"
)

FIVE_SPOT_WELLHEAD_Z_DEFINITION = (
    "five-spot injector well-cell overall z; "
    "produced-stream z = sum of 4 producers' produced / total produced"
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
    accepted_steps: int = 0
    n_newton: int = 0
    n_chop: int = 0
    residual_hists: list[list[float]] | None = None
    inject_n_accepted: int = 0
    produce_n_accepted: int = 0
    inject_residual_hists: list[list[float]] | None = None
    produce_residual_hists: list[list[float]] | None = None

    @property
    def injected(self) -> NDArray[np.float64]:
        return self.inject.injected + self.soak.injected + self.produce.injected

    @property
    def produced(self) -> NDArray[np.float64]:
        return self.inject.produced + self.soak.produced + self.produce.produced


@dataclass
class CycleRecord:
    """One huff-n-puff cycle plus the mole inventory at its start and end."""

    ledger: CycleLedger
    n_start: NDArray[np.float64]
    n_end: NDArray[np.float64]

    @property
    def delta_n(self) -> NDArray[np.float64]:
        return self.n_end.sum(axis=0) - self.n_start.sum(axis=0)


@dataclass
class MultiCycleLedger:
    """Repeated single-well huff-n-puff. ``cycles`` is per-cycle, in order."""

    cycles: list[CycleRecord]
    underflow: bool

    @property
    def injected(self) -> NDArray[np.float64]:
        out = np.zeros_like(self.cycles[0].ledger.injected)
        for rec in self.cycles:
            out = out + rec.ledger.injected
        return out

    @property
    def produced(self) -> NDArray[np.float64]:
        out = np.zeros_like(self.cycles[0].ledger.produced)
        for rec in self.cycles:
            out = out + rec.ledger.produced
        return out


def _n_steps(days: float, step_days: float) -> tuple[int, float]:
    dt = float(step_days) * SECONDS_PER_DAY
    if float(days) <= 0.0:
        return 0, dt
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


def perforated_z_co2(
    fields: CompFields,
    mixture: EosMixture,
    cells: list[int] | tuple[int, ...],
) -> float:
    """Overall CO2 mole fraction of the perforated cells together."""
    idx = np.asarray(cells, dtype=int).ravel()
    n = np.asarray(fields.n[idx], dtype=float).sum(axis=0)
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


def run_huff_and_puff(
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
    """One-well huff-and-puff: inject, soak (shut in), produce. Same cell.

    Not a 1-inj + 1-prod pair and not 1-inject-4-produce. ``pressure_produce``
    is optional produce-period Δp so neighbor fluid can enter the well cell
    and well-cell z_CO2 can fall from the post-inject peak.
    """
    if int(injector.cell) != int(producer.cell):
        raise ValueError("huff-and-puff uses one well; injector.cell must equal producer.cell")
    fields, ledger = run_inject_soak_produce(
        fields,
        T,
        pressure,
        mixture,
        grid,
        permeability,
        injector,
        producer,
        inject_days=inject_days,
        soak_days=soak_days,
        produce_days=produce_days,
        step_days=step_days,
        picard=picard,
        gravity=gravity,
        pressure_produce=pressure_produce,
    )
    return fields, replace(ledger, wellhead_z_definition=HNP_WELLHEAD_Z_DEFINITION)


def run_huff_and_puff_cycles(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injector: RateInjector,
    producer: RateProducer,
    *,
    n_cycles: int = 2,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    step_days: float = STEP_DAYS,
    picard: bool = True,
    gravity: float = 0.0,
    pressure_produce: NDArray[np.float64] | float | None = None,
) -> tuple[CompFields, MultiCycleLedger]:
    """Repeat single-well huff-and-puff ``n_cycles`` times (default 2).

    Each cycle is the documented 2 d / 2 d / 3 d schedule. Per-cycle
    well-cell z_CO2 and ``Δn = injected − produced`` live on ``CycleRecord``.
    """
    if int(n_cycles) < 1:
        raise ValueError("n_cycles must be >= 1")
    records: list[CycleRecord] = []
    underflow = False
    current = fields
    for _ in range(int(n_cycles)):
        n_start = current.n.copy()
        current, ledger = run_huff_and_puff(
            current,
            T,
            pressure,
            mixture,
            grid,
            permeability,
            injector,
            producer,
            inject_days=inject_days,
            soak_days=soak_days,
            produce_days=produce_days,
            step_days=step_days,
            picard=picard,
            gravity=gravity,
            pressure_produce=pressure_produce,
        )
        records.append(CycleRecord(ledger=ledger, n_start=n_start, n_end=current.n.copy()))
        underflow = underflow or ledger.underflow
    return current, MultiCycleLedger(cycles=records, underflow=underflow)


def run_horizontal_huff_and_puff(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    step_days: float = STEP_DAYS,
    picard: bool = True,
    gravity: float = 0.0,
    pressure_produce: NDArray[np.float64] | float | None = None,
) -> tuple[CompFields, CycleLedger]:
    """One horizontal well: inject, soak, produce on the same perforated cells.

    Not a 1-inject-4-produce pattern. Connections must match and number > 1.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    inj_cells = tuple(int(w.cell) for w in inj)
    prod_cells = tuple(int(w.cell) for w in prod)
    if len(set(inj_cells)) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if sorted(inj_cells) != sorted(prod_cells):
        raise ValueError("horizontal HnP uses the same perforated cells for inject and produce")
    z0 = perforated_z_co2(fields, mixture, inj_cells)
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
        fields, pressure=pressure, **common, dt=dt, n_steps=n_inj, injectors=inj, producers=None
    )
    z_after_inject = perforated_z_co2(fields, mixture, inj_cells)
    n_soak, dt = _n_steps(soak_days, step_days)
    fields, led_soak = run_steps(
        fields, pressure=pressure, **common, dt=dt, n_steps=n_soak, injectors=None, producers=None
    )
    z_after_soak = perforated_z_co2(fields, mixture, inj_cells)
    n_prod, dt = _n_steps(produce_days, step_days)
    p_prod = pressure if pressure_produce is None else pressure_produce
    fields, led_prod = run_steps(
        fields, pressure=p_prod, **common, dt=dt, n_steps=n_prod, injectors=None, producers=prod
    )
    underflow = led_inj.underflow or led_soak.underflow or led_prod.underflow
    return fields, CycleLedger(
        inject=led_inj,
        soak=led_soak,
        produce=led_prod,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=perforated_z_co2(fields, mixture, inj_cells),
        z_co2_produced_stream=produced_stream_z_co2(led_prod, mixture),
        wellhead_z_definition=HZ_WELLHEAD_Z_DEFINITION,
    )


def run_huff_and_puff_implicit(
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
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
    pressure_produce: NDArray[np.float64] | float | None = None,
) -> tuple[CompFields, CycleLedger]:
    """Single-well HnP with implicit Newton steps (lagged p). Same 2/2/3 days.

    ``dt`` may grow or hold after an accepted Newton step; failed Newton
    chops ``dt``. Not 1-inject-4-produce. Not wired into FIM.
    """
    if int(injector.cell) != int(producer.cell):
        raise ValueError("huff-and-puff uses one well; injector.cell must equal producer.cell")
    z0 = injector_well_cell_z_co2(fields, mixture, injector.cell)
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        gravity=gravity,
        dt_init=dt_init,
        dt_max=dt_max,
    )
    fields, per_inj = run_implicit_period(
        fields,
        pressure=pressure,
        duration=float(inject_days) * SECONDS_PER_DAY,
        injectors=(injector,),
        producers=None,
        **common,
    )
    z_after_inject = injector_well_cell_z_co2(fields, mixture, injector.cell)
    fields, per_soak = run_implicit_period(
        fields,
        pressure=pressure,
        duration=float(soak_days) * SECONDS_PER_DAY,
        injectors=None,
        producers=None,
        **common,
    )
    z_after_soak = injector_well_cell_z_co2(fields, mixture, injector.cell)
    p_prod = pressure if pressure_produce is None else pressure_produce
    fields, per_prod = run_implicit_period(
        fields,
        pressure=p_prod,
        duration=float(produce_days) * SECONDS_PER_DAY,
        injectors=None,
        producers=(producer,),
        **common,
    )
    underflow = per_inj.underflow or per_soak.underflow or per_prod.underflow
    return fields, CycleLedger(
        inject=per_inj.ledger,
        soak=per_soak.ledger,
        produce=per_prod.ledger,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=injector_well_cell_z_co2(fields, mixture, injector.cell),
        z_co2_produced_stream=produced_stream_z_co2(per_prod.ledger, mixture),
        wellhead_z_definition=HNP_WELLHEAD_Z_DEFINITION,
        accepted_steps=per_inj.n_accepted + per_soak.n_accepted + per_prod.n_accepted,
        n_newton=per_inj.n_newton + per_soak.n_newton + per_prod.n_newton,
        n_chop=per_inj.n_chop + per_soak.n_chop + per_prod.n_chop,
        residual_hists=per_inj.residual_hists + per_soak.residual_hists + per_prod.residual_hists,
    )


def run_horizontal_huff_and_puff_implicit(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
    pressure_produce: NDArray[np.float64] | float | None = None,
) -> tuple[CompFields, CycleLedger]:
    """Horizontal-well HnP with implicit Newton (lagged p). Same 2/2/3 days.

    Same perforated cells inject then produce. Not 1-inject-4-produce.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    inj_cells = tuple(int(w.cell) for w in inj)
    prod_cells = tuple(int(w.cell) for w in prod)
    if len(set(inj_cells)) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if sorted(inj_cells) != sorted(prod_cells):
        raise ValueError("horizontal HnP uses the same perforated cells for inject and produce")
    z0 = perforated_z_co2(fields, mixture, inj_cells)
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        gravity=gravity,
        dt_init=dt_init,
        dt_max=dt_max,
    )
    fields, per_inj = run_implicit_period(
        fields,
        pressure=pressure,
        duration=float(inject_days) * SECONDS_PER_DAY,
        injectors=inj,
        producers=None,
        **common,
    )
    z_after_inject = perforated_z_co2(fields, mixture, inj_cells)
    fields, per_soak = run_implicit_period(
        fields,
        pressure=pressure,
        duration=float(soak_days) * SECONDS_PER_DAY,
        injectors=None,
        producers=None,
        **common,
    )
    z_after_soak = perforated_z_co2(fields, mixture, inj_cells)
    p_prod = pressure if pressure_produce is None else pressure_produce
    fields, per_prod = run_implicit_period(
        fields,
        pressure=p_prod,
        duration=float(produce_days) * SECONDS_PER_DAY,
        injectors=None,
        producers=prod,
        **common,
    )
    underflow = per_inj.underflow or per_soak.underflow or per_prod.underflow
    return fields, CycleLedger(
        inject=per_inj.ledger,
        soak=per_soak.ledger,
        produce=per_prod.ledger,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=perforated_z_co2(fields, mixture, inj_cells),
        z_co2_produced_stream=produced_stream_z_co2(per_prod.ledger, mixture),
        wellhead_z_definition=HZ_WELLHEAD_Z_DEFINITION,
        accepted_steps=per_inj.n_accepted + per_soak.n_accepted + per_prod.n_accepted,
        n_newton=per_inj.n_newton + per_soak.n_newton + per_prod.n_newton,
        n_chop=per_inj.n_chop + per_soak.n_chop + per_prod.n_chop,
        residual_hists=per_inj.residual_hists + per_soak.residual_hists + per_prod.residual_hists,
    )


def run_horizontal_huff_and_puff_np(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
) -> tuple[CompFields, CycleLedger]:
    """HZ HnP with coupled ``(n_i, p)`` Newton. ``T`` prescribed. Same 2/2/3.

    Pressure is a Newton unknown. Volume constraint: see
    ``reservoir_backend.comp.implicit_p.VOLUME_CONSTRAINT``.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    inj_cells = tuple(int(w.cell) for w in inj)
    prod_cells = tuple(int(w.cell) for w in prod)
    if len(set(inj_cells)) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if sorted(inj_cells) != sorted(prod_cells):
        raise ValueError("horizontal HnP uses the same perforated cells for inject and produce")
    z0 = perforated_z_co2(fields, mixture, inj_cells)
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    p = np.asarray(pressure, dtype=float).ravel()
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        gravity=gravity,
        dt_init=dt_init,
        dt_max=dt_max,
        pore_volume=pore_volume,
    )
    fields, per_inj = run_implicit_period_np(
        fields, pressure=p, duration=float(inject_days) * SECONDS_PER_DAY, injectors=inj, producers=None, **common
    )
    p = per_inj.pressure if per_inj.pressure is not None else p
    z_after_inject = perforated_z_co2(fields, mixture, inj_cells)
    fields, per_soak = run_implicit_period_np(
        fields, pressure=p, duration=float(soak_days) * SECONDS_PER_DAY, injectors=None, producers=None, **common
    )
    p = per_soak.pressure if per_soak.pressure is not None else p
    z_after_soak = perforated_z_co2(fields, mixture, inj_cells)
    fields, per_prod = run_implicit_period_np(
        fields, pressure=p, duration=float(produce_days) * SECONDS_PER_DAY, injectors=None, producers=prod, **common
    )
    underflow = per_inj.underflow or per_soak.underflow or per_prod.underflow
    return fields, CycleLedger(
        inject=per_inj.ledger,
        soak=per_soak.ledger,
        produce=per_prod.ledger,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=perforated_z_co2(fields, mixture, inj_cells),
        z_co2_produced_stream=produced_stream_z_co2(per_prod.ledger, mixture),
        wellhead_z_definition=HZ_WELLHEAD_Z_DEFINITION,
        accepted_steps=per_inj.n_accepted + per_soak.n_accepted + per_prod.n_accepted,
        n_newton=per_inj.n_newton + per_soak.n_newton + per_prod.n_newton,
        n_chop=per_inj.n_chop + per_soak.n_chop + per_prod.n_chop,
        residual_hists=per_inj.residual_hists + per_soak.residual_hists + per_prod.residual_hists,
    )


def run_horizontal_huff_and_puff_bhp(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
) -> tuple[CompFields, CycleLedger]:
    """HZ HnP with coupled ``(n_i, p, p_wf)`` Newton. ``T`` prescribed. Same 2/2/3.

    Rate control: ``p_wf`` is a Newton unknown (specified rate vs Peaceman).
    Soak shuts the well and drops ``p_wf``. See
    ``reservoir_backend.comp.implicit_bhp.WELL_RATE_CONSTRAINT``.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    inj_cells = tuple(int(w.cell) for w in inj)
    prod_cells = tuple(int(w.cell) for w in prod)
    if len(set(inj_cells)) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if sorted(inj_cells) != sorted(prod_cells):
        raise ValueError("horizontal HnP uses the same perforated cells for inject and produce")
    z0 = perforated_z_co2(fields, mixture, inj_cells)
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    p = np.asarray(pressure, dtype=float).ravel()
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        gravity=gravity,
        dt_init=dt_init,
        dt_max=dt_max,
        pore_volume=pore_volume,
    )
    fields, per_inj = run_implicit_period_bhp(
        fields, pressure=p, duration=float(inject_days) * SECONDS_PER_DAY, injectors=inj, producers=None, **common
    )
    p = per_inj.pressure if per_inj.pressure is not None else p
    z_after_inject = perforated_z_co2(fields, mixture, inj_cells)
    fields, per_soak = run_implicit_period_bhp(
        fields, pressure=p, duration=float(soak_days) * SECONDS_PER_DAY, injectors=None, producers=None, **common
    )
    p = per_soak.pressure if per_soak.pressure is not None else p
    z_after_soak = perforated_z_co2(fields, mixture, inj_cells)
    fields, per_prod = run_implicit_period_bhp(
        fields, pressure=p, duration=float(produce_days) * SECONDS_PER_DAY, injectors=None, producers=prod, **common
    )
    underflow = per_inj.underflow or per_soak.underflow or per_prod.underflow
    return fields, CycleLedger(
        inject=per_inj.ledger,
        soak=per_soak.ledger,
        produce=per_prod.ledger,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=perforated_z_co2(fields, mixture, inj_cells),
        z_co2_produced_stream=produced_stream_z_co2(per_prod.ledger, mixture),
        wellhead_z_definition=HZ_WELLHEAD_Z_DEFINITION,
        accepted_steps=per_inj.n_accepted + per_soak.n_accepted + per_prod.n_accepted,
        n_newton=per_inj.n_newton + per_soak.n_newton + per_prod.n_newton,
        n_chop=per_inj.n_chop + per_soak.n_chop + per_prod.n_chop,
        residual_hists=per_inj.residual_hists + per_soak.residual_hists + per_prod.residual_hists,
    )


def run_horizontal_huff_and_puff_bhp_spec(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
) -> tuple[CompFields, CycleLedger]:
    """HZ HnP with specified-BHP (Dirichlet ``p_wf``). Same 2/2/3.

    Unknowns are ``(n_i, p)`` only. Mass source is Peaceman
    ``q(p_c, p_wf_spec)``. Soak shuts the well. See
    ``reservoir_backend.comp.implicit_bhp.WELL_BHP_CONSTRAINT``.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    if any(getattr(w, "bhp", None) is None for w in inj):
        raise ValueError("specified-BHP cycle needs injector.bhp")
    if any(w.bhp is None or w.molar_rate is not None for w in prod):
        raise ValueError("specified-BHP cycle needs producer.bhp and no molar_rate")
    return run_horizontal_huff_and_puff_bhp(
        fields,
        T,
        pressure,
        mixture,
        grid,
        permeability,
        inj,
        prod,
        pore_volume,
        inject_days=inject_days,
        soak_days=soak_days,
        produce_days=produce_days,
        dt_init_days=dt_init_days,
        dt_max_days=dt_max_days,
        gravity=gravity,
    )


def run_horizontal_huff_and_puff_mixed(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.25,
    dt_max_days: float = 1.0,
    gravity: float = 0.0,
) -> tuple[CompFields, CycleLedger]:
    """HZ HnP mixed control: rate inject, shut soak, specified-BHP produce.

    Same 2/2/3 days, same perforated cells. See
    ``reservoir_backend.comp.implicit_bhp.MIXED_CONTROL``.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    if any(getattr(w, "bhp", None) is not None for w in inj):
        raise ValueError("mixed cycle inject is rate control; injector.bhp must be None")
    if any(w.bhp is None or w.molar_rate is not None for w in prod):
        raise ValueError("mixed cycle produce is specified-BHP; set producer.bhp and no molar_rate")
    return run_horizontal_huff_and_puff_bhp(
        fields,
        T,
        pressure,
        mixture,
        grid,
        permeability,
        inj,
        prod,
        pore_volume,
        inject_days=inject_days,
        soak_days=soak_days,
        produce_days=produce_days,
        dt_init_days=dt_init_days,
        dt_max_days=dt_max_days,
        gravity=gravity,
    )


def run_five_spot_huff_and_puff(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    inject_days: float = INJECT_DAYS,
    soak_days: float = SOAK_DAYS,
    produce_days: float = PRODUCE_DAYS,
    dt_init_days: float = 0.125,
    dt_max_days: float = 0.25,
    gravity: float = 0.0,
) -> tuple[CompFields, CycleLedger]:
    """1-inject-4-produce EXAMPLE cycle. Opposite wells shut. Same 2/2/3.

    Inject: specified-rate injector on, four producers shut.
    Soak: all shut.
    Produce: four specified-BHP producers on, injector shut.
    Produce BHP is 1 Pa below the post-soak producer-cell pressures
    (one Dirichlet value for all four). ``dt`` is held (grow=1) so the
    two-region 3×3 streak does not grow into a failed Newton. One residual
    ``(n_i, p)``; injector ``p_wf`` is a Newton unknown only while injecting.
    See ``reservoir_backend.comp.implicit_bhp.FIVE_SPOT_CONTROL``.
    """
    inj = tuple(injectors)
    prod = tuple(producers)
    if len(inj) != 1:
        raise ValueError("five-spot EXAMPLE needs exactly 1 injector")
    if len(prod) != 4:
        raise ValueError("five-spot EXAMPLE needs exactly 4 producers")
    inj_cells = {int(w.cell) for w in inj}
    prod_cells = tuple(int(w.cell) for w in prod)
    if inj_cells & set(prod_cells):
        raise ValueError("five-spot injector must be distinct from the 4 producers")
    if any(getattr(w, "bhp", None) is not None for w in inj):
        raise ValueError("five-spot inject is rate control; injector.bhp must be None")
    if any(w.bhp is None or w.molar_rate is not None for w in prod):
        raise ValueError("five-spot produce is specified-BHP; set producer.bhp and no molar_rate")
    z0 = injector_well_cell_z_co2(fields, mixture, inj[0].cell)
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    p = np.asarray(pressure, dtype=float).ravel()
    common = dict(
        T=T,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        gravity=gravity,
        dt_init=dt_init,
        dt_max=dt_max,
        pore_volume=pore_volume,
        grow=1.0,
    )
    # Inject: producers shut.
    fields, per_inj = run_implicit_period_bhp(
        fields, pressure=p, duration=float(inject_days) * SECONDS_PER_DAY, injectors=inj, producers=None, **common
    )
    p = per_inj.pressure if per_inj.pressure is not None else p
    z_after_inject = injector_well_cell_z_co2(fields, mixture, inj[0].cell)
    # Soak: all shut.
    fields, per_soak = run_implicit_period_bhp(
        fields, pressure=p, duration=float(soak_days) * SECONDS_PER_DAY, injectors=None, producers=None, **common
    )
    p = per_soak.pressure if per_soak.pressure is not None else p
    z_after_soak = injector_well_cell_z_co2(fields, mixture, inj[0].cell)
    # Specified BHP is 1 Pa below the current producer-cell pressures.
    # Keeping the initial-p BHP after rate inject draws ~300 Pa on the
    # streak and asks for more moles than a cell holds (Newton hangs).
    p_arr = np.asarray(p, dtype=float).ravel()
    p_prod = [float(p_arr[int(w.cell)] if p_arr.size > 1 else p_arr[0]) for w in prod]
    produce_wells = tuple(replace(w, bhp=min(p_prod) - 1.0) for w in prod)
    # Produce: injector shut.
    fields, per_prod = run_implicit_period_bhp(
        fields, pressure=p, duration=float(produce_days) * SECONDS_PER_DAY, injectors=None, producers=produce_wells, **common
    )
    underflow = per_inj.underflow or per_soak.underflow or per_prod.underflow
    return fields, CycleLedger(
        inject=per_inj.ledger,
        soak=per_soak.ledger,
        produce=per_prod.ledger,
        underflow=underflow,
        z_co2_well_cell_initial=z0,
        z_co2_well_cell_after_inject=z_after_inject,
        z_co2_well_cell_after_soak=z_after_soak,
        z_co2_well_cell_after_produce=injector_well_cell_z_co2(fields, mixture, inj[0].cell),
        z_co2_produced_stream=produced_stream_z_co2(per_prod.ledger, mixture),
        wellhead_z_definition=FIVE_SPOT_WELLHEAD_Z_DEFINITION,
        accepted_steps=per_inj.n_accepted + per_soak.n_accepted + per_prod.n_accepted,
        n_newton=per_inj.n_newton + per_soak.n_newton + per_prod.n_newton,
        n_chop=per_inj.n_chop + per_soak.n_chop + per_prod.n_chop,
        residual_hists=per_inj.residual_hists + per_soak.residual_hists + per_prod.residual_hists,
        inject_n_accepted=per_inj.n_accepted,
        produce_n_accepted=per_prod.n_accepted,
        inject_residual_hists=list(per_inj.residual_hists),
        produce_residual_hists=list(per_prod.residual_hists),
    )
