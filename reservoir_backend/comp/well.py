"""Peaceman-style 1 injector + 1 producer for the standalone comp kernel.

Well index (vertical well, isotropic Cartesian cell, skin ``s``):

    r_e = 0.14 * sqrt(dx² + dy²)     Peaceman, SPEJ 1983
    WI  = 2 π k h / (ln(r_e / r_w) + s)

``k`` in m², ``h = dz`` in m, so ``WI`` is m³ (``WI λ Δp`` would be m³/s).
Injector: rate-controlled EXAMPLE stream. Producer: rate or BHP-style
outflow ``q = ξ_mix WI λ_t max(p − p_bhp, 0)`` with produced composition
equal to the well-cell molar phase mix. One injector and one producer —
not a 1-inject-4-produce pattern. Not industrial-grade, not GEM.
Do not import from ``solver/fi.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

# Peaceman (1983) isotropic equivalent-radius factor.
_PEACEMAN_RE = 0.14


@dataclass(frozen=True)
class RateInjector:
    """Injection into one cell.

    Rate control: ``rate`` is mol/s (>0) and ``bhp`` is None.
    Specified-BHP: ``bhp`` is set (Pa); ``rate`` is unused by the coupled
    Newton (Peaceman ``q(p_c, p_wf)`` is the mass source).
    """

    cell: int
    rate: float
    z_inj: NDArray[np.float64]
    well_index: float
    r_e: float
    r_w: float
    marker: str = ""
    bhp: float | None = None  # Pa; if set, specified-BHP (Dirichlet p_wf)


def peaceman_equivalent_radius(dx: float, dy: float) -> float:
    """``r_e = 0.14 sqrt(dx² + dy²)`` for an isotropic Cartesian cell."""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive (m)")
    return _PEACEMAN_RE * float(np.sqrt(dx * dx + dy * dy))


def peaceman_wi(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    *,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(WI [m³], r_e [m], r_w [m])`` for a vertical well in ``cell``.

    Default ``r_w`` is 10% of min(dx, dy), clipped below ``r_e``.
    """
    i, j, k = grid.ijk(int(cell))
    dx, dy, dz = float(grid.dx[i]), float(grid.dy[j]), float(grid.dz[k])
    r_e = peaceman_equivalent_radius(dx, dy)
    rw = float(r_w) if r_w is not None else 0.10 * min(dx, dy)
    if rw <= 0.0:
        raise ValueError("r_w must be positive (m)")
    if rw >= r_e:
        raise ValueError(f"r_w={rw} must be < Peaceman r_e={r_e}")
    denom = float(np.log(r_e / rw) + skin)
    if denom <= 0.0:
        raise ValueError("ln(r_e/r_w) + skin must be positive")
    if permeability < 0.0:
        raise ValueError("permeability must be non-negative (m²)")
    wi = 2.0 * np.pi * float(permeability) * dz / denom
    return float(wi), r_e, rw


def example_co2_rich_stream(mixture: EosMixture) -> NDArray[np.float64]:
    """EXAMPLE CO2-rich injectate: z_CO2=0.85, remainder C1. Not a field card."""
    if "CO2" not in mixture.names:
        raise ValueError("EXAMPLE stream needs CO2 in the mixture")
    z = np.zeros(mixture.n_components, dtype=float)
    z[mixture.names.index("CO2")] = 0.85
    if "C1" in mixture.names:
        z[mixture.names.index("C1")] = 0.15
    else:
        others = [i for i, name in enumerate(mixture.names) if name != "CO2"]
        if not others:
            z[mixture.names.index("CO2")] = 1.0
        else:
            z[others[0]] = 0.15
    return z / z.sum()


def example_rate_injector(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    mixture: EosMixture,
    *,
    rate: float,
    stream: str = "CO2",
    z_stream: NDArray[np.float64] | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
    bhp: float | None = None,
) -> RateInjector:
    """EXAMPLE injector: rate-controlled, or specified-BHP when ``bhp`` is set."""
    if rate < 0.0:
        raise ValueError("injection rate must be non-negative (mol/s)")
    if bhp is not None and float(bhp) <= 0.0:
        raise ValueError("injector bhp must be positive (Pa)")
    if z_stream is not None:
        z_inj = np.asarray(z_stream, dtype=float).ravel()
        if z_inj.size != mixture.n_components:
            raise ValueError("z_stream size != n_components")
        if float(z_inj.sum()) <= 0.0:
            raise ValueError("z_stream sums to zero")
        z_inj = z_inj / z_inj.sum()
    else:
        if stream not in mixture.names:
            raise ValueError(f"{stream!r} is not in the EXAMPLE mixture")
        z_inj = np.zeros(mixture.n_components, dtype=float)
        z_inj[mixture.names.index(stream)] = 1.0
    wi, r_e, rw = peaceman_wi(grid, cell, permeability, r_w=r_w, skin=skin)
    return RateInjector(
        cell=int(cell),
        rate=float(rate),
        z_inj=z_inj,
        well_index=wi,
        r_e=r_e,
        r_w=rw,
        marker=mixture.marker,
        bhp=None if bhp is None else float(bhp),
    )


def injection_moles(injector: RateInjector, dt: float) -> NDArray[np.float64]:
    """Component moles added in ``dt`` seconds."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    z = np.asarray(injector.z_inj, dtype=float)
    total = float(z.sum())
    if total <= 0.0:
        raise ValueError("injector stream sums to zero")
    return float(injector.rate) * float(dt) * (z / total)


@dataclass(frozen=True)
class RateProducer:
    """Outflow from one cell. Set ``molar_rate`` or ``bhp``, not both.

    Produced composition is the well-cell molar phase mix
    ``(ξ_L S_L x + ξ_V S_V y) / (ξ_L S_L + ξ_V S_V)``.
    """

    cell: int
    well_index: float
    r_e: float
    r_w: float
    marker: str = ""
    molar_rate: float | None = None  # mol/s
    bhp: float | None = None  # Pa


def well_cell_molar_z(cell: CellFlash) -> NDArray[np.float64]:
    """Overall molar composition of the flashed well cell (phase mix)."""
    num = cell.xi_liquid * cell.S_liquid * cell.x + cell.xi_vapor * cell.S_vapor * cell.y
    den = float(num.sum())
    if den <= 0.0:
        return cell.z.copy()
    return num / den


def example_horizontal_well(
    grid: CartesianGrid,
    cells: list[int] | tuple[int, ...],
    permeability: float,
    mixture: EosMixture,
    *,
    inject_rate: float,
    produce_rate: float,
    z_stream: NDArray[np.float64] | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[tuple[RateInjector, ...], tuple[RateProducer, ...]]:
    """One EXAMPLE horizontal well: several perforations, same cells inject then produce.

    Total inject/produce rates are split equally across connections.
    Not a 1-inject-4-produce pattern (no separate producers off the well).
    """
    perfs = tuple(int(c) for c in cells)
    if len(perfs) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if len(set(perfs)) != len(perfs):
        raise ValueError("horizontal well perforations must be unique cells")
    n_perf = len(perfs)
    z = example_co2_rich_stream(mixture) if z_stream is None else z_stream
    injectors = tuple(
        example_rate_injector(
            grid,
            cell,
            permeability,
            mixture,
            rate=float(inject_rate) / n_perf,
            z_stream=z,
            r_w=r_w,
            skin=skin,
        )
        for cell in perfs
    )
    producers = tuple(
        example_producer(
            grid,
            cell,
            permeability,
            mixture,
            molar_rate=float(produce_rate) / n_perf,
            r_w=r_w,
            skin=skin,
        )
        for cell in perfs
    )
    return injectors, producers


def example_horizontal_well_bhp(
    grid: CartesianGrid,
    cells: list[int] | tuple[int, ...],
    permeability: float,
    mixture: EosMixture,
    *,
    inject_bhp: float,
    produce_bhp: float,
    z_stream: NDArray[np.float64] | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[tuple[RateInjector, ...], tuple[RateProducer, ...]]:
    """One EXAMPLE horizontal well on specified BHP. Same cells inject then produce.

    ``p_wf`` is Dirichlet. Connection rate is Peaceman ``q(p_c, p_wf)``.
    Not a 1-inject-4-produce pattern.
    """
    perfs = tuple(int(c) for c in cells)
    if len(perfs) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if len(set(perfs)) != len(perfs):
        raise ValueError("horizontal well perforations must be unique cells")
    if float(inject_bhp) <= 0.0 or float(produce_bhp) <= 0.0:
        raise ValueError("specified BHP must be positive (Pa)")
    z = example_co2_rich_stream(mixture) if z_stream is None else z_stream
    injectors = tuple(
        example_rate_injector(
            grid,
            cell,
            permeability,
            mixture,
            rate=0.0,
            z_stream=z,
            r_w=r_w,
            skin=skin,
            bhp=float(inject_bhp),
        )
        for cell in perfs
    )
    producers = tuple(
        example_producer(
            grid,
            cell,
            permeability,
            mixture,
            bhp=float(produce_bhp),
            r_w=r_w,
            skin=skin,
        )
        for cell in perfs
    )
    return injectors, producers


def example_horizontal_well_mixed(
    grid: CartesianGrid,
    cells: list[int] | tuple[int, ...],
    permeability: float,
    mixture: EosMixture,
    *,
    inject_rate: float,
    produce_bhp: float,
    z_stream: NDArray[np.float64] | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[tuple[RateInjector, ...], tuple[RateProducer, ...]]:
    """One EXAMPLE HZ well: rate-controlled inject, specified-BHP produce.

    Same perforated cells. Not a 1-inject-4-produce pattern.
    Inject: ``p_wf`` is a Newton unknown. Produce: ``p_wf`` is Dirichlet.
    """
    perfs = tuple(int(c) for c in cells)
    if len(perfs) < 2:
        raise ValueError("horizontal well needs at least two perforations")
    if len(set(perfs)) != len(perfs):
        raise ValueError("horizontal well perforations must be unique cells")
    if float(inject_rate) < 0.0:
        raise ValueError("injection rate must be non-negative (mol/s)")
    if float(produce_bhp) <= 0.0:
        raise ValueError("produce BHP must be positive (Pa)")
    n_perf = len(perfs)
    z = example_co2_rich_stream(mixture) if z_stream is None else z_stream
    injectors = tuple(
        example_rate_injector(
            grid,
            cell,
            permeability,
            mixture,
            rate=float(inject_rate) / n_perf,
            z_stream=z,
            r_w=r_w,
            skin=skin,
        )
        for cell in perfs
    )
    producers = tuple(
        example_producer(
            grid,
            cell,
            permeability,
            mixture,
            bhp=float(produce_bhp),
            r_w=r_w,
            skin=skin,
        )
        for cell in perfs
    )
    return injectors, producers


def example_huff_n_puff_well(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    mixture: EosMixture,
    *,
    inject_rate: float,
    produce_rate: float,
    z_stream: NDArray[np.float64] | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[RateInjector, RateProducer]:
    """One EXAMPLE well used first as injector, then as producer (huff-n-puff).

    Default injectate is the CO2-rich EXAMPLE stream. Same cell, same WI.
    Not a 1-inj + 1-prod pair.
    """
    z = example_co2_rich_stream(mixture) if z_stream is None else z_stream
    inj = example_rate_injector(
        grid, cell, permeability, mixture, rate=inject_rate, z_stream=z, r_w=r_w, skin=skin
    )
    prod = example_producer(
        grid, cell, permeability, mixture, molar_rate=produce_rate, r_w=r_w, skin=skin
    )
    return inj, prod


def example_producer(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    mixture: EosMixture,
    *,
    molar_rate: float | None = None,
    bhp: float | None = None,
    r_w: float | None = None,
    skin: float = 0.0,
) -> RateProducer:
    """Single EXAMPLE producer (rate or BHP). Not a multi-well pattern."""
    if (molar_rate is None) == (bhp is None):
        raise ValueError("set exactly one of molar_rate (mol/s) or bhp (Pa)")
    if molar_rate is not None and molar_rate < 0.0:
        raise ValueError("producer molar_rate must be non-negative (mol/s)")
    wi, r_e, rw = peaceman_wi(grid, cell, permeability, r_w=r_w, skin=skin)
    return RateProducer(
        cell=int(cell),
        well_index=wi,
        r_e=r_e,
        r_w=rw,
        marker=mixture.marker,
        molar_rate=None if molar_rate is None else float(molar_rate),
        bhp=None if bhp is None else float(bhp),
    )


def production_moles(
    producer: RateProducer,
    cell: CellFlash,
    p_cell: float,
    dt: float,
    *,
    mu_liquid: float,
    mu_vapor: float,
) -> NDArray[np.float64]:
    """Component moles removed in ``dt`` seconds (uncapped)."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    z_prod = well_cell_molar_z(cell)
    if producer.molar_rate is not None:
        q = float(producer.molar_rate)
    else:
        if producer.bhp is None:
            raise ValueError("producer needs molar_rate or bhp")
        xi_mix = cell.xi_liquid * cell.S_liquid + cell.xi_vapor * cell.S_vapor
        lam = max(cell.S_liquid, 0.0) / max(mu_liquid, 1.0e-30) + max(cell.S_vapor, 0.0) / max(
            mu_vapor, 1.0e-30
        )
        q_vol = producer.well_index * lam * max(float(p_cell) - float(producer.bhp), 0.0)
        q = xi_mix * q_vol
    return q * float(dt) * z_prod
