"""Explicit component mole update (closed except optional 1 inj + 1 prod).

Not a pressure solver and not the FIM residual. Pressure and temperature
are prescribed; moles are transported, ``z`` is renormalized, cells re-flash.

Without wells, ``Σ_cells n_i`` is conserved. With wells, the change in
totals equals injected moles minus produced moles. Production is capped
to available cell moles so ``dt`` is not chopped to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash, component_moles, flash_cell
from reservoir_backend.comp.flux import EXAMPLE_MU_LIQUID, EXAMPLE_MU_VAPOR, interior_faces, phase_molar_flux
from reservoir_backend.comp.well import RateInjector, RateProducer, injection_moles, production_moles
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

# Below this, a step would be reported as underflow. We cap production
# instead of collapsing dt.
DT_MIN = 1.0e-18


@dataclass
class CompFields:
    """Per-cell compositional fields. ``z`` is ``(n_cells, n_comp)``."""

    z: NDArray[np.float64]
    n: NDArray[np.float64]
    cells: list[CellFlash]


@dataclass
class StepReport:
    """One explicit/Picard step, including well ledger and underflow flag."""

    fields: CompFields
    injected: NDArray[np.float64]
    produced: NDArray[np.float64]
    dt_used: float
    underflow: bool


@dataclass
class WellLedger:
    """Cumulative injection / production over a short run."""

    injected: NDArray[np.float64]
    produced: NDArray[np.float64]
    dt_used: list[float] = field(default_factory=list)
    underflow: bool = False


def accumulate_system(
    z: NDArray[np.float64],
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    pore_volume: NDArray[np.float64],
) -> CompFields:
    """Flash every cell and form ``n_i`` from the accumulation formula."""
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
    n_cells = z_arr.shape[0]
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    if z_arr.shape[1] != mixture.n_components:
        raise ValueError("z columns must match mixture.n_components")
    if p.size != n_cells or vp.size != n_cells:
        raise ValueError("pressure and pore_volume must match n_cells")
    cells = [flash_cell(z_arr[c], float(T), float(p[c]), mixture) for c in range(n_cells)]
    n = np.stack([component_moles(cells[c], float(vp[c])) for c in range(n_cells)], axis=0)
    z_from_n = np.array([cell.z for cell in cells], dtype=float)
    return CompFields(z=z_from_n, n=n, cells=cells)


def _apply_divergence(
    n: NDArray[np.float64],
    flux: NDArray[np.float64],
    faces,
    dt: float,
) -> NDArray[np.float64]:
    out = n.copy()
    for f_idx, face in enumerate(faces):
        q = flux[f_idx] * float(dt)
        out[face.left] -= q
        out[face.right] += q
    return out


def _fields_from_moles(
    n: NDArray[np.float64],
    T: float,
    pressure: NDArray[np.float64],
    mixture: EosMixture,
) -> CompFields:
    n_clip = np.clip(n, 0.0, None)
    totals = n_clip.sum(axis=1, keepdims=True)
    z = np.divide(n_clip, totals, out=np.zeros_like(n_clip), where=totals > 0.0)
    cells = [flash_cell(z[c], float(T), float(pressure[c]), mixture) for c in range(z.shape[0])]
    z = np.array([cell.z for cell in cells], dtype=float)
    return CompFields(z=z, n=n_clip, cells=cells)


def _apply_injectors(
    n: NDArray[np.float64],
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    out = n.copy()
    added_total = np.zeros(out.shape[1], dtype=float)
    n_cells, n_comp = out.shape
    for inj in injectors:
        if not 0 <= int(inj.cell) < n_cells:
            raise ValueError(f"injector cell {inj.cell} out of range")
        added = injection_moles(inj, dt)
        if added.size != n_comp:
            raise ValueError("injector stream size != n_components")
        out[int(inj.cell)] += added
        added_total += added
    return out, added_total


def _apply_producers(
    n: NDArray[np.float64],
    cells: list[CellFlash],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pressure: NDArray[np.float64],
    dt: float,
    *,
    mu_liquid: float,
    mu_vapor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove produced moles. Cap to the cell inventory; do not cut ``dt``."""
    out = n.copy()
    produced_total = np.zeros(out.shape[1], dtype=float)
    n_cells = out.shape[0]
    for prod in producers:
        c = int(prod.cell)
        if not 0 <= c < n_cells:
            raise ValueError(f"producer cell {prod.cell} out of range")
        raw = production_moles(
            prod, cells[c], float(pressure[c]), dt, mu_liquid=mu_liquid, mu_vapor=mu_vapor
        )
        available = np.clip(out[c], 0.0, None)
        taken = np.minimum(raw, available)
        out[c] -= taken
        produced_total += taken
    return out, produced_total


def step_once(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    dt: float,
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    picard: bool = False,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
) -> StepReport:
    """One explicit or Picard step. ``dt`` is never chopped to zero."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    underflow = bool(dt > 0.0 and dt < DT_MIN)
    n_cells = fields.n.shape[0]
    n_comp = fields.n.shape[1]
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    faces = interior_faces(grid, permeability)
    z_center = grid.cell_centers()[:, 2]
    flux = phase_molar_flux(
        faces,
        fields.cells,
        p,
        z_center,
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
    )
    n_pred = _apply_divergence(fields.n, flux, faces, dt)
    if picard and dt > 0.0 and faces:
        pred = _fields_from_moles(n_pred, T, p, mixture)
        flux2 = phase_molar_flux(
            faces,
            pred.cells,
            p,
            z_center,
            gravity=gravity,
            mu_liquid=mu_liquid,
            mu_vapor=mu_vapor,
        )
        n_pred = _apply_divergence(fields.n, 0.5 * (flux + flux2), faces, dt)
    injected = np.zeros(n_comp, dtype=float)
    produced = np.zeros(n_comp, dtype=float)
    if injectors:
        n_pred, injected = _apply_injectors(n_pred, injectors, dt)
    if producers:
        mid = _fields_from_moles(n_pred, T, p, mixture)
        n_pred, produced = _apply_producers(
            n_pred, mid.cells, producers, p, dt, mu_liquid=mu_liquid, mu_vapor=mu_vapor
        )
    fields_out = _fields_from_moles(n_pred, T, p, mixture)
    return StepReport(
        fields=fields_out,
        injected=injected,
        produced=produced,
        dt_used=float(dt),
        underflow=underflow,
    )


def explicit_step(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    dt: float,
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    picard: bool = False,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
) -> CompFields:
    """Advance cell moles by ``dt`` [s]. See :func:`step_once` for the ledger."""
    return step_once(
        fields,
        T,
        pressure,
        mixture,
        grid,
        permeability,
        dt,
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
        picard=picard,
        injectors=injectors,
        producers=producers,
    ).fields


def run_steps(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    dt: float,
    n_steps: int,
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    picard: bool = False,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
) -> tuple[CompFields, WellLedger]:
    """Run ``n_steps`` short steps. ``underflow`` is True only if ``dt < DT_MIN``."""
    n_comp = fields.n.shape[1]
    ledger = WellLedger(injected=np.zeros(n_comp, dtype=float), produced=np.zeros(n_comp, dtype=float))
    current = fields
    for _ in range(int(n_steps)):
        report = step_once(
            current,
            T,
            pressure,
            mixture,
            grid,
            permeability,
            dt,
            gravity=gravity,
            mu_liquid=mu_liquid,
            mu_vapor=mu_vapor,
            picard=picard,
            injectors=injectors,
            producers=producers,
        )
        current = report.fields
        ledger.injected += report.injected
        ledger.produced += report.produced
        ledger.dt_used.append(report.dt_used)
        ledger.underflow = ledger.underflow or report.underflow
    return current, ledger
