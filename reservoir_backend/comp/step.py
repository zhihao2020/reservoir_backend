"""Explicit closed-domain component mole update (one Picard optional).

Not a pressure solver and not the FIM residual. Pressure and temperature
are prescribed; moles are transported, ``z`` is renormalized, cells re-flash.

Closed domain (no wells / no-flow outer faces): ``Σ_cells n_i`` is conserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash, component_moles, flash_cell
from reservoir_backend.comp.flux import EXAMPLE_MU_LIQUID, EXAMPLE_MU_VAPOR, interior_faces, phase_molar_flux
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid


@dataclass
class CompFields:
    """Per-cell compositional fields. ``z`` is ``(n_cells, n_comp)``."""

    z: NDArray[np.float64]
    n: NDArray[np.float64]
    cells: list[CellFlash]


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
) -> CompFields:
    """Advance cell moles by ``dt`` [s] on a closed Cartesian domain.

    If ``picard`` is true, fluxes are re-evaluated after one explicit
    predictor and averaged (one Picard correction).
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    n_cells = fields.n.shape[0]
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
    return _fields_from_moles(n_pred, T, p, mixture)
