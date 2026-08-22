"""Immiscible EXAMPLE aqueous phase for the standalone comp kernel.

Oil/gas still come from the existing Peng–Robinson flash
(``flash_cell``). Water is a third, immiscible phase: no HC in water,
no water in the HC flash. Capillary pressure is zero (all phases share
``p``). Relative permeabilities are linear in saturation (``k_{rα} = S_α``),
the same first-cut mobility already used for oil/gas.

Hydrocarbon saturations occupy the remaining pore:

    S_o = S_L^{flash} (1 − S_w),   S_g = S_V^{flash} (1 − S_w),
    S_o + S_g + S_w = 1.

Water accumulation (mol):

    n_w = V_pore ξ_w S_w

with constant EXAMPLE ``ξ_w`` (incompressible textbook water). HC moles
are the existing formula scaled by ``(1 − S_w)``.

Water TPFA (capillary-free, phase-potential upwind):

    Q_w = T ξ_w^{up} (S_w^{up} / μ_w) (Φ_{w,L} − Φ_{w,R})
    Φ_w = p + ρ_w g z

Conceptual rewrite of the published two-point / mobility-split idea
(open-source compositional codes such as Open-DARTS, GEOS, MRST). Does
not import ``references/``. Not a GEM aqueous card. Not wired into FIM.

ASSUMPTIONS (EXAMPLE, not field-validated):
- Immiscible water; no solubility, no vaporization, no salinity.
- No capillary pressure; single pressure for o/g/w.
- Linear relative permeability; no residual saturations.
- Incompressible water (constant ξ_w, ρ_w).
- Pressure remains prescribed on the explicit path (same as two-phase
  ``explicit_step``). Volume constraint is not solved here.
- Water properties are textbook 20 °C liquid water, labeled EXAMPLE.
  Not formation brine, not Jiyang / GEM aqueous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash, component_moles, flash_cell
from reservoir_backend.comp.flux import (
    EXAMPLE_MU_LIQUID,
    EXAMPLE_MU_VAPOR,
    InteriorFace,
    interior_faces,
    phase_molar_flux,
)
from reservoir_backend.comp.step import CompFields, _apply_divergence, _fields_from_moles
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

# Textbook liquid water near 20 °C, 1 atm (NIST Chemistry WebBook style).
# Not formation brine. Not a GEM / Jiyang aqueous card.
EXAMPLE_WATER_MW_KG_MOL = 0.01801528
EXAMPLE_WATER_RHO_KG_M3 = 998.2
EXAMPLE_WATER_XI_MOL_M3 = EXAMPLE_WATER_RHO_KG_M3 / EXAMPLE_WATER_MW_KG_MOL
EXAMPLE_WATER_MU_PA_S = 1.0e-3
EXAMPLE_AQUEOUS_MARKER = (
    "EXAMPLE aqueous (immiscible, no Pc, linear kr); "
    "textbook 20 C water; NOT a Jiyang GEM card; NOT site-calibrated"
)
EXAMPLE_AQUEOUS_ASSUMPTIONS = (
    "immiscible water; So+Sg+Sw=1 with So,Sg from PR flash scaled by (1-Sw); "
    "capillary-free TPFA mobility split; incompressible ξ_w; prescribed p"
)


@dataclass(frozen=True)
class ThreePhaseState:
    """HC flash plus immiscible water. Saturations satisfy So+Sg+Sw=1."""

    hc: CompFields
    n_water: NDArray[np.float64]
    s_water: NDArray[np.float64]
    s_oil: NDArray[np.float64]
    s_gas: NDArray[np.float64]
    marker: str = EXAMPLE_AQUEOUS_MARKER


def water_molar_density(_p: float | NDArray[np.float64] | None = None) -> float:
    """EXAMPLE incompressible ξ_w [mol/m³]. ``p`` accepted and ignored."""
    return float(EXAMPLE_WATER_XI_MOL_M3)


def water_moles(s_water: float, pore_volume: float) -> float:
    """``n_w = V_pore ξ_w S_w`` [mol]."""
    sw = float(s_water)
    if sw < 0.0 or sw > 1.0:
        raise ValueError("S_w must be in [0, 1]")
    if pore_volume < 0.0:
        raise ValueError("pore volume must be non-negative (m³)")
    return float(pore_volume) * water_molar_density() * sw


def three_phase_saturations(cell: CellFlash, s_water: float) -> tuple[float, float, float]:
    """``(S_o, S_g, S_w)`` with ``S_o + S_g + S_w = 1``.

    ``S_o = S_L (1−S_w)``, ``S_g = S_V (1−S_w)`` from the HC flash.
    """
    sw = float(s_water)
    if sw < 0.0 or sw > 1.0:
        raise ValueError("S_w must be in [0, 1]")
    so = float(cell.S_liquid) * (1.0 - sw)
    sg = float(cell.S_vapor) * (1.0 - sw)
    return so, sg, sw


def hydrocarbon_moles_with_water(cell: CellFlash, pore_volume: float, s_water: float) -> NDArray[np.float64]:
    """HC accumulation on the remaining pore: ``(1−S_w) V_p (ξ_L S_L x + ξ_V S_V y)``."""
    sw = float(s_water)
    if sw < 0.0 or sw > 1.0:
        raise ValueError("S_w must be in [0, 1]")
    return (1.0 - sw) * component_moles(cell, pore_volume)


def _saturations_from_cells(
    cells: list[CellFlash],
    s_water: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    so = np.empty(len(cells), dtype=float)
    sg = np.empty(len(cells), dtype=float)
    for i, cell in enumerate(cells):
        so[i], sg[i], _sw = three_phase_saturations(cell, float(s_water[i]))
    return so, sg


def accumulate_three_phase(
    z: NDArray[np.float64],
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    pore_volume: NDArray[np.float64],
    s_water: NDArray[np.float64] | float,
) -> ThreePhaseState:
    """Flash HC at ``(T, p)`` and pack immiscible water. EXAMPLE aqueous only."""
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim == 1:
        z_arr = z_arr.reshape(1, -1)
    n_cells = z_arr.shape[0]
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    sw = np.asarray(s_water, dtype=float).ravel()
    if sw.size == 1:
        sw = np.full(n_cells, float(sw[0]), dtype=float)
    if sw.size != n_cells or vp.size != n_cells or p.size != n_cells:
        raise ValueError("s_water, pressure, and pore_volume must match n_cells")
    if np.any(sw < 0.0) or np.any(sw > 1.0):
        raise ValueError("S_w must be in [0, 1]")
    cells = [flash_cell(z_arr[c], float(T), float(p[c]), mixture) for c in range(n_cells)]
    n_hc = np.stack(
        [hydrocarbon_moles_with_water(cells[c], float(vp[c]), float(sw[c])) for c in range(n_cells)],
        axis=0,
    )
    n_w = np.array([water_moles(float(sw[c]), float(vp[c])) for c in range(n_cells)], dtype=float)
    z_from = np.array([cell.z for cell in cells], dtype=float)
    hc = CompFields(z=z_from, n=n_hc, cells=cells, p=p.copy())
    so, sg = _saturations_from_cells(cells, sw)
    return ThreePhaseState(hc=hc, n_water=n_w, s_water=sw.copy(), s_oil=so, s_gas=sg)


def water_molar_flux(
    faces: list[InteriorFace],
    s_water: NDArray[np.float64],
    pressure: NDArray[np.float64],
    z_center: NDArray[np.float64],
    *,
    gravity: float = 0.0,
    mu_water: float = EXAMPLE_WATER_MU_PA_S,
) -> NDArray[np.float64]:
    """Water molar rate L→R on each face [mol/s]. Capillary-free TPFA."""
    p = np.asarray(pressure, dtype=float).ravel()
    sw = np.asarray(s_water, dtype=float).ravel()
    zc = np.asarray(z_center, dtype=float).ravel()
    flux = np.zeros(len(faces), dtype=float)
    g = float(gravity)
    mu = max(float(mu_water), 1.0e-30)
    xi = water_molar_density()
    rho = float(EXAMPLE_WATER_RHO_KG_M3)
    for f_idx, face in enumerate(faces):
        left, right = face.left, face.right
        dphi = float(p[left] - p[right]) + rho * g * float(zc[left] - zc[right])
        sw_up = float(sw[left] if dphi >= 0.0 else sw[right])
        lam = max(sw_up, 0.0) / mu
        flux[f_idx] = face.transmissibility * xi * lam * dphi
    return flux


def _sw_from_moles(
    n_water: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
) -> NDArray[np.float64]:
    vp = np.maximum(np.asarray(pore_volume, dtype=float).ravel(), 1.0e-30)
    sw = np.asarray(n_water, dtype=float).ravel() / (vp * water_molar_density())
    return np.clip(sw, 0.0, 1.0)


def explicit_step_three_phase(
    state: ThreePhaseState,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    pore_volume: NDArray[np.float64],
    dt: float,
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    mu_water: float = EXAMPLE_WATER_MU_PA_S,
) -> ThreePhaseState:
    """One explicit three-phase mole step. Closed domain (no wells)."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    n_cells = state.hc.n.shape[0]
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    faces = interior_faces(grid, permeability)
    z_center = grid.cell_centers()[:, 2]
    hc_flux = phase_molar_flux(
        faces,
        state.hc.cells,
        p,
        z_center,
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
        s_water=state.s_water,
    )
    w_flux = water_molar_flux(
        faces, state.s_water, p, z_center, gravity=gravity, mu_water=mu_water
    )
    n_hc = _apply_divergence(state.hc.n, hc_flux, faces, dt)
    n_w = _apply_divergence(state.n_water.reshape(-1, 1), w_flux.reshape(-1, 1), faces, dt)
    n_w = np.clip(n_w.ravel(), 0.0, None)
    hc = _fields_from_moles(n_hc, T, p, mixture)
    hc.p = p.copy()
    sw = _sw_from_moles(n_w, vp)
    so, sg = _saturations_from_cells(hc.cells, sw)
    return ThreePhaseState(hc=hc, n_water=n_w, s_water=sw, s_oil=so, s_gas=sg)
