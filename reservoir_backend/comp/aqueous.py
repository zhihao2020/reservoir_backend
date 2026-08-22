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
- Pressure is prescribed on this first Newton cut (lagged ``p``, same
  as ``implicit_newton_step``). Volume constraint is not solved here.
- Water moles ``n_w`` sit in the same implicit-Euler Newton residual as
  ``n_i`` (not a post-process). ``S_w = n_w / (V_pore ξ_w)``.
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
from reservoir_backend.comp.cycle import (
    HZ_1INJ4PROD_INJECT_DAYS,
    HZ_1INJ4PROD_PRODUCE_DAYS,
    HZ_1INJ4PROD_SOAK_DAYS,
    HZ_1INJ4PROD_WELLHEAD_Z_DEFINITION,
    SECONDS_PER_DAY,
    CycleLedger,
    CycleRecord,
    MultiCycleLedger,
    perforated_z_co2,
    produced_stream_z_co2,
)
from reservoir_backend.comp.implicit import DT_CHOP, DT_GROW, NEWTON_MAX, NEWTON_TOL
from reservoir_backend.comp.step import (
    CompFields,
    WellLedger,
    _apply_divergence,
    _apply_injectors,
    _apply_producers,
    _fields_from_moles,
)
from reservoir_backend.comp.well import RateInjector, RateProducer
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
    "capillary-free TPFA mobility split; incompressible ξ_w; prescribed p; "
    "n_w in the implicit-Euler Newton residual with n_i"
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


@dataclass
class ThreePhaseNewtonReport:
    """One implicit Euler Newton step on ``(n_i, n_w)``. Lagged ``p``."""

    state: ThreePhaseState
    newton_converged: bool
    n_newton: int
    residual_hist: list[float]
    n_unknowns: int
    injected: NDArray[np.float64] | None = None
    produced: NDArray[np.float64] | None = None


def _state_from_moles(
    n_hc: NDArray[np.float64],
    n_water: NDArray[np.float64],
    T: float,
    p: NDArray[np.float64],
    mixture: EosMixture,
    pore_volume: NDArray[np.float64],
) -> ThreePhaseState:
    hc = _fields_from_moles(n_hc, T, p, mixture)
    hc.p = np.asarray(p, dtype=float).ravel().copy()
    n_w = np.clip(np.asarray(n_water, dtype=float).ravel(), 0.0, None)
    sw = _sw_from_moles(n_w, pore_volume)
    so, sg = _saturations_from_cells(hc.cells, sw)
    return ThreePhaseState(hc=hc, n_water=n_w, s_water=sw, s_oil=so, s_gas=sg)


def _pack_nw(n_hc: NDArray[np.float64], n_water: NDArray[np.float64]) -> NDArray[np.float64]:
    n_cells, n_comp = n_hc.shape
    u = np.empty(n_cells * (n_comp + 1), dtype=float)
    for c in range(n_cells):
        i0 = c * (n_comp + 1)
        u[i0 : i0 + n_comp] = n_hc[c]
        u[i0 + n_comp] = float(n_water[c])
    return u


def _unpack_nw(
    u: NDArray[np.float64], n_cells: int, n_comp: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_hc = np.zeros((n_cells, n_comp), dtype=float)
    n_w = np.zeros(n_cells, dtype=float)
    for c in range(n_cells):
        i0 = c * (n_comp + 1)
        n_hc[c] = np.clip(u[i0 : i0 + n_comp], 0.0, None)
        n_w[c] = max(0.0, float(u[i0 + n_comp]))
    return n_hc, n_w


def _rhs_three_phase(
    n_hc: NDArray[np.float64],
    n_water: NDArray[np.float64],
    n_hc_old: NDArray[np.float64],
    n_w_old: NDArray[np.float64],
    T: float,
    p: NDArray[np.float64],
    mixture: EosMixture,
    faces,
    z_center: NDArray[np.float64],
    dt: float,
    pore_volume: NDArray[np.float64],
    *,
    gravity: float,
    mu_liquid: float,
    mu_vapor: float,
    mu_water: float,
    injectors,
    producers,
) -> tuple[NDArray[np.float64], NDArray[np.float64], ThreePhaseState, NDArray[np.float64], NDArray[np.float64]]:
    """Implicit Euler hats: n_old − dt Div(flux) + inject − produce (HC); water flux only."""
    state = _state_from_moles(n_hc, n_water, T, p, mixture, pore_volume)
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
    n_hc_hat = _apply_divergence(n_hc_old, hc_flux, faces, dt)
    n_w_hat = _apply_divergence(n_w_old.reshape(-1, 1), w_flux.reshape(-1, 1), faces, dt).ravel()
    injected = np.zeros(n_hc_old.shape[1], dtype=float)
    produced = np.zeros(n_hc_old.shape[1], dtype=float)
    if injectors:
        n_hc_hat, injected = _apply_injectors(n_hc_hat, injectors, dt)
    if producers:
        n_hc_hat, produced = _apply_producers(
            n_hc_hat,
            state.hc.cells,
            producers,
            p,
            dt,
            mu_liquid=mu_liquid,
            mu_vapor=mu_vapor,
        )
    return n_hc_hat, n_w_hat, state, injected, produced


def _residual_nw(
    n_hc: NDArray[np.float64],
    n_water: NDArray[np.float64],
    n_hc_hat: NDArray[np.float64],
    n_w_hat: NDArray[np.float64],
    n_hc_ref: float,
    n_w_ref: float,
) -> NDArray[np.float64]:
    n_cells, n_comp = n_hc.shape
    r = np.empty(n_cells * (n_comp + 1), dtype=float)
    r_hc = (n_hc - n_hc_hat) / n_hc_ref
    r_w = (n_water - n_w_hat) / n_w_ref
    for c in range(n_cells):
        i0 = c * (n_comp + 1)
        r[i0 : i0 + n_comp] = r_hc[c]
        r[i0 + n_comp] = r_w[c]
    return r


def implicit_newton_step_three_phase(
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
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
    tol: float = NEWTON_TOL,
    max_iter: int = NEWTON_MAX,
) -> ThreePhaseNewtonReport:
    """One implicit Euler Newton step. Unknowns are ``(n_i, n_w)`` per cell.

    ``p`` is lagged. Water accumulation is in the residual, not a
    post-process. Wells (if any) add/remove HC moles only. EXAMPLE aqueous.
    """
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    n_hc_old = np.asarray(state.hc.n, dtype=float)
    n_w_old = np.asarray(state.n_water, dtype=float).ravel()
    n_cells, n_comp = n_hc_old.shape
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    faces = interior_faces(grid, permeability)
    z_center = grid.cell_centers()[:, 2]
    kw = dict(
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
        mu_water=mu_water,
        injectors=injectors,
        producers=producers,
    )
    n_hc_ref = max(1.0, float(np.mean(np.abs(n_hc_old))))
    n_w_ref = max(1.0, float(np.mean(np.abs(n_w_old))))
    n_hc = n_hc_old.copy()
    n_w = n_w_old.copy()
    n_hc_hat, n_w_hat, trial, injected, produced = _rhs_three_phase(
        n_hc, n_w, n_hc_old, n_w_old, T, p, mixture, faces, z_center, dt, vp, **kw
    )
    R = _residual_nw(n_hc, n_w, n_hc_hat, n_w_hat, n_hc_ref, n_w_ref)
    hist = [float(np.max(np.abs(R)))]
    n_unknowns = n_cells * (n_comp + 1)
    if dt == 0.0 or hist[0] < tol:
        return ThreePhaseNewtonReport(trial, True, 0, hist, n_unknowns, injected, produced)

    u = _pack_nw(n_hc, n_w)
    n_newton = 0
    for it in range(1, int(max_iter) + 1):
        n_newton = it
        J = np.zeros((n_unknowns, n_unknowns), dtype=float)
        for j in range(n_unknowns):
            eps = 1.0e-6 * max(1.0, abs(float(u[j])))
            u_p = u.copy()
            u_p[j] += eps
            n_p, w_p = _unpack_nw(u_p, n_cells, n_comp)
            hat_n, hat_w, _, _, _ = _rhs_three_phase(
                n_p, w_p, n_hc_old, n_w_old, T, p, mixture, faces, z_center, dt, vp, **kw
            )
            J[:, j] = (_residual_nw(n_p, w_p, hat_n, hat_w, n_hc_ref, n_w_ref) - R) / eps
        try:
            du = np.linalg.solve(J + 1.0e-12 * np.eye(n_unknowns), -R)
        except np.linalg.LinAlgError:
            return ThreePhaseNewtonReport(trial, False, n_newton, hist, n_unknowns, injected, produced)
        alpha = 1.0
        r0 = float(np.max(np.abs(R)))
        accepted = False
        for _ in range(8):
            n_try, w_try = _unpack_nw(u + alpha * du, n_cells, n_comp)
            n_hc_hat, n_w_hat, trial, injected, produced = _rhs_three_phase(
                n_try, w_try, n_hc_old, n_w_old, T, p, mixture, faces, z_center, dt, vp, **kw
            )
            R_try = _residual_nw(n_try, w_try, n_hc_hat, n_w_hat, n_hc_ref, n_w_ref)
            if float(np.max(np.abs(R_try))) <= r0 * (1.0 - 1.0e-4 * alpha) or alpha < 0.05:
                u = _pack_nw(n_try, w_try)
                n_hc, n_w = n_try, w_try
                R = R_try
                hist.append(float(np.max(np.abs(R))))
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            return ThreePhaseNewtonReport(trial, False, n_newton, hist, n_unknowns, injected, produced)
        if float(np.max(np.abs(R))) < tol:
            return ThreePhaseNewtonReport(trial, True, n_newton, hist, n_unknowns, injected, produced)
    return ThreePhaseNewtonReport(trial, False, n_newton, hist, n_unknowns, injected, produced)


def run_implicit_period_three_phase(
    state: ThreePhaseState,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    pore_volume: NDArray[np.float64],
    duration: float,
    *,
    dt_init: float,
    dt_max: float,
    gravity: float = 0.0,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
    grow: float = DT_GROW,
    chop: float = DT_CHOP,
) -> tuple[ThreePhaseState, WellLedger, list[list[float]], int]:
    """Advance ``duration`` with lagged-p three-phase Newton. Returns ledger, residual hists, n_accepted."""
    n_comp = state.hc.n.shape[1]
    ledger = WellLedger(injected=np.zeros(n_comp, dtype=float), produced=np.zeros(n_comp, dtype=float))
    if float(duration) <= 0.0:
        return state, ledger, [], 0
    current = state
    t = 0.0
    dt = min(float(dt_init), float(duration), float(dt_max))
    n_accepted = 0
    residual_hists: list[list[float]] = []
    p = np.asarray(pressure, dtype=float).ravel()
    while t < float(duration) - 1.0e-12:
        dt = min(dt, float(duration) - t, float(dt_max))
        report = implicit_newton_step_three_phase(
            current,
            T,
            p,
            mixture,
            grid,
            permeability,
            pore_volume,
            dt,
            gravity=gravity,
            injectors=injectors,
            producers=producers,
        )
        if not report.newton_converged:
            dt = float(chop) * dt
            if dt < 1.0e-18:
                ledger.underflow = True
                break
            continue
        current = report.state
        t += float(dt)
        inj = report.injected if report.injected is not None else np.zeros(n_comp, dtype=float)
        prd = report.produced if report.produced is not None else np.zeros(n_comp, dtype=float)
        ledger.injected += inj
        ledger.produced += prd
        ledger.dt_used.append(float(dt))
        residual_hists.append(list(report.residual_hist))
        n_accepted += 1
        dt = min(float(grow) * dt, float(dt_max))
    return current, ledger, residual_hists, n_accepted


def run_hz_1inj4prod_three_phase(
    state: ThreePhaseState,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    injectors: tuple[RateInjector, ...] | list[RateInjector],
    producers: tuple[RateProducer, ...] | list[RateProducer],
    pore_volume: NDArray[np.float64] | float,
    *,
    n_cycles: int = 1,
    inject_days: float = HZ_1INJ4PROD_INJECT_DAYS,
    soak_days: float = HZ_1INJ4PROD_SOAK_DAYS,
    produce_days: float = HZ_1INJ4PROD_PRODUCE_DAYS,
    dt_init_days: float = 0.125,
    dt_max_days: float = 0.125,
    gravity: float = 0.0,
) -> tuple[ThreePhaseState, MultiCycleLedger]:
    """Short HZ 1+4 huff-n-puff with immiscible water in the Newton residual.

    Opposite wells shut. Lagged ``p``. Same tiny mesh / schedule as the
    two-phase EXAMPLE cycle. Water is not injected or produced (HC wells
    only). Not FIM, not a GEM aqueous card.
    """
    if int(n_cycles) < 1:
        raise ValueError("n_cycles must be >= 1")
    inj = tuple(injectors)
    prod = tuple(producers)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    dt_init = float(dt_init_days) * SECONDS_PER_DAY
    dt_max = float(dt_max_days) * SECONDS_PER_DAY
    p = np.asarray(pressure, dtype=float).ravel()
    records: list[CycleRecord] = []
    underflow = False
    current = state
    common = dict(
        T=T,
        pressure=p,
        mixture=mixture,
        grid=grid,
        permeability=permeability,
        pore_volume=vp,
        dt_init=dt_init,
        dt_max=dt_max,
        gravity=gravity,
    )
    for _ in range(int(n_cycles)):
        n_start = current.hc.n.copy()
        inj_cells = tuple(int(w.cell) for w in inj)
        z0 = perforated_z_co2(current.hc, mixture, inj_cells)
        current, led_inj, hist_inj, n_inj = run_implicit_period_three_phase(
            current, duration=float(inject_days) * SECONDS_PER_DAY, injectors=inj, producers=None, **common
        )
        z_inj = perforated_z_co2(current.hc, mixture, inj_cells)
        current, led_soak, hist_soak, n_soak = run_implicit_period_three_phase(
            current, duration=float(soak_days) * SECONDS_PER_DAY, injectors=None, producers=None, **common
        )
        z_soak = perforated_z_co2(current.hc, mixture, inj_cells)
        current, led_prod, hist_prod, n_prod = run_implicit_period_three_phase(
            current, duration=float(produce_days) * SECONDS_PER_DAY, injectors=None, producers=prod, **common
        )
        uf = led_inj.underflow or led_soak.underflow or led_prod.underflow
        underflow = underflow or uf
        ledger = CycleLedger(
            inject=led_inj,
            soak=led_soak,
            produce=led_prod,
            underflow=uf,
            z_co2_well_cell_initial=z0,
            z_co2_well_cell_after_inject=z_inj,
            z_co2_well_cell_after_soak=z_soak,
            z_co2_well_cell_after_produce=perforated_z_co2(current.hc, mixture, inj_cells),
            z_co2_produced_stream=produced_stream_z_co2(led_prod, mixture),
            wellhead_z_definition=HZ_1INJ4PROD_WELLHEAD_Z_DEFINITION,
            accepted_steps=n_inj + n_soak + n_prod,
            residual_hists=hist_inj + hist_soak + hist_prod,
            inject_n_accepted=n_inj,
            produce_n_accepted=n_prod,
            inject_residual_hists=list(hist_inj),
            produce_residual_hists=list(hist_prod),
        )
        records.append(CycleRecord(ledger=ledger, n_start=n_start, n_end=current.hc.n.copy()))
    return current, MultiCycleLedger(cycles=records, underflow=underflow)
