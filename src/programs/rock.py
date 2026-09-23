"""Program 4: conservation-constrained k and phi, then mean-constrain to core values."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidPermeability, InvalidSaturation
from ..core.cartesian import CartesianGrid
from .interpolate import (
    default_variogram,
    fit_variogram,
    interpolate_field,
    interpolation_weights,
    ordinary_kriging,
)
from .mesh import PointMap, WellMap, _map_wells, map_points

_LOGIT_LO = 1.0e-4
_LOGIT_HI = 0.25

# exp overflow/underflow guards. np.exp overflows to inf above ~709 and
# underflows to 0 below ~-745; k is clamped to >=1e-30 afterwards anyway.
_EXP_MAX = 709.0
_EXP_MIN = -745.0


def _harmonic_mean(k1: NDArray[np.float64], k2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Harmonic mean ``2*k1*k2/(k1+k2)``, zero where ``k1+k2 == 0``.

    No additive epsilon: ``k`` is clamped to ``>= 1e-30`` upstream so the
    denominator is always positive, and an absolute ``1e-30`` floor would bias
    the result by ~33% at the clamp.
    """
    den = k1 + k2
    return np.divide(2.0 * k1 * k2, den, out=np.zeros_like(k1), where=den > 0.0)


def _clip_exp(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """``exp`` clipped so an unconstrained kriging weight cannot overflow to inf."""
    return np.exp(np.clip(np.asarray(x, dtype=float), _EXP_MIN, _EXP_MAX))


def _theta_bounds(n_probe: int, n_rel: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Box bounds on the inversion parameters ``theta``.

    Layout is ``[log_k (n_probe), phi_logit (n_probe), relperm_log (n_rel)]``.
    ``log_k`` keeps a wide range so the ``exp`` in ``map_fields`` cannot overflow
    (the phi sigmoid clips its input to [-20, 20] and rel-perm clips via ``exp``
    to [1e-3, 50]).
    """
    n = 2 * n_probe + n_rel
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    lb[:n_probe] = np.log(1.0e-30)
    ub[:n_probe] = np.log(1.0e3)
    lb[n_probe : 2 * n_probe] = -30.0
    ub[n_probe : 2 * n_probe] = 30.0
    if n_rel:
        lb[2 * n_probe :] = -10.0
        ub[2 * n_probe :] = 5.0
    return lb, ub


@dataclass(frozen=True)
class RelpermTable:
    """Tabular two-phase relative-permeability curves (CMG ``*SGT`` / ``*SWT``).

    ``sg``/``krg``/``krog`` are the gas-oil table (gas saturation, gas relperm,
    oil relperm in the presence of gas); ``sw``/``krw``/``krow`` the water-oil
    table. The three-phase oil relperm uses Stone I (``kro = krog * krow``), which
    collapses to ``krog`` when ``sw`` is at connate water.
    """

    sg: NDArray[np.float64]
    krg: NDArray[np.float64]
    krog: NDArray[np.float64]
    sw: NDArray[np.float64]
    krw: NDArray[np.float64]
    krow: NDArray[np.float64]

    @classmethod
    def from_rows(cls, sgt: list[list[float]], swt: list[list[float]]) -> RelpermTable:
        sgt = np.asarray(sgt, dtype=float)
        swt = np.asarray(swt, dtype=float)
        if sgt.shape[1] < 3 or swt.shape[1] < 3:
            raise InvalidSaturation("relperm tables need [saturation, kr_self, kr_other] columns")
        return cls(sgt[:, 0], sgt[:, 1], sgt[:, 2], swt[:, 0], swt[:, 1], swt[:, 2])


@dataclass(frozen=True)
class FluidParams:
    mu_w: float = 5.0e-4
    mu_o: float = 2.0e-3
    mu_g: float = 2.0e-5
    krw_end: float = 0.3
    kro_end: float = 0.8
    krg_end: float = 0.9
    nw: float = 2.0
    no: float = 2.0
    ng: float = 2.0
    swc: float = 0.05
    sor: float = 0.15
    sgc: float = 0.02
    ct: float = 1.0e-9
    # Solution-gas (CO2-in-oil) extension. ``rs_slope`` is the Henry's-law slope
    # ``Rs = rs_slope * p`` (m3 dissolved gas / m3 oil / Pa) used for the *output*
    # dissolved-gas metric (a throughput measure). ``rs_eq_slope`` is the
    # *equilibrium* solubility slope for the forward model's phase split (a
    # thermodynamic bound); 0.0 falls back to ``rs_slope`` for backward
    # compatibility. ``bo_slope`` is the linear oil-swelling factor (reserved).
    rs_slope: float = 0.0
    rs_eq_slope: float = 0.0
    bo_slope: float = 0.0
    # When True, the equilibrium Rs comes from the Peng-Robinson EOS bubble-point
    # solubility ``Rs_sat(p)`` (``core.pr_eos.rs_sat_interp``), the thermodynamic
    # bound GEM actually uses, instead of the Henry's-law ``rs_slope``/``rs_quad``.
    # This reproduces the correct free-gas fraction (the injected CO2 beyond the
    # solubility stays free gas), which the fitted Rs=187 throughput metric gets
    # wrong (it dissolves everything).
    rs_eos: bool = False
    # Quadratic solubility coefficient (1/Pa^2) so the equilibrium Rs can be a
    # *nonlinear* ``Rs(p) = rs_slope*p + rs_quad*p^2``. The GEM Peng-Robinson EOS
    # gives a solubility that is more pressure-sensitive than Henry's law near
    # the phase boundary (~3.6x the linear slope at 20 MPa / 120 C), so a PVT
    # table (or this quadratic) is a more faithful equilibrium bound. 0.0 keeps
    # the backward-compatible Henry's law.
    rs_quad: float = 0.0
    # Phase gravity heads ``rho*g`` (Pa/m) for the forward model's buoyancy
    # (denser phases sink). 0.0 = no gravity (backward compatible). At high
    # pressure the CO2-rich phase can be *denser* than the oil (density
    # inversion), so ``rho_g > rho_o`` makes the free gas segregate to the bottom.
    rho_w: float = 0.0
    rho_o: float = 0.0
    rho_g: float = 0.0
    # Solvent (CO2) density head ``rho_s*g`` (Pa/m) for the FCM miscible model.
    # The CO2-rich phase can be denser than the oil (density inversion), in which
    # case ``rho_s > rho_o`` makes the mixture sink.
    rho_s: float = 0.0
    # Solubility threshold ``c_sat`` (solvent volume fraction) for the FCM phase
    # split: below it all CO2 is dissolved (sg=0), above it the excess is free gas.
    c_sat: float = 0.66
    # Gas formation-volume factor ``Bg`` (reservoir gas volume / surface gas
    # volume). At high pressure the gas is compressed (Bg << 1). The conserved
    # CO2 component ``C = sg/Bg + Rs*so`` is in *surface* volume; ``Bg`` maps the
    # reservoir free-gas volume ``sg`` to that surface measure (and ``1/Bg`` maps
    # the reservoir gas *flux* back to surface). The well gas rate ``qg`` is
    # already surface volume (GEM ``*BHF``), so it enters the C source as-is.
    bg: float = 1.0
    # Optional tabular rel-perm (CMG *SGT / *SWT). When set, the forward model
    # interpolates these curves instead of the Corey power-law above.
    relperm_table: RelpermTable | None = None
    # Kinetic dissolution rate (1/s) of CO2 into the oil:
    # ``d(Cd)/dt = k_diss * (Rs*No - Cd)``. 0.0 = instantaneous equilibrium
    # (backward compatible); >0 makes the free gas dissolve over ~1/k_diss.
    k_diss: float = 0.0


@dataclass(frozen=True)
class WellModelParams:
    """Well completion model parameters.

    ``rw`` wellbore radius (m), ``skin`` dimensionless skin/damage factor,
    ``kv_kh`` vertical-to-horizontal permeability ratio, ``rho_g`` the fluid
    density times gravity (Pa/m) for the along-wellbore hydrostatic head (0 =
    no head).
    """

    rw: float = 0.005
    skin: float = 0.0
    kv_kh: float = 1.0
    rho_g: float = 0.0


@dataclass
class RockDiagnostics:
    mass_residual_rms: float = float("nan")
    mass_residual_rel: float = float("nan")
    phi_identifiable: bool = False
    phi_mean_error: float = float("nan")
    k_mean_error: float = float("nan")
    kriging_k: dict[str, Any] = field(default_factory=dict)
    kriging_phi: dict[str, Any] = field(default_factory=dict)
    relperm: dict[str, float] = field(default_factory=dict)
    theta: list[float] = field(default_factory=list)
    converged: bool = False
    status: int = -1
    message: str = ""
    cost: float = float("nan")
    optimality: float = float("nan")
    nfev: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mass_residual_rms": self.mass_residual_rms,
            "mass_residual_rel": self.mass_residual_rel,
            "phi_identifiable": self.phi_identifiable,
            "phi_mean_error": self.phi_mean_error,
            "k_mean_error": self.k_mean_error,
            "kriging_k": self.kriging_k,
            "kriging_phi": self.kriging_phi,
            "relperm": self.relperm,
            "theta": self.theta,
            "converged": self.converged,
            "status": self.status,
            "message": self.message,
            "cost": self.cost,
            "optimality": self.optimality,
            "nfev": self.nfev,
        }


def corey_phase_mobilities(
    sw: NDArray[np.float64] | float,
    so: NDArray[np.float64] | float,
    sg: NDArray[np.float64] | float,
    params: FluidParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return per-phase mobilities ``(lam_w, lam_o, lam_g)`` from Corey rel-perm."""
    sw_a = np.asarray(sw, dtype=float)
    so_a = np.asarray(so, dtype=float)
    sg_a = np.asarray(sg, dtype=float)
    denom = 1.0 - params.swc - params.sor - params.sgc
    if denom <= 0.0:
        raise InvalidSaturation("Corey residual saturations leave no mobile range")
    swe = np.clip((sw_a - params.swc) / denom, 0.0, 1.0)
    soe = np.clip((so_a - params.sor) / denom, 0.0, 1.0)
    sge = np.clip((sg_a - params.sgc) / denom, 0.0, 1.0)
    krw = params.krw_end * np.power(swe, params.nw)
    kro = params.kro_end * np.power(soe, params.no)
    krg = params.krg_end * np.power(sge, params.ng)
    return krw / params.mu_w, kro / params.mu_o, krg / params.mu_g


def tabular_phase_mobilities(
    sw: NDArray[np.float64] | float,
    so: NDArray[np.float64] | float,
    sg: NDArray[np.float64] | float,
    table: RelpermTable,
    params: FluidParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-phase mobilities from tabular rel-perm (Stone I for the oil).

    The rel-perm curves are interpolated with PCHIP (monotone, C1-smooth) rather
    than piecewise-linear ``np.interp``, which has a discontinuous ``dkr/dS`` at
    every table point. The smooth derivative is more Newton-friendly (the values
    at the table points are unchanged).
    """
    from scipy.interpolate import PchipInterpolator

    sw_a = np.asarray(sw, dtype=float)
    sg_a = np.asarray(sg, dtype=float)
    krw = PchipInterpolator(table.sw, table.krw)(sw_a)
    krow = PchipInterpolator(table.sw, table.krow)(sw_a)
    krog = PchipInterpolator(table.sg, table.krog)(sg_a)
    krg = PchipInterpolator(table.sg, table.krg)(sg_a)
    kro = krog * krow  # Stone I (connate-water oil relperm is ~1)
    return krw / params.mu_w, kro / params.mu_o, krg / params.mu_g


def phase_mobilities(
    sw: NDArray[np.float64] | float,
    so: NDArray[np.float64] | float,
    sg: NDArray[np.float64] | float,
    params: FluidParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Per-phase mobilities: tabular when ``params.relperm_table`` is set, else Corey."""
    if params.relperm_table is not None:
        return tabular_phase_mobilities(sw, so, sg, params.relperm_table, params)
    return corey_phase_mobilities(sw, so, sg, params)


def corey_total_mobility(
    sw: NDArray[np.float64] | float,
    so: NDArray[np.float64] | float,
    sg: NDArray[np.float64] | float,
    params: FluidParams,
) -> NDArray[np.float64]:
    lam_w, lam_o, lam_g = corey_phase_mobilities(sw, so, sg, params)
    return lam_w + lam_o + lam_g


def solution_gas_ratio(
    pressure: NDArray[np.float64] | float,
    params: FluidParams,
) -> NDArray[np.float64]:
    """Solution gas-oil ratio ``Rs`` from Henry's law (linear in pressure).

    ``Rs = rs_slope * p`` is the dissolved-gas (CO2) volume per unit oil volume at
    reservoir conditions. ``rs_slope == 0`` reduces to the dead-oil / permanent
    free-gas black-oil model, so callers that pass the default params are
    unaffected.
    """
    rs = params.rs_slope * np.asarray(pressure, dtype=float)
    return np.maximum(rs, 0.0)


def scale_mean(
    field: NDArray[np.float64],
    target: float,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    arr = np.asarray(field, dtype=float)
    if target <= 0.0:
        raise InvalidPermeability("core mean constraint must be positive")
    if weights is None:
        mean = float(np.mean(arr))
    else:
        w = np.asarray(weights, dtype=float)
        mean = float(np.average(arr, weights=w))
    if not np.isfinite(mean) or mean <= 0.0:
        return np.full(arr.shape, float(target), dtype=float)
    return arr * (float(target) / mean)


def well_cell_rates(
    grid: CartesianGrid,
    wells: WellMap,
    well_rate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Place well volumetric rates across the well's completion cells.

    The total rate is split uniformly over the cells of the well trajectory;
    injection is positive.
    """
    q = np.zeros(grid.n_cells, dtype=float)
    rates = np.asarray(well_rate, dtype=float).ravel()
    for i, cells in enumerate(wells.cells):
        if i >= rates.size or not np.isfinite(rates[i]) or cells.size == 0:
            continue
        q[cells] += float(rates[i]) / float(cells.size)
    return q


def peaceman_wi(
    grid: CartesianGrid,
    k_field: NDArray[np.float64],
    well_cells: NDArray[np.int64],
    direction: NDArray[np.float64] | tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    rw: float = 0.005,
    skin: float = 0.0,
    kv_kh: float = 1.0,
) -> NDArray[np.float64]:
    """Anisotropic Peaceman well index per completion cell.

    Permeabilities are ``kx = ky = k_field`` and ``kz = kv_kh * k_field``; the
    well ``direction`` (unit heel->toe) selects the axis the well runs along and
    hence the perpendicular effective permeability and equivalent radius. A point
    well (direction ~ 0) uses an isotropic geometric-mean form.
    """
    k = np.asarray(k_field, dtype=float).ravel()
    kz = max(float(kv_kh), 0.0) * k
    cells = np.asarray(well_cells, dtype=np.int64).ravel()
    ijk = np.array([grid.ijk(int(c)) for c in cells])
    dx_c = grid.dx[ijk[:, 0]]
    dy_c = grid.dy[ijk[:, 1]]
    dz_c = grid.dz[ijk[:, 2]]
    d = np.asarray(direction, dtype=float)
    ax = int(np.argmax(np.abs(d))) if float(np.linalg.norm(d)) > 1.0e-12 else -1
    r = np.sqrt(kz[cells] / np.maximum(k[cells], 1.0e-30))  # sqrt(kz/k_h)
    if ax == 2:  # vertical well (along z)
        kh = k[cells]
        re = 0.14 * np.sqrt(dx_c**2 + dy_c**2)
        h = dz_c
    elif ax == 0:  # horizontal well (along x)
        kh = np.sqrt(k[cells] * kz[cells])
        re = 0.28 * np.sqrt(r**2 * dy_c**2 + dz_c**2) / (r + 1.0)
        h = dx_c
    elif ax == 1:  # horizontal well (along y)
        kh = np.sqrt(k[cells] * kz[cells])
        re = 0.28 * np.sqrt(r**2 * dx_c**2 + dz_c**2) / (r + 1.0)
        h = dy_c
    else:  # point well / no direction: isotropic geometric mean
        kh = (k[cells] ** 2 * kz[cells]) ** (1.0 / 3.0)
        h = (dx_c * dy_c * dz_c) ** (1.0 / 3.0)
        re = 0.2 * h
    denom = np.log(np.maximum(re, 1.0e-12) / max(float(rw), 1.0e-12)) + float(skin)
    return 2.0 * np.pi * kh * h / np.maximum(denom, 1.0e-12)


def well_cell_rates_weighted(
    grid: CartesianGrid,
    wells: WellMap,
    well_rate: NDArray[np.float64],
    well_bhp: NDArray[np.float64],
    k_field: NDArray[np.float64],
    mobility: NDArray[np.float64],
    pressure: NDArray[np.float64],
    *,
    rw: float = 0.005,
    skin: float = 0.0,
    kv_kh: float = 1.0,
    rho_g: float = 0.0,
) -> NDArray[np.float64]:
    """Allocate well rates across completions by ``Peaceman WI * mobility * |drawdown|``.

    WI is anisotropic (``kv_kh``) and honours the well direction, skin enters the
    WI, and an optional wellbore hydrostatic head ``rho_g`` shifts the effective
    BHP per completion depth. Falls back to a uniform split when the drawdown is
    degenerate. Injection is positive.
    """
    q = np.zeros(grid.n_cells, dtype=float)
    rates = np.asarray(well_rate, dtype=float).ravel()
    bhp = np.asarray(well_bhp, dtype=float).ravel()
    k = np.asarray(k_field, dtype=float).ravel()
    lam = np.asarray(mobility, dtype=float).ravel()
    p = np.asarray(pressure, dtype=float).ravel()
    centers = grid.cell_centers()
    directions = wells.directions if wells.directions else [np.zeros(3)] * len(wells.cells)
    for i, cells in enumerate(wells.cells):
        if i >= rates.size or not np.isfinite(rates[i]) or cells.size == 0:
            continue
        direction = directions[i] if i < len(directions) else np.zeros(3)
        wi = peaceman_wi(grid, k, cells, direction, rw=rw, skin=skin, kv_kh=kv_kh)
        # wellbore head: BHP_ref + rho_g * (z_ref - z_cell)
        z_ref = float(wells.xyz[i, 2])
        bhp_eff = float(bhp[i]) + float(rho_g) * (z_ref - centers[cells, 2])
        drawdown = np.abs(p[cells] - bhp_eff)
        w = wi * lam[cells] * drawdown
        wsum = float(np.sum(w))
        if wsum > 0.0:
            q[cells] += float(rates[i]) * (w / wsum)
        else:
            q[cells] += float(rates[i]) / float(cells.size)
    return q


def darcy_divergence(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    mobility: NDArray[np.float64],
    pressure: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cell divergence of ``q = -k λ ∇p`` with no-flow outer faces."""
    coefs = _face_coefficients(grid, mobility, pressure)
    return _divergence_from_coefs(grid, permeability, coefs)


def _face_coefficients(
    grid: CartesianGrid,
    mobility: NDArray[np.float64],
    pressure: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Precompute the permeability-independent part of the Darcy face flux.

    For one time slice, returns ``(gx, gy, gz)`` where the flux across a face is
    ``harmonic_mean(k) * g``. ``mobility`` and ``pressure`` are flat
    ``(n_cells,)`` arrays; ``g`` is ``None`` for a degenerate axis.
    """
    lam = np.asarray(mobility, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    return _face_coefficients_from_lam(grid, lam, _face_geometry(grid, pressure))


def _face_geometry(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
) -> tuple:
    """Parameter-independent face-flux geometry.

    Returns ``(coef_x, up_x, coef_y, up_y, coef_z, up_z)`` where ``coef`` is the
    ``-area*dp/dist`` term and ``up`` is the upwind mask (``True`` = the low
    side is upstream). Only the upwind *mobility* then changes with the unknown
    parameters, so this can be precomputed once per time slice.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p = np.asarray(pressure, dtype=float).reshape((nz, ny, nx))
    coef_x = up_x = coef_y = up_y = coef_z = up_z = None
    if nx > 1:
        dist = grid.center_distance_x().reshape((1, 1, nx - 1))
        area = grid.face_area_x()[:, :, 1:-1]
        dp = p[:, :, 1:] - p[:, :, :-1]
        coef_x = -area * dp / np.maximum(dist, 1.0e-18)
        up_x = dp < 0.0
    if ny > 1:
        dist = grid.center_distance_y().reshape((1, ny - 1, 1))
        area = grid.face_area_y()[:, 1:-1, :]
        dp = p[:, 1:, :] - p[:, :-1, :]
        coef_y = -area * dp / np.maximum(dist, 1.0e-18)
        up_y = dp < 0.0
    if nz > 1:
        dist = grid.center_distance_z().reshape((nz - 1, 1, 1))
        area = grid.face_area_z()[1:-1, :, :]
        dp = p[1:, :, :] - p[:-1, :, :]
        coef_z = -area * dp / np.maximum(dist, 1.0e-18)
        up_z = dp < 0.0
    return coef_x, up_x, coef_y, up_y, coef_z, up_z


def _face_coefficients_from_lam(
    grid: CartesianGrid,
    lam3: NDArray[np.float64],
    geom: tuple,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Face coefficients ``(gx, gy, gz)`` from a mobility field ``lam3`` (nz,ny,nx)
    and precomputed ``_face_geometry``."""
    coef_x, up_x, coef_y, up_y, coef_z, up_z = geom
    gx = gy = gz = None
    if coef_x is not None:
        lam_f = np.where(up_x, lam3[:, :, :-1], lam3[:, :, 1:])
        gx = lam_f * coef_x
    if coef_y is not None:
        lam_f = np.where(up_y, lam3[:, :-1, :], lam3[:, 1:, :])
        gy = lam_f * coef_y
    if coef_z is not None:
        lam_f = np.where(up_z, lam3[:-1, :, :], lam3[1:, :, :])
        gz = lam_f * coef_z
    return gx, gy, gz


def _divergence_from_coefs(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    coefs: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None],
) -> NDArray[np.float64]:
    """Cell divergence of ``-k λ ∇p`` using precomputed face coefficients."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k3 = np.asarray(permeability, dtype=float).reshape((nz, ny, nx))
    gx, gy, gz = coefs
    div = np.zeros((nz, ny, nx), dtype=float)
    if gx is not None:
        kh = _harmonic_mean(k3[:, :, :-1], k3[:, :, 1:])
        flux = kh * gx
        div[:, :, :-1] += flux
        div[:, :, 1:] -= flux
    if gy is not None:
        kh = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :])
        flux = kh * gy
        div[:, :-1, :] += flux
        div[:, 1:, :] -= flux
    if gz is not None:
        kh = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :])
        flux = kh * gz
        div[:-1, :, :] += flux
        div[1:, :, :] -= flux
    return div.ravel()


def _divergence_all_times(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    coefs_list: list[tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]],
) -> list[NDArray[np.float64]]:
    """Divergence for every time slice, sharing the harmonic-k face means."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k3 = np.asarray(permeability, dtype=float).reshape((nz, ny, nx))
    gx0, gy0, gz0 = coefs_list[0]
    khx = khy = khz = None
    if gx0 is not None:
        khx = _harmonic_mean(k3[:, :, :-1], k3[:, :, 1:])
    if gy0 is not None:
        khy = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :])
    if gz0 is not None:
        khz = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :])
    out: list[NDArray[np.float64]] = []
    for gx, gy, gz in coefs_list:
        div = np.zeros((nz, ny, nx), dtype=float)
        if gx is not None:
            flux = khx * gx
            div[:, :, :-1] += flux
            div[:, :, 1:] -= flux
        if gy is not None:
            flux = khy * gy
            div[:, :-1, :] += flux
            div[:, 1:, :] -= flux
        if gz is not None:
            flux = khz * gz
            div[:-1, :, :] += flux
            div[1:, :, :] -= flux
        out.append(div.ravel())
    return out


def _divergence_directional_deriv(
    grid: CartesianGrid,
    k_field: NDArray[np.float64],
    direction: NDArray[np.float64],
    coefs: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None],
) -> NDArray[np.float64]:
    """Directional derivative of the harmonic-mean TPFA divergence w.r.t. ``k``,
    in the direction ``direction``. Same layout as ``_divergence_from_coefs``."""
    faces = _divergence_deriv_faces(grid, k_field, direction)
    return _divergence_deriv_from_faces(grid, faces, coefs)


def _divergence_deriv_faces(
    grid: CartesianGrid,
    k_field: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """k/direction-dependent directional-derivative face factors ``(dkh_x, dkh_y,
    dkh_z)``. Independent of the face coefficients, so reusable across time."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k3 = np.asarray(k_field, dtype=float).reshape((nz, ny, nx))
    d3 = np.asarray(direction, dtype=float).reshape((nz, ny, nx))
    dkh_x = dkh_y = dkh_z = None
    if nx > 1:
        k_lo, k_hi = k3[:, :, :-1], k3[:, :, 1:]
        d_lo, d_hi = d3[:, :, :-1], d3[:, :, 1:]
        s = k_lo + k_hi
        dkh_x = (2.0 * k_hi**2 / s**2) * d_lo + (2.0 * k_lo**2 / s**2) * d_hi
    if ny > 1:
        k_lo, k_hi = k3[:, :-1, :], k3[:, 1:, :]
        d_lo, d_hi = d3[:, :-1, :], d3[:, 1:, :]
        s = k_lo + k_hi
        dkh_y = (2.0 * k_hi**2 / s**2) * d_lo + (2.0 * k_lo**2 / s**2) * d_hi
    if nz > 1:
        k_lo, k_hi = k3[:-1, :, :], k3[1:, :, :]
        d_lo, d_hi = d3[:-1, :, :], d3[1:, :, :]
        s = k_lo + k_hi
        dkh_z = (2.0 * k_hi**2 / s**2) * d_lo + (2.0 * k_lo**2 / s**2) * d_hi
    return dkh_x, dkh_y, dkh_z


def _divergence_deriv_from_faces(
    grid: CartesianGrid,
    faces: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None],
    coefs: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None],
) -> NDArray[np.float64]:
    """Divergence derivative from precomputed face factors and face coefficients."""
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dkh_x, dkh_y, dkh_z = faces
    gx, gy, gz = coefs
    div = np.zeros((nz, ny, nx), dtype=float)
    if gx is not None:
        flux = dkh_x * gx
        div[:, :, :-1] += flux
        div[:, :, 1:] -= flux
    if gy is not None:
        flux = dkh_y * gy
        div[:, :-1, :] += flux
        div[:, 1:, :] -= flux
    if gz is not None:
        flux = dkh_z * gz
        div[:-1, :, :] += flux
        div[1:, :, :] -= flux
    return div.ravel()


def _divergence_directional_deriv_all_times(
    grid: CartesianGrid,
    k_field: NDArray[np.float64],
    direction: NDArray[np.float64],
    coefs_list: list[tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None]],
) -> list[NDArray[np.float64]]:
    """Directional derivative for every time slice, sharing the face factors."""
    faces = _divergence_deriv_faces(grid, k_field, direction)
    return [_divergence_deriv_from_faces(grid, faces, coefs) for coefs in coefs_list]


@functools.lru_cache(maxsize=8)
def _face_cell_pairs(nx: int, ny: int, nz: int) -> tuple:
    """(lo_cell, hi_cell) index arrays for x/y/z interior faces (cached)."""
    c1_x = (np.arange(nz)[:, None, None] * ny * nx + np.arange(ny)[None, :, None] * nx + np.arange(nx - 1)[None, None, :]).ravel().astype(np.int64)
    c2_x = (np.arange(nz)[:, None, None] * ny * nx + np.arange(ny)[None, :, None] * nx + (np.arange(nx - 1) + 1)[None, None, :]).ravel().astype(np.int64)
    c1_y = (np.arange(nz)[:, None, None] * ny * nx + np.arange(ny - 1)[None, :, None] * nx + np.arange(nx)[None, None, :]).ravel().astype(np.int64)
    c2_y = (np.arange(nz)[:, None, None] * ny * nx + (np.arange(ny - 1) + 1)[None, :, None] * nx + np.arange(nx)[None, None, :]).ravel().astype(np.int64)
    c1_z = (np.arange(nz - 1)[:, None, None] * ny * nx + np.arange(ny)[None, :, None] * nx + np.arange(nx)[None, None, :]).ravel().astype(np.int64)
    c2_z = ((np.arange(nz - 1) + 1)[:, None, None] * ny * nx + np.arange(ny)[None, :, None] * nx + np.arange(nx)[None, None, :]).ravel().astype(np.int64)
    return (c1_x, c2_x), (c1_y, c2_y), (c1_z, c2_z)


@functools.lru_cache(maxsize=8)
def _divergence_deriv_pattern(nx: int, ny: int, nz: int) -> tuple:
    """(rows, cols) sparsity pattern of the divergence-derivative matrix (cached)."""
    (c1_x, c2_x), (c1_y, c2_y), (c1_z, c2_z) = _face_cell_pairs(nx, ny, nz)
    rows = np.concatenate([c1_x, c1_x, c2_x, c2_x, c1_y, c1_y, c2_y, c2_y, c1_z, c1_z, c2_z, c2_z])
    cols = np.concatenate([c1_x, c2_x, c1_x, c2_x, c1_y, c2_y, c1_y, c2_y, c1_z, c2_z, c1_z, c2_z])
    return rows, cols


@functools.lru_cache(maxsize=8)
def _divergence_deriv_structure(nx: int, ny: int, nz: int) -> tuple:
    """Preallocated CSR structure + COO->CSR position map (cached).

    Returns ``(indptr, indices, pos, n)`` where ``pos[i]`` maps COO entry ``i``
    (in ``_divergence_deriv_pattern`` order) to its summed CSR data slot."""
    from scipy.sparse import coo_matrix

    rows, cols = _divergence_deriv_pattern(nx, ny, nz)
    n = nx * ny * nz
    template = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    order = np.lexsort((cols, rows))
    sorted_key = rows[order].astype(np.int64) * n + cols[order]
    unique_key, _ = np.unique(sorted_key, return_index=True)
    pos = np.searchsorted(unique_key, rows.astype(np.int64) * n + cols)
    return template.indptr, template.indices, pos, n


def _divergence_deriv_matrix(
    grid: CartesianGrid,
    k_field: NDArray[np.float64],
    coefs: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, NDArray[np.float64] | None],
):
    """Sparse ``(n_cells, n_cells)`` matrix ``D`` such that ``D @ direction`` is
    the directional derivative of the harmonic-mean TPFA divergence. Uses a
    preallocated CSR structure, filling only the data array."""
    from scipy.sparse import csr_matrix

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k3 = np.asarray(k_field, dtype=float).reshape((nz, ny, nx))
    gx, gy, gz = coefs
    indptr, indices, pos, n = _divergence_deriv_structure(nx, ny, nz)
    parts: list[NDArray[np.float64]] = []

    def add_axis(g, k_lo, k_hi):
        s = k_lo + k_hi
        dlo = ((2.0 * k_hi**2 / s**2) * g).ravel()
        dhi = ((2.0 * k_lo**2 / s**2) * g).ravel()
        parts.extend([dlo, dhi, -dlo, -dhi])

    if gx is not None:
        add_axis(gx, k3[:, :, :-1], k3[:, :, 1:])
    if gy is not None:
        add_axis(gy, k3[:, :-1, :], k3[:, 1:, :])
    if gz is not None:
        add_axis(gz, k3[:-1, :, :], k3[1:, :, :])
    data = np.zeros(indices.size, dtype=float)
    np.add.at(data, pos, np.concatenate(parts))
    return csr_matrix((data, indices, indptr), shape=(n, n))


def _sigmoid_phi(logit: NDArray[np.float64]) -> NDArray[np.float64]:
    s = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
    return _LOGIT_LO + (_LOGIT_HI - _LOGIT_LO) * s


def _logit_phi(phi: float) -> float:
    x = (float(phi) - _LOGIT_LO) / (_LOGIT_HI - _LOGIT_LO)
    x = min(max(x, 1.0e-8), 1.0 - 1.0e-8)
    return float(np.log(x / (1.0 - x)))


_RELPERM_NAMES = ("nw", "no", "ng", "krw_end", "kro_end", "krg_end")


def transient_weights(
    times: NDArray[np.float64],
    *,
    tau: float | None = None,
) -> NDArray[np.float64]:
    """Early-time emphasis weights (exponential decay) for transient inversion.

    The pressure transient carries porosity / near-well permeability information;
    weighting the early residuals up makes the inversion exploit it. ``tau``
    defaults to a third of the time span.
    """
    t = np.asarray(times, dtype=float).ravel()
    if tau is None:
        span = float(t.max() - t.min())
        tau = span / 3.0 if span > 0.0 else 1.0
    return np.exp(-(t - float(t.min())) / max(tau, 1.0e-12))


def _record_convergence(diagnostics: RockDiagnostics, fit: Any) -> None:
    """Copy the least-squares termination state into the diagnostics.

    ``fit`` is the ``scipy.optimize.OptimizeResult``, or ``None`` when the
    solver raised and the inversion fell back to the initial guess. Recording
    ``success/status/message/cost/optimality/nfev`` makes a silent fallback
    visible in ``validation.json`` instead of returning a uniform field with no
    trace of the failure.
    """
    if fit is None:
        diagnostics.converged = False
        diagnostics.status = -1
        diagnostics.message = "least_squares raised; fell back to initial guess"
        diagnostics.cost = float("nan")
        diagnostics.optimality = float("nan")
        diagnostics.nfev = 0
        return
    diagnostics.converged = bool(getattr(fit, "success", False))
    diagnostics.status = int(getattr(fit, "status", -1))
    diagnostics.message = str(getattr(fit, "message", ""))
    diagnostics.cost = float(getattr(fit, "cost", float("nan")))
    diagnostics.optimality = float(getattr(fit, "optimality", float("nan")))
    diagnostics.nfev = int(getattr(fit, "nfev", 0))


def invert_rock(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    times: NDArray[np.float64],
    probes: PointMap,
    wells: WellMap,
    well_rate: NDArray[np.float64],
    *,
    phi0: float,
    k0: float,
    params: FluidParams | None = None,
    relperm: tuple[str, ...] = (),
    time_weights: NDArray[np.float64] | None = None,
    theta0: NDArray[np.float64] | None = None,
    max_nfev: int = 60,
    well_bhp: NDArray[np.float64] | None = None,
    well_params: WellModelParams | None = None,
    smoothness: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], RockDiagnostics]:
    """Jointly invert static k, phi and optional Corey rel-perm parameters.

    ``relperm`` names Corey parameters (from nw/no/ng/krw_end/kro_end/krg_end) to
    invert jointly with k/phi in log space. ``time_weights`` (n_times,) can
    emphasise the pressure transient for porosity identifiability.
    """
    oil = params or FluidParams()
    relperm = tuple(p for p in relperm if p in _RELPERM_NAMES)
    n_rel = len(relperm)
    if time_weights is not None:
        time_weights = np.asarray(time_weights, dtype=float).ravel()
    p = np.asarray(pressure, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    sw_a = np.asarray(sw, dtype=float)
    so_a = np.asarray(so, dtype=float)
    sg_a = np.asarray(sg, dtype=float)
    if sw_a.ndim == 1:
        sw_a, so_a, sg_a = sw_a[None, :], so_a[None, :], np.asarray(sg, dtype=float)[None, :]
    rates = np.asarray(well_rate, dtype=float)
    if rates.ndim == 1:
        rates = rates[None, :]
    if well_bhp is not None:
        well_bhp = np.asarray(well_bhp, dtype=float)
        if well_bhp.ndim == 1:
            well_bhp = well_bhp[None, :]
    wp = well_params or WellModelParams()
    n_t = int(p.shape[0])
    volumes = grid.cell_volumes()
    n_probe = int(probes.xyz.shape[0])
    diagnostics = RockDiagnostics()
    if n_probe == 0:
        phi = scale_mean(np.full(grid.n_cells, float(phi0)), phi0, volumes)
        k = scale_mean(np.full(grid.n_cells, float(k0)), k0, volumes)
        return phi, k, diagnostics

    sources = np.stack([well_cell_rates(grid, wells, rates[t]) for t in range(n_t)])
    # Face-flux geometry is parameter-independent; precompute it once per time
    # slice. With rel-perm inversion the mobility changes with theta, so only
    # the upwind mobility is recomputed per residual evaluation.
    geom = [_face_geometry(grid, p[t]) for t in range(n_t)]
    if n_rel == 0:
        mobility0 = np.stack(
            [corey_total_mobility(sw_a[t], so_a[t], sg_a[t], oil) for t in range(n_t)]
        )
        div_coefs = [_face_coefficients(grid, mobility0[t], p[t]) for t in range(n_t)]
    vol_norm = volumes / float(np.sum(volumes))
    nearest = _nearest_probe(probes.xyz) if n_probe > 1 else np.zeros(0, dtype=np.int64)
    q_scale = float(np.max(np.abs(sources)))
    if not np.isfinite(q_scale) or q_scale <= 0.0:
        # Shut-in / zero-rate step: keep residuals in absolute Darcy units
        # instead of dividing by ~0, so the mean-constraint dominates.
        q_scale = 1.0
    dt = np.diff(np.asarray(times, dtype=float).ravel()) if n_t > 1 else np.zeros(0)
    accum_scale = 0.0
    if n_t > 1:
        dpdt = np.abs(np.diff(p, axis=0)) / np.maximum(dt[:, None], 1.0e-18)
        accum_scale = float(np.max(dpdt) * oil.ct * np.max(volumes) * phi0)
    diagnostics.phi_identifiable = bool(accum_scale > 0.05 * q_scale)

    centers = grid.cell_centers()
    k_model = default_variogram(probes.xyz, np.full(n_probe, np.log(k0)))

    # Precompute kriging weights so the least-squares inner loop is a
    # matrix-vector product instead of re-solving kriging each residual eval.
    phi_model = default_variogram(probes.xyz, np.full(n_probe, _logit_phi(phi0)))
    k_weights = interpolation_weights(probes.xyz, centers, method="kriging", model=k_model)
    phi_weights = interpolation_weights(probes.xyz, centers, method="kriging", model=phi_model)

    def map_fields(theta: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        log_k = theta[:n_probe]
        phi_probe = _sigmoid_phi(theta[n_probe : 2 * n_probe])
        k_field = _clip_exp(log_k @ k_weights)
        phi_field = phi_probe @ phi_weights
        return np.maximum(k_field, 1.0e-30), np.clip(phi_field, _LOGIT_LO, _LOGIT_HI)

    def _oil_of(theta: NDArray[np.float64]) -> FluidParams:
        if n_rel == 0:
            return oil
        overrides = {}
        for j, pname in enumerate(relperm):
            overrides[pname] = float(np.clip(np.exp(theta[2 * n_probe + j]), 1.0e-3, 50.0))
        return replace(oil, **overrides)

    def residuals(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        k_field, phi_field = map_fields(theta)
        oil_t = _oil_of(theta)
        if n_rel:
            mobility_t = np.stack(
                [corey_total_mobility(sw_a[t], so_a[t], sg_a[t], oil_t) for t in range(n_t)]
            )
            coefs = [
                _face_coefficients_from_lam(grid, mobility_t[t].reshape((grid.nz, grid.ny, grid.nx)), geom[t])
                for t in range(n_t)
            ]
            divs = _divergence_all_times(grid, k_field, coefs)
        else:
            mobility_t = mobility0
            divs = _divergence_all_times(grid, k_field, div_coefs)
        chunks: list[NDArray[np.float64]] = []
        for t in range(n_t):
            src = sources[t]
            if well_bhp is not None:
                src = well_cell_rates_weighted(
                    grid, wells, rates[t], well_bhp[t], k_field, mobility_t[t], p[t],
                    rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
                )
            r = divs[t] - src
            if t > 0 and dt[t - 1] > 0.0:
                r = r + phi_field * oil_t.ct * volumes * (p[t] - p[t - 1]) / dt[t - 1]
            if time_weights is not None:
                r = r * time_weights[t]
            chunks.append(r / q_scale)
        k_mean = float(k_field @ vol_norm)
        phi_mean = float(phi_field @ vol_norm)
        chunks.append(np.array([(k_mean - k0) / k0 * 8.0, (phi_mean - phi0) / max(phi0, 1.0e-12) * 8.0]))
        if n_probe > 1:
            log_k = theta[:n_probe]
            chunks.append(smoothness * (log_k - log_k[nearest]))
        return np.concatenate(chunks)

    def jacobian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytic Jacobian of ``residuals``: the divergence derivative w.r.t.
        log-k is exact (harmonic-mean TPFA), the porosity accumulation is linear,
        and the (small) rel-perm block is finite-differenced."""
        k_field, phi_field = map_fields(theta)
        oil_t = _oil_of(theta)
        if n_rel:
            mobility_t = np.stack(
                [corey_total_mobility(sw_a[t], so_a[t], sg_a[t], oil_t) for t in range(n_t)]
            )
            coefs = [
                _face_coefficients_from_lam(grid, mobility_t[t].reshape((grid.nz, grid.ny, grid.nx)), geom[t])
                for t in range(n_t)
            ]
        else:
            coefs = div_coefs
        n_c = grid.n_cells
        m = n_t * n_c + 2 + (n_probe if n_probe > 1 else 0)
        n_p = 2 * n_probe + n_rel
        J = np.zeros((m, n_p))
        dphi = (phi_field - _LOGIT_LO) * (_LOGIT_HI - phi_field) / (_LOGIT_HI - _LOGIT_LO)

        Kdir = k_field[:, None] * k_weights.T  # (n_cells, n_probe): d(k_field)/d(log_k_j)
        for t in range(n_t):
            block = (_divergence_deriv_matrix(grid, k_field, coefs[t]) @ Kdir) / q_scale
            if time_weights is not None:
                block = block * time_weights[t]
            J[t * n_c : (t + 1) * n_c, :n_probe] = block
        for j in range(n_probe):
            J[n_t * n_c, j] = ((k_field * k_weights[j, :]) @ vol_norm) / k0 * 8.0

        for j in range(n_probe):
            dphi_j = dphi * phi_weights[j, :]
            for t in range(n_t):
                if t > 0 and dt[t - 1] > 0.0:
                    col = J[t * n_c : (t + 1) * n_c, n_probe + j]
                    col[:] = dphi_j * oil_t.ct * volumes * (p[t] - p[t - 1]) / dt[t - 1] / q_scale
                    if time_weights is not None:
                        col *= time_weights[t]
            J[n_t * n_c + 1, n_probe + j] = (dphi_j @ vol_norm) / max(phi0, 1.0e-12) * 8.0

        if n_probe > 1:
            for j in range(n_probe):
                row = n_t * n_c + 2 + j
                J[row, j] = smoothness
                J[row, nearest[j]] -= smoothness

        if n_rel:
            for r in range(n_rel):
                col = 2 * n_probe + r
                h = 1.0e-6 * max(1.0, abs(float(theta[col])))
                tp = theta.copy()
                tm = theta.copy()
                tp[col] += h
                tm[col] -= h
                J[:, col] = (residuals(tp) - residuals(tm)) / (2.0 * h)
        return J

    if theta0 is None:
        theta0 = np.concatenate(
            [np.full(n_probe, np.log(max(k0, 1.0e-30))), np.full(n_probe, _logit_phi(phi0))]
        )
        if n_rel:
            theta0 = np.concatenate([theta0, [np.log(float(getattr(oil, pname))) for pname in relperm]])
    else:
        theta0 = np.asarray(theta0, dtype=float).copy()
    fit = None
    try:
        from scipy.optimize import least_squares

        lb, ub = _theta_bounds(n_probe, n_rel)
        fit = least_squares(residuals, theta0, jac=jacobian, method="trf", max_nfev=max_nfev, ftol=1.0e-8, xtol=1.0e-8, x_scale="jac", bounds=(lb, ub))
        theta = np.asarray(fit.x, dtype=float)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        theta = theta0
    _record_convergence(diagnostics, fit)

    k_probe = np.exp(theta[:n_probe])
    phi_probe = _sigmoid_phi(theta[n_probe : 2 * n_probe])
    oil_final = _oil_of(theta)
    k_report = fit_variogram(probes.xyz, np.log(np.maximum(k_probe, 1.0e-30)))
    phi_logit = np.log(np.clip((phi_probe - _LOGIT_LO) / (_LOGIT_HI - _LOGIT_LO), 1.0e-8, 1.0 - 1.0e-8))
    phi_report = fit_variogram(probes.xyz, phi_logit)
    log_k_field, k_var = ordinary_kriging(probes.xyz, np.log(np.maximum(k_probe, 1.0e-30)), centers, model=k_report.model)
    k_field = np.exp(log_k_field)
    phi_logit_field, _ = ordinary_kriging(probes.xyz, phi_logit, centers, model=phi_report.model)
    phi_field = _sigmoid_phi(phi_logit_field)
    k_report.variance_mean = float(np.mean(k_var))
    k_field = np.maximum(k_field, 1.0e-30)
    phi_field = np.clip(phi_field, _LOGIT_LO, _LOGIT_HI)
    k_field = scale_mean(k_field, k0, volumes)
    phi_field = scale_mean(phi_field, phi0, volumes)

    mobility_f = np.stack(
        [corey_total_mobility(sw_a[t], so_a[t], sg_a[t], oil_final) for t in range(n_t)]
    )
    coefs_f = [_face_coefficients(grid, mobility_f[t], p[t]) for t in range(n_t)]
    divs_f = _divergence_all_times(grid, k_field, coefs_f)
    mass = []
    for t in range(n_t):
        src = sources[t]
        if well_bhp is not None:
            # Keep the diagnostic residual consistent with the fitted residual:
            # honour the Peaceman-weighted allocation when BHP controls the well.
            src = well_cell_rates_weighted(
                grid, wells, rates[t], well_bhp[t], k_field, mobility_f[t], p[t],
                rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
            )
        r = divs_f[t] - src
        if t > 0 and n_t > 1 and dt[t - 1] > 0.0:
            r = r + phi_field * oil_final.ct * volumes * (p[t] - p[t - 1]) / dt[t - 1]
        mass.append(r)
    mass_arr = np.concatenate(mass) if mass else np.zeros(1)
    diagnostics.mass_residual_rms = float(np.sqrt(np.mean(mass_arr**2)))
    diagnostics.mass_residual_rel = diagnostics.mass_residual_rms / q_scale
    diagnostics.k_mean_error = abs(float(np.average(k_field, weights=volumes)) - k0) / k0
    diagnostics.phi_mean_error = abs(float(np.average(phi_field, weights=volumes)) - phi0) / phi0
    diagnostics.kriging_k = k_report.as_dict()
    diagnostics.kriging_phi = phi_report.as_dict()
    diagnostics.relperm = {pname: float(getattr(oil_final, pname)) for pname in relperm}
    diagnostics.theta = np.asarray(theta, dtype=float).tolist()
    if not diagnostics.phi_identifiable:
        diagnostics.kriging_phi["note"] = "pressure change is too small to identify porosity; field is mean-constrained"
    return phi_field, k_field, diagnostics


def invert_rock_three_phase(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    times: NDArray[np.float64],
    probes: PointMap,
    wells: WellMap,
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
    *,
    phi0: float,
    k0: float,
    params: FluidParams | None = None,
    relperm: tuple[str, ...] = (),
    time_weights: NDArray[np.float64] | None = None,
    fractional_weight: float = 1.0,
    theta0: NDArray[np.float64] | None = None,
    max_nfev: int = 80,
    well_bhp: NDArray[np.float64] | None = None,
    well_params: WellModelParams | None = None,
    smoothness: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], RockDiagnostics]:
    """Jointly invert static k, phi and Corey rel-perm from three-phase black-oil
    mass balance.

    Uses per-phase (water/oil/gas) conservation, so the phase mobilities enter
    separately from the permeability (unlike the total-mobility model), making
    the Corey rel-perm identifiable. The saturation-change accumulation term
    ``phi * dS_alpha/dt`` provides porosity identifiability during transients.

    ``fractional_weight`` adds a residual matching the observed phase-rate split
    (fractional flow ``f_alpha = lambda_alpha / lambda_total``) at production
    wells — this is rel-perm-sensitive and independent of permeability, so it
    disambiguates the Corey parameters from k.
    """
    oil = params or FluidParams()
    relperm = tuple(p for p in relperm if p in _RELPERM_NAMES)
    n_rel = len(relperm)
    if time_weights is not None:
        time_weights = np.asarray(time_weights, dtype=float).ravel()

    p = np.asarray(pressure, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    sw_a = np.asarray(sw, dtype=float)
    so_a = np.asarray(so, dtype=float)
    sg_a = np.asarray(sg, dtype=float)
    if sw_a.ndim == 1:
        sw_a, so_a, sg_a = sw_a[None, :], so_a[None, :], np.asarray(sg, dtype=float)[None, :]
    qw = np.asarray(well_qw, dtype=float)
    qo = np.asarray(well_qo, dtype=float)
    qg = np.asarray(well_qg, dtype=float)
    if qw.ndim == 1:
        qw, qo, qg = qw[None, :], qo[None, :], qg[None, :]
    if well_bhp is not None:
        well_bhp = np.asarray(well_bhp, dtype=float)
        if well_bhp.ndim == 1:
            well_bhp = well_bhp[None, :]
    wp = well_params or WellModelParams()

    n_t = int(p.shape[0])
    volumes = grid.cell_volumes()
    n_probe = int(probes.xyz.shape[0])
    diagnostics = RockDiagnostics()
    if n_probe == 0:
        phi = scale_mean(np.full(grid.n_cells, float(phi0)), phi0, volumes)
        k = scale_mean(np.full(grid.n_cells, float(k0)), k0, volumes)
        return phi, k, diagnostics

    sources_w = np.stack([well_cell_rates(grid, wells, qw[t]) for t in range(n_t)])
    sources_o = np.stack([well_cell_rates(grid, wells, qo[t]) for t in range(n_t)])
    sources_g = np.stack([well_cell_rates(grid, wells, qg[t]) for t in range(n_t)])
    q_scale = max(
        float(np.max(np.abs(sources_w))),
        float(np.max(np.abs(sources_o))),
        float(np.max(np.abs(sources_g))),
    )
    if not np.isfinite(q_scale) or q_scale <= 0.0:
        q_scale = 1.0
    q_total = qw + qo + qg
    producer_wells = np.flatnonzero(np.sum(q_total, axis=0) < 0.0)

    geom = [_face_geometry(grid, p[t]) for t in range(n_t)]
    vol_norm = volumes / float(np.sum(volumes))
    nearest = _nearest_probe(probes.xyz) if n_probe > 1 else np.zeros(0, dtype=np.int64)
    dt = np.diff(np.asarray(times, dtype=float).ravel()) if n_t > 1 else np.zeros(0)
    dsw = np.diff(sw_a, axis=0)
    dso = np.diff(so_a, axis=0)
    dsg = np.diff(sg_a, axis=0)
    accum_scale = 0.0
    if n_t > 1:
        accum_scale = float(
            np.max((np.abs(dsw) + np.abs(dso) + np.abs(dsg)) / np.maximum(dt[:, None], 1.0e-18))
            * np.max(volumes) * phi0
        )
    diagnostics.phi_identifiable = bool(accum_scale > 0.05 * q_scale)

    centers = grid.cell_centers()
    k_model = default_variogram(probes.xyz, np.full(n_probe, np.log(k0)))

    phi_model = default_variogram(probes.xyz, np.full(n_probe, _logit_phi(phi0)))
    k_weights = interpolation_weights(probes.xyz, centers, method="kriging", model=k_model)
    phi_weights = interpolation_weights(probes.xyz, centers, method="kriging", model=phi_model)

    def map_fields(theta):
        log_k = theta[:n_probe]
        phi_probe = _sigmoid_phi(theta[n_probe : 2 * n_probe])
        k_field = _clip_exp(log_k @ k_weights)
        phi_field = phi_probe @ phi_weights
        return np.maximum(k_field, 1.0e-30), np.clip(phi_field, _LOGIT_LO, _LOGIT_HI)

    def _oil_of(theta):
        if n_rel == 0:
            return oil
        overrides = {}
        for j, pname in enumerate(relperm):
            overrides[pname] = float(np.clip(np.exp(theta[2 * n_probe + j]), 1.0e-3, 50.0))
        return replace(oil, **overrides)

    def residuals(theta):
        k_field, phi_field = map_fields(theta)
        oil_t = _oil_of(theta)
        chunks: list[NDArray[np.float64]] = []
        for t in range(n_t):
            lam_w, lam_o, lam_g = corey_phase_mobilities(sw_a[t], so_a[t], sg_a[t], oil_t)
            coefs_w = _face_coefficients_from_lam(grid, lam_w.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
            coefs_o = _face_coefficients_from_lam(grid, lam_o.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
            coefs_g = _face_coefficients_from_lam(grid, lam_g.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
            div_w = _divergence_from_coefs(grid, k_field, coefs_w)
            div_o = _divergence_from_coefs(grid, k_field, coefs_o)
            div_g = _divergence_from_coefs(grid, k_field, coefs_g)
            acc_w = acc_o = acc_g = 0.0
            if t > 0 and dt[t - 1] > 0.0:
                acc_w = phi_field * volumes * dsw[t - 1] / dt[t - 1]
                acc_o = phi_field * volumes * dso[t - 1] / dt[t - 1]
                acc_g = phi_field * volumes * dsg[t - 1] / dt[t - 1]
            if well_bhp is not None:
                src_w = well_cell_rates_weighted(
                    grid, wells, qw[t], well_bhp[t], k_field, lam_w, p[t],
                    rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
                )
                src_o = well_cell_rates_weighted(
                    grid, wells, qo[t], well_bhp[t], k_field, lam_o, p[t],
                    rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
                )
                src_g = well_cell_rates_weighted(
                    grid, wells, qg[t], well_bhp[t], k_field, lam_g, p[t],
                    rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
                )
            else:
                src_w, src_o, src_g = sources_w[t], sources_o[t], sources_g[t]
            r_w = (div_w - src_w + acc_w) / q_scale
            r_o = (div_o - src_o + acc_o) / q_scale
            r_g = (div_g - src_g + acc_g) / q_scale
            if time_weights is not None:
                r_w = r_w * time_weights[t]
                r_o = r_o * time_weights[t]
                r_g = r_g * time_weights[t]
            chunks.extend([r_w, r_o, r_g])
            if fractional_weight > 0.0:
                for wi in producer_wells:
                    cells = wells.cells[wi]
                    if cells.size == 0 or float(q_total[t, wi]) >= 0.0:
                        continue
                    lw = float(np.mean(lam_w[cells]))
                    lo = float(np.mean(lam_o[cells]))
                    lg = float(np.mean(lam_g[cells]))
                    ltot = lw + lo + lg
                    if ltot <= 0.0:
                        continue
                    qt = float(q_total[t, wi])
                    chunks.append(
                        fractional_weight
                        * np.array([lw / ltot - qw[t, wi] / qt, lo / ltot - qo[t, wi] / qt, lg / ltot - qg[t, wi] / qt])
                    )
        k_mean = float(k_field @ vol_norm)
        phi_mean = float(phi_field @ vol_norm)
        chunks.append(np.array([(k_mean - k0) / k0 * 8.0, (phi_mean - phi0) / max(phi0, 1.0e-12) * 8.0]))
        if n_probe > 1:
            log_k = theta[:n_probe]
            chunks.append(smoothness * (log_k - log_k[nearest]))
        return np.concatenate(chunks)

    def jacobian(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytic Jacobian for the mass-balance part (exact divergence + linear
        porosity); the fractional-flow and rel-perm rows are finite-differenced."""
        k_field, phi_field = map_fields(theta)
        oil_t = _oil_of(theta)
        n_c = grid.n_cells
        coefs_w = []
        coefs_o = []
        coefs_g = []
        for t in range(n_t):
            lam_w, lam_o, lam_g = corey_phase_mobilities(sw_a[t], so_a[t], sg_a[t], oil_t)
            coefs_w.append(_face_coefficients_from_lam(grid, lam_w.reshape((grid.nz, grid.ny, grid.nx)), geom[t]))
            coefs_o.append(_face_coefficients_from_lam(grid, lam_o.reshape((grid.nz, grid.ny, grid.nx)), geom[t]))
            coefs_g.append(_face_coefficients_from_lam(grid, lam_g.reshape((grid.nz, grid.ny, grid.nx)), geom[t]))

        n_frac = 0
        frac_before = np.zeros(n_t, dtype=np.int64)
        if fractional_weight > 0.0:
            for t in range(n_t):
                frac_before[t] = n_frac
                for wi in producer_wells:
                    cells = wells.cells[wi]
                    if cells.size > 0 and float(q_total[t, wi]) < 0.0:
                        n_frac += 3
        m = 3 * n_t * n_c + n_frac + 2 + (n_probe if n_probe > 1 else 0)
        n_p = 2 * n_probe + n_rel
        J = np.zeros((m, n_p))
        dphi = (phi_field - _LOGIT_LO) * (_LOGIT_HI - phi_field) / (_LOGIT_HI - _LOGIT_LO)
        mean_row = 3 * n_t * n_c + n_frac

        Kdir = k_field[:, None] * k_weights.T
        for t in range(n_t):
            base = 3 * t * n_c + int(frac_before[t])
            bw = (_divergence_deriv_matrix(grid, k_field, coefs_w[t]) @ Kdir) / q_scale
            bo = (_divergence_deriv_matrix(grid, k_field, coefs_o[t]) @ Kdir) / q_scale
            bg = (_divergence_deriv_matrix(grid, k_field, coefs_g[t]) @ Kdir) / q_scale
            if time_weights is not None:
                bw = bw * time_weights[t]
                bo = bo * time_weights[t]
                bg = bg * time_weights[t]
            J[base : base + n_c, :n_probe] = bw
            J[base + n_c : base + 2 * n_c, :n_probe] = bo
            J[base + 2 * n_c : base + 3 * n_c, :n_probe] = bg
        for j in range(n_probe):
            J[mean_row, j] = ((k_field * k_weights[j, :]) @ vol_norm) / k0 * 8.0

        for j in range(n_probe):
            dphi_j = dphi * phi_weights[j, :]
            for t in range(n_t):
                if t > 0 and dt[t - 1] > 0.0:
                    base = 3 * t * n_c + int(frac_before[t])
                    J[base : base + n_c, n_probe + j] = dphi_j * volumes * dsw[t - 1] / dt[t - 1] / q_scale
                    J[base + n_c : base + 2 * n_c, n_probe + j] = dphi_j * volumes * dso[t - 1] / dt[t - 1] / q_scale
                    J[base + 2 * n_c : base + 3 * n_c, n_probe + j] = dphi_j * volumes * dsg[t - 1] / dt[t - 1] / q_scale
                    if time_weights is not None:
                        J[base : base + 3 * n_c, n_probe + j] *= time_weights[t]
            J[mean_row + 1, n_probe + j] = (dphi_j @ vol_norm) / max(phi0, 1.0e-12) * 8.0

        if n_probe > 1:
            for j in range(n_probe):
                row = mean_row + 2 + j
                J[row, j] = smoothness
                J[row, nearest[j]] -= smoothness

        if n_rel:
            for r in range(n_rel):
                col = 2 * n_probe + r
                h = 1.0e-6 * max(1.0, abs(float(theta[col])))
                tp = theta.copy()
                tm = theta.copy()
                tp[col] += h
                tm[col] -= h
                J[:, col] = (residuals(tp) - residuals(tm)) / (2.0 * h)
        return J

    if theta0 is None:
        theta0 = np.concatenate(
            [np.full(n_probe, np.log(max(k0, 1.0e-30))), np.full(n_probe, _logit_phi(phi0))]
        )
        if n_rel:
            theta0 = np.concatenate([theta0, [np.log(float(getattr(oil, pname))) for pname in relperm]])
    else:
        theta0 = np.asarray(theta0, dtype=float).copy()
    fit = None
    try:
        from scipy.optimize import least_squares

        lb, ub = _theta_bounds(n_probe, n_rel)
        fit = least_squares(residuals, theta0, jac=jacobian, method="trf", max_nfev=max_nfev, ftol=1.0e-8, xtol=1.0e-8, x_scale="jac", bounds=(lb, ub))
        theta = np.asarray(fit.x, dtype=float)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        theta = theta0
    _record_convergence(diagnostics, fit)

    k_probe = np.exp(theta[:n_probe])
    phi_probe = _sigmoid_phi(theta[n_probe : 2 * n_probe])
    oil_final = _oil_of(theta)
    k_report = fit_variogram(probes.xyz, np.log(np.maximum(k_probe, 1.0e-30)))
    phi_logit = np.log(np.clip((phi_probe - _LOGIT_LO) / (_LOGIT_HI - _LOGIT_LO), 1.0e-8, 1.0 - 1.0e-8))
    phi_report = fit_variogram(probes.xyz, phi_logit)
    log_k_field, k_var = ordinary_kriging(probes.xyz, np.log(np.maximum(k_probe, 1.0e-30)), centers, model=k_report.model)
    k_field = np.exp(log_k_field)
    phi_logit_field, _ = ordinary_kriging(probes.xyz, phi_logit, centers, model=phi_report.model)
    phi_field = _sigmoid_phi(phi_logit_field)
    k_report.variance_mean = float(np.mean(k_var))
    k_field = np.maximum(k_field, 1.0e-30)
    phi_field = np.clip(phi_field, _LOGIT_LO, _LOGIT_HI)
    k_field = scale_mean(k_field, k0, volumes)
    phi_field = scale_mean(phi_field, phi0, volumes)

    mass = []
    for t in range(n_t):
        lam_w, lam_o, lam_g = corey_phase_mobilities(sw_a[t], so_a[t], sg_a[t], oil_final)
        coefs_w = _face_coefficients_from_lam(grid, lam_w.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
        coefs_o = _face_coefficients_from_lam(grid, lam_o.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
        coefs_g = _face_coefficients_from_lam(grid, lam_g.reshape((grid.nz, grid.ny, grid.nx)), geom[t])
        div_w = _divergence_from_coefs(grid, k_field, coefs_w)
        div_o = _divergence_from_coefs(grid, k_field, coefs_o)
        div_g = _divergence_from_coefs(grid, k_field, coefs_g)
        if well_bhp is not None:
            # Consistent with the fitted residual: Peaceman-weighted allocation.
            src_w = well_cell_rates_weighted(
                grid, wells, qw[t], well_bhp[t], k_field, lam_w, p[t],
                rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
            )
            src_o = well_cell_rates_weighted(
                grid, wells, qo[t], well_bhp[t], k_field, lam_o, p[t],
                rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
            )
            src_g = well_cell_rates_weighted(
                grid, wells, qg[t], well_bhp[t], k_field, lam_g, p[t],
                rw=wp.rw, skin=wp.skin, kv_kh=wp.kv_kh, rho_g=wp.rho_g,
            )
        else:
            src_w, src_o, src_g = sources_w[t], sources_o[t], sources_g[t]
        acc_w = acc_o = acc_g = 0.0
        if t > 0 and n_t > 1 and dt[t - 1] > 0.0:
            acc_w = phi_field * volumes * dsw[t - 1] / dt[t - 1]
            acc_o = phi_field * volumes * dso[t - 1] / dt[t - 1]
            acc_g = phi_field * volumes * dsg[t - 1] / dt[t - 1]
        mass.extend([div_w - src_w + acc_w, div_o - src_o + acc_o, div_g - src_g + acc_g])
    mass_arr = np.concatenate(mass) if mass else np.zeros(1)
    diagnostics.mass_residual_rms = float(np.sqrt(np.mean(mass_arr**2)))
    diagnostics.mass_residual_rel = diagnostics.mass_residual_rms / q_scale
    diagnostics.k_mean_error = abs(float(np.average(k_field, weights=volumes)) - k0) / k0
    diagnostics.phi_mean_error = abs(float(np.average(phi_field, weights=volumes)) - phi0) / phi0
    diagnostics.kriging_k = k_report.as_dict()
    diagnostics.kriging_phi = phi_report.as_dict()
    diagnostics.relperm = {pname: float(getattr(oil_final, pname)) for pname in relperm}
    diagnostics.theta = np.asarray(theta, dtype=float).tolist()
    if not diagnostics.phi_identifiable:
        diagnostics.kriging_phi["note"] = "saturation change is too small to identify porosity; field is mean-constrained"
    return phi_field, k_field, diagnostics


def _nearest_probe(xyz: NDArray[np.float64]) -> NDArray[np.int64]:
    dist = np.sqrt(np.sum((xyz[:, None, :] - xyz[None, :, :]) ** 2, axis=2))
    dist.flat[:: xyz.shape[0] + 1] = np.inf
    return np.argmin(dist, axis=1).astype(np.int64)


def _coarsen_grid(grid: CartesianGrid, factor: int = 2) -> CartesianGrid:
    nx = max(1, (grid.nx + factor - 1) // factor)
    ny = max(1, (grid.ny + factor - 1) // factor)
    nz = max(1, (grid.nz + factor - 1) // factor)

    def merge(arr, n):
        out = [float(np.sum(arr[i : i + factor])) for i in range(0, arr.size, factor)]
        return np.array(out[:n], dtype=float)

    return CartesianGrid(
        nx=nx, ny=ny, nz=nz,
        dx=merge(grid.dx, nx), dy=merge(grid.dy, ny), dz=merge(grid.dz, nz),
        origin=grid.origin,
    )


def _coarsen_matrix(nx: int, ny: int, nz: int, factor: int = 2):
    """Sparse averaging matrix ``R`` (n_coarse, n_fine): ``coarse = R @ fine``."""
    from scipy.sparse import coo_matrix, diags

    nx_c = max(1, (nx + factor - 1) // factor)
    ny_c = max(1, (ny + factor - 1) // factor)
    nz_c = max(1, (nz + factor - 1) // factor)
    ii = np.arange(nx)
    jj = np.arange(ny)
    kk = np.arange(nz)
    ic = np.minimum(ii // factor, nx_c - 1)
    jc = np.minimum(jj // factor, ny_c - 1)
    kc = np.minimum(kk // factor, nz_c - 1)
    fine = (kk[:, None, None] * ny * nx + jj[None, :, None] * nx + ii[None, None, :]).ravel()
    coarse = (kc[:, None, None] * ny_c * nx_c + jc[None, :, None] * nx_c + ic[None, None, :]).ravel()
    R = coo_matrix((np.ones(fine.size), (coarse, fine)), shape=(nx_c * ny_c * nz_c, nx * ny * nz)).tocsr()
    row_sum = np.asarray(R.sum(axis=1)).ravel()
    return diags(1.0 / np.maximum(row_sum, 1.0)) @ R


def _coarsen_field(R, field: NDArray[np.float64]) -> NDArray[np.float64]:
    f = np.asarray(field, dtype=float)
    if f.ndim == 1:
        return R @ f
    return (R @ f.T).T


def invert_rock_coarse_to_fine(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    times: NDArray[np.float64],
    probes: PointMap,
    wells: WellMap,
    well_rate: NDArray[np.float64],
    *,
    phi0: float,
    k0: float,
    params: FluidParams | None = None,
    relperm: tuple[str, ...] = (),
    time_weights: NDArray[np.float64] | None = None,
    factor: int = 2,
    coarse_max_nfev: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64], RockDiagnostics]:
    """Coarse-to-fine inversion: solve on a coarsened grid for an initial
    probe-value guess, then refine on the full grid from that guess (fewer
    fine-grid iterations for large grids)."""
    if factor <= 1 or min(grid.nx, grid.ny, grid.nz) <= factor:
        return invert_rock(
            grid, pressure, sw, so, sg, times, probes, wells, well_rate,
            phi0=phi0, k0=k0, params=params,
            relperm=relperm, time_weights=time_weights,
        )
    coarse_grid = _coarsen_grid(grid, factor)
    R = _coarsen_matrix(grid.nx, grid.ny, grid.nz, factor)
    p_c = _coarsen_field(R, pressure)
    sw_c = _coarsen_field(R, sw)
    so_c = _coarsen_field(R, so)
    sg_c = _coarsen_field(R, sg)
    probes_c = map_points(coarse_grid, probes.ids, probes.xyz)
    wells_c = _map_wells(coarse_grid, wells.ids, wells.xyz)
    _, _, diag_c = invert_rock(
        coarse_grid, p_c, sw_c, so_c, sg_c, times, probes_c, wells_c, well_rate,
        phi0=phi0, k0=k0, params=params,
        relperm=relperm, time_weights=time_weights, max_nfev=coarse_max_nfev,
    )
    theta0 = np.asarray(diag_c.theta, dtype=float)
    return invert_rock(
        grid, pressure, sw, so, sg, times, probes, wells, well_rate,
        phi0=phi0, k0=k0, params=params,
        relperm=relperm, time_weights=time_weights, theta0=theta0,
    )
