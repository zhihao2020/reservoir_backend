"""Coupled implicit Newton on ``(n_i, p)`` per cell. ``T`` is prescribed.

Not the FIM residual and not a rewrite of ``solver/fi.py``. Pressure is a
Newton unknown (not a lagged explicit update).

Volume / pressure constraint (documented, EXAMPLE first cut):

    R_p,c = n_tot,c * v_mix(T, p_c, z_c) − V_pore,c

    v_mix = ν v_V + (1 − ν) v_L

from ``flash_tp`` at the trial ``(T, p, z)``. ``z = n / Σ n``. Component
mass residuals are implicit Euler on ``n_i`` (flux evaluated at trial
``n`` and trial ``p``). Unknown vector per cell: ``(n_0, …, n_{n_c−1}, p)``.

Standalone EXAMPLE kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash
from reservoir_backend.comp.flux import EXAMPLE_MU_LIQUID, EXAMPLE_MU_VAPOR, interior_faces
from reservoir_backend.comp.implicit import (
    DT_CHOP,
    DT_GROW,
    ImplicitPeriodLedger,
    ImplicitStepReport,
    _rhs,
)
from reservoir_backend.comp.step import DT_MIN, CompFields, WellLedger, _fields_from_moles
from reservoir_backend.comp.well import RateInjector, RateProducer
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

VOLUME_CONSTRAINT = (
    "R_p = n_tot * v_mix(T, p, z) - V_pore, with "
    "v_mix = nu * v_V + (1 - nu) * v_L from flash_tp at the trial (T, p, z). "
    "T is prescribed. p is a Newton unknown in the same vector as n_i."
)

P_REF = 1.0e6  # Pa, scales p in the unknown vector
P_MIN = 1.0e4
P_MAX = 1.0e8
NEWTON_TOL = 1.0e-8
NEWTON_MAX = 20


def _v_mix(cell: CellFlash) -> float:
    return float(cell.nu * cell.v_vapor + (1.0 - cell.nu) * cell.v_liquid)


def _pack_np(n: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
    n_cells, n_comp = n.shape
    u = np.empty(n_cells * (n_comp + 1), dtype=float)
    for c in range(n_cells):
        i0 = c * (n_comp + 1)
        u[i0 : i0 + n_comp] = n[c]
        u[i0 + n_comp] = float(p[c]) / P_REF
    return u


def _unpack_np(u: NDArray[np.float64], n_cells: int, n_comp: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = np.zeros((n_cells, n_comp), dtype=float)
    p = np.zeros(n_cells, dtype=float)
    for c in range(n_cells):
        i0 = c * (n_comp + 1)
        n[c] = np.clip(u[i0 : i0 + n_comp], 0.0, None)
        p[c] = float(np.clip(u[i0 + n_comp] * P_REF, P_MIN, P_MAX))
    return n, p


def _residual_np(
    n_trial: NDArray[np.float64],
    p_trial: NDArray[np.float64],
    n_old: NDArray[np.float64],
    T: float,
    mixture: EosMixture,
    faces,
    z_center: NDArray[np.float64],
    dt: float,
    pore_volume: NDArray[np.float64],
    n_ref: float,
    **kw,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], CompFields]:
    n_hat, injected, produced = _rhs(
        n_trial, n_old, T, p_trial, mixture, faces, z_center, dt, **kw
    )
    fld = _fields_from_moles(n_trial, T, p_trial, mixture)
    fld.p = p_trial.copy()
    r_n = (n_trial - n_hat) / n_ref
    vp = np.maximum(np.asarray(pore_volume, dtype=float).ravel(), 1.0e-30)
    r_p = np.empty(n_trial.shape[0], dtype=float)
    for c, cell in enumerate(fld.cells):
        r_p[c] = (float(n_trial[c].sum()) * _v_mix(cell) - vp[c]) / vp[c]
    r = np.empty(n_trial.shape[0] * (n_trial.shape[1] + 1), dtype=float)
    nc = n_trial.shape[1]
    for c in range(n_trial.shape[0]):
        i0 = c * (nc + 1)
        r[i0 : i0 + nc] = r_n[c]
        r[i0 + nc] = r_p[c]
    return r, injected, produced, fld


def implicit_newton_step_np(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    dt: float,
    pore_volume: NDArray[np.float64] | float,
    *,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
    tol: float = NEWTON_TOL,
    max_iter: int = NEWTON_MAX,
) -> ImplicitStepReport:
    """One implicit Euler Newton step. Unknowns are ``(n_i, p)`` per cell."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    underflow = bool(dt > 0.0 and dt < DT_MIN)
    n_old = np.asarray(fields.n, dtype=float)
    n_cells, n_comp = n_old.shape
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    vp = np.asarray(pore_volume, dtype=float).ravel()
    if vp.size == 1:
        vp = np.full(n_cells, float(vp[0]), dtype=float)
    if vp.size != n_cells:
        raise ValueError("pore_volume must match n_cells")
    faces = interior_faces(grid, permeability)
    z_center = grid.cell_centers()[:, 2]
    kw = dict(
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
        injectors=injectors,
        producers=producers,
    )
    n_ref = max(1.0, float(np.mean(np.abs(n_old))))
    n_trial = n_old.copy()
    p_trial = p.copy()
    R, injected, produced, fld = _residual_np(
        n_trial, p_trial, n_old, T, mixture, faces, z_center, dt, vp, n_ref, **kw
    )
    hist = [float(np.max(np.abs(R)))]
    n_unknowns = n_cells * (n_comp + 1)
    extra = dict(pressure=p_trial.copy(), has_pressure_unknown=True, n_unknowns=n_unknowns)
    if dt == 0.0 or hist[0] < tol:
        fld.p = p_trial.copy()
        return ImplicitStepReport(
            fld, injected, produced, float(dt), underflow, True, 0, residual_hist=hist, **extra
        )

    u = _pack_np(n_trial, p_trial)
    n_newton = 0
    for it in range(1, int(max_iter) + 1):
        n_newton = it
        J = np.zeros((n_unknowns, n_unknowns), dtype=float)
        for j in range(n_unknowns):
            eps = 1.0e-6 * max(1.0, abs(float(u[j])))
            u_p = u.copy()
            u_p[j] += eps
            n_p, p_p = _unpack_np(u_p, n_cells, n_comp)
            R_p, _, _, _ = _residual_np(n_p, p_p, n_old, T, mixture, faces, z_center, dt, vp, n_ref, **kw)
            J[:, j] = (R_p - R) / eps
        try:
            du = np.linalg.solve(J + 1.0e-12 * np.eye(n_unknowns), -R)
        except np.linalg.LinAlgError:
            fld.p = p_trial.copy()
            return ImplicitStepReport(
                fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **extra
            )
        alpha = 1.0
        r0 = float(np.max(np.abs(R)))
        accepted = False
        for _ in range(8):
            n_try, p_try = _unpack_np(u + alpha * du, n_cells, n_comp)
            R_try, injected, produced, fld = _residual_np(
                n_try, p_try, n_old, T, mixture, faces, z_center, dt, vp, n_ref, **kw
            )
            if float(np.max(np.abs(R_try))) <= r0 * (1.0 - 1.0e-4 * alpha) or alpha < 0.05:
                u = _pack_np(n_try, p_try)
                n_trial, p_trial = n_try, p_try
                R = R_try
                hist.append(float(np.max(np.abs(R))))
                accepted = True
                break
            alpha *= 0.5
        extra = dict(pressure=p_trial.copy(), has_pressure_unknown=True, n_unknowns=n_unknowns)
        if not accepted:
            fld.p = p_trial.copy()
            return ImplicitStepReport(
                fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **extra
            )
        if float(np.max(np.abs(R))) < tol:
            fld.p = p_trial.copy()
            return ImplicitStepReport(
                fld, injected, produced, float(dt), underflow, True, n_newton, residual_hist=hist, **extra
            )

    fld.p = p_trial.copy()
    return ImplicitStepReport(
        fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **extra
    )


def run_implicit_period_np(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    duration: float,
    pore_volume: NDArray[np.float64] | float,
    *,
    dt_init: float,
    dt_max: float,
    gravity: float = 0.0,
    mu_liquid: float = EXAMPLE_MU_LIQUID,
    mu_vapor: float = EXAMPLE_MU_VAPOR,
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
    grow: float = DT_GROW,
    chop: float = DT_CHOP,
) -> tuple[CompFields, ImplicitPeriodLedger]:
    """Advance ``duration`` with coupled ``(n_i, p)`` Newton; grow/hold/chop ``dt``."""
    n_comp = fields.n.shape[1]
    n_cells = fields.n.shape[0]
    ledger = WellLedger(injected=np.zeros(n_comp, dtype=float), produced=np.zeros(n_comp, dtype=float))
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    if float(duration) <= 0.0:
        empty = ImplicitPeriodLedger(ledger, 0, 0, 0, False, float(dt_init), float(dt_max), [])
        empty.pressure = p
        return fields, empty
    current = fields
    t = 0.0
    dt = min(float(dt_init), float(duration), float(dt_max))
    n_accepted = 0
    n_newton = 0
    n_chop = 0
    underflow = False
    dts: list[float] = []
    residual_hists: list[list[float]] = []
    while t < float(duration) - 1.0e-12:
        dt = min(dt, float(duration) - t, float(dt_max))
        if dt < DT_MIN:
            underflow = True
            break
        report = implicit_newton_step_np(
            current,
            T,
            p,
            mixture,
            grid,
            permeability,
            dt,
            pore_volume,
            gravity=gravity,
            mu_liquid=mu_liquid,
            mu_vapor=mu_vapor,
            injectors=injectors,
            producers=producers,
        )
        n_newton += report.n_newton
        if not report.newton_converged:
            n_chop += 1
            dt = float(chop) * dt
            continue
        current = report.fields
        if report.pressure is not None:
            p = np.asarray(report.pressure, dtype=float).ravel()
        t += report.dt_used
        ledger.injected += report.injected
        ledger.produced += report.produced
        ledger.dt_used.append(report.dt_used)
        ledger.underflow = ledger.underflow or report.underflow
        dts.append(report.dt_used)
        residual_hists.append(list(report.residual_hist))
        n_accepted += 1
        dt = min(float(grow) * report.dt_used, float(dt_max))
    underflow = underflow or ledger.underflow
    out = ImplicitPeriodLedger(
        ledger=ledger,
        n_accepted=n_accepted,
        n_newton=n_newton,
        n_chop=n_chop,
        underflow=underflow,
        dt_init=float(dt_init),
        dt_max=float(dt_max),
        dt_used=dts,
        residual_hists=residual_hists,
    )
    out.pressure = p
    return current, out
