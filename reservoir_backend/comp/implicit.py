"""Implicit Euler + Newton on component moles. Lagged / prescribed pressure.

Not a pressure solver and not the FIM residual. ``p`` and ``T`` are held;
the unknown is cell moles ``n_i``. Newton uses a dense finite-difference
Jacobian (tiny EXAMPLE grids). Failed Newton chops ``dt``; accepted steps
may grow or hold ``dt``. Production is still capped to available moles.

Standalone EXAMPLE kernel. Do not import from ``solver/fi.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.flux import EXAMPLE_MU_LIQUID, EXAMPLE_MU_VAPOR, interior_faces, phase_molar_flux
from reservoir_backend.comp.step import (
    DT_MIN,
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

NEWTON_TOL = 1.0e-9
NEWTON_MAX = 20
DT_GROW = 2.0
DT_CHOP = 0.5


@dataclass
class ImplicitStepReport:
    """One attempted or accepted implicit step."""

    fields: CompFields
    injected: NDArray[np.float64]
    produced: NDArray[np.float64]
    dt_used: float
    underflow: bool
    newton_converged: bool
    n_newton: int
    n_chop: int = 0
    residual_hist: list[float] = field(default_factory=list)
    pressure: NDArray[np.float64] | None = None
    has_pressure_unknown: bool = False
    n_unknowns: int = 0
    has_bhp_unknown: bool = False
    bhp: float | None = None  # p_wf [Pa]; set when BHP is a Newton unknown


@dataclass
class ImplicitPeriodLedger:
    """Adaptive implicit run over a time interval."""

    ledger: WellLedger
    n_accepted: int
    n_newton: int
    n_chop: int
    underflow: bool
    dt_init: float
    dt_max: float
    dt_used: list[float] = field(default_factory=list)
    residual_hists: list[list[float]] = field(default_factory=list)
    pressure: NDArray[np.float64] | None = None
    bhp: float | None = None


def _pack(n: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(n, dtype=float).ravel()


def _unpack(vec: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    return np.clip(np.asarray(vec, dtype=float).reshape(shape), 0.0, None)


def _rhs(
    n_trial: NDArray[np.float64],
    n_old: NDArray[np.float64],
    T: float,
    p: NDArray[np.float64],
    mixture: EosMixture,
    faces,
    z_center: NDArray[np.float64],
    dt: float,
    *,
    gravity: float,
    mu_liquid: float,
    mu_vapor: float,
    injectors,
    producers,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Implicit Euler right-hand side: n_old − dt Div(flux) + inject − produce."""
    fld = _fields_from_moles(n_trial, T, p, mixture)
    flux = phase_molar_flux(
        faces, fld.cells, p, z_center, gravity=gravity, mu_liquid=mu_liquid, mu_vapor=mu_vapor
    )
    n_hat = _apply_divergence(n_old, flux, faces, dt)
    injected = np.zeros(n_old.shape[1], dtype=float)
    produced = np.zeros(n_old.shape[1], dtype=float)
    if injectors:
        n_hat, injected = _apply_injectors(n_hat, injectors, dt)
    if producers:
        n_hat, produced = _apply_producers(
            n_hat, fld.cells, producers, p, dt, mu_liquid=mu_liquid, mu_vapor=mu_vapor
        )
    return n_hat, injected, produced


def _residual(n_trial: NDArray[np.float64], n_hat: NDArray[np.float64]) -> NDArray[np.float64]:
    return _pack(n_trial) - _pack(n_hat)


def implicit_newton_step(
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
    injectors: tuple[RateInjector, ...] | list[RateInjector] | None = None,
    producers: tuple[RateProducer, ...] | list[RateProducer] | None = None,
    tol: float = NEWTON_TOL,
    max_iter: int = NEWTON_MAX,
) -> ImplicitStepReport:
    """One implicit Euler Newton step at a fixed ``dt``. Lagged ``p``."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    underflow = bool(dt > 0.0 and dt < DT_MIN)
    n_old = np.asarray(fields.n, dtype=float)
    shape = n_old.shape
    n_comp = shape[1]
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(shape[0], float(p[0]), dtype=float)
    faces = interior_faces(grid, permeability)
    z_center = grid.cell_centers()[:, 2]
    kw = dict(
        gravity=gravity,
        mu_liquid=mu_liquid,
        mu_vapor=mu_vapor,
        injectors=injectors,
        producers=producers,
    )
    # Start from n_old so the first residual is the implicit defect, not an
    # explicit predictor (that would look like "a large explicit step").
    n_trial = n_old.copy()
    n_hat, injected, produced = _rhs(n_trial, n_old, T, p, mixture, faces, z_center, dt, **kw)
    n_vec = _pack(n_trial)
    R = _residual(n_trial, n_hat)
    hist = [float(np.max(np.abs(R)))]
    scale = max(1.0, float(np.max(np.abs(n_old))))
    n_newton = 0
    if dt == 0.0 or hist[0] < tol * scale:
        out = _fields_from_moles(n_trial, T, p, mixture)
        return ImplicitStepReport(
            out, injected, produced, float(dt), underflow, True, 0, residual_hist=hist
        )

    dim = n_vec.size
    for it in range(1, int(max_iter) + 1):
        n_newton = it
        J = np.zeros((dim, dim), dtype=float)
        for j in range(dim):
            eps = 1.0e-6 * max(1.0, abs(float(n_vec[j])))
            n_p = n_vec.copy()
            n_p[j] += eps
            n_hat_p, _, _ = _rhs(_unpack(n_p, shape), n_old, T, p, mixture, faces, z_center, dt, **kw)
            J[:, j] = (_residual(_unpack(n_p, shape), n_hat_p) - R) / eps
        try:
            dn = np.linalg.solve(J + 1.0e-14 * np.eye(dim), -R)
        except np.linalg.LinAlgError:
            out = _fields_from_moles(n_trial, T, p, mixture)
            return ImplicitStepReport(
                out, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist
            )
        alpha = 1.0
        r0 = float(np.max(np.abs(R)))
        accepted = False
        for _ in range(8):
            n_try = _unpack(n_vec + alpha * dn, shape)
            n_hat, injected, produced = _rhs(n_try, n_old, T, p, mixture, faces, z_center, dt, **kw)
            R_try = _residual(n_try, n_hat)
            if float(np.max(np.abs(R_try))) <= r0 * (1.0 - 1.0e-4 * alpha) or alpha < 0.05:
                n_vec = _pack(n_try)
                n_trial = n_try
                R = R_try
                hist.append(float(np.max(np.abs(R))))
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            out = _fields_from_moles(n_trial, T, p, mixture)
            return ImplicitStepReport(
                out, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist
            )
        if float(np.max(np.abs(R))) < tol * scale:
            out = _fields_from_moles(n_trial, T, p, mixture)
            return ImplicitStepReport(
                out, injected, produced, float(dt), underflow, True, n_newton, residual_hist=hist
            )

    out = _fields_from_moles(n_trial, T, p, mixture)
    return ImplicitStepReport(
        out, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist
    )


def run_implicit_period(
    fields: CompFields,
    T: float,
    pressure: NDArray[np.float64] | float,
    mixture: EosMixture,
    grid: CartesianGrid,
    permeability: NDArray[np.float64] | float,
    duration: float,
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
    """Advance ``duration`` seconds with implicit Newton; grow/hold/chop ``dt``."""
    n_comp = fields.n.shape[1]
    ledger = WellLedger(injected=np.zeros(n_comp, dtype=float), produced=np.zeros(n_comp, dtype=float))
    if float(duration) <= 0.0:
        return fields, ImplicitPeriodLedger(ledger, 0, 0, 0, False, float(dt_init), float(dt_max), [])
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
        report = implicit_newton_step(
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
            injectors=injectors,
            producers=producers,
        )
        n_newton += report.n_newton
        if not report.newton_converged:
            n_chop += 1
            dt = float(chop) * dt
            continue
        current = report.fields
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
    return current, ImplicitPeriodLedger(
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
