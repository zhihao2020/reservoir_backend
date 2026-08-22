"""Coupled implicit Newton with well control. ``T`` is prescribed.

Extends ``implicit_p``. Not the FIM residual and not a rewrite of
``solver/fi.py``.

Rate control (``p_wf`` is a Newton unknown):

    R_wf = Q_spec − Σ_perfs q_PI(p_c, p_wf)

Specified-BHP (``p_wf`` is Dirichlet, not an unknown):

    unknowns are (n_i, p) only
    mass source uses q_PI(p_c, p_wf_spec)
    rate is the residual outcome, not a Newton unknown

    q_PI = ξ WI λ Δp     (signed Peaceman / linear PI)

    inject:  Δp = p_wf − p_c     (ξ = injectate 1/v_mix at (T, p_wf))
    produce: Δp = p_c − p_wf     (ξ = well-cell molar density)

Shut-in (soak): well is off; ``p_wf`` is dropped.

Standalone EXAMPLE kernel.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.accumulation import CellFlash, flash_cell
from reservoir_backend.comp.flux import EXAMPLE_MU_LIQUID, EXAMPLE_MU_VAPOR, interior_faces
from reservoir_backend.comp.implicit import (
    DT_CHOP,
    DT_GROW,
    ImplicitPeriodLedger,
    ImplicitStepReport,
)
from reservoir_backend.comp.implicit_p import (
    NEWTON_MAX,
    P_MAX,
    P_MIN,
    P_REF,
    _pack_np,
    _residual_np,
    _unpack_np,
    _v_mix,
)

# Combined (n, p, p_wf) system is stiffer than lagged-p (n, p); 1e-6 is
# still decades below the O(1) well residual at a cell-pressure first guess.
NEWTON_TOL = 1.0e-6
from reservoir_backend.comp.step import DT_MIN, CompFields, WellLedger, _fields_from_moles
from reservoir_backend.comp.well import RateInjector, RateProducer, well_cell_molar_z
from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

WELL_RATE_CONSTRAINT = (
    "R_wf = Q_spec - sum_perfs q_PI(p_c, p_wf), with "
    "q_PI = xi * WI * lambda * Delta p (signed Peaceman / linear PI). "
    "Rate control: p_wf is a Newton unknown in the same vector as (n_i, p). "
    "Inject Delta p = p_wf - p_c (xi = injectate 1/v_mix at T, p_wf); "
    "produce Delta p = p_c - p_wf (xi = well-cell mix). "
    "Specified BHP is Dirichlet (p_wf not unknown). "
    "Shut-in (soak): p_wf is dropped from the unknown vector."
)

WELL_BHP_CONSTRAINT = (
    "Specified-BHP: p_wf is Dirichlet (not a Newton unknown). "
    "Unknowns are (n_i, p) only. "
    "Mass source is Peaceman q_PI(p_c, p_wf_spec) = xi * WI * lambda * Delta p "
    "(signed linear PI). Rate is the residual outcome, not an unknown. "
    "Soak shuts the well (p_wf dropped)."
)

MIXED_CONTROL = (
    "One well, mixed control: inject is rate control (p_wf Newton unknown, "
    "R_wf = Q_spec - sum q_PI); soak shuts the well and drops p_wf; "
    "produce is specified-BHP (p_wf Dirichlet, mass source q_PI). "
    "Same perforated cells. Not 1-inject-4-produce."
)

Q_REF = 1.0e-8  # mol/s, floors the well-residual scale


def _mobility(cell: CellFlash, mu_liquid: float, mu_vapor: float) -> float:
    return max(cell.S_liquid, 0.0) / max(mu_liquid, 1.0e-30) + max(cell.S_vapor, 0.0) / max(
        mu_vapor, 1.0e-30
    )


def _well_mode(injectors, producers) -> str:
    """``inject`` | ``produce`` | ``inject_bhp`` | ``produce_bhp`` | ``shut``."""
    if injectors:
        if all(getattr(w, "bhp", None) is not None for w in injectors):
            return "inject_bhp"
        return "inject"
    if producers:
        if all(p.molar_rate is not None for p in producers):
            return "produce"
        return "produce_bhp"
    return "shut"


def _bhp_from_wells(mode: str, injectors, producers) -> float:
    if mode == "inject_bhp":
        return float(injectors[0].bhp)
    if mode == "produce_bhp":
        return float(producers[0].bhp)
    raise ValueError(f"{mode} is not a specified-BHP well mode")


def _specified_rate(mode: str, injectors, producers) -> float:
    if mode == "inject":
        return float(sum(float(w.rate) for w in injectors))
    if mode == "produce":
        return float(sum(float(w.molar_rate) for w in producers))
    return 0.0


def _peaceman_rate(
    mode: str,
    fld: CompFields,
    p: NDArray[np.float64],
    p_wf: float,
    T: float,
    mixture: EosMixture,
    injectors,
    producers,
    mu_liquid: float,
    mu_vapor: float,
) -> float:
    """Total signed Peaceman / PI molar rate [mol/s] at trial ``p_wf``."""
    if mode == "inject":
        z_inj = np.asarray(injectors[0].z_inj, dtype=float)
        inj_flash = flash_cell(z_inj, T, float(p_wf), mixture)
        xi_inj = 1.0 / max(_v_mix(inj_flash), 1.0e-30)
        q = 0.0
        for w in injectors:
            cell = fld.cells[int(w.cell)]
            lam = _mobility(cell, mu_liquid, mu_vapor)
            q += xi_inj * float(w.well_index) * lam * (float(p_wf) - float(p[int(w.cell)]))
        return float(q)
    if mode == "produce":
        q = 0.0
        for w in producers:
            c = int(w.cell)
            cell = fld.cells[c]
            xi = cell.xi_liquid * cell.S_liquid + cell.xi_vapor * cell.S_vapor
            lam = _mobility(cell, mu_liquid, mu_vapor)
            q += max(xi, 0.0) * float(w.well_index) * lam * (float(p[c]) - float(p_wf))
        return float(q)
    return 0.0


def _apply_peaceman_bhp(
    n_hat: NDArray[np.float64],
    fld: CompFields,
    p: NDArray[np.float64],
    p_wf: float,
    T: float,
    mixture: EosMixture,
    mode: str,
    injectors,
    producers,
    dt: float,
    mu_liquid: float,
    mu_vapor: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Add specified-BHP Peaceman sources to ``n_hat``. Cap so moles stay non-negative."""
    out = np.asarray(n_hat, dtype=float).copy()
    injected = np.zeros(out.shape[1], dtype=float)
    produced = np.zeros(out.shape[1], dtype=float)
    if mode == "inject_bhp":
        z = np.asarray(injectors[0].z_inj, dtype=float)
        z = z / max(float(z.sum()), 1.0e-30)
        xi = 1.0 / max(_v_mix(flash_cell(z, T, float(p_wf), mixture)), 1.0e-30)
        for w in injectors:
            c = int(w.cell)
            lam = _mobility(fld.cells[c], mu_liquid, mu_vapor)
            q = xi * float(w.well_index) * lam * (float(p_wf) - float(p[c]))
            dn = q * float(dt) * z
            if q >= 0.0:
                out[c] += dn
                injected += dn
            else:
                taken = np.minimum(-dn, np.clip(out[c], 0.0, None))
                out[c] -= taken
                produced += taken
    elif mode == "produce_bhp":
        for w in producers:
            c = int(w.cell)
            cell = fld.cells[c]
            xi = cell.xi_liquid * cell.S_liquid + cell.xi_vapor * cell.S_vapor
            lam = _mobility(cell, mu_liquid, mu_vapor)
            q = max(xi, 0.0) * float(w.well_index) * lam * (float(p[c]) - float(p_wf))
            z = well_cell_molar_z(cell)
            dn = q * float(dt) * z
            if q >= 0.0:
                taken = np.minimum(dn, np.clip(out[c], 0.0, None))
                out[c] -= taken
                produced += taken
            else:
                out[c] += -dn
                injected += -dn
    return out, injected, produced


def _n_hat_from_residual(
    n_trial: NDArray[np.float64], R: NDArray[np.float64], n_ref: float
) -> NDArray[np.float64]:
    """Implicit Euler state ``n_hat`` so inventory matches the well ledger."""
    nc = n_trial.shape[1]
    n_hat = np.empty_like(n_trial)
    for c in range(n_trial.shape[0]):
        i0 = c * (nc + 1)
        n_hat[c] = n_trial[c] - np.asarray(R[i0 : i0 + nc], dtype=float) * n_ref
    return np.clip(n_hat, 0.0, None)


def _pack_bhp(n: NDArray[np.float64], p: NDArray[np.float64], p_wf: float | None) -> NDArray[np.float64]:
    u = _pack_np(n, p)
    if p_wf is not None:
        u = np.append(u, float(p_wf) / P_REF)
    return u


def _unpack_bhp(
    u: NDArray[np.float64], n_cells: int, n_comp: int, has_bhp: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
    n_np = n_cells * (n_comp + 1)
    n, p = _unpack_np(np.asarray(u[:n_np], dtype=float), n_cells, n_comp)
    p_wf = float(np.clip(float(u[-1]) * P_REF, P_MIN, P_MAX)) if has_bhp else None
    return n, p, p_wf


def _residual_bhp(
    n_trial: NDArray[np.float64],
    p_trial: NDArray[np.float64],
    p_wf: float | None,
    n_old: NDArray[np.float64],
    T: float,
    mixture: EosMixture,
    faces,
    z_center: NDArray[np.float64],
    dt: float,
    pore_volume: NDArray[np.float64],
    n_ref: float,
    mode: str,
    **kw,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], CompFields]:
    if mode in ("inject_bhp", "produce_bhp"):
        if p_wf is None:
            raise ValueError("specified-BHP residual needs p_wf")
        kw_shut = dict(kw, injectors=None, producers=None)
        r, _, _, fld = _residual_np(
            n_trial, p_trial, n_old, T, mixture, faces, z_center, dt, pore_volume, n_ref, **kw_shut
        )
        nc = n_trial.shape[1]
        r_n = np.zeros_like(n_trial)
        for c in range(n_trial.shape[0]):
            i0 = c * (nc + 1)
            r_n[c] = r[i0 : i0 + nc]
        n_hat = n_trial - r_n * n_ref
        n_hat, injected, produced = _apply_peaceman_bhp(
            n_hat,
            fld,
            p_trial,
            float(p_wf),
            T,
            mixture,
            mode,
            kw.get("injectors"),
            kw.get("producers"),
            dt,
            float(kw.get("mu_liquid", EXAMPLE_MU_LIQUID)),
            float(kw.get("mu_vapor", EXAMPLE_MU_VAPOR)),
        )
        r_n = (n_trial - n_hat) / n_ref
        for c in range(n_trial.shape[0]):
            i0 = c * (nc + 1)
            r[i0 : i0 + nc] = r_n[c]
        return r, injected, produced, fld
    r, injected, produced, fld = _residual_np(
        n_trial, p_trial, n_old, T, mixture, faces, z_center, dt, pore_volume, n_ref, **kw
    )
    if mode not in ("inject", "produce") or p_wf is None:
        return r, injected, produced, fld
    q_spec = _specified_rate(mode, kw.get("injectors"), kw.get("producers"))
    q_pea = _peaceman_rate(
        mode,
        fld,
        p_trial,
        float(p_wf),
        T,
        mixture,
        kw.get("injectors"),
        kw.get("producers"),
        float(kw.get("mu_liquid", EXAMPLE_MU_LIQUID)),
        float(kw.get("mu_vapor", EXAMPLE_MU_VAPOR)),
    )
    r_wf = (q_spec - q_pea) / max(abs(q_spec), Q_REF)
    return np.append(r, r_wf), injected, produced, fld


def _init_pwf(mode: str, p: NDArray[np.float64], p_wf: float | None) -> float | None:
    if mode not in ("inject", "produce"):
        return None
    if p_wf is not None:
        return float(np.clip(p_wf, P_MIN, P_MAX))
    # Start on the cell-pressure side so R_wf is O(1) (Q_pea ≈ 0) and the
    # first Newton step has to move p_wf.
    if mode == "inject":
        return float(np.max(p))
    return float(np.min(p))


def implicit_newton_step_bhp(
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
    p_wf: float | None = None,
    tol: float = NEWTON_TOL,
    max_iter: int = NEWTON_MAX,
) -> ImplicitStepReport:
    """One implicit Euler Newton step.

    Rate control: unknowns ``(n_i, p, p_wf)``. Specified-BHP: unknowns
    ``(n_i, p)`` only; ``p_wf`` is Dirichlet.
    """
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
    mode = _well_mode(injectors, producers)
    has_bhp = mode in ("inject", "produce")
    if mode in ("inject_bhp", "produce_bhp"):
        p_wf_trial = _bhp_from_wells(mode, injectors, producers)
    else:
        p_wf_trial = _init_pwf(mode, p, p_wf)
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
    R, injected, produced, fld = _residual_bhp(
        n_trial, p_trial, p_wf_trial, n_old, T, mixture, faces, z_center, dt, vp, n_ref, mode, **kw
    )
    hist = [float(np.max(np.abs(R)))]
    n_unknowns = n_cells * (n_comp + 1) + (1 if has_bhp else 0)

    def _extra() -> dict:
        return dict(
            pressure=p_trial.copy(),
            has_pressure_unknown=True,
            n_unknowns=n_unknowns,
            has_bhp_unknown=has_bhp,
            bhp=None if p_wf_trial is None else float(p_wf_trial),
        )

    def _commit(fld_in: CompFields, R_now: NDArray[np.float64]) -> CompFields:
        out = _fields_from_moles(_n_hat_from_residual(n_trial, R_now, n_ref), T, p_trial, mixture)
        out.p = p_trial.copy()
        return out

    if dt == 0.0 or hist[0] < tol:
        return ImplicitStepReport(
            _commit(fld, R), injected, produced, float(dt), underflow, True, 0, residual_hist=hist, **_extra()
        )

    u = _pack_bhp(n_trial, p_trial, p_wf_trial if has_bhp else None)
    n_newton = 0
    for it in range(1, int(max_iter) + 1):
        n_newton = it
        J = np.zeros((n_unknowns, n_unknowns), dtype=float)
        for j in range(n_unknowns):
            if has_bhp and j == n_unknowns - 1:
                eps = 1.0e-8 * max(1.0, abs(float(u[j])))
            else:
                eps = 1.0e-6 * max(1.0, abs(float(u[j])))
            u_p = u.copy()
            u_p[j] += eps
            n_p, p_p, wf_p = _unpack_bhp(u_p, n_cells, n_comp, has_bhp)
            if not has_bhp:
                wf_p = p_wf_trial
            R_p, _, _, _ = _residual_bhp(
                n_p, p_p, wf_p, n_old, T, mixture, faces, z_center, dt, vp, n_ref, mode, **kw
            )
            J[:, j] = (R_p - R) / eps
        try:
            du = np.linalg.solve(J + 1.0e-12 * np.eye(n_unknowns), -R)
        except np.linalg.LinAlgError:
            fld.p = p_trial.copy()
            return ImplicitStepReport(
                fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **_extra()
            )
        alpha = 1.0
        r0 = float(np.max(np.abs(R)))
        accepted = False
        for _ in range(8):
            n_try, p_try, wf_try = _unpack_bhp(u + alpha * du, n_cells, n_comp, has_bhp)
            if not has_bhp:
                wf_try = p_wf_trial
            R_try, injected, produced, fld = _residual_bhp(
                n_try, p_try, wf_try, n_old, T, mixture, faces, z_center, dt, vp, n_ref, mode, **kw
            )
            r_try = float(np.max(np.abs(R_try)))
            r_np_try = float(np.max(np.abs(R_try[:-1]))) if has_bhp else r_try
            r_np_0 = float(np.max(np.abs(R[:-1]))) if has_bhp else r0
            # Accept a mass/volume drop even if R_wf jumps (PI coefficients
            # update with the new flash). Next iteration drives R_wf down.
            if (
                r_try <= r0 * (1.0 - 1.0e-4 * alpha)
                or (has_bhp and r_np_try <= r_np_0 / 10.0)
                or (alpha < 0.05 and r_try <= r0)
            ):
                u = _pack_bhp(n_try, p_try, wf_try if has_bhp else None)
                n_trial, p_trial = n_try, p_try
                if has_bhp:
                    p_wf_trial = wf_try
                R = R_try
                hist.append(float(np.max(np.abs(R))))
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            fld.p = p_trial.copy()
            return ImplicitStepReport(
                fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **_extra()
            )
        if float(np.max(np.abs(R))) < tol:
            return ImplicitStepReport(
                _commit(fld, R), injected, produced, float(dt), underflow, True, n_newton, residual_hist=hist, **_extra()
            )

    fld.p = p_trial.copy()
    return ImplicitStepReport(
        fld, injected, produced, float(dt), underflow, False, n_newton, residual_hist=hist, **_extra()
    )


def run_implicit_period_bhp(
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
    p_wf: float | None = None,
    grow: float = DT_GROW,
    chop: float = DT_CHOP,
) -> tuple[CompFields, ImplicitPeriodLedger]:
    """Advance ``duration`` with coupled ``(n_i, p, p_wf)`` Newton; soak drops ``p_wf``."""
    n_comp = fields.n.shape[1]
    n_cells = fields.n.shape[0]
    ledger = WellLedger(injected=np.zeros(n_comp, dtype=float), produced=np.zeros(n_comp, dtype=float))
    p = np.asarray(pressure, dtype=float).ravel()
    if p.size == 1:
        p = np.full(n_cells, float(p[0]), dtype=float)
    wf = p_wf
    if float(duration) <= 0.0:
        empty = ImplicitPeriodLedger(ledger, 0, 0, 0, False, float(dt_init), float(dt_max), [])
        empty.pressure = p
        empty.bhp = wf
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
    last_bhp = wf
    while t < float(duration) - 1.0e-12:
        dt = min(dt, float(duration) - t, float(dt_max))
        if dt < DT_MIN:
            underflow = True
            break
        report = implicit_newton_step_bhp(
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
            p_wf=wf,
        )
        n_newton += report.n_newton
        if not report.newton_converged:
            n_chop += 1
            dt = float(chop) * dt
            continue
        current = report.fields
        if report.pressure is not None:
            p = np.asarray(report.pressure, dtype=float).ravel()
        wf = report.bhp
        last_bhp = report.bhp
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
    out.bhp = last_bhp
    return current, out
