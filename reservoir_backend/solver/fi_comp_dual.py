"""Fully implicit compositional DPDP Newton. Does not edit ``fi_comp.py``.

Unknowns: fracture (n_f[0:Nc], p_f) then matrix (n_m[0:Nc], p_m) per cell.
Colored FD Jacobian is stored as CSR; linear solve is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from numpy.typing import NDArray
from reservoir_backend.comp.dual_residual import dual_residual, pack_dual, unpack_dual
from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.properties import flash_state, last_flash_seconds, moles_from_z
from reservoir_backend.comp.wells import well_molar_sources
from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import PhysicsConvergenceError, TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.dpdp_blocks import assemble_block_jacobian
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.dpdp_jacobian import fill_column_slice, residual_scales
from reservoir_backend.solver.fi import clip_dt_to_report_times, dt_from_newton_iters, index_nearest_time
from reservoir_backend.solver.fi_comp import _control_map, _mass_pack
from reservoir_backend.solver.impes import StepReport, Trajectory
from reservoir_backend.solver.linear import solve_newton_system


@dataclass
class DualCompStepResult:
    state: DualCompositionalState
    newton_iters: int
    port_rates: dict[str, float]
    port_bhp: dict[str, float]
    q_src_fracture: NDArray[np.float64]
    q_src_matrix: NDArray[np.float64]
    max_mass_residual: float = 0.0
    max_volume_residual: float = 0.0
    newton_norm: float = 0.0
    jac_s: float = 0.0
    solve_s: float = 0.0
    resid_s: float = 0.0
    flash_s: float = 0.0
    flash_jac_s: float = 0.0
    n_flash_main: int = 0
    n_flash_thermo_jac: int = 0
    n_flash_line_search: int = 0
    n_jac_reuse: int = 0
    linear_iterations: int = 0
    linear_residual: float = 0.0
    linear_setup_s: float = 0.0
    linear_method: str = ""


def initialize_dual_state(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    p_fracture: float,
    p_matrix: float | None = None,
) -> DualCompositionalState:
    n = grid.n_cells
    vol = grid.cell_volumes()
    pf = np.full(n, float(p_fracture))
    pm = np.full(n, float(p_fracture if p_matrix is None else p_matrix))
    n_f = moles_from_z(spec, pf, spec.z_init, dual_rock.fracture.porosity * vol)
    n_m = moles_from_z(spec, pm, spec.z_init, dual_rock.matrix.porosity * vol)
    return DualCompositionalState(
        fracture=CompositionalContinuumState(pf, n_f),
        matrix=CompositionalContinuumState(pm, n_m),
        time_s=0.0,
    )


def dual_to_state(
    spec: CompSpec,
    dual: DualCompositionalState,
    dual_rock: DualRock | None = None,
    *,
    props_f=None,
    props_m=None,
) -> State:
    """Keep both continua on State so H can select fracture / matrix / bulk."""
    held = props_f is not None and props_m is not None
    if props_f is None:
        props_f = flash_state(spec, dual.fracture.pressure, dual.fracture.moles)
    if props_m is None:
        props_m = flash_state(spec, dual.matrix.pressure, dual.matrix.moles)
    phi_f = None if dual_rock is None else np.asarray(dual_rock.fracture.porosity, dtype=float)
    phi_m = None if dual_rock is None else np.asarray(dual_rock.matrix.porosity, dtype=float)
    return State(
        pressure=dual.fracture.pressure.copy(),
        sw=props_f.sw.copy(),
        sg=props_f.sv.copy(),
        moles=dual.fracture.moles.copy(),
        moles_matrix=dual.matrix.moles.copy(),
        time_s=float(dual.time_s),
        pressure_matrix=dual.matrix.pressure.copy(),
        sw_matrix=props_m.sw.copy(),
        sg_matrix=props_m.sv.copy(),
        phi_fracture=None if phi_f is None else phi_f.copy(),
        phi_matrix=None if phi_m is None else phi_m.copy(),
        saturations_held=bool(held),
    )


def dual_from_visual_state(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    state: State,
) -> DualCompositionalState:
    """Restore DualCompositionalState. Matrix moles are required for a lossless restart."""
    if state.moles is not None and state.moles_matrix is not None:
        pm = state.pressure_matrix if state.pressure_matrix is not None else state.pressure
        return DualCompositionalState(
            fracture=CompositionalContinuumState(np.asarray(state.pressure, dtype=float).copy(), np.asarray(state.moles, dtype=float).copy()),
            matrix=CompositionalContinuumState(np.asarray(pm, dtype=float).copy(), np.asarray(state.moles_matrix, dtype=float).copy()),
            time_s=float(state.time_s),
        )
    if float(state.time_s) > 1.0e-15:
        raise ValueError("lossless DPDP restart requires moles_matrix at t>0")
    dual = initialize_dual_state(grid, dual_rock, spec, float(np.mean(state.pressure)))
    dual.time_s = float(state.time_s)
    dual.fracture.pressure = np.asarray(state.pressure, dtype=float).ravel().copy()
    if state.moles is not None:
        dual.fracture.moles = np.asarray(state.moles, dtype=float).copy()
    if state.pressure_matrix is not None:
        dual.matrix.pressure = np.asarray(state.pressure_matrix, dtype=float).ravel().copy()
    return dual


def _state_from_u(u: NDArray[np.float64], n_cells: int, nc: int, time_s: float) -> DualCompositionalState:
    nf, pf, nm, pm = unpack_dual(u, n_cells, nc)
    return DualCompositionalState(
        fracture=CompositionalContinuumState(pf, nf),
        matrix=CompositionalContinuumState(pm, nm),
        time_s=time_s,
    )


def _wells(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    state: DualCompositionalState,
    ports: list[FlowPort],
    cmap: dict[tuple[str, str], ControlSeries],
    t_eval: float,
    props_f,
    props_m,
    *,
    need_bhp: bool,
):
    q_f = np.zeros((grid.n_cells, spec.nc))
    q_m = np.zeros((grid.n_cells, spec.nc))
    rates: dict[str, float] = {}
    bhp: dict[str, float] = {}
    if not ports:
        return q_f, q_m, rates, bhp
    for port in ports:
        coupling = str(getattr(port, "continuum_coupling", "fracture"))
        frac = float(getattr(port, "fracture_fraction", 1.0))
        if coupling == "matrix":
            rock, p, props, dest_f = dual_rock.matrix, state.matrix.pressure, props_m, 0.0
        else:
            rock, p, props, dest_f = dual_rock.fracture, state.fracture.pressure, props_f, 1.0 if coupling == "fracture" else frac
        q, r, b = well_molar_sources(
            grid, rock, [port], cmap, p, props, spec, t_eval, need_bhp=need_bhp
        )
        if coupling == "matrix":
            q_m = q_m + q
        elif coupling == "split":
            q_f = q_f + dest_f * q
            q_m = q_m + (1.0 - dest_f) * q
        else:
            q_f = q_f + q
        rates.update(r)
        bhp.update(b)
    return q_f, q_m, rates, bhp


def _residual(
    grid,
    dual_rock,
    spec,
    state,
    old,
    dt,
    transfer,
    t_f,
    t_m,
    ports,
    cmap,
    t_eval,
    *,
    props_f=None,
    props_m=None,
    reflash_f=None,
    reflash_m=None,
    need_bhp=False,
    reflash_all=False,
):
    flash_s = 0.0
    hint = getattr(state, "flash", None)
    kf = None if hint is None or hint.fracture is None else hint.fracture.k
    km = None if hint is None or hint.matrix is None else hint.matrix.k
    if props_f is None or reflash_all:
        props_f = flash_state(spec, state.fracture.pressure, state.fracture.moles, out=props_f, k_hint=kf)
        flash_s += last_flash_seconds()
    elif reflash_f is not None:
        flash_state(spec, state.fracture.pressure, state.fracture.moles, cells=reflash_f, out=props_f)
        flash_s += last_flash_seconds()
    if props_m is None or reflash_all:
        props_m = flash_state(spec, state.matrix.pressure, state.matrix.moles, out=props_m, k_hint=km)
        flash_s += last_flash_seconds()
    elif reflash_m is not None:
        flash_state(spec, state.matrix.pressure, state.matrix.moles, cells=reflash_m, out=props_m)
        flash_s += last_flash_seconds()
    from reservoir_backend.eos.flash_cache import DualFlashCache, FlashCache

    state.flash = DualFlashCache(FlashCache.from_props(props_f), FlashCache.from_props(props_m))
    q_f, q_m, rates, bhp = _wells(
        grid, dual_rock, spec, state, ports, cmap, t_eval, props_f, props_m, need_bhp=need_bhp
    )
    res, props_f, props_m, _tr = dual_residual(
        grid,
        dual_rock,
        spec,
        state,
        old,
        dt,
        transfer,
        q_src_fracture=q_f,
        q_src_matrix=q_m,
        t_fracture=t_f,
        t_matrix=t_m,
        props_fracture=props_f,
        props_matrix=props_m,
    )
    return res, props_f, props_m, rates, bhp, q_f, q_m, flash_s


def _coloring_jacobian(
    ctx: DPDPModelContext,
    spec: CompSpec,
    dual_rock: DualRock,
    old: DualCompositionalState,
    dt: float,
    transfer: ComponentTransfer,
    t_f,
    t_m,
    ports,
    cmap,
    t1: float,
    u: NDArray[np.float64],
    res0: NDArray[np.float64],
    props_f0,
    props_m0,
    n_scale: float,
    p_scale: float,
):
    """Colored FD into the cached CSR pattern."""
    grid = ctx.grid
    n_cells = grid.n_cells
    nc = spec.nc
    nu = nc + 1
    half = n_cells * nu
    pattern = ctx.pattern
    data = pattern.empty_data()
    eps_n = 1.0e-8 * max(n_scale, 1.0)
    eps_p = 1.0e-8 * max(p_scale, 1.0e5)
    nf0, pf0, nm0, pm0 = unpack_dual(u, n_cells, nc)
    t_flash = 0.0
    for cells in ctx.color_cells:
        if cells.size == 0:
            continue
        for cont in (0, 1):
            offset = cont * half
            for slot in range(nu):
                nf, pf, nm, pm = nf0.copy(), pf0.copy(), nm0.copy(), pm0.copy()
                pf_props = props_f0.copy()
                pm_props = props_m0.copy()
                eps = eps_n if slot < nc else eps_p
                if cont == 0:
                    if slot < nc:
                        nf[cells, slot] = nf[cells, slot] + eps
                    else:
                        pf[cells] = pf[cells] + eps
                    reflash_f, reflash_m = cells, None
                else:
                    if slot < nc:
                        nm[cells, slot] = nm[cells, slot] + eps
                    else:
                        pm[cells] = pm[cells] + eps
                    reflash_f, reflash_m = None, cells
                trial = DualCompositionalState(
                    fracture=CompositionalContinuumState(pf, nf),
                    matrix=CompositionalContinuumState(pm, nm),
                    time_s=t1,
                )
                r2, _, _, _, _, _, _, fls = _residual(
                    grid,
                    dual_rock,
                    spec,
                    trial,
                    old,
                    dt,
                    transfer,
                    t_f,
                    t_m,
                    ports,
                    cmap,
                    t1,
                    props_f=pf_props,
                    props_m=pm_props,
                    reflash_f=reflash_f,
                    reflash_m=reflash_m,
                    need_bhp=False,
                )
                t_flash += float(fls)
                dres = (r2 - res0) / eps
                for c in cells:
                    fill_column_slice(pattern, data, offset + int(c) * nu + slot, dres)
    return pattern.to_csr(data), t_flash


def _residual_stats(res: NDArray[np.float64], n_cells: int, nc: int, scale: NDArray[np.float64]) -> tuple[float, float, float]:
    block = np.asarray(res, dtype=float).reshape(2, n_cells, nc + 1)
    mass = float(np.max(np.abs(block[:, :, :nc])))
    vol = float(np.max(np.abs(block[:, :, nc])))
    nrm = float(np.max(np.abs(res * scale)))
    return mass, vol, nrm


def _well_source_fd(
    grid,
    dual_rock,
    spec,
    state,
    dt,
    ports,
    cmap,
    t1,
    props_f,
    props_m,
    n_scale,
    p_scale,
):
    from scipy import sparse

    n_cells = grid.n_cells
    nc = spec.nc
    nu = nc + 1
    n_u = 2 * n_cells * nu
    eps_n = 1.0e-8 * max(float(n_scale), 1.0)
    eps_p = 1.0e-8 * max(float(p_scale), 1.0e5)
    qf0, qm0, _, _ = _wells(grid, dual_rock, spec, state, ports, cmap, t1, props_f, props_m, need_bhp=False)
    cells = {int(c) for port in ports for c in port.cell_ids}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    nf = state.fracture.moles
    pf = state.fracture.pressure
    nm = state.matrix.moles
    pm = state.matrix.pressure
    for c in cells:
        for cont in (0, 1):
            for slot in range(nu):
                n_f, p_f, n_m, p_m = nf.copy(), pf.copy(), nm.copy(), pm.copy()
                eps = eps_n if slot < nc else eps_p
                if cont == 0:
                    if slot < nc:
                        n_f[c, slot] = n_f[c, slot] + eps
                    else:
                        p_f[c] = p_f[c] + eps
                else:
                    if slot < nc:
                        n_m[c, slot] = n_m[c, slot] + eps
                    else:
                        p_m[c] = p_m[c] + eps
                trial = DualCompositionalState(
                    fracture=CompositionalContinuumState(p_f, n_f),
                    matrix=CompositionalContinuumState(p_m, n_m),
                    time_s=state.time_s,
                )
                pf_p, pm_p = props_f.copy(), props_m.copy()
                if cont == 0:
                    flash_state(spec, p_f, n_f, cells=np.array([c]), out=pf_p)
                else:
                    flash_state(spec, p_m, n_m, cells=np.array([c]), out=pm_p)
                qf, qm, _, _ = _wells(grid, dual_rock, spec, trial, ports, cmap, t1, pf_p, pm_p, need_bhp=False)
                col = cont * n_cells * nu + c * nu + slot
                for cont_r, block in ((0, (qf - qf0) / eps), (1, (qm - qm0) / eps)):
                    hit = np.argwhere(np.abs(block) > 1.0e-18)
                    for cc, i in hit:
                        val = -float(dt) * float(block[int(cc), int(i)])
                        rows.append(cont_r * n_cells * nu + int(cc) * nu + int(i))
                        cols.append(col)
                        data.append(val)
    if not data:
        return sparse.csc_matrix((n_u, n_u))
    return sparse.csc_matrix((data, (rows, cols)), shape=(n_u, n_u))


def _clip_newton_step(
    step: NDArray[np.float64],
    u: NDArray[np.float64],
    n_cells: int,
    nc: int,
    *,
    dp_max: float = 5.0e6,
    dz_max: float = 0.25,
) -> NDArray[np.float64]:
    """Cap |Δp| and |Δz| on the Newton increment before line search."""
    w = np.asarray(step, dtype=float).reshape(2, n_cells, nc + 1)
    w[:, :, nc] = np.clip(w[:, :, nc], -float(dp_max), float(dp_max))
    n = np.asarray(u, dtype=float).reshape(2, n_cells, nc + 1)[:, :, :nc]
    n_tot = np.maximum(np.sum(np.abs(n), axis=2, keepdims=True), 1.0e-18)
    w[:, :, :nc] = np.clip(w[:, :, :nc], -float(dz_max) * n_tot, float(dz_max) * n_tot)
    return w.ravel()


def solve_dual_comp_step(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    state: DualCompositionalState,
    dt: float,
    transfer: ComponentTransfer,
    *,
    ports: list[FlowPort] | None = None,
    controls: list[ControlSeries] | dict[tuple[str, str], ControlSeries] | None = None,
    max_newton: int = 20,
    tol: float = 1.0e-8,
    context: DPDPModelContext | None = None,
) -> DualCompStepResult:
    """One fully implicit DPDP step. Raises on Newton failure (caller chops Δt)."""
    ctx = context if context is not None else DPDPModelContext.build(grid, spec.nc)
    old = state.copy()
    n_cells = grid.n_cells
    nc = spec.nc
    t_f, t_m = ctx.transmissibilities(dual_rock)
    u = pack_dual(state)
    t1 = float(state.time_s) + float(dt)
    ports = list(ports or [])
    if isinstance(controls, dict):
        cmap = controls
    else:
        cmap = _control_map(list(controls or []))
    last_res = None
    last_rates: dict[str, float] = {}
    last_bhp: dict[str, float] = {}
    last_qf = np.zeros((n_cells, nc))
    last_qm = np.zeros((n_cells, nc))
    t_jac = 0.0
    t_solve = 0.0
    t_res = 0.0
    t_flash = 0.0
    t_flash_jac = 0.0
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0e-6)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)
    pv = np.asarray(dual_rock.fracture.porosity, dtype=float).ravel() * grid.cell_volumes()
    pv_scale = max(float(np.mean(pv)), 1.0e-12)
    row_s = residual_scales(n_cells, nc, n_scale, pv_scale)
    props_f = None
    props_m = None
    jac_hold = None
    jac_s_hold = None
    phase_hold = None
    reuse_left = 0
    n_flash_main = 0
    n_flash_thermo_jac = 0
    n_flash_line_search = 0
    n_jac_reuse = 0
    n_cells_flash = 2 * n_cells
    jacobian_reuse_max = 1 if n_cells > 4 else 0
    nrm_prev = None

    linear_it = 0
    linear_setup = 0.0
    linear_res = 0.0
    linear_method = ""
    for it in range(int(max_newton)):
        trial = _state_from_u(u, n_cells, nc, t1)
        t_r0 = time.perf_counter()
        res, props_f, props_m, last_rates, last_bhp, last_qf, last_qm, fls = _residual(
            grid, dual_rock, spec, trial, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=True
        )
        t_flash += float(fls)
        n_flash_main += n_cells_flash
        t_res += time.perf_counter() - t_r0
        last_res = res
        mass_r, vol_r, nrm = _residual_stats(res, n_cells, nc, row_s)
        if nrm < float(tol):
            trial.time_s = t1
            return DualCompStepResult(
                state=trial,
                newton_iters=it + 1,
                port_rates=dict(last_rates),
                port_bhp=dict(last_bhp),
                q_src_fracture=last_qf,
                q_src_matrix=last_qm,
                max_mass_residual=mass_r,
                max_volume_residual=vol_r,
                newton_norm=nrm,
                jac_s=t_jac,
                solve_s=t_solve,
                resid_s=t_res,
                flash_s=t_flash,
                flash_jac_s=t_flash_jac,
                n_flash_main=n_flash_main,
                n_flash_thermo_jac=n_flash_thermo_jac,
                n_flash_line_search=n_flash_line_search,
                n_jac_reuse=n_jac_reuse,
                linear_iterations=linear_it,
                linear_residual=linear_res,
                linear_setup_s=linear_setup,
                linear_method=linear_method,
            )
        phase_now = np.concatenate([props_f.two_phase, props_m.two_phase])
        drop_ok = nrm_prev is not None and nrm < 0.5 * float(nrm_prev)
        can_reuse = (
            jac_s_hold is not None
            and reuse_left > 0
            and phase_hold is not None
            and bool(np.array_equal(phase_now, phase_hold))
            and (drop_ok or n_cells >= 1000)
        )
        t_j0 = time.perf_counter()
        if can_reuse:
            jac_s = jac_s_hold
            reuse_left -= 1
            n_jac_reuse += 1
            fls_j = 0.0
        else:
            jac, fls_j = assemble_block_jacobian(
                grid, spec, dual_rock, trial, dt, transfer, t_f, t_m, props_f, props_m, n_scale, p_scale
            )
            n_flash_thermo_jac += n_cells_flash * (nc + 1)
            if ports:
                jac = jac + _well_source_fd(
                    grid, dual_rock, spec, trial, dt, ports, cmap, t1, props_f, props_m, n_scale, p_scale
                )
            jac_s_hold = jac.tocsr().multiply(row_s[:, None]).tocsc()
            jac_s = jac_s_hold
            phase_hold = phase_now
            reuse_left = int(jacobian_reuse_max)
        t_flash_jac += float(fls_j)
        t_jac += time.perf_counter() - t_j0
        nrm_prev = nrm
        t_s0 = time.perf_counter()
        lin = solve_newton_system(jac_s, -(res * row_s), n_comp=nc)
        t_solve += time.perf_counter() - t_s0
        linear_it += int(getattr(lin, "iterations", 0) or 0)
        linear_setup += float(getattr(lin, "setup_s", 0.0) or 0.0)
        linear_res = float(getattr(lin, "final_residual", 0.0) or 0.0)
        linear_method = str(getattr(lin, "method", "") or "")
        step = lin.x
        if not np.all(np.isfinite(step)):
            raise PhysicsConvergenceError("DPDP Newton step is not finite")
        step = _clip_newton_step(step, u, n_cells, nc)
        alpha = 1.0
        accepted = False
        r0 = float(np.linalg.norm(res * row_s))
        for _ in range(8):
            u_try = u + alpha * step
            trial2 = _state_from_u(u_try, n_cells, nc, t1)
            trial2.fracture.moles = np.maximum(trial2.fracture.moles, 1.0e-18)
            trial2.matrix.moles = np.maximum(trial2.matrix.moles, 1.0e-18)
            trial2.fracture.pressure = np.clip(trial2.fracture.pressure, 1.0e4, 1.0e9)
            trial2.matrix.pressure = np.clip(trial2.matrix.pressure, 1.0e4, 1.0e9)
            packed = pack_dual(trial2)
            r_try, _, _, rates_try, bhp_try, qf_try, qm_try, fls_try = _residual(
                grid, dual_rock, spec, trial2, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=False
            )
            t_flash += float(fls_try)
            n_flash_line_search += n_cells_flash
            if float(np.linalg.norm(r_try * row_s)) <= (1.0 - 1.0e-4 * alpha) * r0:
                u = packed
                last_rates, last_bhp, last_qf, last_qm = rates_try, bhp_try, qf_try, qm_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            raise PhysicsConvergenceError("DPDP Newton line search failed")
    nrm = float(np.max(np.abs(last_res * row_s))) if last_res is not None else float("inf")
    raise PhysicsConvergenceError(f"DPDP Newton failed, ||R*||_inf={nrm:.3e}")


def simulate_dual_comp(
    grid: CartesianGrid,
    dual_rock: DualRock,
    spec: CompSpec,
    transfer: ComponentTransfer,
    ports: list[FlowPort],
    controls: list[ControlSeries],
    state0: DualCompositionalState | State,
    t_end: float,
    *,
    dt_init: float = 10.0,
    dt_min: float = 1.0e-6,
    dt_max: float = 60.0,
    max_steps: int = 12000,
    report_times: NDArray[np.float64] | None = None,
    context: DPDPModelContext | None = None,
    matrix_intercell: bool = True,
) -> tuple[Trajectory, DualCompositionalState]:
    """Time loop for compositional DPDP. Port coupling selects fracture/matrix/split."""
    ctx = context if context is not None else DPDPModelContext.build(grid, spec.nc, matrix_intercell=matrix_intercell)
    if isinstance(state0, DualCompositionalState):
        dual = state0.copy()
    else:
        dual = dual_from_visual_state(grid, dual_rock, spec, state0)
    moles0 = dual.total_moles()
    injected = np.zeros(spec.nc)
    produced = np.zeros(spec.nc)
    t = float(dual.time_s)
    t_end = float(t_end)
    dt = min(float(dt_init), float(dt_max))
    reports: list[StepReport] = []
    vis = dual_to_state(spec, dual, dual_rock)
    states = [vis]
    times = [t]
    cmap = _control_map(controls)
    if ports:
        props0 = flash_state(spec, dual.fracture.pressure, dual.fracture.moles)
        _, rates0, bhp0 = well_molar_sources(
            grid, dual_rock.fracture, ports, cmap, dual.fracture.pressure, props0, spec, t
        )
    else:
        rates0, bhp0 = {}, {}
    rates_hist = [dict(rates0)]
    bhp_hist = [dict(bhp0)]
    n_acc = 0
    n_reject = 0
    last_its = 5
    sum_jac = 0.0
    sum_solve = 0.0
    sum_res = 0.0
    sum_flash = 0.0
    sum_flash_jac = 0.0
    n_flash_main = 0
    n_flash_thermo_jac = 0
    n_flash_line_search = 0
    n_jac_reuse = 0
    sum_lin_it = 0
    max_dp_all = 0.0
    max_ds_all = 0.0

    while t < t_end - 1.0e-15:
        if n_acc >= int(max_steps):
            raise TimeStepUnderflow(f"DPDP stepper took more than {max_steps} steps")
        dt = min(dt, t_end - t, float(dt_max))
        dt = clip_dt_to_report_times(t, dt, report_times, t_end)
        if dt < float(dt_min):
            if (t_end - t) <= float(dt_min):
                break
            raise TimeStepUnderflow(f"failed to accept a DPDP step at t={t}")
        try:
            nxt = solve_dual_comp_step(
                grid, dual_rock, spec, dual, dt, transfer, ports=ports, controls=cmap, context=ctx
            )
        except PhysicsConvergenceError:
            n_reject += 1
            dt *= 0.5
            continue
        q_tot = nxt.q_src_fracture + nxt.q_src_matrix
        inj = np.sum(np.maximum(q_tot, 0.0), axis=0) * dt
        prod = np.sum(np.maximum(-q_tot, 0.0), axis=0) * dt
        injected = injected + inj
        produced = produced + prod
        prev = dual
        dual = nxt.state
        t = float(dual.time_s)
        n_acc += 1
        mb = _mass_pack(
            moles0.reshape(1, -1),
            dual.total_moles().reshape(1, -1),
            injected,
            produced,
        )
        sum_jac += float(nxt.jac_s)
        sum_solve += float(nxt.solve_s)
        sum_res += float(nxt.resid_s)
        sum_flash += float(nxt.flash_s)
        sum_flash_jac += float(nxt.flash_jac_s)
        n_flash_main += int(nxt.n_flash_main)
        n_flash_thermo_jac += int(nxt.n_flash_thermo_jac)
        n_flash_line_search += int(nxt.n_flash_line_search)
        n_jac_reuse += int(nxt.n_jac_reuse)
        sum_lin_it += int(getattr(nxt, "linear_iterations", 0) or 0)
        reports.append(
            StepReport(
                time_s=t,
                dt=dt,
                max_cfl=0.0,
                max_ds=0.0,
                mass=mb,
                port_rates=dict(nxt.port_rates),
                notes=[
                    f"max_mass_residual={nxt.max_mass_residual:.3e}",
                    f"max_volume_residual={nxt.max_volume_residual:.3e}",
                    f"newton_norm={nxt.newton_norm:.3e}",
                    f"jac_s={nxt.jac_s:.4f}",
                    f"solve_s={nxt.solve_s:.4f}",
                    f"resid_s={nxt.resid_s:.4f}",
                    f"flash_s={nxt.flash_s:.4f}",
                    f"flash_main_s={nxt.flash_s:.4f}",
                    f"flash_jacobian_s={nxt.flash_jac_s:.4f}",
                    f"n_flash_main={nxt.n_flash_main}",
                    f"n_flash_thermo_jac={nxt.n_flash_thermo_jac}",
                    f"n_flash_line_search={nxt.n_flash_line_search}",
                    f"n_reject={n_reject}",
                    f"linear_iterations={getattr(nxt, 'linear_iterations', 0)}",
                    f"linear_method={getattr(nxt, 'linear_method', '')}",
                    f"linear_setup_s={float(getattr(nxt, 'linear_setup_s', 0.0) or 0.0):.4f}",
                    f"linear_residual={float(getattr(nxt, 'linear_residual', 0.0) or 0.0):.3e}",
                ],
                newton_its=nxt.newton_iters,
            )
        )
        vis = dual_to_state(spec, dual, dual_rock)
        ds = 0.0
        if states:
            ds = float(np.max(np.abs(vis.sw - states[-1].sw)))
            if vis.sg is not None and states[-1].sg is not None:
                ds = max(ds, float(np.max(np.abs(vis.sg - states[-1].sg))))
        reports[-1].max_ds = ds
        max_ds_all = max(max_ds_all, ds)
        states.append(vis)
        times.append(t)
        rates_hist.append(dict(nxt.port_rates))
        bhp_hist.append(dict(nxt.port_bhp))
        dt = dt_from_newton_iters(dt, nxt.newton_iters, its0=last_its, dt_min=dt_min, dt_max=dt_max)
        dp = max(
            float(np.max(np.abs(dual.fracture.pressure - prev.fracture.pressure))),
            float(np.max(np.abs(dual.matrix.pressure - prev.matrix.pressure))),
        )
        def _z(moles):
            tot = np.maximum(np.sum(moles, axis=1, keepdims=True), 1.0e-18)
            return moles / tot

        dz = max(
            float(np.max(np.abs(_z(dual.fracture.moles) - _z(prev.fracture.moles)))),
            float(np.max(np.abs(_z(dual.matrix.moles) - _z(prev.matrix.moles)))),
        )
        max_dp_all = max(max_dp_all, dp)
        if dp > 5.0e5 or dz > 0.15 or ds > 0.20:
            dt = max(float(dt_min), 0.5 * dt)
        last_its = nxt.newton_iters

    if reports:
        reports[-1].notes.extend(
            [
                f"sum_jac_s={sum_jac:.4f}",
                f"sum_solve_s={sum_solve:.4f}",
                f"sum_resid_s={sum_res:.4f}",
                f"sum_flash_s={sum_flash:.4f}",
                f"sum_flash_main_s={sum_flash:.4f}",
                f"sum_flash_jacobian_s={sum_flash_jac:.4f}",
                f"n_flash_main={n_flash_main}",
                f"n_flash_thermo_jac={n_flash_thermo_jac}",
                f"n_flash_line_search={n_flash_line_search}",
                f"n_jac_reuse={n_jac_reuse}",
                f"n_accept={n_acc}",
                f"n_reject={n_reject}",
                f"linear_iterations={sum_lin_it}",
                f"max_dp={max_dp_all:.4g}",
                f"max_dS={max_ds_all:.4g}",
            ]
        )

    if report_times is not None:
        need = np.unique(np.asarray(report_times, dtype=float))
        out_t = []
        out_s = []
        out_r = []
        out_b = []
        arr_t = np.asarray(times, dtype=float)
        for tt in need:
            idx = index_nearest_time(arr_t, float(tt))
            out_t.append(arr_t[idx])
            out_s.append(states[idx])
            out_r.append(rates_hist[idx])
            out_b.append(bhp_hist[idx] if idx < len(bhp_hist) else {})
        if out_t:
            times, states, rates_hist, bhp_hist = out_t, out_s, out_r, out_b

    traj = Trajectory(
        times_s=np.asarray(times, dtype=float),
        states=states,
        reports=reports,
        port_rates=rates_hist,
        port_bhp=bhp_hist,
    )
    return traj, dual
