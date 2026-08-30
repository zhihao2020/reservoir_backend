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
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.comp.wells import well_molar_sources
from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import PhysicsConvergenceError, TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.dpdp_jacobian import fill_column_slice, residual_scales
from reservoir_backend.solver.fi import dt_from_newton_iters
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


def dual_to_state(spec: CompSpec, dual: DualCompositionalState, dual_rock: DualRock | None = None) -> State:
    """Keep both continua on State so H can select fracture / matrix / bulk."""
    props_f = flash_state(spec, dual.fracture.pressure, dual.fracture.moles)
    props_m = flash_state(spec, dual.matrix.pressure, dual.matrix.moles)
    phi_f = None if dual_rock is None else np.asarray(dual_rock.fracture.porosity, dtype=float)
    phi_m = None if dual_rock is None else np.asarray(dual_rock.matrix.porosity, dtype=float)
    return State(
        pressure=dual.fracture.pressure.copy(),
        sw=props_f.sw.copy(),
        sg=props_f.sv.copy(),
        moles=dual.fracture.moles.copy(),
        time_s=float(dual.time_s),
        pressure_matrix=dual.matrix.pressure.copy(),
        sw_matrix=props_m.sw.copy(),
        sg_matrix=props_m.sv.copy(),
        phi_fracture=None if phi_f is None else phi_f.copy(),
        phi_matrix=None if phi_m is None else phi_m.copy(),
    )


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
):
    if props_f is None:
        props_f = flash_state(spec, state.fracture.pressure, state.fracture.moles)
    elif reflash_f is not None:
        flash_state(spec, state.fracture.pressure, state.fracture.moles, cells=reflash_f, out=props_f)
    if props_m is None:
        props_m = flash_state(spec, state.matrix.pressure, state.matrix.moles)
    elif reflash_m is not None:
        flash_state(spec, state.matrix.pressure, state.matrix.moles, cells=reflash_m, out=props_m)
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
    return res, props_f, props_m, rates, bhp, q_f, q_m


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
                r2, _, _, _, _, _, _ = _residual(
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
                dres = (r2 - res0) / eps
                for c in cells:
                    fill_column_slice(pattern, data, offset + int(c) * nu + slot, dres)
    return pattern.to_csr(data)


def _residual_stats(res: NDArray[np.float64], n_cells: int, nc: int, scale: NDArray[np.float64]) -> tuple[float, float, float]:
    block = np.asarray(res, dtype=float).reshape(2, n_cells, nc + 1)
    mass = float(np.max(np.abs(block[:, :, :nc])))
    vol = float(np.max(np.abs(block[:, :, nc])))
    nrm = float(np.max(np.abs(res * scale)))
    return mass, vol, nrm


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
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0e-6)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)
    pv = np.asarray(dual_rock.fracture.porosity, dtype=float).ravel() * grid.cell_volumes()
    pv_scale = max(float(np.mean(pv)), 1.0e-12)
    row_s = residual_scales(n_cells, nc, n_scale, pv_scale)

    for it in range(int(max_newton)):
        trial = _state_from_u(u, n_cells, nc, t1)
        t_r0 = time.perf_counter()
        res, props_f, props_m, last_rates, last_bhp, last_qf, last_qm = _residual(
            grid, dual_rock, spec, trial, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=True
        )
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
            )
        t_j0 = time.perf_counter()
        jac = _coloring_jacobian(
            ctx,
            spec,
            dual_rock,
            old,
            dt,
            transfer,
            t_f,
            t_m,
            ports,
            cmap,
            t1,
            u,
            res,
            props_f,
            props_m,
            n_scale,
            p_scale,
        )
        t_jac += time.perf_counter() - t_j0
        jac_s = jac.tocsr().multiply(row_s[:, None]).tocsc()
        t_s0 = time.perf_counter()
        lin = solve_newton_system(jac_s, -(res * row_s))
        t_solve += time.perf_counter() - t_s0
        step = lin.x
        if not np.all(np.isfinite(step)):
            raise PhysicsConvergenceError("DPDP Newton step is not finite")
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
            r_try, _, _, rates_try, bhp_try, qf_try, qm_try = _residual(
                grid, dual_rock, spec, trial2, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=False
            )
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
        dual = initialize_dual_state(grid, dual_rock, spec, float(np.mean(state0.pressure)))
        dual.time_s = float(state0.time_s)
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

    while t < t_end - 1.0e-15:
        if n_acc >= int(max_steps):
            raise TimeStepUnderflow(f"DPDP stepper took more than {max_steps} steps")
        dt = min(dt, t_end - t, float(dt_max))
        if dt < float(dt_min):
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
                    f"n_reject={n_reject}",
                ],
                newton_its=nxt.newton_iters,
            )
        )
        states.append(dual_to_state(spec, dual, dual_rock))
        times.append(t)
        rates_hist.append(dict(nxt.port_rates))
        bhp_hist.append(dict(nxt.port_bhp))
        dt = dt_from_newton_iters(dt, nxt.newton_iters, its0=last_its, dt_min=dt_min, dt_max=dt_max)
        last_its = nxt.newton_iters

    if reports:
        reports[-1].notes.extend(
            [
                f"sum_jac_s={sum_jac:.4f}",
                f"sum_solve_s={sum_solve:.4f}",
                f"sum_resid_s={sum_res:.4f}",
                f"n_accept={n_acc}",
                f"n_reject={n_reject}",
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
            idx = int(np.searchsorted(arr_t, float(tt), side="right") - 1)
            idx = int(np.clip(idx, 0, arr_t.size - 1))
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
