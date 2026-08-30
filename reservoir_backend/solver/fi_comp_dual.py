"""Fully implicit compositional DPDP Newton. Does not edit ``fi_comp.py``.

Unknowns: fracture (n_f[0:Nc], p_f) then matrix (n_m[0:Nc], p_m) per cell.
Coloring Jacobian includes same-cell transfer coupling. Dense FD is used
when the unknown count is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from reservoir_backend.comp.dual_residual import dual_residual, pack_dual, transmissibilities, unpack_dual
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
from reservoir_backend.solver.fi import dt_from_newton_iters
from reservoir_backend.solver.fi_comp import _cell_colors, _control_map, _mass_pack, _neighbor_cells
from reservoir_backend.solver.impes import StepReport, Trajectory


@dataclass
class DualCompStepResult:
    state: DualCompositionalState
    newton_iters: int
    port_rates: dict[str, float]
    port_bhp: dict[str, float]
    q_src_fracture: NDArray[np.float64]


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


def dual_to_state(spec: CompSpec, dual: DualCompositionalState) -> State:
    """Observation-facing state is the fracture continuum."""
    props = flash_state(spec, dual.fracture.pressure, dual.fracture.moles)
    return State(
        pressure=dual.fracture.pressure.copy(),
        sw=props.sw.copy(),
        sg=props.sv.copy(),
        moles=dual.fracture.moles.copy(),
        time_s=float(dual.time_s),
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
    *,
    need_bhp: bool,
):
    if not ports:
        return np.zeros((grid.n_cells, spec.nc)), {}, {}
    q_f, rates, bhp = well_molar_sources(
        grid,
        dual_rock.fracture,
        ports,
        cmap,
        state.fracture.pressure,
        props_f,
        spec,
        t_eval,
        need_bhp=need_bhp,
    )
    return q_f, rates, bhp


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
    q_f, rates, bhp = _wells(grid, dual_rock, spec, state, ports, cmap, t_eval, props_f, need_bhp=need_bhp)
    q_m = np.zeros_like(q_f)
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
    return res, props_f, props_m, rates, bhp, q_f


def _coloring_jacobian(
    grid: CartesianGrid,
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
) -> NDArray[np.float64]:
    """FD Jacobian. Transfer couples the two continua in the same cell."""
    n_cells = grid.n_cells
    nc = spec.nc
    nu = nc + 1
    half = n_cells * nu
    n_u = 2 * half
    colors = _cell_colors(grid)
    n_colors = int(np.max(colors)) + 1
    color_cells = [np.flatnonzero(colors == color) for color in range(n_colors)]
    neighbors = [_neighbor_cells(grid, c) for c in range(n_cells)]
    jac = np.zeros((n_u, n_u))
    eps_n = 1.0e-8 * max(n_scale, 1.0)
    eps_p = 1.0e-8 * max(p_scale, 1.0e5)
    nf0, pf0, nm0, pm0 = unpack_dual(u, n_cells, nc)
    for cells in color_cells:
        if cells.size == 0:
            continue
        for cont in (0, 1):
            offset = cont * half
            other = (1 - cont) * half
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
                r2, _, _, _, _, _ = _residual(
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
                    col = offset + int(c) * nu + slot
                    for cc in neighbors[int(c)]:
                        for blk in range(nu):
                            jac[offset + int(cc) * nu + blk, col] = dres[offset + int(cc) * nu + blk]
                    for blk in range(nu):
                        jac[other + int(c) * nu + blk, col] = dres[other + int(c) * nu + blk]
    return jac


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
) -> DualCompStepResult:
    """One fully implicit DPDP step. Raises on Newton failure."""
    old = state.copy()
    n_cells = grid.n_cells
    nc = spec.nc
    t_f, t_m = transmissibilities(grid, dual_rock)
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
    last_q = np.zeros((n_cells, nc))
    n_scale = max(float(np.mean(np.sum(state.fracture.moles, axis=1))), 1.0e-6)
    p_scale = max(float(np.mean(np.abs(state.fracture.pressure))), 1.0e5)

    for it in range(int(max_newton)):
        trial = _state_from_u(u, n_cells, nc, t1)
        res, props_f, props_m, last_rates, last_bhp, last_q = _residual(
            grid, dual_rock, spec, trial, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=True
        )
        last_res = res
        if float(np.max(np.abs(res))) < float(tol):
            trial.time_s = t1
            return DualCompStepResult(
                state=trial,
                newton_iters=it + 1,
                port_rates=dict(last_rates),
                port_bhp=dict(last_bhp),
                q_src_fracture=last_q,
            )
        jac = _coloring_jacobian(
            grid,
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
        try:
            step = linalg.solve(jac, -res, assume_a="gen")
        except linalg.LinAlgError as exc:
            raise PhysicsConvergenceError("DPDP Newton Jacobian is singular") from exc
        if not np.all(np.isfinite(step)):
            raise PhysicsConvergenceError("DPDP Newton step is not finite")
        alpha = 1.0
        accepted = False
        r0 = float(np.linalg.norm(res))
        for _ in range(8):
            u_try = u + alpha * step
            trial2 = _state_from_u(u_try, n_cells, nc, t1)
            trial2.fracture.moles = np.maximum(trial2.fracture.moles, 1.0e-18)
            trial2.matrix.moles = np.maximum(trial2.matrix.moles, 1.0e-18)
            trial2.fracture.pressure = np.clip(trial2.fracture.pressure, 1.0e4, 1.0e9)
            trial2.matrix.pressure = np.clip(trial2.matrix.pressure, 1.0e4, 1.0e9)
            packed = pack_dual(trial2)
            r_try, _, _, rates_try, bhp_try, q_try = _residual(
                grid, dual_rock, spec, trial2, old, dt, transfer, t_f, t_m, ports, cmap, t1, need_bhp=False
            )
            if float(np.linalg.norm(r_try)) <= (1.0 - 1.0e-4 * alpha) * r0:
                u = packed
                last_rates, last_bhp, last_q = rates_try, bhp_try, q_try
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            u = pack_dual(_state_from_u(u + 0.1 * step, n_cells, nc, t1))
    nrm = float(np.max(np.abs(last_res))) if last_res is not None else float("inf")
    raise PhysicsConvergenceError(f"DPDP Newton failed, max|R|={nrm:.3e}")


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
) -> tuple[Trajectory, DualCompositionalState]:
    """Time loop for compositional DPDP. Wells are on the fracture continuum."""
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
    vis = dual_to_state(spec, dual)
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
    last_its = 5

    while t < t_end - 1.0e-15:
        if n_acc >= int(max_steps):
            raise TimeStepUnderflow(f"DPDP stepper took more than {max_steps} steps")
        dt = min(dt, t_end - t, float(dt_max))
        if dt < float(dt_min):
            raise TimeStepUnderflow(f"failed to accept a DPDP step at t={t}")
        try:
            nxt = solve_dual_comp_step(
                grid, dual_rock, spec, dual, dt, transfer, ports=ports, controls=cmap
            )
        except PhysicsConvergenceError:
            dt *= 0.5
            continue
        inj = np.sum(np.maximum(nxt.q_src_fracture, 0.0), axis=0) * dt
        prod = np.sum(np.maximum(-nxt.q_src_fracture, 0.0), axis=0) * dt
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
        reports.append(
            StepReport(
                time_s=t,
                dt=dt,
                max_cfl=0.0,
                max_ds=0.0,
                mass=mb,
                port_rates=dict(nxt.port_rates),
                notes=[],
                newton_its=nxt.newton_iters,
            )
        )
        states.append(dual_to_state(spec, dual))
        times.append(t)
        rates_hist.append(dict(nxt.port_rates))
        bhp_hist.append(dict(nxt.port_bhp))
        dt = dt_from_newton_iters(dt, nxt.newton_iters, its0=last_its, dt_min=dt_min, dt_max=dt_max)
        last_its = nxt.newton_iters

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
