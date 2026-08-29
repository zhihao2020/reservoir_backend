"""Fully implicit compositional Newton. Unknowns (n_i, p) per cell.

New path: does not edit ``solver/fi.py``. Jacobian is coloring FD of the
same residual used by Newton. Names follow docs/fim_name_map.md (no upstream IDs).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.properties import moles_from_z
from reservoir_backend.comp.residual import coupled_residual, unpack_unknowns
from reservoir_backend.comp.wells import well_molar_sources
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi import dt_from_newton_iters
from reservoir_backend.solver.impes import MassBalance, StepReport, Trajectory


@dataclass
class CompStepResult:
    moles: NDArray[np.float64]
    pressure: NDArray[np.float64]
    newton_iters: int
    port_rates: dict[str, float]
    port_bhp: dict[str, float]
    q_src: NDArray[np.float64]


def _cell_colors(grid: CartesianGrid) -> NDArray[np.int64]:
    n = grid.n_cells
    colors = np.zeros(n, dtype=np.int64)
    for c in range(n):
        i = c % grid.nx
        j = (c // grid.nx) % grid.ny
        k = c // (grid.nx * grid.ny)
        colors[c] = (i % 3) + 3 * (j % 3) + 9 * (k % 3)
    return colors


def _neighbor_cells(grid: CartesianGrid, c: int) -> list[int]:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    i = c % nx
    j = (c // nx) % ny
    k = c // (nx * ny)
    out = [c]
    if i > 0:
        out.append(c - 1)
    if i + 1 < nx:
        out.append(c + 1)
    if j > 0:
        out.append(c - nx)
    if j + 1 < ny:
        out.append(c + nx)
    if k > 0:
        out.append(c - nx * ny)
    if k + 1 < nz:
        out.append(c + nx * ny)
    return out


def _control_map(controls: list[ControlSeries]) -> dict[tuple[str, str], ControlSeries]:
    return {(c.port_name, c.kind): c for c in controls}


def _scale_rows(n_cells: int, nc: int, n_scale: float, pv_scale: float) -> NDArray[np.float64]:
    s = np.ones(n_cells * (nc + 1))
    block = s.reshape(n_cells, nc + 1)
    block[:, :nc] = 1.0 / max(n_scale, 1.0e-12)
    block[:, nc] = 1.0 / max(pv_scale, 1.0e-12)
    return s


def solve_comp_step(
    grid: CartesianGrid,
    rock: Rock,
    spec: CompSpec,
    ports: list[FlowPort],
    controls: dict[tuple[str, str], ControlSeries],
    moles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    dt: float,
    t: float,
    *,
    max_newton: int = 12,
    tol: float = 1.0e-6,
) -> CompStepResult | None:
    """One fully implicit compositional step, or None on Newton failure."""
    n_cells = grid.n_cells
    nc = spec.nc
    n0 = np.asarray(moles, dtype=float).reshape(n_cells, nc).copy()
    p = np.asarray(pressure, dtype=float).ravel().copy()
    n = n0.copy()
    t_geom = geometric_transmissibility(grid, rock.permeability, kz=rock.kz)
    colors = _cell_colors(grid)
    pv = np.asarray(rock.porosity, dtype=float).ravel() * grid.cell_volumes()
    n_scale = max(float(np.mean(np.sum(n0, axis=1))), 1.0e-6)
    pv_scale = max(float(np.mean(pv)), 1.0e-12)
    row_s = _scale_rows(n_cells, nc, n_scale, pv_scale)
    nu = nc + 1
    n_u = n_cells * nu
    use_dense = n_u <= 192
    n_colors = int(np.max(colors)) + 1
    color_cells = [np.flatnonzero(colors == color) for color in range(n_colors)]
    neighbors = [_neighbor_cells(grid, c) for c in range(n_cells)]

    def residual_of(nm, pr, props=None, reflash=None, *, need_bhp=False):
        from reservoir_backend.comp.properties import flash_state

        if props is None:
            props_q = flash_state(spec, pr, nm)
        else:
            props_q = props
            if reflash is not None:
                flash_state(spec, pr, nm, cells=reflash, out=props_q)
        q_src, rates, bhp = well_molar_sources(
            grid, rock, ports, controls, pr, props_q, spec, t + dt, need_bhp=need_bhp
        )
        res, props_out = coupled_residual(
            grid, rock, spec, nm, pr, n0, dt, q_src, t_geom, props=props_q
        )
        return res, props_out, rates, bhp, q_src

    def assemble_jacobian(nm, pr, res0, props0):
        eps_n = 1.0e-7 * max(n_scale, 1.0)
        eps_p = 1.0e-6 * max(float(np.mean(np.abs(pr))), 1.0e5)
        if use_dense:
            jac = np.zeros((n_u, n_u))
        else:
            rows: list[int] = []
            cols: list[int] = []
            data: list[float] = []
        for cells in color_cells:
            if cells.size == 0:
                continue
            for slot in range(nu):
                n_t = nm.copy()
                p_t = pr.copy()
                if slot < nc:
                    n_t[cells, slot] = n_t[cells, slot] + eps_n
                    eps = eps_n
                else:
                    p_t[cells] = p_t[cells] + eps_p
                    eps = eps_p
                res_t, _, _, _, _ = residual_of(n_t, p_t, props=props0.copy(), reflash=cells, need_bhp=False)
                dres = (res_t - res0) / eps
                for c in cells:
                    col = int(c) * nu + slot
                    for cc in neighbors[int(c)]:
                        for blk in range(nu):
                            row = int(cc) * nu + blk
                            val = float(dres[row])
                            if use_dense:
                                jac[row, col] = val
                            else:
                                rows.append(row)
                                cols.append(col)
                                data.append(val)
        if use_dense:
            return row_s[:, None] * jac
        raw = sparse.csr_matrix((data, (rows, cols)), shape=(n_u, n_u))
        return sparse.diags(row_s) @ raw

    try:
        res, props, rates, bhp, q_src = residual_of(n, p, need_bhp=True)
    except Exception:
        return None
    r0 = float(np.linalg.norm(res * row_s))
    if not np.isfinite(r0):
        return None
    r0 = max(r0, 1.0e-18)
    n_its = 0
    jac_s = None
    jac_age = 0
    refresh_every = 2
    for n_its in range(1, int(max_newton) + 1):
        rnorm = float(np.linalg.norm(res * row_s))
        if rnorm / r0 < float(tol) or rnorm < 1.0e-10:
            _, _, rates, bhp, q_src = residual_of(n, p, props=props, need_bhp=True)
            return CompStepResult(
                moles=n, pressure=p, newton_iters=n_its, port_rates=rates, port_bhp=bhp, q_src=q_src
            )
        rebuild = jac_s is None or jac_age >= refresh_every
        if rebuild:
            try:
                jac_s = assemble_jacobian(n, p, res, props)
            except Exception:
                return None
            jac_age = 0
        rhs = -res * row_s
        try:
            if use_dense:
                try:
                    du = np.linalg.solve(jac_s, rhs)
                except np.linalg.LinAlgError:
                    du = np.linalg.lstsq(jac_s, rhs, rcond=None)[0]
                du = np.asarray(du, dtype=float).ravel()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", MatrixRankWarning)
                    du = np.asarray(spsolve(jac_s, rhs), dtype=float).ravel()
                if du.size != n_u or not np.all(np.isfinite(du)):
                    du = np.asarray(
                        lsmr(jac_s, rhs, atol=1.0e-10, btol=1.0e-10, maxiter=max(80, n_u))[0],
                        dtype=float,
                    ).ravel()
        except Exception:
            return None
        if du.size != n_u or not np.all(np.isfinite(du)):
            return None
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            dn, dp = unpack_unknowns(alpha * du, n_cells, nc)
            n_try = np.maximum(n + dn, 1.0e-16)
            p_try = np.clip(p + dp, 1.0e4, 1.0e9)
            try:
                res_try, props_try, rates_try, bhp_try, q_try = residual_of(n_try, p_try, need_bhp=False)
            except Exception:
                continue
            r_try = float(np.linalg.norm(res_try * row_s))
            if np.isfinite(r_try) and r_try < rnorm * (1.0 - 1.0e-4 * alpha):
                n, p, res, props, rates, bhp, q_src = n_try, p_try, res_try, props_try, rates_try, bhp_try, q_try
                improved = True
                break
        if not improved:
            if rebuild:
                return None
            jac_s = None
            continue
        jac_age += 1
    return None


def initialize_state(grid: CartesianGrid, rock: Rock, spec: CompSpec, p_init: float) -> State:
    n_cells = grid.n_cells
    pressure = np.full(n_cells, float(p_init))
    pv = np.asarray(rock.porosity, dtype=float).ravel() * grid.cell_volumes()
    moles = moles_from_z(spec, pressure, spec.z_init, pv)
    from reservoir_backend.comp.properties import flash_state

    props = flash_state(spec, pressure, moles)
    return State(
        pressure=pressure,
        sw=props.sw.copy(),
        sg=props.sv.copy(),
        rs=None,
        moles=moles,
        time_s=0.0,
    )


def _mass_pack(
    moles0: NDArray[np.float64],
    moles1: NDArray[np.float64],
    injected: NDArray[np.float64],
    produced: NDArray[np.float64],
) -> MassBalance:
    n0 = np.sum(moles0, axis=0)
    n1 = np.sum(moles1, axis=0)
    inj = np.asarray(injected, dtype=float).ravel()
    prod = np.asarray(produced, dtype=float).ravel()
    tot0 = float(np.sum(n0))
    tot1 = float(np.sum(n1))
    inj_t = float(np.sum(inj))
    prod_t = float(np.sum(prod))
    err = tot1 - tot0 - inj_t + prod_t
    c1_err = float(n1[0] - n0[0] - inj[0] + prod[0]) if n0.size else 0.0
    return MassBalance(
        initial_mass=tot0,
        final_mass=tot1,
        injected_mass=inj_t,
        produced_mass=prod_t,
        boundary_flux=0.0,
        balance_error=err,
        relative_balance_error=abs(err) / max(abs(tot0), 1.0e-12),
        gas_initial_mass=float(n0[0]) if n0.size else 0.0,
        gas_final_mass=float(n1[0]) if n0.size else 0.0,
        gas_injected_mass=float(inj[0]) if inj.size else 0.0,
        gas_produced_mass=float(prod[0]) if prod.size else 0.0,
        gas_balance_error=c1_err,
        gas_relative_balance_error=abs(c1_err) / max(abs(float(n0[0])) if n0.size else 1.0, 1.0e-12),
    )


def simulate_comp(
    grid: CartesianGrid,
    rock: Rock,
    spec: CompSpec,
    ports: list[FlowPort],
    controls: list[ControlSeries],
    state0: State,
    t_end: float,
    *,
    dt_init: float = 10.0,
    dt_min: float = 1.0e-6,
    dt_max: float = 60.0,
    max_steps: int = 12000,
    report_times: NDArray[np.float64] | None = None,
) -> Trajectory:
    """Time loop. Δt from Newton count. Failure chops; underflow raises."""
    cmap = _control_map(controls)
    if state0.moles is None:
        st = initialize_state(grid, rock, spec, float(np.mean(state0.pressure)))
        st.time_s = float(state0.time_s)
    else:
        st = state0.copy()
    moles = np.asarray(st.moles, dtype=float)
    p = np.asarray(st.pressure, dtype=float).ravel()
    moles0 = moles.copy()
    injected = np.zeros(spec.nc)
    produced = np.zeros(spec.nc)
    t = float(st.time_s)
    t_end = float(t_end)
    dt = min(float(dt_init), float(dt_max))
    reports: list[StepReport] = []
    states = [st.copy()]
    times = [t]
    from reservoir_backend.comp.properties import flash_state

    props0 = flash_state(spec, p, moles)
    _, rates0, bhp0 = well_molar_sources(grid, rock, ports, cmap, p, props0, spec, t)
    rates_hist = [dict(rates0)]
    bhp_hist = [dict(bhp0)]
    n_acc = 0
    last_its = 5

    while t < t_end - 1.0e-15:
        if n_acc >= int(max_steps):
            raise TimeStepUnderflow(f"compositional stepper took more than {max_steps} steps")
        dt = min(dt, t_end - t, float(dt_max))
        if dt < float(dt_min):
            raise TimeStepUnderflow(f"failed to accept a step at t={t}")
        nxt = solve_comp_step(grid, rock, spec, ports, cmap, moles, p, dt, t)
        if nxt is None:
            dt *= 0.5
            continue
        inj = np.sum(np.maximum(nxt.q_src, 0.0), axis=0) * dt
        prod = np.sum(np.maximum(-nxt.q_src, 0.0), axis=0) * dt
        injected = injected + inj
        produced = produced + prod
        moles, p = nxt.moles, nxt.pressure
        t = t + dt
        n_acc += 1
        props = flash_state(spec, p, moles)
        mb = _mass_pack(moles0, moles, injected, produced)
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
        st = State(pressure=p.copy(), sw=props.sw.copy(), sg=props.sv.copy(), moles=moles.copy(), time_s=t)
        # Keep every accepted step so report-time probes see the nearest F state,
        # not t=0 vs t_end only.
        states.append(st.copy())
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
        if not out_t:
            out_t, out_s, out_r, out_b = times, states, rates_hist, bhp_hist
        times, states, rates_hist, bhp_hist = out_t, out_s, out_r, out_b

    return Trajectory(
        times_s=np.asarray(times, dtype=float),
        states=states,
        reports=reports,
        port_rates=rates_hist,
        port_bhp=bhp_hist,
    )
