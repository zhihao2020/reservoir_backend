"""Sequential IMPES: TPFA pressure then explicit black-oil transport.

Conservation is surface volume, as in MRST ``equationsOilWater``:

    (pv bα Sα − pv0 bα0 Sα0)/dt + Div(bα vα) = qα^s

Rate well controls are surface rates. Reservoir sources are q^s / b.
Incompressible B=1, c=0 is the lab special case of the same equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.discretization.tpfa import assemble_pressure, face_fluxes, solve_pressure
from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import InvalidSaturation, TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.physics.rock import Rock, as_cell_field
from reservoir_backend.ports.flow import FlowPort, half_cell_wi
from reservoir_backend.solver.transport import implicit_water


@dataclass
class MassBalance:
    initial_mass: float
    final_mass: float
    injected_mass: float
    produced_mass: float
    boundary_flux: float
    balance_error: float
    relative_balance_error: float
    gas_initial_mass: float = 0.0
    gas_final_mass: float = 0.0
    gas_injected_mass: float = 0.0
    gas_produced_mass: float = 0.0
    gas_balance_error: float = 0.0
    gas_relative_balance_error: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "initial_mass": self.initial_mass,
            "final_mass": self.final_mass,
            "injected_mass": self.injected_mass,
            "produced_mass": self.produced_mass,
            "boundary_flux": self.boundary_flux,
            "balance_error": self.balance_error,
            "relative_balance_error": self.relative_balance_error,
            "gas_initial_mass": self.gas_initial_mass,
            "gas_final_mass": self.gas_final_mass,
            "gas_injected_mass": self.gas_injected_mass,
            "gas_produced_mass": self.gas_produced_mass,
            "gas_balance_error": self.gas_balance_error,
            "gas_relative_balance_error": self.gas_relative_balance_error,
        }


@dataclass
class StepReport:
    time_s: float
    dt: float
    max_cfl: float
    max_ds: float
    mass: MassBalance
    port_rates: dict[str, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    times_s: NDArray[np.float64]
    states: list[State]
    reports: list[StepReport]
    port_rates: list[dict[str, float]]

    def state_at(self, t: float) -> State:
        times = np.asarray(self.times_s, dtype=float)
        if times.size == 0:
            raise ValueError("empty trajectory")
        idx = int(np.argmin(np.abs(times - float(t))))
        return self.states[idx]


def surface_water(
    grid: CartesianGrid,
    rock: Rock,
    sw: NDArray[np.float64],
    pressure: NDArray[np.float64] | None = None,
    pvt: BlackOilPVT | None = None,
) -> float:
    """Surface water volume Σ φ(p) bW(p) Sw V. Reduces to pore-volume Sw when B=1, cr=0."""
    fluid = pvt or BlackOilPVT.incompressible()
    p = np.asarray(sw, dtype=float) * 0.0 + 1.0e5 if pressure is None else np.asarray(pressure, dtype=float)
    acc = rock.porosity * fluid.pv_mult(p) * fluid.b_w(p) * np.asarray(sw, dtype=float) * grid.cell_volumes()
    return float(np.sum(acc))


def water_mass(
    grid: CartesianGrid,
    rock: Rock,
    sw: NDArray[np.float64],
    pressure: NDArray[np.float64] | None = None,
    pvt: BlackOilPVT | None = None,
) -> float:
    return surface_water(grid, rock, sw, pressure=pressure, pvt=pvt)


def surface_gas(
    grid: CartesianGrid,
    rock: Rock,
    sg: NDArray[np.float64] | None,
    pressure: NDArray[np.float64] | None = None,
    pvt: BlackOilPVT | None = None,
) -> float:
    if sg is None:
        return 0.0
    fluid = pvt or BlackOilPVT.incompressible()
    p = np.asarray(sg, dtype=float) * 0.0 + 1.0e5 if pressure is None else np.asarray(pressure, dtype=float)
    acc = rock.porosity * fluid.pv_mult(p) * fluid.b_g(p) * np.asarray(sg, dtype=float) * grid.cell_volumes()
    return float(np.sum(acc))


def gas_mass(
    grid: CartesianGrid,
    rock: Rock,
    sg: NDArray[np.float64] | None,
    pressure: NDArray[np.float64] | None = None,
    pvt: BlackOilPVT | None = None,
) -> float:
    return surface_gas(grid, rock, sg, pressure=pressure, pvt=pvt)


def _control_map(controls: list[ControlSeries]) -> dict[tuple[str, str], ControlSeries]:
    return {(c.port_name, c.kind): c for c in controls}


def _port_value(controls: dict[tuple[str, str], ControlSeries], port: FlowPort, t: float) -> float:
    series = controls.get((port.name, port.control))
    if series is None:
        raise KeyError(f"missing {port.control} control for port {port.name}")
    return series.value_at(t)


def _injection_sw(controls: dict[tuple[str, str], ControlSeries], port: FlowPort, t: float) -> float:
    series = controls.get((port.name, "composition"))
    if series is None:
        return float(port.sw_inj)
    return float(np.clip(series.value_at(t), 0.0, 1.0))


def _injection_sg(controls: dict[tuple[str, str], ControlSeries], port: FlowPort, t: float) -> float:
    series = controls.get((port.name, "gas_composition"))
    if series is None:
        return 0.0
    return float(np.clip(series.value_at(t), 0.0, 1.0))


def _upwind_frac(
    frac_l: NDArray[np.float64],
    frac_r: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(q >= 0.0, q * frac_l, q * frac_r)


def _upwind_b(
    b_l: NDArray[np.float64],
    b_r: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(q >= 0.0, b_l, b_r)


def _capillary_water_flux(
    t_abs_over_lt: NDArray[np.float64],
    sw_l: NDArray[np.float64],
    sw_r: NDArray[np.float64],
    relperm: CoreyTwoPhase,
    capillary,
) -> NDArray[np.float64]:
    if isinstance(capillary, NoCapillary):
        return np.zeros_like(t_abs_over_lt, dtype=float)
    lw_l, lo_l, lt_l = relperm.mobility(sw_l)
    lw_r, lo_r, lt_r = relperm.mobility(sw_r)
    lt = 0.5 * (lt_l + lt_r)
    m_l = np.divide(lw_l * lo_l, lt_l, out=np.zeros_like(lt_l), where=lt_l > 0.0)
    m_r = np.divide(lw_r * lo_r, lt_r, out=np.zeros_like(lt_r), where=lt_r > 0.0)
    m = 0.5 * (m_l + m_r)
    t_abs = np.divide(t_abs_over_lt, np.maximum(lt, 1.0e-30))
    return t_abs * m * (capillary.pc(sw_r) - capillary.pc(sw_l))


def _divergence(
    grid: CartesianGrid,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
) -> NDArray[np.float64]:
    div = fx[:, :, 1:] - fx[:, :, :-1] + fy[:, 1:, :] - fy[:, :-1, :] + fz[1:, :, :] - fz[:-1, :, :]
    return div.ravel()


def _boundary_in(fx: NDArray[np.float64], fy: NDArray[np.float64], fz: NDArray[np.float64]) -> float:
    return float(
        np.sum(fx[:, :, 0])
        - np.sum(fx[:, :, -1])
        + np.sum(fy[:, 0, :])
        - np.sum(fy[:, -1, :])
        + np.sum(fz[0, :, :])
        - np.sum(fz[-1, :, :])
    )


def estimate_dt(
    grid: CartesianGrid,
    rock: Rock,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    max_cfl: float,
    cell_rate: NDArray[np.float64] | None = None,
    pv_scale: NDArray[np.float64] | None = None,
) -> float:
    inflow = np.abs(fx[:, :, :-1]) + np.abs(fx[:, :, 1:])
    inflow = inflow + np.abs(fy[:, :-1, :]) + np.abs(fy[:, 1:, :])
    inflow = inflow + np.abs(fz[:-1, :, :]) + np.abs(fz[1:, :, :])
    scale = np.ones(grid.n_cells, dtype=float) if pv_scale is None else np.asarray(pv_scale, dtype=float).ravel()
    pv = grid.reshape_ijk(rock.porosity * scale * grid.cell_volumes())
    cfl_unit = inflow / np.maximum(pv, 1.0e-30)
    if cell_rate is not None:
        q = np.abs(np.asarray(cell_rate, dtype=float).ravel())
        cfl_unit = np.maximum(cfl_unit, grid.reshape_ijk(q) / np.maximum(pv, 1.0e-30))
    worst = float(np.max(cfl_unit)) if cfl_unit.size else 0.0
    if worst <= 0.0:
        return 1.0e30
    return float(max_cfl / worst)


def _cell_face_outflow(
    grid: CartesianGrid,
    cell: int,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
) -> float:
    i, j, k = grid.ijk(int(cell))
    return float(
        fx[k, j, i + 1]
        - fx[k, j, i]
        + fy[k, j + 1, i]
        - fy[k, j, i]
        + fz[k + 1, j, i]
        - fz[k, j, i]
    )


def _port_total_rate(
    grid: CartesianGrid,
    port: FlowPort,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    cell_rate: NDArray[np.float64],
) -> float:
    q = 0.0
    for cell in port.cell_ids:
        q += _cell_face_outflow(grid, int(cell), fx, fy, fz) - float(cell_rate[int(cell)])
    return q


def _set_cell_outgoing_phase(
    grid: CartesianGrid,
    cell: int,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    qz: NDArray[np.float64],
    frac: float,
) -> None:
    """Outgoing faces of ``cell`` carry volume fraction ``frac`` of this phase."""
    i, j, k = grid.ijk(int(cell))
    f = float(frac)
    if fx[k, j, i + 1] > 0.0:
        qx[k, j, i + 1] = fx[k, j, i + 1] * f
    if fx[k, j, i] < 0.0:
        qx[k, j, i] = fx[k, j, i] * f
    if fy[k, j + 1, i] > 0.0:
        qy[k, j + 1, i] = fy[k, j + 1, i] * f
    if fy[k, j, i] < 0.0:
        qy[k, j, i] = fy[k, j, i] * f
    if fz[k + 1, j, i] > 0.0:
        qz[k + 1, j, i] = fz[k + 1, j, i] * f
    if fz[k, j, i] < 0.0:
        qz[k, j, i] = fz[k, j, i] * f


def _dirichlet_phase_faces(
    grid: CartesianGrid,
    q_x: NDArray[np.float64],
    q_y: NDArray[np.float64],
    q_z: NDArray[np.float64],
    frac_cell: NDArray[np.float64],
    frac_in: float,
    face_dirichlet: dict[str, float],
) -> None:
    if "left" in face_dirichlet:
        q_x[:, :, 0] = np.where(q_x[:, :, 0] >= 0.0, q_x[:, :, 0] * frac_in, q_x[:, :, 0] * frac_cell[:, :, 0])
    if "right" in face_dirichlet:
        q_x[:, :, -1] = np.where(q_x[:, :, -1] >= 0.0, q_x[:, :, -1] * frac_cell[:, :, -1], q_x[:, :, -1] * frac_in)
    if "front" in face_dirichlet:
        q_y[:, 0, :] = np.where(q_y[:, 0, :] >= 0.0, q_y[:, 0, :] * frac_in, q_y[:, 0, :] * frac_cell[:, 0, :])
    if "back" in face_dirichlet:
        q_y[:, -1, :] = np.where(q_y[:, -1, :] >= 0.0, q_y[:, -1, :] * frac_cell[:, -1, :], q_y[:, -1, :] * frac_in)
    if "bottom" in face_dirichlet:
        q_z[0, :, :] = np.where(q_z[0, :, :] >= 0.0, q_z[0, :, :] * frac_in, q_z[0, :, :] * frac_cell[0, :, :])
    if "top" in face_dirichlet:
        q_z[-1, :, :] = np.where(q_z[-1, :, :] >= 0.0, q_z[-1, :, :] * frac_cell[-1, :, :], q_z[-1, :, :] * frac_in)


def _phase_fracs(
    relperm: CoreyTwoPhase,
    three_phase: CoreyThreePhase | None,
    sw_c: float,
    sg_c: float,
) -> tuple[float, float, float]:
    if three_phase is not None:
        fw, fo, fg = three_phase.fractional_flow(sw_c, sg_c)
        return float(fw), float(fo), float(fg)
    fw = float(relperm.fractional_flow(sw_c))
    return fw, 1.0 - fw, 0.0


def _surface_from_reservoir(
    q_res: float,
    b_w: float,
    b_o: float,
    b_g: float,
    fw: float,
    fo: float,
    fg: float,
    sw_src: float,
    sg_src: float,
) -> tuple[float, float]:
    """Surface water and gas rates from a reservoir volume rate."""
    if q_res >= 0.0:
        qw_s = q_res * sw_src * b_w
        qg_s = q_res * sg_src * b_g
        return qw_s, qg_s
    return q_res * fw * b_w, q_res * fg * b_g


def solve_step(
    grid: CartesianGrid,
    rock: Rock,
    relperm: CoreyTwoPhase,
    capillary,
    ports: list[FlowPort],
    controls: list[ControlSeries],
    state: State,
    dt: float,
    *,
    face_dirichlet: dict[str, float] | None = None,
    pvt: BlackOilPVT | None = None,
    single_phase: bool = False,
    mu_single: float = 1.0e-3,
    three_phase: CoreyThreePhase | None = None,
    gravity: float = 0.0,
    face_mult_x: NDArray[np.float64] | None = None,
    face_mult_y: NDArray[np.float64] | None = None,
    face_mult_z: NDArray[np.float64] | None = None,
    implicit: bool = False,
) -> tuple[State, dict[str, float], dict[str, float], dict[str, float], NDArray, NDArray, NDArray]:
    """One IMPES attempt. Port rates are surface water / reservoir liquid."""
    fluid = pvt or BlackOilPVT.incompressible()
    n = grid.n_cells
    cmap = _control_map(controls)
    sw = as_cell_field(state.sw, n, "sw")
    sg = None if state.sg is None else as_cell_field(state.sg, n, "sg")
    if three_phase is not None and sg is None:
        sg = np.zeros(n, dtype=float)
    if np.any(sw < -1.0e-12) or np.any(sw > 1.0 + 1.0e-12):
        raise InvalidSaturation("incoming water saturation out of [0, 1]")
    if sg is not None and (np.any(sg < -1.0e-12) or np.any(sw + sg > 1.0 + 1.0e-12)):
        raise InvalidSaturation("incoming three-phase saturations are inconsistent")
    sw = np.clip(sw, 0.0, 1.0)
    if sg is not None:
        sg = np.clip(sg, 0.0, 1.0 - sw)
    so = 1.0 - sw - (0.0 if sg is None else sg)

    lw_cell: NDArray[np.float64] | None = None
    lo_cell: NDArray[np.float64] | None = None
    if single_phase:
        mobility = np.full(n, 1.0 / max(float(mu_single), 1.0e-30))
        lw_cell = mobility
        lo_cell = np.zeros(n, dtype=float)
    elif three_phase is not None:
        lw_cell, lo_cell, _lg, mobility = three_phase.mobility(sw, sg)
    else:
        lw_cell, lo_cell, mobility = relperm.mobility(sw)

    p_old = as_cell_field(state.pressure, n, "pressure")
    b_w = np.asarray(fluid.b_w(p_old), dtype=float).ravel()
    b_o = np.asarray(fluid.b_o(p_old), dtype=float).ravel()
    b_g = np.asarray(fluid.b_g(p_old), dtype=float).ravel()

    cell_rate = np.zeros(n, dtype=float)
    cell_qw_s = np.zeros(n, dtype=float)
    cell_qg_s = np.zeros(n, dtype=float)
    well_index: dict[int, tuple[float, float]] = {}
    cell_dirichlet: dict[int, float] = {}
    k_field = as_cell_field(rock.permeability, n, "k")
    t_ctrl = state.time_s + dt
    for port in ports:
        val = _port_value(cmap, port, t_ctrl)
        if port.control == "rate":
            share_s = float(val) / float(port.cell_ids.size)
            sw_src = _injection_sw(cmap, port, t_ctrl)
            sg_src = _injection_sg(cmap, port, t_ctrl) if three_phase is not None else 0.0
            for c in port.cell_ids:
                c = int(c)
                if single_phase:
                    cell_rate[c] += share_s
                    cell_qw_s[c] += share_s
                    continue
                if share_s >= 0.0:
                    qw_s = share_s * sw_src
                    qg_s = share_s * sg_src
                    qo_s = share_s * max(0.0, 1.0 - sw_src - sg_src)
                    q_res = qw_s / max(b_w[c], 1.0e-30) + qo_s / max(b_o[c], 1.0e-30)
                    q_res += qg_s / max(b_g[c], 1.0e-30)
                else:
                    fw, fo, fg = _phase_fracs(relperm, three_phase, float(sw[c]), 0.0 if sg is None else float(sg[c]))
                    bmix = b_w[c] * fw + b_o[c] * fo + b_g[c] * fg
                    q_res = share_s / max(bmix, 1.0e-30)
                    qw_s = q_res * fw * b_w[c]
                    qg_s = q_res * fg * b_g[c]
                cell_rate[c] += q_res
                cell_qw_s[c] += qw_s
                cell_qg_s[c] += qg_s
        elif port.use_productivity:
            mu_ref = float(mu_single) if single_phase else float(relperm.mu_w)
            for c in port.cell_ids:
                c = int(c)
                wi = float(port.wi_multiplier) * half_cell_wi(grid, c, float(k_field[c])) / max(mu_ref, 1.0e-30)
                if wi > 0.0:
                    well_index[c] = (wi, float(val))
        else:
            for c in port.cell_ids:
                cell_dirichlet[int(c)] = float(val)

    # Incompressible all-rate systems have a constant null space.
    # Storage regularizes the step, but dt=0 snapshots have no storage term.
    if (not fluid.has_storage() or dt <= 0.0) and not cell_dirichlet and not well_index and not face_dirichlet:
        datum = 0
        for port in ports:
            if port.role == "producer":
                datum = int(port.cell_ids[0])
                break
        cell_dirichlet[datum] = float(p_old[datum])

    storage = None
    p_prev = None
    if fluid.has_storage() and dt > 0.0:
        pv_ref = rock.porosity * grid.cell_volumes()
        storage = pv_ref * np.asarray(fluid.pv_mult(p_old), dtype=float).ravel() * fluid.ct(sw, so, sg) / float(dt)
        p_prev = p_old

    rho_w = float(fluid.rho_w_sc) * float(np.mean(b_w))
    rho_o = float(fluid.rho_o_sc) * float(np.mean(b_o))
    system = assemble_pressure(
        grid,
        rock.permeability,
        mobility,
        cell_dirichlet=cell_dirichlet or None,
        face_dirichlet=face_dirichlet,
        cell_rate=cell_rate,
        well_index=well_index or None,
        storage=storage,
        pressure_prev=p_prev,
        kz=rock.vertical_permeability(),
        mult_x=face_mult_x,
        mult_y=face_mult_y,
        mult_z=face_mult_z,
        lw=lw_cell,
        lo=lo_cell,
        gravity=float(gravity),
        rho_w=rho_w,
        rho_o=rho_o,
    )
    pressure = solve_pressure(system)
    b_w1 = np.asarray(fluid.b_w(pressure), dtype=float).ravel()
    b_g1 = np.asarray(fluid.b_g(pressure), dtype=float).ravel()
    wi_port: dict[int, FlowPort] = {}
    for port in ports:
        if port.use_productivity:
            for c in port.cell_ids:
                wi_port[int(c)] = port
    for c, (wi, pbhp) in well_index.items():
        q_res = float(wi) * (float(pbhp) - float(pressure[int(c)]))
        cell_rate[int(c)] += q_res
        port = wi_port.get(int(c))
        sw_src = _injection_sw(cmap, port, t_ctrl) if port is not None else 1.0
        sg_src = _injection_sg(cmap, port, t_ctrl) if port is not None and three_phase is not None else 0.0
        fw, _fo, fg = _phase_fracs(relperm, three_phase, float(sw[int(c)]), 0.0 if sg is None else float(sg[int(c)]))
        qw_s, qg_s = _surface_from_reservoir(
            q_res, float(b_w1[int(c)]), 1.0, float(b_g1[int(c)]), fw, 1.0 - fw, fg, sw_src, sg_src
        )
        cell_qw_s[int(c)] += qw_s
        cell_qg_s[int(c)] += qg_s
    fx, fy, fz = face_fluxes(
        grid,
        pressure,
        system.t_x,
        system.t_y,
        system.t_z,
        face_dirichlet=face_dirichlet,
        k=rock.permeability,
        mobility=mobility,
        g_x=system.g_x,
        g_y=system.g_y,
        g_z=system.g_z,
        kz=rock.vertical_permeability(),
    )

    port_liquid: dict[str, float] = {}
    port_water: dict[str, float] = {}
    port_gas: dict[str, float] = {}
    for port in ports:
        if port.control == "pressure" and not port.use_productivity:
            q_res = _port_total_rate(grid, port, fx, fy, fz, cell_rate)
        else:
            q_res = float(np.sum(cell_rate[port.cell_ids]))
        cells = port.cell_ids
        bw_m = float(np.mean(b_w1[cells]))
        bg_m = float(np.mean(b_g1[cells]))
        if single_phase:
            qw_s, qg_s = q_res, 0.0
        elif q_res >= 0.0:
            sw_src = _injection_sw(cmap, port, t_ctrl)
            sg_src = _injection_sg(cmap, port, t_ctrl) if three_phase else 0.0
            qw_s, qg_s = q_res * sw_src * bw_m, q_res * sg_src * bg_m
        else:
            if three_phase is not None:
                fw, _fo, fg = three_phase.fractional_flow(sw[cells], sg[cells])
                fw_m, fg_m = float(np.mean(fw)), float(np.mean(fg))
            else:
                fw_m, fg_m = float(np.mean(relperm.fractional_flow(sw[cells]))), 0.0
            qw_s, qg_s = q_res * fw_m * bw_m, q_res * fg_m * bg_m
        port_liquid[port.name] = q_res
        port_water[port.name] = qw_s
        port_gas[port.name] = qg_s

    extras = {
        "gas_rates": port_gas,
        "boundary_water": 0.0,
        "boundary_gas": 0.0,
        "cell_rate": cell_rate,
    }
    cfl_rate = np.asarray(cell_rate, dtype=float).copy()
    for port in ports:
        if port.control != "pressure" or port.use_productivity or port.role != "producer":
            continue
        for c in port.cell_ids:
            c = int(c)
            q_out = _cell_face_outflow(grid, c, fx, fy, fz)
            if q_out < 0.0:
                cfl_rate[c] = q_out
    extras["cell_rate"] = cfl_rate
    extras["implicit_ok"] = False
    if single_phase or dt <= 0.0:
        new_state = State(
            pressure=pressure,
            sw=sw.copy(),
            sg=None if sg is None else sg.copy(),
            time_s=state.time_s + dt,
        )
        return new_state, port_water, port_liquid, extras, fx, fy, fz

    vol = grid.cell_volumes()
    pv0 = rock.porosity * np.asarray(fluid.pv_mult(p_old), dtype=float).ravel() * vol
    pv1 = rock.porosity * np.asarray(fluid.pv_mult(pressure), dtype=float).ravel() * vol

    if implicit and three_phase is None and dt >= 30.0:
        pinned_ids = []
        injector_fw: dict[int, float] = {}
        for port in ports:
            if port.control != "pressure" or port.use_productivity or port.role != "injector":
                continue
            sw_src = _injection_sw(cmap, port, t_ctrl)
            for c in port.cell_ids:
                if _cell_face_outflow(grid, int(c), fx, fy, fz) > 0.0:
                    pinned_ids.append(int(c))
                    injector_fw[int(c)] = sw_src
        producer_q: dict[int, float] = {}
        for port in ports:
            if port.control != "pressure" or port.use_productivity or port.role != "producer":
                continue
            for c in port.cell_ids:
                q_out = _cell_face_outflow(grid, int(c), fx, fy, fz)
                if q_out < 0.0:
                    producer_q[int(c)] = q_out
        sw_imp = implicit_water(
            grid,
            relperm,
            sw,
            pv0 * b_w * sw,
            pv1,
            b_w1,
            fx,
            fy,
            fz,
            cell_qw_s,
            dt,
            pinned=np.asarray(pinned_ids, dtype=np.int64) if pinned_ids else None,
            injector_fw=injector_fw or None,
            producer_q=producer_q or None,
        )
        if sw_imp is not None:
            extras["implicit_ok"] = True
            extras["boundary_water"] = 0.0
            return (
                State(pressure=pressure, sw=sw_imp, sg=None, time_s=state.time_s + dt),
                port_water,
                port_liquid,
                extras,
                fx,
                fy,
                fz,
            )

    sw_ijk = grid.reshape_ijk(sw)
    sg_ijk = None if sg is None else grid.reshape_ijk(sg)
    qw_x = np.zeros_like(fx)
    qw_y = np.zeros_like(fy)
    qw_z = np.zeros_like(fz)
    qg_x = np.zeros_like(fx)
    qg_y = np.zeros_like(fy)
    qg_z = np.zeros_like(fz)

    if three_phase is None:
        fw = relperm.fractional_flow(sw_ijk)
        if grid.nx > 1:
            qw_x[:, :, 1:-1] = _upwind_frac(fw[:, :, :-1], fw[:, :, 1:], fx[:, :, 1:-1])
            qw_x[:, :, 1:-1] += _capillary_water_flux(
                system.t_x, sw_ijk[:, :, :-1], sw_ijk[:, :, 1:], relperm, capillary
            )
        if grid.ny > 1:
            qw_y[:, 1:-1, :] = _upwind_frac(fw[:, :-1, :], fw[:, 1:, :], fy[:, 1:-1, :])
            qw_y[:, 1:-1, :] += _capillary_water_flux(
                system.t_y, sw_ijk[:, :-1, :], sw_ijk[:, 1:, :], relperm, capillary
            )
        if grid.nz > 1:
            qw_z[1:-1, :, :] = _upwind_frac(fw[:-1, :, :], fw[1:, :, :], fz[1:-1, :, :])
            qw_z[1:-1, :, :] += _capillary_water_flux(
                system.t_z, sw_ijk[:-1, :, :], sw_ijk[1:, :, :], relperm, capillary
            )
        if face_dirichlet:
            _dirichlet_phase_faces(grid, qw_x, qw_y, qw_z, fw, float(relperm.fractional_flow(1.0 - relperm.sor)), face_dirichlet)
    else:
        fw, _fo, fg = three_phase.fractional_flow(sw_ijk, sg_ijk)
        if grid.nx > 1:
            qw_x[:, :, 1:-1] = _upwind_frac(fw[:, :, :-1], fw[:, :, 1:], fx[:, :, 1:-1])
            qg_x[:, :, 1:-1] = _upwind_frac(fg[:, :, :-1], fg[:, :, 1:], fx[:, :, 1:-1])
        if grid.ny > 1:
            qw_y[:, 1:-1, :] = _upwind_frac(fw[:, :-1, :], fw[:, 1:, :], fy[:, 1:-1, :])
            qg_y[:, 1:-1, :] = _upwind_frac(fg[:, :-1, :], fg[:, 1:, :], fy[:, 1:-1, :])
        if grid.nz > 1:
            qw_z[1:-1, :, :] = _upwind_frac(fw[:-1, :, :], fw[1:, :, :], fz[1:-1, :, :])
            qg_z[1:-1, :, :] = _upwind_frac(fg[:-1, :, :], fg[1:, :, :], fz[1:-1, :, :])
        if face_dirichlet:
            _dirichlet_phase_faces(grid, qw_x, qw_y, qw_z, fw, 0.0, face_dirichlet)
            _dirichlet_phase_faces(grid, qg_x, qg_y, qg_z, fg, 0.0, face_dirichlet)

    pinned_injectors: list[int] = []
    for port in ports:
        if port.control != "pressure" or port.use_productivity or port.role != "injector":
            continue
        sw_src = _injection_sw(cmap, port, t_ctrl)
        sg_src = _injection_sg(cmap, port, t_ctrl) if three_phase is not None else 0.0
        for c in port.cell_ids:
            c = int(c)
            if _cell_face_outflow(grid, c, fx, fy, fz) <= 0.0:
                continue
            _set_cell_outgoing_phase(grid, c, fx, fy, fz, qw_x, qw_y, qw_z, sw_src)
            if three_phase is not None:
                _set_cell_outgoing_phase(grid, c, fx, fy, fz, qg_x, qg_y, qg_z, sg_src)
            pinned_injectors.append(c)

    b_w1_ijk = grid.reshape_ijk(b_w1)
    b_g1_ijk = grid.reshape_ijk(b_g1)
    qw_s_x = qw_x * _upwind_b(
        np.concatenate([b_w1_ijk[:, :, :1], b_w1_ijk], axis=2),
        np.concatenate([b_w1_ijk, b_w1_ijk[:, :, -1:]], axis=2),
        qw_x,
    )
    qw_s_y = qw_y * _upwind_b(
        np.concatenate([b_w1_ijk[:, :1, :], b_w1_ijk], axis=1),
        np.concatenate([b_w1_ijk, b_w1_ijk[:, -1:, :]], axis=1),
        qw_y,
    )
    qw_s_z = qw_z * _upwind_b(
        np.concatenate([b_w1_ijk[:1, :, :], b_w1_ijk], axis=0),
        np.concatenate([b_w1_ijk, b_w1_ijk[-1:, :, :]], axis=0),
        qw_z,
    )
    qg_s_x = qg_x * _upwind_b(
        np.concatenate([b_g1_ijk[:, :, :1], b_g1_ijk], axis=2),
        np.concatenate([b_g1_ijk, b_g1_ijk[:, :, -1:]], axis=2),
        qg_x,
    )
    qg_s_y = qg_y * _upwind_b(
        np.concatenate([b_g1_ijk[:, :1, :], b_g1_ijk], axis=1),
        np.concatenate([b_g1_ijk, b_g1_ijk[:, -1:, :]], axis=1),
        qg_y,
    )
    qg_s_z = qg_z * _upwind_b(
        np.concatenate([b_g1_ijk[:1, :, :], b_g1_ijk], axis=0),
        np.concatenate([b_g1_ijk, b_g1_ijk[-1:, :, :]], axis=0),
        qg_z,
    )

    extras["boundary_water"] = _boundary_in(qw_s_x, qw_s_y, qw_s_z)
    extras["boundary_gas"] = _boundary_in(qg_s_x, qg_s_y, qg_s_z)

    acc_w = pv0 * b_w * sw - float(dt) * _divergence(grid, qw_s_x, qw_s_y, qw_s_z) + float(dt) * cell_qw_s
    acc_g = None
    if sg is not None:
        acc_g = pv0 * b_g * sg - float(dt) * _divergence(grid, qg_s_x, qg_s_y, qg_s_z) + float(dt) * cell_qg_s

    for port in ports:
        if port.control != "pressure" or port.use_productivity or port.role != "producer":
            continue
        for c in port.cell_ids:
            c = int(c)
            q_out = _cell_face_outflow(grid, c, fx, fy, fz)
            if q_out >= 0.0:
                continue
            fw_c, _fo_c, fg_c = _phase_fracs(relperm, three_phase, float(sw[c]), 0.0 if sg is None else float(sg[c]))
            acc_w[c] += float(dt) * q_out * fw_c * float(b_w1[c])
            if acc_g is not None:
                acc_g[c] += float(dt) * q_out * fg_c * float(b_g1[c])

    denom_w = np.maximum(pv1 * b_w1, 1.0e-30)
    raw_w = acc_w / denom_w
    raw_g = None
    if acc_g is not None:
        raw_g = acc_g / np.maximum(pv1 * b_g1, 1.0e-30)

    for c in pinned_injectors:
        raw_w[c] = sw[c]
        if raw_g is not None and sg is not None:
            raw_g[c] = sg[c]

    new_state = State(pressure=pressure, sw=raw_w, sg=raw_g, time_s=state.time_s + dt)
    return new_state, port_water, port_liquid, extras, fx, fy, fz


def simulate(
    grid: CartesianGrid,
    rock: Rock,
    relperm: CoreyTwoPhase,
    ports: list[FlowPort],
    controls: list[ControlSeries],
    state0: State,
    t_end: float,
    *,
    capillary=None,
    face_dirichlet: dict[str, float] | None = None,
    pvt: BlackOilPVT | None = None,
    single_phase: bool = False,
    mu_single: float = 1.0e-3,
    three_phase: CoreyThreePhase | None = None,
    gravity: float = 0.0,
    face_mult_x: NDArray[np.float64] | None = None,
    face_mult_y: NDArray[np.float64] | None = None,
    face_mult_z: NDArray[np.float64] | None = None,
    implicit: bool = False,
    dt_init: float = 1.0,
    dt_min: float = 1.0e-6,
    dt_max: float | None = None,
    max_cfl: float = 0.5,
    max_ds: float = 0.15,
    max_steps: int = 12000,
    report_times: NDArray[np.float64] | None = None,
) -> Trajectory:
    """Advance from ``state0.time_s`` to ``t_end`` with adaptive dt."""
    if capillary is None:
        capillary = NoCapillary()
    fluid = pvt or BlackOilPVT.incompressible()
    t_end = float(t_end)
    state = state0.copy()
    if three_phase is not None and state.sg is None:
        state.sg = np.zeros(grid.n_cells, dtype=float)
    dt = float(dt_init)
    if dt_max is None:
        dt_max = max(dt_init, 1.0)
    targets: list[float] = []
    if report_times is not None:
        targets = [
            float(x)
            for x in np.asarray(report_times, dtype=float)
            if state.time_s + 1.0e-15 < float(x) <= t_end + 1.0e-12
        ]
    if not targets or targets[-1] < t_end - 1.0e-12:
        targets.append(t_end)
    targets = sorted(set(targets))

    times = [state.time_s]
    states = [state.copy()]
    reports: list[StepReport] = []
    _, port_w0, _, extras0, _, _, _ = solve_step(
        grid,
        rock,
        relperm,
        capillary,
        ports,
        controls,
        state,
        0.0,
        face_dirichlet=face_dirichlet,
        pvt=fluid,
        single_phase=single_phase,
        mu_single=mu_single,
        three_phase=three_phase,
        gravity=gravity,
        face_mult_x=face_mult_x,
        face_mult_y=face_mult_y,
        face_mult_z=face_mult_z,
        implicit=implicit,
    )
    rates_hist: list[dict[str, float]] = [dict(port_w0)]
    mass0 = surface_water(grid, rock, state.sw, pressure=state.pressure, pvt=fluid)
    gas0 = surface_gas(grid, rock, state.sg, pressure=state.pressure, pvt=fluid)
    injected = 0.0
    produced = 0.0
    boundary = 0.0
    gas_inj = 0.0
    gas_prod = 0.0

    for target in targets:
        while state.time_s + 1.0e-12 < target:
            if len(reports) > int(max_steps):
                raise TimeStepUnderflow(f"more than {max_steps} steps at t={state.time_s}")
            dt_try = min(dt, target - state.time_s, dt_max)
            accepted = False
            note = ""
            hard_floor = float(dt_min) if dt_min < 1.0e-3 else max(1.0e-4, 0.02 * float(dt_min))
            for _ in range(24):
                if dt_try < hard_floor:
                    raise TimeStepUnderflow(f"dt {dt_try} < dt_min {dt_min} at t={state.time_s}")
                trial, port_w, _port_l, extras, fx, fy, fz = solve_step(
                    grid,
                    rock,
                    relperm,
                    capillary,
                    ports,
                    controls,
                    state,
                    dt_try,
                    face_dirichlet=face_dirichlet,
                    pvt=fluid,
                    single_phase=single_phase,
                    mu_single=mu_single,
                    three_phase=three_phase,
                    gravity=gravity,
                    face_mult_x=face_mult_x,
                    face_mult_y=face_mult_y,
                    face_mult_z=face_mult_z,
                    implicit=implicit,
                )
                dt_cfl = estimate_dt(
                    grid,
                    rock,
                    fx,
                    fy,
                    fz,
                    max_cfl,
                    cell_rate=extras.get("cell_rate"),
                    pv_scale=np.asarray(fluid.pv_mult(trial.pressure), dtype=float).ravel(),
                )
                # Implicit BE is unconditionally stable in dt; CFL only bounds explicit fallback.
                if not extras.get("implicit_ok") and dt_try > dt_cfl * 1.01:
                    dt_try = 0.8 * dt_cfl
                    note = "cfl"
                    continue
                ds = float(np.max(np.abs(trial.sw - state.sw))) if trial.sw.size else 0.0
                if trial.sg is not None and state.sg is not None:
                    ds = max(ds, float(np.max(np.abs(trial.sg - state.sg))))
                if ds > max_ds and dt_try > hard_floor * 2.0:
                    dt_try *= 0.5
                    note = "ds"
                    continue
                so = trial.so()
                illegal = (
                    np.any(trial.sw < -1.0e-8)
                    or np.any(trial.sw > 1.0 + 1.0e-8)
                    or np.any(so < -1.0e-8)
                )
                if trial.sg is not None:
                    illegal = illegal or np.any(trial.sg < -1.0e-8) or np.any(trial.sg > 1.0 + 1.0e-8)
                if illegal:
                    dt_try *= 0.5
                    note = "bounds"
                    continue
                trial.sw = np.clip(trial.sw, 0.0, 1.0)
                if trial.sg is not None:
                    trial.sg = np.clip(trial.sg, 0.0, 1.0 - trial.sw)

                inj_step = sum(max(v, 0.0) * dt_try for v in port_w.values())
                prod_step = sum(max(-v, 0.0) * dt_try for v in port_w.values())
                b_step = float(extras.get("boundary_water", 0.0)) * dt_try
                injected += inj_step
                produced += prod_step
                boundary += b_step
                g_rates = extras.get("gas_rates") or {}
                gas_inj += sum(max(v, 0.0) * dt_try for v in g_rates.values())
                gas_prod += sum(max(-v, 0.0) * dt_try for v in g_rates.values())
                mass1 = surface_water(grid, rock, trial.sw, pressure=trial.pressure, pvt=fluid)
                gas1 = surface_gas(grid, rock, trial.sg, pressure=trial.pressure, pvt=fluid)
                scale = max(abs(mass0), abs(injected), abs(produced), 1.0e-12)
                err = (mass1 - mass0) - (injected - produced) - boundary
                g_err = (gas1 - gas0) - (gas_inj - gas_prod)
                g_scale = max(abs(gas0), abs(gas_inj), abs(gas_prod), 1.0e-12)
                mb = MassBalance(
                    initial_mass=mass0,
                    final_mass=mass1,
                    injected_mass=injected,
                    produced_mass=produced,
                    boundary_flux=boundary,
                    balance_error=err,
                    relative_balance_error=abs(err) / scale,
                    gas_initial_mass=gas0,
                    gas_final_mass=gas1,
                    gas_injected_mass=gas_inj,
                    gas_produced_mass=gas_prod,
                    gas_balance_error=g_err,
                    gas_relative_balance_error=abs(g_err) / g_scale,
                )
                cfl = dt_try / dt_cfl if np.isfinite(dt_cfl) and dt_cfl > 0 else 0.0
                reports.append(
                    StepReport(
                        time_s=trial.time_s,
                        dt=dt_try,
                        max_cfl=float(cfl),
                        max_ds=ds,
                        mass=mb,
                        port_rates=dict(port_w),
                        notes=[note] if note else [],
                    )
                )
                state = trial
                dt = min(dt_try * 1.25, dt_max)
                accepted = True
                break
            if not accepted:
                raise TimeStepUnderflow(f"failed to accept a step at t={state.time_s}")
        times.append(state.time_s)
        states.append(state.copy())
        rates_hist.append(dict(reports[-1].port_rates) if reports else {})

    return Trajectory(
        times_s=np.asarray(times, dtype=float),
        states=states,
        reports=reports,
        port_rates=rates_hist,
    )
