"""Sequential IMPES: TPFA pressure then explicit black-oil transport.

Conservation is surface volume:

    (pv bα Sα − pv0 bα0 Sα0)/dt + Div(bα vα) = qα^s

Rate well controls are surface rates. Reservoir sources are q^s / b.
Incompressible B=1, c=0 is the lab special case of the same equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.discretization.tpfa import (
    assemble_pressure,
    face_fluxes,
    geometric_transmissibility,
    interior_transmissibility,
    phase_interior_fluxes,
    solve_pressure,
)
from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import InvalidSaturation, TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.physics.rock import Rock, as_cell_field
from reservoir_backend.ports.flow import FlowPort, half_cell_wi, peaceman_wi
from reservoir_backend.solver.fi import FiStepResult, _lambda, _well_surface_rates, dt_from_newton_iters, solve_fi_step
from reservoir_backend.solver.seqtools import (
    cross_flow_mixture,
    cross_flow_mixture_density,
    iteration_count_timestep,
    outer_converged,
    sequential_gravity_face,
    sequential_transport_extras,
    state_change_timestep,
)
from reservoir_backend.solver.transport import implicit_blackoil, implicit_water


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
    sw: NDArray[np.float64] | None = None,
) -> float:
    """Free gas plus dissolved Rs b_o So when ``sw`` and live-oil tables are present."""
    if sg is None:
        return 0.0
    fluid = pvt or BlackOilPVT.incompressible()
    p = np.asarray(sg, dtype=float) * 0.0 + 1.0e5 if pressure is None else np.asarray(pressure, dtype=float)
    sg_a = np.asarray(sg, dtype=float)
    if sw is not None:
        hold = fluid.surface_gas_holdup(sw, sg_a, p)
    else:
        hold = fluid.b_g(p) * sg_a
    acc = rock.porosity * fluid.pv_mult(p) * hold * grid.cell_volumes()
    return float(np.sum(acc))


def gas_mass(
    grid: CartesianGrid,
    rock: Rock,
    sg: NDArray[np.float64] | None,
    pressure: NDArray[np.float64] | None = None,
    pvt: BlackOilPVT | None = None,
    sw: NDArray[np.float64] | None = None,
) -> float:
    return surface_gas(grid, rock, sg, pressure=pressure, pvt=pvt, sw=sw)


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


def _upwind_b(
    b_l: NDArray[np.float64],
    b_r: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.where(q >= 0.0, b_l, b_r)


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


def _segregation_extra(
    grid: CartesianGrid,
    fx: NDArray[np.float64],
    fy: NDArray[np.float64],
    fz: NDArray[np.float64],
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    qz: NDArray[np.float64],
    frac_cell: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    f_ijk = grid.reshape_ijk(frac_cell)
    extra_x = np.zeros_like(fx)
    extra_y = np.zeros_like(fy)
    extra_z = np.zeros_like(fz)
    if grid.nx > 1:
        vx = fx[:, :, 1:-1]
        extra_x[:, :, 1:-1] = qx[:, :, 1:-1] - np.where(vx >= 0.0, f_ijk[:, :, :-1], f_ijk[:, :, 1:]) * vx
    if grid.ny > 1:
        vy = fy[:, 1:-1, :]
        extra_y[:, 1:-1, :] = qy[:, 1:-1, :] - np.where(vy >= 0.0, f_ijk[:, :-1, :], f_ijk[:, 1:, :]) * vy
    if grid.nz > 1:
        vz = fz[1:-1, :, :]
        extra_z[1:-1, :, :] = qz[1:-1, :, :] - np.where(vz >= 0.0, f_ijk[:-1, :, :], f_ijk[1:, :, :]) * vz
    return extra_x, extra_y, extra_z


def _well_transport_sources(
    well_index: dict[int, tuple[float, float]],
    pressure: NDArray[np.float64],
    lw: NDArray[np.float64],
    lo: NDArray[np.float64],
    lg: NDArray[np.float64] | None,
    b_w1: NDArray[np.float64],
    b_o1: NDArray[np.float64],
    b_g1: NDArray[np.float64],
    rs1: NDArray[np.float64],
    wi_port: dict[int, FlowPort],
    cmap: dict,
    t_ctrl: float,
    three_phase,
    qw_rate: NDArray[np.float64],
    qg_rate: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[int, float]]:
    """Freeze well qT and split by mobility in transport.

    Injectors keep deck composition unless the same well has producing
    perforations (``crossFlowMixture``). Producers return reservoir qT so
    the implicit transport residual can use the current fw/fo/fg.
    """
    src_w = np.asarray(qw_rate, dtype=float).copy()
    src_g = np.asarray(qg_rate, dtype=float).copy()
    prod_q: dict[int, float] = {}
    cells = [int(c) for c in well_index]
    if not cells:
        return src_w, src_g, prod_q
    names: list[str] = []
    name_to_idx: dict[str, int] = {}
    for c in cells:
        port = wi_port.get(c)
        name = port.name if port is not None else f"_cell_{c}"
        if name not in name_to_idx:
            name_to_idx[name] = len(name_to_idx)
        names.append(name)
    n_wells = len(name_to_idx)
    nperf = len(cells)
    flux = np.zeros((nperf, 3), dtype=float)
    compi = np.zeros((n_wells, 3), dtype=float)
    q_res_c = np.zeros(nperf, dtype=float)
    perf2well = np.zeros(nperf, dtype=np.int64)
    for i, c in enumerate(cells):
        wi, pbhp = well_index[c]
        q_res_c[i] = float(wi) * (float(pbhp) - float(pressure[c]))
        dp = float(pbhp) - float(pressure[c])
        flux[i, 0] = float(lw[c]) * dp
        flux[i, 1] = float(lo[c]) * dp
        flux[i, 2] = 0.0 if lg is None else float(lg[c]) * dp
        wid = name_to_idx[names[i]]
        perf2well[i] = wid
        port = wi_port.get(c)
        sw_src = _injection_sw(cmap, port, t_ctrl) if port is not None else 1.0
        sg_src = _injection_sg(cmap, port, t_ctrl) if port is not None and three_phase is not None else 0.0
        compi[wid] = [sw_src, max(0.0, 1.0 - sw_src - sg_src), sg_src]
    mixed = cross_flow_mixture(flux, compi, perf2well, n_wells)
    for i, c in enumerate(cells):
        q_res = float(q_res_c[i])
        wid = int(perf2well[i])
        sw_src, _so_src, sg_src = (float(x) for x in mixed[wid])
        if q_res >= 0.0:
            lt_c = max(float(lw[c] + lo[c] + (0.0 if lg is None else lg[c])), 1.0e-30)
            fw, fo, fg = float(lw[c] / lt_c), float(lo[c] / lt_c), 0.0 if lg is None else float(lg[c] / lt_c)
            qw_s, qg_s = _surface_from_reservoir(
                q_res,
                float(b_w1[c]),
                float(b_o1[c]),
                float(b_g1[c]),
                fw,
                fo,
                fg,
                sw_src,
                sg_src,
                float(rs1[c]),
            )
            src_w[c] += qw_s
            src_g[c] += qg_s
        else:
            prod_q[c] = q_res
    return src_w, src_g, prod_q


def _hybrid_gravity_fluxes(
    grid: CartesianGrid,
    rock: Rock,
    pressure: NDArray[np.float64],
    lw: NDArray[np.float64],
    lo: NDArray[np.float64],
    lg: NDArray[np.float64] | None,
    *,
    gravity: float,
    rho_w,
    rho_o,
    rho_g,
    pc,
    face_mult_x,
    face_mult_y,
    face_mult_z,
    fx=None,
    fy=None,
    fz=None,
    upwind: str = "potential",
) -> tuple:
    """Sequential transport extras ``q_α − f_α(vT) vT``.

    ``potential`` uses one Brenier–Jaffré flag on (G, vT). ``hybrid``
    keeps viscous vT-upwind and gravity at vT=0.
    """
    n = grid.n_cells
    g = float(gravity)
    t_gx, t_gy, t_gz = geometric_transmissibility(
        grid,
        rock.permeability,
        kz=rock.vertical_permeability(),
        mult_x=face_mult_x,
        mult_y=face_mult_y,
        mult_z=face_mult_z,
    )
    lw_i = grid.reshape_ijk(as_cell_field(lw, n, "lw"))
    lo_i = grid.reshape_ijk(as_cell_field(lo, n, "lo"))
    lg_a = np.zeros(n) if lg is None else as_cell_field(lg, n, "lg")
    lg_i = grid.reshape_ijk(lg_a)
    z_i = grid.reshape_ijk(grid.cell_centers()[:, 2])
    pc_i = None if pc is None else grid.reshape_ijk(as_cell_field(np.asarray(pc, dtype=float).ravel(), n, "pc"))

    def _rho(val, sl_l, sl_r):
        a = np.asarray(val, dtype=float)
        if a.size == 1:
            return float(a)
        r = grid.reshape_ijk(as_cell_field(a.ravel(), n, "rho"))
        return 0.5 * (r[sl_l] + r[sl_r])

    def _axis(t_geom, sl_l, sl_r, vt_face=None):
        dz = z_i[sl_l] - z_i[sl_r]
        dpc = 0.0 if pc_i is None else (pc_i[sl_l] - pc_i[sl_r])
        gw = (-dpc + _rho(rho_w, sl_l, sl_r) * g * dz).ravel()
        go = (_rho(rho_o, sl_l, sl_r) * g * dz).ravel()
        gg = (_rho(rho_g, sl_l, sl_r) * g * dz).ravel()
        pot = np.column_stack([gw, go, gg])
        mob_l = np.column_stack([lw_i[sl_l].ravel(), lo_i[sl_l].ravel(), lg_i[sl_l].ravel()])
        mob_r = np.column_stack([lw_i[sl_r].ravel(), lo_i[sl_r].ravel(), lg_i[sl_r].ravel()])
        if vt_face is None:
            q = sequential_gravity_face(t_geom, pot, mob_l, mob_r)
        else:
            q = sequential_transport_extras(vt_face, t_geom, pot, mob_l, mob_r, upwind=upwind)
        return q[:, 0].reshape(t_geom.shape), q[:, 1].reshape(t_geom.shape), q[:, 2].reshape(t_geom.shape)

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    qw_x = np.zeros((nz, ny, nx + 1))
    qw_y = np.zeros((nz, ny + 1, nx))
    qw_z = np.zeros((nz + 1, ny, nx))
    qo_x, qo_y, qo_z = np.zeros_like(qw_x), np.zeros_like(qw_y), np.zeros_like(qw_z)
    qg_x, qg_y, qg_z = np.zeros_like(qw_x), np.zeros_like(qw_y), np.zeros_like(qw_z)
    if nx > 1:
        vtx = None if fx is None else np.asarray(fx, dtype=float)[:, :, 1:-1]
        qw_x[:, :, 1:-1], qo_x[:, :, 1:-1], qg_x[:, :, 1:-1] = _axis(
            t_gx,
            (slice(None), slice(None), slice(None, -1)),
            (slice(None), slice(None), slice(1, None)),
            vtx,
        )
    if ny > 1:
        vty = None if fy is None else np.asarray(fy, dtype=float)[:, 1:-1, :]
        qw_y[:, 1:-1, :], qo_y[:, 1:-1, :], qg_y[:, 1:-1, :] = _axis(
            t_gy,
            (slice(None), slice(None, -1), slice(None)),
            (slice(None), slice(1, None), slice(None)),
            vty,
        )
    if nz > 1:
        vtz = None if fz is None else np.asarray(fz, dtype=float)[1:-1, :, :]
        qw_z[1:-1, :, :], qo_z[1:-1, :, :], qg_z[1:-1, :, :] = _axis(
            t_gz,
            (slice(None, -1), slice(None), slice(None)),
            (slice(1, None), slice(None), slice(None)),
            vtz,
        )
    return qw_x, qw_y, qw_z, qo_x, qo_y, qo_z, qg_x, qg_y, qg_z


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


def _fracs_from_lambda(
    lw: NDArray[np.float64],
    lo: NDArray[np.float64],
    lg: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    gas = np.zeros_like(lw) if lg is None else lg
    lt = np.maximum(lw + lo + gas, 1.0e-30)
    return lw / lt, lo / lt, gas / lt


def _cell_mobility(
    relperm: CoreyTwoPhase,
    three_phase: CoreyThreePhase | None,
    sw: NDArray[np.float64],
    sg: NDArray[np.float64] | None,
    fluid: BlackOilPVT,
    pressure: NDArray[np.float64],
    *,
    single_phase: bool,
    mu_single: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64]]:
    n = sw.size
    if single_phase:
        mobility = np.full(n, 1.0 / max(float(mu_single), 1.0e-30))
        return mobility, np.zeros(n, dtype=float), None, mobility
    use_table_mu = fluid.has_live_oil()
    mu_o = fluid.viscosity_o(pressure) if use_table_mu else None
    mu_g = fluid.viscosity_g(pressure) if use_table_mu else None
    if three_phase is not None:
        krw, kro, krg = three_phase.kr(sw, sg)
        lw = krw / float(three_phase.mu_w)
        lo = kro / (np.maximum(mu_o, 1.0e-30) if mu_o is not None else float(three_phase.mu_o))
        lg = krg / (np.maximum(mu_g, 1.0e-30) if mu_g is not None else float(three_phase.mu_g))
        return lw, lo, lg, lw + lo + lg
    krw, kro = relperm.kr(sw)
    lw = krw / float(relperm.mu_w)
    lo = kro / (np.maximum(mu_o, 1.0e-30) if mu_o is not None else float(relperm.mu_o))
    return lw, lo, None, lw + lo


def _wellbore_density(
    port: FlowPort,
    cells: NDArray[np.int64],
    lw: NDArray[np.float64],
    lo: NDArray[np.float64],
    lg: NDArray[np.float64] | None,
    mobility: NDArray[np.float64],
    rho_w: float,
    rho_o: float,
    rho_g: float,
    sw_inj: float,
) -> float:
    """Wellbore fluid density. Injector uses the injected stream.

    Producers use inflow mass / inflow volume over the well's
    perforations (λ-weighted cells if no inflow).
    """
    if port.role == "injector":
        sg_inj = 0.0
        so_inj = max(0.0, 1.0 - float(sw_inj) - sg_inj)
        return float(sw_inj) * float(rho_w) + so_inj * float(rho_o)
    ids = np.asarray(cells, dtype=np.int64).ravel()
    if ids.size == 0:
        return float(rho_o)
    nperf = int(ids.size)
    vol = np.zeros(nperf, dtype=float)
    mass = np.zeros((nperf, 1), dtype=float)
    for i, c in enumerate(ids):
        c = int(c)
        lt = max(float(mobility[c]), 1.0e-30)
        fw = float(lw[c] / lt)
        fo = float(lo[c] / lt)
        fg = 0.0 if lg is None else float(lg[c] / lt)
        rho_c = fw * float(rho_w) + fo * float(rho_o) + fg * float(rho_g)
        q = -lt
        vol[i] = q
        mass[i, 0] = q * rho_c
    rho = cross_flow_mixture_density(
        mass,
        vol,
        np.zeros((1, 1)),
        np.zeros(nperf, dtype=np.int64),
        1,
    )
    val = float(rho[0, 0])
    if not np.isfinite(val) or val <= 0.0:
        return float(rho_o)
    return val


def _connection_bhp(
    grid: CartesianGrid,
    cells: NDArray[np.int64],
    bhp: float,
    rho_wb: float,
    gravity: float,
) -> dict[int, float]:
    """IMEX *K well: BHP at the top connection, hydrostatic head down the wellbore."""
    ids = np.asarray(cells, dtype=np.int64).ravel()
    if ids.size == 0:
        return {}
    if abs(float(gravity)) < 1.0e-15 or ids.size == 1:
        return {int(c): float(bhp) for c in ids}
    z = grid.cell_centers()[:, 2]
    z_ref = float(np.max(z[ids]))
    return {
        int(c): float(bhp) + float(rho_wb) * float(gravity) * (z_ref - float(z[int(c)]))
        for c in ids
    }


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
    rs: float = 0.0,
) -> tuple[float, float]:
    """Surface water and gas rates from a reservoir volume rate.

    Produced gas is free gas plus dissolved gas in the oil stream.
    """
    if q_res >= 0.0:
        qw_s = q_res * sw_src * b_w
        qg_s = q_res * sg_src * b_g
        return qw_s, qg_s
    return q_res * fw * b_w, q_res * (fg * b_g + fo * b_o * rs)


def _sfi_pressure_flux(
    grid: CartesianGrid,
    rock: Rock,
    relperm,
    three_phase,
    fluid: BlackOilPVT,
    capillary,
    sw: NDArray[np.float64],
    sg: NDArray[np.float64] | None,
    p_old: NDArray[np.float64],
    dt: float,
    gravity: float,
    *,
    cell_dirichlet: dict[int, float],
    face_dirichlet: dict[str, float] | None,
    rate_src: NDArray[np.float64],
    wi_base: dict[int, tuple[float, float]],
    face_mult_x,
    face_mult_y,
    face_mult_z,
    single_phase: bool,
    mu_single: float,
):
    """Pressure + total flux from current saturations (SFI reupdate).

    Live oil repeats the linear solve until ``max|Δp| / range < 1e-3``,
    matching sequential pressure increment tolerance (properties at p).
    """
    so = 1.0 - sw - (0.0 if sg is None else sg)
    p_prop = np.asarray(p_old, dtype=float).ravel().copy()
    n_inc = 4 if fluid.has_live_oil() else 1
    pressure = p_prop
    lw = lo = lg = mob = None
    rho_w = rho_o = rho_g = None
    pc_cell = None
    well_index: dict[int, tuple[float, float]] = {}
    system = None
    for it in range(n_inc):
        lw, lo, lg, mob = _cell_mobility(
            relperm, three_phase, sw, sg, fluid, p_prop, single_phase=single_phase, mu_single=mu_single
        )
        well_index = {c: (base * float(mob[c]), pbhp) for c, (base, pbhp) in wi_base.items()}
        b_w = np.asarray(fluid.b_w(p_prop), dtype=float).ravel()
        b_o = np.asarray(fluid.b_o(p_prop), dtype=float).ravel()
        b_g = np.asarray(fluid.b_g(p_prop), dtype=float).ravel()
        rho_w = fluid.rho_w_sc * b_w
        rho_o = fluid.rho_o_sc * b_o
        rho_g = fluid.rho_g_sc * b_g
        storage = None
        p_prev = None
        if fluid.has_storage() and dt > 0.0:
            pv_ref = rock.porosity * grid.cell_volumes()
            storage = (
                pv_ref
                * np.asarray(fluid.pv_mult(p_old), dtype=float).ravel()
                * fluid.ct(sw, so, sg, p=p_prop)
                / float(dt)
            )
            p_prev = p_old
        pc_cell = None if isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw), dtype=float).ravel()
        system = assemble_pressure(
            grid,
            rock.permeability,
            mob,
            cell_dirichlet=cell_dirichlet or None,
            face_dirichlet=face_dirichlet,
            cell_rate=rate_src,
            well_index=well_index or None,
            storage=storage,
            pressure_prev=p_prev,
            kz=rock.vertical_permeability(),
            mult_x=face_mult_x,
            mult_y=face_mult_y,
            mult_z=face_mult_z,
            lw=lw,
            lo=lo,
            lg=lg,
            gravity=float(gravity),
            rho_w=rho_w,
            rho_o=rho_o,
            rho_g=rho_g,
            pc=pc_cell,
        )
        pressure = solve_pressure(system)
        rng = max(float(np.max(pressure) - np.min(pressure)), 1.0)
        if it > 0 and float(np.max(np.abs(pressure - p_prop))) / rng < 1.0e-3:
            break
        p_prop = pressure
    fx, fy, fz = face_fluxes(
        grid,
        pressure,
        system.t_x,
        system.t_y,
        system.t_z,
        face_dirichlet=face_dirichlet,
        k=rock.permeability,
        mobility=mob,
        g_x=system.g_x,
        g_y=system.g_y,
        g_z=system.g_z,
        kz=rock.vertical_permeability(),
    )
    return pressure, fx, fy, fz, lw, lo, lg, mob, rho_w, rho_o, rho_g, pc_cell, well_index


def _picard_pressure(
    grid: CartesianGrid,
    rock: Rock,
    relperm,
    three_phase,
    fluid: BlackOilPVT,
    capillary,
    sw: NDArray[np.float64],
    sg: NDArray[np.float64] | None,
    p_old: NDArray[np.float64],
    pressure: NDArray[np.float64],
    dt: float,
    gravity: float,
    *,
    cell_dirichlet: dict[int, float],
    face_dirichlet: dict[str, float] | None,
    cell_rate: NDArray[np.float64],
    well_index: dict[int, tuple[float, float]],
    mobility: NDArray[np.float64],
    face_mult_x,
    face_mult_y,
    face_mult_z,
    single_phase: bool,
    mu_single: float,
    max_it: int = 3,
    inc_tol: float = 1.0e-3,
):
    """Reassemble pressure with properties at the new p until increment is small."""
    so = 1.0 - sw - (0.0 if sg is None else sg)
    wi_base = {
        int(c): (float(wi) / max(float(mobility[int(c)]), 1.0e-30), float(pbhp))
        for c, (wi, pbhp) in well_index.items()
    }
    p_prop = np.asarray(pressure, dtype=float).ravel().copy()
    lw = lo = lg = mob = None
    rho_w = rho_o = rho_g = None
    pc_cell = None
    system = None
    p_out = p_prop
    for it in range(max(1, int(max_it))):
        lw, lo, lg, mob = _cell_mobility(
            relperm, three_phase, sw, sg, fluid, p_prop, single_phase=single_phase, mu_single=mu_single
        )
        well_index = {c: (base * float(mob[c]), pbhp) for c, (base, pbhp) in wi_base.items()}
        b_w = np.asarray(fluid.b_w(p_prop), dtype=float).ravel()
        b_o = np.asarray(fluid.b_o(p_prop), dtype=float).ravel()
        b_g = np.asarray(fluid.b_g(p_prop), dtype=float).ravel()
        rho_w = fluid.rho_w_sc * b_w
        rho_o = fluid.rho_o_sc * b_o
        rho_g = fluid.rho_g_sc * b_g
        storage = None
        p_prev = None
        if fluid.has_storage() and dt > 0.0:
            pv_ref = rock.porosity * grid.cell_volumes()
            storage = (
                pv_ref
                * np.asarray(fluid.pv_mult(p_old), dtype=float).ravel()
                * fluid.ct(sw, so, sg, p=p_prop)
                / float(dt)
            )
            p_prev = p_old
        pc_cell = None if isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw), dtype=float).ravel()
        system = assemble_pressure(
            grid,
            rock.permeability,
            mob,
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
            lw=lw,
            lo=lo,
            lg=lg,
            gravity=float(gravity),
            rho_w=rho_w,
            rho_o=rho_o,
            rho_g=rho_g,
            pc=pc_cell,
        )
        p_out = solve_pressure(system)
        rng = max(float(np.max(p_out) - np.min(p_out)), 1.0)
        if float(np.max(np.abs(p_out - p_prop))) / rng < float(inc_tol):
            break
        p_prop = p_out
    return p_out, system, lw, lo, lg, mob, rho_w, rho_o, rho_g, pc_cell, well_index


def _apply_reupdate(
    grid: CartesianGrid,
    rock: Rock,
    relperm,
    three_phase,
    fluid: BlackOilPVT,
    capillary,
    sw: NDArray[np.float64],
    sg: NDArray[np.float64] | None,
    pressure: NDArray[np.float64],
    p_old: NDArray[np.float64],
    dt: float,
    gravity: float,
    extras: dict,
    *,
    cell_dirichlet: dict[int, float],
    face_dirichlet: dict[str, float] | None,
    rate_src: NDArray[np.float64],
    wi_base: dict[int, tuple[float, float]],
    face_mult_x,
    face_mult_y,
    face_mult_z,
    single_phase: bool,
    mu_single: float,
) -> tuple:
    """After transport: new S, one pressure/flux solve, no second transport."""
    packed = _sfi_pressure_flux(
        grid,
        rock,
        relperm,
        three_phase,
        fluid,
        capillary,
        sw,
        sg,
        pressure,
        dt,
        float(gravity),
        cell_dirichlet=cell_dirichlet,
        face_dirichlet=face_dirichlet,
        rate_src=rate_src,
        wi_base=wi_base,
        face_mult_x=face_mult_x,
        face_mult_y=face_mult_y,
        face_mult_z=face_mult_z,
        single_phase=single_phase,
        mu_single=mu_single,
    )
    pressure_n, fx, fy, fz, lw, lo, lg, mob, rho_w, rho_o, rho_g, pc_cell, well_index = packed
    sg_n = sg
    if sg is not None and fluid.has_live_oil():
        g_pv = fluid.surface_gas_holdup(sw, sg, pressure_n)
        sg_n = fluid.flash_from_total(sw, g_pv, pressure_n)
    extras["reupdate_pressure"] = True
    extras["dp_rel"] = max(
        float(extras.get("dp_rel", 0.0)),
        float(np.max(np.abs(pressure_n - p_old)) / max(float(np.mean(np.abs(p_old))), 1.0)),
    )
    return pressure_n, fx, fy, fz, sw, sg_n, lw, lo, lg, mob, rho_w, rho_o, rho_g, pc_cell, well_index


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
    sfi_outer: int = 0,
    reupdate_pressure: bool = True,
    upwind_type: str = "potential",
    fully_implicit: bool = False,
) -> tuple[State, dict[str, float], dict[str, float], dict[str, float], NDArray, NDArray, NDArray]:
    """One sequential step. Port rates are surface water / reservoir liquid.

    ``sfi_outer`` is the sequential outer count: after transport, update
    pressure (and vT) with the new saturations and repeat transport. 0 is a
    single pressure/transport split.

    ``reupdate_pressure`` is P→T→P: after transport, one pressure/flux
    solve with the new saturations and no second transport.

    ``fully_implicit`` is a coupled (p, Sw, Sg) Newton on the same
    conservation residuals. Failure rejects the step so the driver
    halves dt. It does not switch the step to sequential transport.
    """
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
    sw_beg = sw.copy()

    p_old = as_cell_field(state.pressure, n, "pressure")
    lw_cell, lo_cell, lg_cell, mobility = _cell_mobility(
        relperm, three_phase, sw, sg, fluid, p_old, single_phase=single_phase, mu_single=mu_single
    )
    b_w = np.asarray(fluid.b_w(p_old), dtype=float).ravel()
    b_o = np.asarray(fluid.b_o(p_old), dtype=float).ravel()
    b_g = np.asarray(fluid.b_g(p_old), dtype=float).ravel()
    rs0 = np.asarray(fluid.rs(p_old), dtype=float).ravel()
    rho_w = fluid.rho_w_sc * b_w
    rho_o = fluid.rho_o_sc * b_o
    rho_g = fluid.rho_g_sc * b_g
    rho_w_mean = float(np.mean(rho_w))
    rho_o_mean = float(np.mean(rho_o))
    rho_g_mean = float(np.mean(rho_g))

    cell_rate = np.zeros(n, dtype=float)
    cell_qw_s = np.zeros(n, dtype=float)
    cell_qo_s = np.zeros(n, dtype=float)
    cell_qg_s = np.zeros(n, dtype=float)
    well_index: dict[int, tuple[float, float]] = {}
    well_datum: dict[int, float] = {}
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
                    lt_c = max(float(mobility[c]), 1.0e-30)
                    fw = float(lw_cell[c] / lt_c)
                    fo = float(lo_cell[c] / lt_c)
                    fg = 0.0 if lg_cell is None else float(lg_cell[c] / lt_c)
                    bmix = b_w[c] * fw + b_o[c] * fo + b_g[c] * fg
                    q_res = share_s / max(bmix, 1.0e-30)
                    qw_s = q_res * fw * b_w[c]
                    qo_s = q_res * fo * b_o[c]
                    qg_s = q_res * (fg * b_g[c] + fo * b_o[c] * rs0[c])
                cell_rate[c] += q_res
                cell_qw_s[c] += qw_s
                cell_qo_s[c] += qo_s
                cell_qg_s[c] += qg_s
        elif port.use_productivity:
            sw_src = _injection_sw(cmap, port, t_ctrl)
            rho_wb = _wellbore_density(
                port,
                port.cell_ids,
                lw_cell,
                lo_cell,
                lg_cell,
                mobility,
                rho_w_mean,
                rho_o_mean,
                rho_g_mean,
                sw_src,
            )
            p_conn = _connection_bhp(grid, port.cell_ids, float(val), rho_wb, float(gravity))
            for c in port.cell_ids:
                c = int(c)
                rw = float(getattr(port, "rw_m", 0.0) or 0.0)
                if rw > 0.0:
                    wi_geom = peaceman_wi(
                        grid,
                        c,
                        float(k_field[c]),
                        rw,
                        skin=float(getattr(port, "skin", 0.0) or 0.0),
                        geofac=float(getattr(port, "geofac", 0.0) or 0.0),
                    )
                    wi = float(port.wi_multiplier) * wi_geom * float(mobility[c])
                else:
                    wi = float(port.wi_multiplier) * half_cell_wi(grid, c, float(k_field[c])) * float(mobility[c])
                if wi > 0.0:
                    well_index[c] = (wi, p_conn[c])
                    well_datum[c] = float(val)
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
        storage = pv_ref * np.asarray(fluid.pv_mult(p_old), dtype=float).ravel() * fluid.ct(sw, so, sg, p=p_old) / float(dt)
        p_prev = p_old

    pc_cell = None if isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw), dtype=float).ravel()
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
        lg=lg_cell,
        gravity=float(gravity),
        rho_w=rho_w,
        rho_o=rho_o,
        rho_g=rho_g,
        pc=pc_cell,
    )
    pressure = solve_pressure(system)
    sg_old = None if sg is None else sg.copy()
    rs_cell = None if state.rs is None else as_cell_field(state.rs, n, "rs")
    if rs_cell is None and fluid.has_live_oil():
        rs_cell = np.asarray(fluid.rs(p_old), dtype=float).ravel()
    rs_beg = None if rs_cell is None else rs_cell.copy()
    if (
        three_phase is not None
        and sg is not None
        and fluid.has_live_oil()
        and dt > 0.0
        and not fully_implicit
    ):
        # Sequential only: flash after IMPES p, then Picard with λg.
        # FIM must start from (p^n, S^n); a flashed Picard p is too low
        # for S^n and Newton keeps the extra liberation.
        sg = fluid.flash_sg(sw, sg, pressure, p_old, rs=rs_beg)
        rs_sat_now = np.asarray(fluid.rs(pressure), dtype=float).ravel()
        rs_cell = np.where(fluid.vo_unsat(sg), np.minimum(rs_beg, rs_sat_now), rs_sat_now)
        sg = np.clip(sg, 0.0, 1.0 - sw)
        so = 1.0 - sw - sg
        liberated = float(np.max(np.abs(sg - sg_old))) > 1.0e-8
        if liberated:
            (
                pressure,
                system,
                lw_cell,
                lo_cell,
                lg_cell,
                mobility,
                rho_w,
                rho_o,
                rho_g,
                pc_cell,
                well_index,
            ) = _picard_pressure(
                grid,
                rock,
                relperm,
                three_phase,
                fluid,
                capillary,
                sw,
                sg,
                p_old,
                pressure,
                dt,
                float(gravity),
                cell_dirichlet=cell_dirichlet,
                face_dirichlet=face_dirichlet,
                cell_rate=cell_rate,
                well_index=well_index,
                mobility=mobility,
                face_mult_x=face_mult_x,
                face_mult_y=face_mult_y,
                face_mult_z=face_mult_z,
                single_phase=single_phase,
                mu_single=mu_single,
            )
            sg = fluid.flash_sg(sw, sg, pressure, p_old)
            sg = np.clip(sg, 0.0, 1.0 - sw)
            so = 1.0 - sw - sg
            lw_cell, lo_cell, lg_cell, mobility = _cell_mobility(
                relperm, three_phase, sw, sg, fluid, pressure, single_phase=single_phase, mu_single=mu_single
            )
    b_w1 = np.asarray(fluid.b_w(pressure), dtype=float).ravel()
    b_o1 = np.asarray(fluid.b_o(pressure), dtype=float).ravel()
    b_g1 = np.asarray(fluid.b_g(pressure), dtype=float).ravel()
    rs1 = np.asarray(fluid.rs(pressure), dtype=float).ravel()
    wi_port: dict[int, FlowPort] = {}
    for port in ports:
        if port.use_productivity:
            for c in port.cell_ids:
                wi_port[int(c)] = port
    rate_src = cell_rate.copy()
    qw_rate = cell_qw_s.copy()
    qo_rate = cell_qo_s.copy()
    qg_rate = cell_qg_s.copy()
    wi_base = {
        int(c): (float(wi) / max(float(mobility[int(c)]), 1.0e-30), float(pbhp))
        for c, (wi, pbhp) in well_index.items()
    }
    for c, (wi, pbhp) in well_index.items():
        q_res = float(wi) * (float(pbhp) - float(pressure[int(c)]))
        cell_rate[int(c)] += q_res
        port = wi_port.get(int(c))
        sw_src = _injection_sw(cmap, port, t_ctrl) if port is not None else 1.0
        sg_src = _injection_sg(cmap, port, t_ctrl) if port is not None and three_phase is not None else 0.0
        lt_c = max(float(mobility[int(c)]), 1.0e-30)
        fw = float(lw_cell[int(c)] / lt_c)
        fo = float(lo_cell[int(c)] / lt_c)
        fg = 0.0 if lg_cell is None else float(lg_cell[int(c)] / lt_c)
        qw_s, qg_s = _surface_from_reservoir(
            q_res,
            float(b_w1[int(c)]),
            float(b_o1[int(c)]),
            float(b_g1[int(c)]),
            fw,
            fo,
            fg,
            sw_src,
            sg_src,
            float(rs1[int(c)]),
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
        bo_m = float(np.mean(b_o1[cells]))
        rs_m = float(np.mean(rs1[cells]))
        if single_phase:
            qw_s, qg_s = q_res, 0.0
        elif q_res >= 0.0:
            sw_src = _injection_sw(cmap, port, t_ctrl)
            sg_src = _injection_sg(cmap, port, t_ctrl) if three_phase else 0.0
            qw_s, qg_s = q_res * sw_src * bw_m, q_res * sg_src * bg_m
        else:
            fw_c, fo_c, fg_c = _fracs_from_lambda(lw_cell[cells], lo_cell[cells], None if lg_cell is None else lg_cell[cells])
            fw_m, fo_m, fg_m = float(np.mean(fw_c)), float(np.mean(fo_c)), float(np.mean(fg_c))
            qw_s, qg_s = q_res * fw_m * bw_m, q_res * (fg_m * bg_m + fo_m * bo_m * rs_m)
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
    extras["fi_ok"] = False
    extras["fi_failed"] = False
    extras["scheme"] = "sequential"
    extras["dp_rel"] = float(np.max(np.abs(pressure - p_old)) / max(float(np.mean(np.abs(p_old))), 1.0))
    if fully_implicit and implicit and three_phase is not None and sg is not None and dt > 0.0:
        wi_comp = {}
        for c in well_index:
            port = wi_port.get(int(c))
            sw_src = _injection_sw(cmap, port, t_ctrl) if port is not None else 1.0
            sg_src = _injection_sg(cmap, port, t_ctrl) if port is not None else 0.0
            wi_comp[int(c)] = (sw_src, sg_src)
        wi_group = {
            int(c): i
            for i, port in enumerate(ports)
            if port.use_productivity
            for c in port.cell_ids
            if int(c) in well_index
        }
        # Freeze connection total mobility without free gas (oil+water at t^n).
        sg_lt = sg_old if sg_old is not None else sg
        lw0, lo0, lg0, _mob0 = _cell_mobility(
            relperm,
            three_phase,
            sw_beg,
            sg_lt,
            fluid,
            p_old,
            single_phase=single_phase,
            mu_single=mu_single,
        )
        lt_fixed = np.maximum(np.asarray(lw0, dtype=float) + np.asarray(lo0, dtype=float), 1.0e-30)
        fi = solve_fi_step(
            grid,
            rock,
            three_phase,
            fluid,
            capillary,
            sw_beg,
            sg_old if sg_old is not None else sg,
            p_old,
            pressure,
            dt,
            float(gravity),
            src_w=qw_rate,
            src_o=qo_rate,
            src_g=qg_rate,
            rs0=rs_beg,
            wi_base=wi_base,
            wi_comp=wi_comp or None,
            wi_group=wi_group,
            wi_datum=well_datum or None,
            lt_fixed=lt_fixed,
            cell_dirichlet=cell_dirichlet or None,
            face_mult_x=face_mult_x,
            face_mult_y=face_mult_y,
            face_mult_z=face_mult_z,
        )
        extras["scheme"] = "fim"
        if fi is not None:
            if not isinstance(fi, FiStepResult):
                # defensive: older tuple shape
                p_fi, sw_fi, sg_fi, rs_fi = fi[:4]
                fx_fi, fy_fi, fz_fi = fx, fy, fz
                n_newt = 0
            else:
                p_fi, sw_fi, sg_fi, rs_fi = fi.pressure, fi.sw, fi.sg, fi.rs
                fx_fi, fy_fi, fz_fi = fi.fx, fi.fy, fi.fz
                n_newt = int(fi.newton_iters)
            sat_fi = None if not fluid.has_live_oil() else ~fluid.vo_unsat(sg_fi)
            lw_f, lo_f, lg_f = _lambda(three_phase, fluid, sw_fi, sg_fi, p_fi, rs=rs_fi, saturated=sat_fi)
            bw_f = fluid.b_w(p_fi)
            bo_f = fluid.b_o(p_fi, rs=rs_fi, saturated=sat_fi)
            bg_f = fluid.b_g(p_fi)
            rho_w_f = fluid.density_w(p_fi, bw=bw_f)
            rho_o_f = fluid.density_o(p_fi, rs=rs_fi, bo=bo_f)
            rho_g_f = fluid.density_g(p_fi, bg=bg_f)
            z_cell = np.asarray(grid.cell_centers()[:, 2], dtype=float).ravel()
            dw_f, _do_f, dg_f = _well_surface_rates(
                wi_base,
                wi_comp or None,
                wi_group,
                p_fi,
                lw_f,
                lo_f,
                lg_f,
                bw_f,
                bo_f,
                bg_f,
                rs_fi,
                wi_datum=well_datum or None,
                z=z_cell,
                gravity=float(gravity),
                rho_w=rho_w_f,
                rho_o=rho_o_f,
                rho_g=rho_g_f,
                lt_fixed=lt_fixed,
            )
            port_water = {}
            port_liquid = {}
            port_gas = {}
            cell_rate_fi = np.zeros(n, dtype=float)
            for port in ports:
                cells = np.asarray(port.cell_ids, dtype=np.int64)
                qw_s = float(np.sum(dw_f[cells]))
                qg_s = float(np.sum(dg_f[cells]))
                q_res = 0.0
                for c in cells:
                    c = int(c)
                    if c not in wi_base:
                        continue
                    base, pbhp = wi_base[c]
                    lt = float(lt_fixed[c]) if lt_fixed is not None else float(lw_f[c] + lo_f[c] + lg_f[c])
                    qi = float(base) * lt * (float(pbhp) - float(p_fi[c]))
                    q_res += qi
                    cell_rate_fi[c] += qi
                port_water[port.name] = qw_s
                port_liquid[port.name] = q_res
                port_gas[port.name] = qg_s
            extras["implicit_ok"] = True
            extras["fi_ok"] = True
            extras["newton_its"] = n_newt
            extras["boundary_water"] = 0.0
            extras["boundary_gas"] = 0.0
            extras["gas_rates"] = port_gas
            extras["cell_rate"] = cell_rate_fi
            extras["dp_rel"] = float(np.max(np.abs(p_fi - p_old)) / max(float(np.mean(np.abs(p_old))), 1.0))
            extras["vo_unsat"] = float(np.mean(fluid.vo_unsat(sg_fi))) if fluid.has_live_oil() else 0.0
            return (
                State(pressure=p_fi, sw=sw_fi, sg=sg_fi, rs=rs_fi, time_s=state.time_s + dt),
                port_water,
                port_liquid,
                extras,
                fx_fi,
                fy_fi,
                fz_fi,
            )
        extras["fi_failed"] = True
        extras["implicit_ok"] = False
        return (
            State(
                pressure=p_old,
                sw=sw_beg,
                sg=sg_old if sg_old is not None else sg,
                rs=rs_beg,
                time_s=state.time_s,
            ),
            port_water,
            port_liquid,
            extras,
            fx,
            fy,
            fz,
        )
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

    qw_x, qw_y, qw_z, _qo_x, _qo_y, _qo_z, qg_x, qg_y, qg_z = phase_interior_fluxes(
        grid,
        pressure,
        rock.permeability,
        lw_cell,
        lo_cell,
        lg=lg_cell,
        kz=rock.vertical_permeability(),
        mult_x=face_mult_x,
        mult_y=face_mult_y,
        mult_z=face_mult_z,
        gravity=float(gravity),
        rho_w=rho_w,
        rho_o=rho_o,
        rho_g=rho_g,
        pc=pc_cell,
    )
    fw_c, fo_c, fg_c = _fracs_from_lambda(lw_cell, lo_cell, lg_cell)
    hyb_w_x, hyb_w_y, hyb_w_z, hyb_o_x, hyb_o_y, hyb_o_z, hyb_g_x, hyb_g_y, hyb_g_z = _hybrid_gravity_fluxes(
        grid,
        rock,
        pressure,
        lw_cell,
        lo_cell,
        lg_cell,
        gravity=float(gravity),
        rho_w=rho_w,
        rho_o=rho_o,
        rho_g=rho_g,
        pc=pc_cell,
        face_mult_x=face_mult_x,
        face_mult_y=face_mult_y,
        face_mult_z=face_mult_z,
        fx=fx,
        fy=fy,
        fz=fz,
        upwind=upwind_type,
    )
    extra_x, extra_y, extra_z = hyb_w_x, hyb_w_y, hyb_w_z

    if implicit:
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
        if three_phase is not None and sg is not None:
            extra_gx, extra_gy, extra_gz = hyb_g_x, hyb_g_y, hyb_g_z
            extra_ox, extra_oy, extra_oz = hyb_o_x, hyb_o_y, hyb_o_z
            injector_fg: dict[int, float] = {}
            for port in ports:
                if port.control != "pressure" or port.use_productivity or port.role != "injector":
                    continue
                sg_src = _injection_sg(cmap, port, t_ctrl)
                for c in port.cell_ids:
                    if _cell_face_outflow(grid, int(c), fx, fy, fz) > 0.0:
                        injector_fg[int(c)] = sg_src
            acc_g0 = pv0 * fluid.surface_gas_holdup(
                sw_beg, sg_old if sg_old is not None else sg, p_old, rs=rs_beg
            )
            src_w, src_g, wi_prod_q = _well_transport_sources(
                well_index,
                pressure,
                lw_cell,
                lo_cell,
                lg_cell,
                b_w1,
                b_o1,
                b_g1,
                rs1,
                wi_port,
                cmap,
                t_ctrl,
                three_phase,
                qw_rate,
                qg_rate,
            )
            well_prod = {**producer_q, **wi_prod_q}

            def _project_vo(sw_a, sg_a):
                if not fluid.has_live_oil():
                    return sw_a, sg_a
                g_pv = fluid.surface_gas_holdup(sw_a, sg_a, pressure)
                return sw_a, fluid.flash_from_total(sw_a, g_pv, pressure)

            def _refresh_extras(sw_a, sg_a):
                lw_n, lo_n, lg_n, _mob_n = _cell_mobility(
                    relperm, three_phase, sw_a, sg_a, fluid, pressure,
                    single_phase=single_phase, mu_single=mu_single,
                )
                rho_wn = fluid.rho_w_sc * np.asarray(fluid.b_w(pressure), dtype=float).ravel()
                rho_on = fluid.rho_o_sc * np.asarray(fluid.b_o(pressure), dtype=float).ravel()
                rho_gn = fluid.rho_g_sc * np.asarray(fluid.b_g(pressure), dtype=float).ravel()
                pc_n = None if isinstance(capillary, NoCapillary) else np.asarray(capillary.pc(sw_a), dtype=float).ravel()
                ewx, ewy, ewz, eox, eoy, eoz, egx, egy, egz = _hybrid_gravity_fluxes(
                    grid,
                    rock,
                    pressure,
                    lw_n,
                    lo_n,
                    lg_n,
                    gravity=float(gravity),
                    rho_w=rho_wn,
                    rho_o=rho_on,
                    rho_g=rho_gn,
                    pc=pc_n,
                    face_mult_x=face_mult_x,
                    face_mult_y=face_mult_y,
                    face_mult_z=face_mult_z,
                    fx=fx,
                    fy=fy,
                    fz=fz,
                    upwind=upwind_type,
                )
                return ewx, ewy, ewz, egx, egy, egz, eox, eoy, eoz

            nls_stats: dict = {}
            pair = implicit_blackoil(
                grid,
                three_phase,
                sw,
                sg,
                pv0 * b_w * sw_beg,
                acc_g0,
                pv1,
                b_w1,
                b_g1,
                rs1 * b_o1,
                fx,
                fy,
                fz,
                src_w,
                src_g,
                dt,
                extra_w_x=extra_x,
                extra_w_y=extra_y,
                extra_w_z=extra_z,
                extra_g_x=extra_gx,
                extra_g_y=extra_gy,
                extra_g_z=extra_gz,
                extra_o_x=extra_ox,
                extra_o_y=extra_oy,
                extra_o_z=extra_oz,
                pinned=np.asarray(pinned_ids, dtype=np.int64) if pinned_ids else None,
                injector_fw=injector_fw or None,
                injector_fg=injector_fg or None,
                producer_q=well_prod or None,
                refresh_extras=_refresh_extras,
                project=_project_vo,
                rs0=None,
                rs_sat=None,
                b_o1=b_o1,
                acc_o0=pv0 * b_o * np.clip(1.0 - sw_beg - (sg_old if sg_old is not None else sg), 0.0, 1.0),
                rs_out=rs_cell,
                stats=nls_stats,
            )
            if pair is not None:
                extras["implicit_ok"] = True
                extras["boundary_water"] = 0.0
                extras["boundary_gas"] = 0.0
                extras["newton_its"] = int(nls_stats.get("newton_its", 0))
                sw_imp, sg_imp = pair
                if fluid.has_live_oil():
                    g_pv = fluid.surface_gas_holdup(sw_imp, sg_imp, pressure)
                    sg_imp = fluid.flash_from_total(sw_imp, g_pv, pressure)
                n_outer = max(0, int(sfi_outer))
                extras["sfi_used"] = 0
                if n_outer <= 0:
                    if reupdate_pressure:
                        (
                            pressure,
                            fx,
                            fy,
                            fz,
                            sw_imp,
                            sg_imp,
                            _lw,
                            _lo,
                            _lg,
                            _mob,
                            _rw,
                            _ro,
                            _rg,
                            _pc,
                            _wi,
                        ) = _apply_reupdate(
                            grid,
                            rock,
                            relperm,
                            three_phase,
                            fluid,
                            capillary,
                            sw_imp,
                            sg_imp,
                            pressure,
                            p_old,
                            dt,
                            float(gravity),
                            extras,
                            cell_dirichlet=cell_dirichlet,
                            face_dirichlet=face_dirichlet,
                            rate_src=rate_src,
                            wi_base=wi_base,
                            face_mult_x=face_mult_x,
                            face_mult_y=face_mult_y,
                            face_mult_z=face_mult_z,
                            single_phase=single_phase,
                            mu_single=mu_single,
                        )
                    return (
                        State(pressure=pressure, sw=sw_imp, sg=sg_imp, rs=rs_cell, time_s=state.time_s + dt),
                        port_water,
                        port_liquid,
                        extras,
                        fx,
                        fy,
                        fz,
                    )
                for _it in range(n_outer):
                    (
                        pressure,
                        fx,
                        fy,
                        fz,
                        lw_cell,
                        lo_cell,
                        lg_cell,
                        mobility,
                        rho_w,
                        rho_o,
                        rho_g,
                        pc_cell,
                        well_index,
                    ) = _sfi_pressure_flux(
                        grid,
                        rock,
                        relperm,
                        three_phase,
                        fluid,
                        capillary,
                        sw_imp,
                        sg_imp,
                        p_old,
                        dt,
                        float(gravity),
                        cell_dirichlet=cell_dirichlet,
                        face_dirichlet=face_dirichlet,
                        rate_src=rate_src,
                        wi_base=wi_base,
                        face_mult_x=face_mult_x,
                        face_mult_y=face_mult_y,
                        face_mult_z=face_mult_z,
                        single_phase=single_phase,
                        mu_single=mu_single,
                    )
                    b_w1 = np.asarray(fluid.b_w(pressure), dtype=float).ravel()
                    b_o1 = np.asarray(fluid.b_o(pressure), dtype=float).ravel()
                    b_g1 = np.asarray(fluid.b_g(pressure), dtype=float).ravel()
                    rs1 = np.asarray(fluid.rs(pressure), dtype=float).ravel()
                    pv1 = rock.porosity * np.asarray(fluid.pv_mult(pressure), dtype=float).ravel() * vol
                    cell_qw_s = qw_rate.copy()
                    cell_qg_s = qg_rate.copy()
                    for c, (wi, pbhp) in well_index.items():
                        q_res = float(wi) * (float(pbhp) - float(pressure[int(c)]))
                        port = wi_port.get(int(c))
                        sw_src = _injection_sw(cmap, port, t_ctrl) if port is not None else 1.0
                        sg_src = _injection_sg(cmap, port, t_ctrl) if port is not None else 0.0
                        lt_c = max(float(mobility[int(c)]), 1.0e-30)
                        fw = float(lw_cell[int(c)] / lt_c)
                        fo = float(lo_cell[int(c)] / lt_c)
                        fg = 0.0 if lg_cell is None else float(lg_cell[int(c)] / lt_c)
                        qw_s, qg_s = _surface_from_reservoir(
                            q_res,
                            float(b_w1[int(c)]),
                            float(b_o1[int(c)]),
                            float(b_g1[int(c)]),
                            fw,
                            fo,
                            fg,
                            sw_src,
                            sg_src,
                            float(rs1[int(c)]),
                        )
                        cell_qw_s[int(c)] += qw_s
                        cell_qg_s[int(c)] += qg_s
                    qw_x, qw_y, qw_z, _qo_x, _qo_y, _qo_z, qg_x, qg_y, qg_z = phase_interior_fluxes(
                        grid,
                        pressure,
                        rock.permeability,
                        lw_cell,
                        lo_cell,
                        lg=lg_cell,
                        kz=rock.vertical_permeability(),
                        mult_x=face_mult_x,
                        mult_y=face_mult_y,
                        mult_z=face_mult_z,
                        gravity=float(gravity),
                        rho_w=rho_w,
                        rho_o=rho_o,
                        rho_g=rho_g,
                        pc=pc_cell,
                    )
                    extra_x, extra_y, extra_z, extra_ox, extra_oy, extra_oz, extra_gx, extra_gy, extra_gz = (
                        _hybrid_gravity_fluxes(
                            grid,
                            rock,
                            pressure,
                            lw_cell,
                            lo_cell,
                            lg_cell,
                            gravity=float(gravity),
                            rho_w=rho_w,
                            rho_o=rho_o,
                            rho_g=rho_g,
                            pc=pc_cell,
                            face_mult_x=face_mult_x,
                            face_mult_y=face_mult_y,
                            face_mult_z=face_mult_z,
                            fx=fx,
                            fy=fy,
                            fz=fz,
                            upwind=upwind_type,
                        )
                    )
                    pair2 = implicit_blackoil(
                        grid,
                        three_phase,
                        sw_imp,
                        sg_imp,
                        pv0 * b_w * sw_beg,
                        acc_g0,
                        pv1,
                        b_w1,
                        b_g1,
                        rs1 * b_o1,
                        fx,
                        fy,
                        fz,
                        src_w,
                        src_g,
                        dt,
                        extra_w_x=extra_x,
                        extra_w_y=extra_y,
                        extra_w_z=extra_z,
                        extra_g_x=extra_gx,
                        extra_g_y=extra_gy,
                        extra_g_z=extra_gz,
                        extra_o_x=extra_ox,
                        extra_o_y=extra_oy,
                        extra_o_z=extra_oz,
                        pinned=np.asarray(pinned_ids, dtype=np.int64) if pinned_ids else None,
                        injector_fw=injector_fw or None,
                        injector_fg=injector_fg or None,
                        producer_q=well_prod or None,
                        refresh_extras=_refresh_extras,
                        project=_project_vo,
                        b_o1=b_o1,
                        acc_o0=pv0 * b_o * np.clip(1.0 - sw_beg - (sg_old if sg_old is not None else sg), 0.0, 1.0),
                    )
                    if pair2 is None:
                        break
                    sw2, sg2 = pair2
                    extras["sfi_used"] = int(_it) + 1
                    done = outer_converged(sw2, sw_imp, sg2, sg_imp)
                    sw_imp, sg_imp = sw2, sg2
                    if done:
                        break
                pressure, fx, fy, fz, lw_cell, lo_cell, lg_cell, mobility, rho_w, rho_o, rho_g, pc_cell, well_index = (
                    _sfi_pressure_flux(
                        grid,
                        rock,
                        relperm,
                        three_phase,
                        fluid,
                        capillary,
                        sw_imp,
                        sg_imp,
                        p_old,
                        dt,
                        float(gravity),
                        cell_dirichlet=cell_dirichlet,
                        face_dirichlet=face_dirichlet,
                        rate_src=rate_src,
                        wi_base=wi_base,
                        face_mult_x=face_mult_x,
                        face_mult_y=face_mult_y,
                        face_mult_z=face_mult_z,
                        single_phase=single_phase,
                        mu_single=mu_single,
                    )
                )
                b_w1 = np.asarray(fluid.b_w(pressure), dtype=float).ravel()
                b_o1 = np.asarray(fluid.b_o(pressure), dtype=float).ravel()
                b_g1 = np.asarray(fluid.b_g(pressure), dtype=float).ravel()
                rs1 = np.asarray(fluid.rs(pressure), dtype=float).ravel()
                port_liquid = {}
                port_water = {}
                port_gas = {}
                cell_rate_out = rate_src.copy()
                for c, (wi, pbhp) in well_index.items():
                    cell_rate_out[int(c)] += float(wi) * (float(pbhp) - float(pressure[int(c)]))
                for port in ports:
                    if port.control == "pressure" and not port.use_productivity:
                        q_res = _port_total_rate(grid, port, fx, fy, fz, cell_rate_out)
                    else:
                        q_res = float(np.sum(cell_rate_out[port.cell_ids]))
                    cells = port.cell_ids
                    bw_m = float(np.mean(b_w1[cells]))
                    bg_m = float(np.mean(b_g1[cells]))
                    bo_m = float(np.mean(b_o1[cells]))
                    rs_m = float(np.mean(rs1[cells]))
                    if q_res >= 0.0:
                        sw_src = _injection_sw(cmap, port, t_ctrl)
                        sg_src = _injection_sg(cmap, port, t_ctrl)
                        qw_s, qg_s = q_res * sw_src * bw_m, q_res * sg_src * bg_m
                    else:
                        fw_c, fo_c, fg_c = _fracs_from_lambda(
                            lw_cell[cells], lo_cell[cells], None if lg_cell is None else lg_cell[cells]
                        )
                        fw_m, fo_m, fg_m = float(np.mean(fw_c)), float(np.mean(fo_c)), float(np.mean(fg_c))
                        qw_s, qg_s = q_res * fw_m * bw_m, q_res * (fg_m * bg_m + fo_m * bo_m * rs_m)
                    port_liquid[port.name] = q_res
                    port_water[port.name] = qw_s
                    port_gas[port.name] = qg_s
                extras["gas_rates"] = port_gas
                extras["sfi_outer"] = n_outer
                return (
                    State(pressure=pressure, sw=sw_imp, sg=sg_imp, rs=rs_cell, time_s=state.time_s + dt),
                    port_water,
                    port_liquid,
                    extras,
                    fx,
                    fy,
                    fz,
                )
        else:
            src_w2, _src_g2, wi_prod2 = _well_transport_sources(
                well_index,
                pressure,
                lw_cell,
                lo_cell,
                None,
                b_w1,
                b_o1,
                b_g1,
                rs1,
                wi_port,
                cmap,
                t_ctrl,
                None,
                qw_rate,
                qg_rate,
            )
            nls_stats_w: dict = {}
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
                src_w2,
                dt,
                pinned=np.asarray(pinned_ids, dtype=np.int64) if pinned_ids else None,
                injector_fw=injector_fw or None,
                producer_q={**producer_q, **wi_prod2} or None,
                extra_x=extra_x,
                extra_y=extra_y,
                extra_z=extra_z,
                stats=nls_stats_w,
            )
            if sw_imp is not None:
                extras["implicit_ok"] = True
                extras["boundary_water"] = 0.0
                extras["newton_its"] = int(nls_stats_w.get("newton_its", 0))
                if reupdate_pressure:
                    (
                        pressure,
                        fx,
                        fy,
                        fz,
                        sw_imp,
                        _sg_n,
                        _lw,
                        _lo,
                        _lg,
                        _mob,
                        _rw,
                        _ro,
                        _rg,
                        _pc,
                        _wi,
                    ) = _apply_reupdate(
                        grid,
                        rock,
                        relperm,
                        None,
                        fluid,
                        capillary,
                        sw_imp,
                        None,
                        pressure,
                        p_old,
                        dt,
                        float(gravity),
                        extras,
                        cell_dirichlet=cell_dirichlet,
                        face_dirichlet=face_dirichlet,
                        rate_src=rate_src,
                        wi_base=wi_base,
                        face_mult_x=face_mult_x,
                        face_mult_y=face_mult_y,
                        face_mult_z=face_mult_z,
                        single_phase=single_phase,
                        mu_single=mu_single,
                    )
                return (
                    State(pressure=pressure, sw=sw_imp, sg=None, time_s=state.time_s + dt),
                    port_water,
                    port_liquid,
                    extras,
                    fx,
                    fy,
                    fz,
                )

    qo_x = fx - qw_x - qg_x
    qo_y = fy - qw_y - qg_y
    qo_z = fz - qw_z - qg_z
    if face_dirichlet:
        fw, fo, fg = _fracs_from_lambda(
            grid.reshape_ijk(lw_cell),
            grid.reshape_ijk(lo_cell),
            None if lg_cell is None else grid.reshape_ijk(lg_cell),
        )
        inj_fw = 0.0 if three_phase is not None else float(relperm.fractional_flow(1.0 - relperm.sor))
        _dirichlet_phase_faces(grid, qw_x, qw_y, qw_z, fw, inj_fw, face_dirichlet)
        if three_phase is not None:
            _dirichlet_phase_faces(grid, qg_x, qg_y, qg_z, fg, 0.0, face_dirichlet)
            _dirichlet_phase_faces(grid, qo_x, qo_y, qo_z, fo, 0.0, face_dirichlet)

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
    rs_bo = grid.reshape_ijk(rs1 * b_o1)
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
    if sg is not None:
        qg_s_x = qg_s_x + qo_x * _upwind_b(
            np.concatenate([rs_bo[:, :, :1], rs_bo], axis=2),
            np.concatenate([rs_bo, rs_bo[:, :, -1:]], axis=2),
            qo_x,
        )
        qg_s_y = qg_s_y + qo_y * _upwind_b(
            np.concatenate([rs_bo[:, :1, :], rs_bo], axis=1),
            np.concatenate([rs_bo, rs_bo[:, -1:, :]], axis=1),
            qo_y,
        )
        qg_s_z = qg_s_z + qo_z * _upwind_b(
            np.concatenate([rs_bo[:1, :, :], rs_bo], axis=0),
            np.concatenate([rs_bo, rs_bo[-1:, :, :]], axis=0),
            qo_z,
        )

    extras["boundary_water"] = _boundary_in(qw_s_x, qw_s_y, qw_s_z)
    extras["boundary_gas"] = _boundary_in(qg_s_x, qg_s_y, qg_s_z)

    acc_w = pv0 * b_w * sw - float(dt) * _divergence(grid, qw_s_x, qw_s_y, qw_s_z) + float(dt) * cell_qw_s
    acc_g = None
    if sg is not None:
        g0 = pv0 * fluid.surface_gas_holdup(
            sw_beg, sg_old if sg_old is not None else sg, p_old, rs=rs_beg
        )
        acc_g = g0 - float(dt) * _divergence(grid, qg_s_x, qg_s_y, qg_s_z) + float(dt) * cell_qg_s

    for port in ports:
        if port.control != "pressure" or port.use_productivity or port.role != "producer":
            continue
        for c in port.cell_ids:
            c = int(c)
            q_out = _cell_face_outflow(grid, c, fx, fy, fz)
            if q_out >= 0.0:
                continue
            lt_c = max(float(mobility[c]), 1.0e-30)
            fw_c = float(lw_cell[c] / lt_c)
            fo_c = float(lo_cell[c] / lt_c)
            fg_c = 0.0 if lg_cell is None else float(lg_cell[c] / lt_c)
            acc_w[c] += float(dt) * q_out * fw_c * float(b_w1[c])
            if acc_g is not None:
                acc_g[c] += float(dt) * q_out * (fg_c * float(b_g1[c]) + fo_c * float(b_o1[c]) * float(rs1[c]))

    denom_w = np.maximum(pv1 * b_w1, 1.0e-30)
    if extras.get("implicit_ok") and sg is not None:
        raw_w = sw
    else:
        raw_w = acc_w / denom_w
    if acc_g is None:
        raw_g = None
    else:
        raw_g = fluid.flash_from_total(raw_w, acc_g / np.maximum(pv1, 1.0e-30), pressure)

    for c in pinned_injectors:
        raw_w[c] = sw[c]
        if raw_g is not None and sg is not None:
            raw_g[c] = sg[c]

    extras["dp_rel"] = float(np.max(np.abs(pressure - p_old)) / max(float(np.mean(np.abs(p_old))), 1.0))
    if raw_g is not None and rs_cell is not None and fluid.has_live_oil():
        rs_sat_now = np.asarray(fluid.rs(pressure), dtype=float).ravel()
        rs_cell = np.where(fluid.vo_unsat(raw_g), np.minimum(rs_cell, rs_sat_now), rs_sat_now)
    new_state = State(pressure=pressure, sw=raw_w, sg=raw_g, rs=rs_cell, time_s=state.time_s + dt)
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
    sfi_outer: int = 0,
    reupdate_pressure: bool = True,
    upwind_type: str = "potential",
    fully_implicit: bool = False,
    dt_init: float = 1.0,
    dt_min: float = 1.0e-6,
    dt_max: float | None = None,
    max_cfl: float = 0.5,
    max_ds: float = 0.15,
    max_steps: int = 12000,
    report_times: NDArray[np.float64] | None = None,
) -> Trajectory:
    """Advance from ``state0.time_s`` to ``t_end`` with adaptive dt.

    ``max_steps`` is a safety fuse. Implicit transport is not sized to an
    explicit CFL step budget; chop on Newton / ``max_ds`` / bounds instead.
    """
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
    dt_prev: float | None = None
    its_prev: int | None = None
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
        sfi_outer=sfi_outer,
        reupdate_pressure=reupdate_pressure,
        upwind_type=upwind_type,
        fully_implicit=fully_implicit,
    )
    rates_hist: list[dict[str, float]] = [dict(port_w0)]
    mass0 = surface_water(grid, rock, state.sw, pressure=state.pressure, pvt=fluid)
    gas0 = surface_gas(grid, rock, state.sg, pressure=state.pressure, pvt=fluid, sw=state.sw)
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
                    sfi_outer=sfi_outer,
                    reupdate_pressure=reupdate_pressure,
                    upwind_type=upwind_type,
                    fully_implicit=fully_implicit,
                )
                if extras.get("fi_failed"):
                    dt_try *= 0.5
                    note = "fim"
                    continue
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
                dp_rel = float(extras.get("dp_rel", 0.0))
                if dp_rel > 0.55 and dt_try > hard_floor * 2.0:
                    dt_try *= 0.5
                    note = "dp"
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
                gas1 = surface_gas(grid, rock, trial.sg, pressure=trial.pressure, pvt=fluid, sw=trial.sw)
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
                its_now = extras.get("newton_its")
                dt_state = state_change_timestep(
                    dt_try,
                    ds,
                    dp_rel,
                    target_ds=float(max_ds),
                    dt_min=hard_floor,
                    dt_max=dt_max,
                )
                if extras.get("implicit_ok") and its_now is not None:
                    if extras.get("scheme") == "fim":
                        dt_its = dt_from_newton_iters(
                            dt_try,
                            int(its_now),
                            dt0=dt_prev,
                            its0=its_prev,
                            dt_min=hard_floor,
                            dt_max=float(dt_max) if dt_max is not None else 1.0e30,
                        )
                    else:
                        dt_its = iteration_count_timestep(
                            dt_try,
                            int(its_now),
                            dt0=dt_prev,
                            its0=its_prev,
                            dt_min=hard_floor,
                            dt_max=dt_max,
                        )
                    dt = min(dt_its, dt_state)
                    dt_prev = dt_try
                    its_prev = int(its_now)
                else:
                    dt = min(dt_try * 1.25, dt_state, dt_max)
                    if implicit and not extras.get("implicit_ok"):
                        dt = min(float(dt_max), max(dt, float(dt_init)))
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
