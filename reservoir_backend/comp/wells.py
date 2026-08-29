"""Rate and BHP wells as molar sources. Rate wells report implied p_wf (injectivity)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.fluid import CompSpec
from reservoir_backend.comp.properties import PhaseProps
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort, half_cell_wi, peaceman_wi


_SC_P = 1.01325e5
_SC_T = 288.15
_VM_STD_GAS = 0.023645  # m3/mol, same as jiyang YAML STG conversion


def _surface_oil_gas(spec: CompSpec, n_dot: NDArray[np.float64]) -> tuple[float, float]:
    """Stock-tank oil and gas volumetric rates (m3/s) from molar wellstream."""
    n_dot = np.maximum(np.asarray(n_dot, dtype=float).ravel(), 0.0)
    tot = float(np.sum(n_dot))
    if tot <= 1.0e-18:
        return 0.0, 0.0
    z = n_dot / tot
    try:
        fl = flash_tp(spec.eos, _SC_P, _SC_T, z)
        q_oil = tot * (1.0 - float(fl.vapor_frac)) * max(float(fl.v_liq), 0.0)
        q_gas = tot * float(fl.vapor_frac) * max(float(fl.v_vap), 0.0)
        return float(q_oil), float(q_gas)
    except Exception:
        mw = np.asarray(spec.eos.mw, dtype=float).ravel()
        oil_i = int(np.argmax(mw)) if mw.size == n_dot.size else n_dot.size - 1
        rho = 730.0
        v_oil = float(mw[oil_i]) / rho if oil_i < mw.size else 2.0e-4
        q_oil = float(n_dot[oil_i]) * v_oil
        q_gas = float(np.sum(n_dot) - n_dot[oil_i]) * _VM_STD_GAS
        return q_oil, q_gas


def _wi(grid: CartesianGrid, rock: Rock, port: FlowPort, cell: int) -> float:
    k = float(rock.permeability[int(cell)])
    if port.use_productivity:
        return float(
            peaceman_wi(
                grid,
                int(cell),
                k,
                rw_m=port.rw_m,
                skin=port.skin,
                geofac=port.geofac,
                axis=getattr(port, "axis", "k"),
            )
        )
    return float(half_cell_wi(grid, int(cell), k)) * float(port.wi_multiplier)


def _implied_rate_bhp(
    spec: CompSpec,
    z: NDArray[np.float64],
    q_mol: float,
    p_cells: NDArray[np.float64],
    wi_lam: NDArray[np.float64],
    fw: float = 0.0,
) -> float:
    """p_wf such that Σ WI λ (p_wf − p) = Q · v. Rate is control; p_wf is data."""
    w = np.maximum(np.asarray(wi_lam, dtype=float).ravel(), 1.0e-30)
    p_cells = np.asarray(p_cells, dtype=float).ravel()
    p_guess = float(np.average(p_cells, weights=w))
    p_wf = p_guess
    fw = float(np.clip(fw, 0.0, 1.0))
    for _ in range(4):
        pw = max(p_wf, 1.0e4)
        q_vol = 0.0
        if fw < 1.0:
            fl = flash_tp(spec.eos, pw, float(spec.temperature_k), z)
            q_vol += float(q_mol) * (1.0 - fw) * max(fl.v_mix, 1.0e-12)
        if fw > 0.0:
            q_vol += float(q_mol) * fw * float(spec.water_vw(pw))
        p_wf = (q_vol + float(np.sum(w * p_cells))) / float(np.sum(w))
        p_wf = float(np.clip(p_wf, 1.0e4, 1.0e9))
    return p_wf


def well_molar_sources(
    grid: CartesianGrid,
    rock: Rock,
    ports: list[FlowPort],
    controls: dict[tuple[str, str], ControlSeries],
    pressure: NDArray[np.float64],
    props: PhaseProps,
    spec: CompSpec,
    t: float,
    *,
    need_bhp: bool = True,
) -> tuple[NDArray[np.float64], dict[str, float], dict[str, float]]:
    """q[c, i] moles/s into the cell. rates: net molar into domain. bhp: wellbore pressure.

    Implied injector p_wf is an observation, not a residual unknown. Skip it
    on Jacobian / line-search residuals (need_bhp=False).
    """
    n = grid.n_cells
    n_hc = spec.n_hc
    q = np.zeros((n, spec.nc))
    rates: dict[str, float] = {p.name: 0.0 for p in ports}
    bhp: dict[str, float] = {p.name: 0.0 for p in ports}
    p = np.asarray(pressure, dtype=float).ravel()
    z_inj = np.asarray(spec.z_inj, dtype=float).ravel()
    z_inj = z_inj / max(float(np.sum(z_inj)), 1.0e-18)

    def ctrl(port: FlowPort, kind: str, default: float | None = None) -> float:
        series = controls.get((port.name, kind))
        if series is None:
            if default is None:
                raise KeyError(f"missing control {(port.name, kind)}")
            return float(default)
        return float(series.value_at(t))

    for port in ports:
        cells = np.asarray(port.cell_ids, dtype=np.int64)
        wi = np.array([_wi(grid, rock, port, int(c)) for c in cells], dtype=float)
        lam_t = props.lam_l[cells] + props.lam_v[cells] + props.lam_w[cells]
        w = np.maximum(wi * lam_t, 1.0e-30)
        if port.control == "rate":
            q_tot = ctrl(port, "rate")
            z = z_inj
            series = controls.get((port.name, "composition"))
            if series is not None:
                z0 = float(np.clip(series.value_at(t), 0.0, 1.0))
                z = np.array([z0, 1.0 - z0], dtype=float) if n_hc == 2 else z_inj
            fw = float(np.clip(port.sw_inj, 0.0, 1.0)) if spec.has_water else 0.0
            share = w / float(np.sum(w))
            q_hc = q_tot * (1.0 - fw)
            q_w = q_tot * fw
            for c, s in zip(cells, share):
                q[int(c), :n_hc] += q_hc * s * z
                if spec.has_water:
                    q[int(c), n_hc] += q_w * s
            rates[port.name] = float(q_tot)
            n_dot = q_hc * z
            q_oil, q_gas = _surface_oil_gas(spec, n_dot)
            rates[port.name + ":q_oil"] = q_oil
            rates[port.name + ":q_gas"] = q_gas
            rates[port.name + ":q_inj"] = float(q_tot) * _VM_STD_GAS
            if need_bhp:
                bhp[port.name] = _implied_rate_bhp(spec, z, q_tot, p[cells], w, fw=fw)
            else:
                bhp[port.name] = float(np.average(p[cells], weights=w))
            continue
        p_wf = ctrl(port, "pressure")
        q_l = wi * props.lam_l[cells] * (p_wf - p[cells])
        q_v = wi * props.lam_v[cells] * (p_wf - p[cells])
        xi_l = props.xi_l[cells]
        xi_v = props.xi_v[cells]
        src = xi_l[:, None] * props.x[cells] * q_l[:, None] + xi_v[:, None] * props.y[cells] * q_v[:, None]
        water_src = 0.0
        for i, c in enumerate(cells):
            q[int(c), :n_hc] += src[i]
            if spec.has_water:
                q_wv = wi[i] * props.lam_w[int(c)] * (p_wf - p[int(c)])
                dw = props.xi_w[int(c)] * q_wv
                q[int(c), n_hc] += dw
                water_src += float(dw)
        n_dot = np.sum(src, axis=0)
        rates[port.name] = float(np.sum(src) + water_src)
        q_oil, q_gas = _surface_oil_gas(spec, n_dot)
        rates[port.name + ":q_oil"] = q_oil
        rates[port.name + ":q_gas"] = q_gas
        rates[port.name + ":q_inj"] = 0.0
        bhp[port.name] = float(p_wf)
    return q, rates, bhp

