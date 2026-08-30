import numpy as np
import pytest

from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.frozen_pressure import step_frozen_pressure


def test_frozen_step_reduces_matrix_fracture_gap() -> None:
    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    dual = DualRock.from_cf(1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.0e7])
    pm = np.array([1.2e7])
    lam = np.array([1.0e-3])
    pf2, pm2 = step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, dt=10.0)
    assert abs(float(pm2[0] - pf2[0])) < abs(float(pm[0] - pf[0]))
    assert np.all(np.isfinite(pf2)) and np.all(np.isfinite(pm2))


def test_frozen_step_finite_on_two_cells() -> None:
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.15e7, 1.00e7])
    pm = np.array([1.20e7, 1.18e7])
    lam = np.array([1.0e-3, 1.0e-3])
    pf2, pm2 = step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, dt=1.0)
    assert pf2.shape == (2,)
    assert np.all(np.isfinite(pf2))
    assert np.all(np.isfinite(pm2))
    _ = pytest


def test_frozen_bhp_well_raises_inlet_pressure() -> None:
    from reservoir_backend.domain.types import ControlSeries
    from reservoir_backend.ports.flow import FlowPort

    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.15e7, 1.00e7])
    pm = np.array([1.20e7, 1.18e7])
    lam = np.array([1.0e-3, 1.0e-3])
    ct = np.full(2, 2.0e-9)
    port = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.05, 0.05, 0.05))
    ctrl = ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([1.50e7, 1.50e7]))
    pf2, _pm2 = step_frozen_pressure(
        grid,
        ctx,
        dual,
        tr,
        pf,
        pm,
        lam,
        lam,
        dt=1.0,
        ct_fracture=ct,
        ct_matrix=ct,
        ports=[port],
        controls=[ctrl],
        t_eval=1.0,
    )
    assert float(pf2[0]) > float(pf[0])


def _fast_full_err(dt: float) -> float:
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import flash_compressibility, flash_state
    from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp

    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=spec.nc)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.25e7)
    _, full = simulate_dual_comp(
        grid, dual, spec, tr, [], [], state, t_end=dt, dt_init=dt, dt_max=dt, max_steps=8, context=ctx
    )
    pf = flash_state(spec, state.fracture.pressure, state.fracture.moles)
    pm = flash_state(spec, state.matrix.pressure, state.matrix.moles)
    pff, _pmm = step_frozen_pressure(
        grid,
        ctx,
        dual,
        tr,
        state.fracture.pressure,
        state.matrix.pressure,
        pf.lam_l + pf.lam_v + pf.lam_w,
        pm.lam_l + pm.lam_v + pm.lam_w,
        dt=dt,
        ct_fracture=flash_compressibility(spec, state.fracture.pressure, state.fracture.moles, pf),
        ct_matrix=flash_compressibility(spec, state.matrix.pressure, state.matrix.moles, pm),
    )
    pref = max(float(np.mean(np.abs(full.fracture.pressure))), 1.0)
    return abs(float(pff[0] - full.fracture.pressure[0])) / pref


def test_fast_vs_full_pressure_1s_5s_30s() -> None:
    e1 = _fast_full_err(1.0)
    e5 = _fast_full_err(5.0)
    e30 = _fast_full_err(30.0)
    assert e1 < 0.25
    assert e5 < 0.35
    assert e30 < 0.50


def test_fast_vs_full_pressure_rate_and_split_wells() -> None:
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import flash_compressibility, flash_state
    from reservoir_backend.domain.types import ControlSeries
    from reservoir_backend.ports.flow import FlowPort
    from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp

    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=spec.nc)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    state = initialize_dual_state(grid, dual, spec, 1.20e7, p_matrix=1.22e7)
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.05, 0.05, 0.05))
    inj.continuum_coupling = "split"
    inj.fracture_fraction = 0.7
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.15, 0.05, 0.05))
    times = np.array([0.0, 2.0])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([1.0e-6, 1.0e-6])),
        ControlSeries("PROD", "pressure", times, np.array([1.10e7, 1.10e7])),
    ]
    _, full = simulate_dual_comp(
        grid, dual, spec, tr, [inj, prod], controls, state, t_end=1.0, dt_init=1.0, dt_max=1.0, max_steps=8, context=ctx
    )
    pf = flash_state(spec, state.fracture.pressure, state.fracture.moles)
    pm = flash_state(spec, state.matrix.pressure, state.matrix.moles)
    pff, _pmm = step_frozen_pressure(
        grid,
        ctx,
        dual,
        tr,
        state.fracture.pressure,
        state.matrix.pressure,
        pf.lam_l + pf.lam_v + pf.lam_w,
        pm.lam_l + pm.lam_v + pm.lam_w,
        dt=1.0,
        ct_fracture=flash_compressibility(spec, state.fracture.pressure, state.fracture.moles, pf),
        ct_matrix=flash_compressibility(spec, state.matrix.pressure, state.matrix.moles, pm),
        ports=[inj, prod],
        controls=controls,
        t_eval=1.0,
        v_mix_fracture=pf.v_mix,
        v_mix_matrix=pm.v_mix,
    )
    pref = max(float(np.mean(np.abs(full.fracture.pressure))), 1.0)
    err = float(np.max(np.abs(pff - full.fracture.pressure))) / pref
    assert np.all(np.isfinite(pff))
    assert err < 0.50
