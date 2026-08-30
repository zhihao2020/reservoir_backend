import numpy as np

from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.physics.transfer import ComponentTransfer
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.dpdp_context import DPDPModelContext
from reservoir_backend.solver.frozen_pressure import FrozenPressureContext, step_frozen_pressure


def _setup():
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    dual = DualRock.from_cf(2, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    pf = np.array([1.15e7, 1.00e7])
    pm = np.array([1.20e7, 1.18e7])
    lam = np.array([1.0e-3, 1.0e-3])
    return grid, dual, ctx, tr, pf, pm, lam


def test_bhp_setpoint_reuses_factor() -> None:
    grid, dual, ctx, tr, pf, pm, lam = _setup()
    port = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.05, 0.05, 0.05))
    c1 = ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([1.50e7, 1.50e7]))
    c2 = ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([1.55e7, 1.55e7]))
    factor = FrozenPressureContext()
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, 1.0, ports=[port], controls=[c1], t_eval=1.0, factor=factor)
    n0 = factor.n_factor
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, 1.0, ports=[port], controls=[c2], t_eval=1.0, factor=factor)
    assert factor.n_factor == n0
    assert factor.n_reuse >= 1


def test_control_mode_rebuilds_factor() -> None:
    grid, dual, ctx, tr, pf, pm, lam = _setup()
    p_bhp = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.05, 0.05, 0.05))
    p_rate = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.05, 0.05, 0.05))
    cb = ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([1.50e7, 1.50e7]))
    cq = ControlSeries("INJ", "rate", np.array([0.0, 10.0]), np.array([1.0e-6, 1.0e-6]))
    factor = FrozenPressureContext()
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, 1.0, ports=[p_bhp], controls=[cb], t_eval=1.0, factor=factor)
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, 1.0, ports=[p_rate], controls=[cq], t_eval=1.0, factor=factor)
    assert factor.n_factor == 2


def test_lambda_change_rebuilds_factor() -> None:
    grid, dual, ctx, tr, pf, pm, lam = _setup()
    factor = FrozenPressureContext()
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam, lam, 1.0, factor=factor)
    lam2 = lam * 2.0
    step_frozen_pressure(grid, ctx, dual, tr, pf, pm, lam2, lam2, 1.0, factor=factor)
    assert factor.n_factor == 2
