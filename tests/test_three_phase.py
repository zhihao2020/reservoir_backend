import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.relperm import CoreyThreePhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.impes import simulate


def test_three_phase_closure_and_bounds() -> None:
    rp = CoreyThreePhase()
    sw = np.array([0.25, 0.40, 0.55])
    sg = np.array([0.10, 0.15, 0.05])
    fw, fo, fg = rp.fractional_flow(sw, sg)
    assert np.allclose(fw + fo + fg, 1.0, atol=1.0e-12)
    krw, kro, krg = rp.kr(rp.swi, 0.2)
    assert float(np.asarray(krw)) == 0.0
    assert float(np.asarray(krg)) >= 0.0
    _krw0, kro0, _krg0 = rp.kr(0.3, rp.sgr)
    assert float(np.asarray(kro0)) >= 0.0


def test_three_phase_transport_conserves_and_closes() -> None:
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.2)
    rp = CoreyThreePhase()
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=0.7)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04))
    q = 8.0e-9
    t_end = 20.0
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([q, q])),
        ControlSeries("INJ", "composition", times, np.array([0.7, 0.7])),
        ControlSeries("INJ", "gas_composition", times, np.array([0.1, 0.1])),
        ControlSeries("PROD", "pressure", times, np.array([1.0e5, 1.0e5])),
    ]
    from reservoir_backend.physics.relperm import CoreyTwoPhase

    state0 = State(
        pressure=np.full(grid.n_cells, 1.1e5),
        sw=np.full(grid.n_cells, 0.22),
        sg=np.full(grid.n_cells, 0.06),
    )
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        [inj, prod],
        controls,
        state0,
        t_end,
        three_phase=rp,
        dt_init=1.0,
        dt_max=2.0,
        max_cfl=0.4,
    )
    last = traj.states[-1]
    so = last.so()
    assert last.sg is not None
    assert np.allclose(last.sw + so + last.sg, 1.0, atol=1.0e-8)
    assert np.all(last.sw >= -1.0e-12) and np.all(last.sg >= -1.0e-12) and np.all(so >= -1.0e-12)
    assert traj.reports[-1].mass.relative_balance_error < 0.08
    assert traj.reports[-1].mass.gas_relative_balance_error < 0.12
