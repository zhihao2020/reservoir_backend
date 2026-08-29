import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.impes import simulate, water_mass


def test_closed_injection_mass_balance() -> None:
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.2)
    relperm = CoreyTwoPhase()
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=0.8)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04))
    q = 1.0e-8
    t_end = 30.0
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([q, q])),
        ControlSeries("INJ", "composition", times, np.array([0.8, 0.8])),
        ControlSeries("PROD", "pressure", times, np.array([1.0e5, 1.0e5])),
    ]
    state0 = State(pressure=np.full(grid.n_cells, 1.1e5), sw=np.full(grid.n_cells, 0.2))
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        controls,
        state0,
        t_end,
        capillary=NoCapillary(),
        dt_init=1.0,
        dt_max=3.0,
        max_cfl=0.45,
    )
    mb = traj.reports[-1].mass
    assert mb.relative_balance_error < 5.0e-2
    assert water_mass(grid, rock, traj.states[-1].sw) == mb.final_mass
    assert np.all(traj.states[-1].sw >= 0.0) and np.all(traj.states[-1].sw <= 1.0)
