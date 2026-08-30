import numpy as np

from reservoir_backend.domain.types import State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.impes import simulate


def test_1d_dirichlet_linear_pressure() -> None:
    nx = 20
    grid = CartesianGrid(nx=nx, ny=1, nz=1, dx=np.full(nx, 0.01), dy=np.array([0.01]), dz=np.array([0.01]))
    rock = Rock.uniform(grid.n_cells, k=1.0e-12, phi=0.2)
    p_l, p_r = 2.0e5, 1.0e5
    state0 = State(pressure=np.full(grid.n_cells, p_r), sw=np.full(grid.n_cells, 0.2))
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        ports=[],
        controls=[],
        state0=state0,
        t_end=1.0,
        face_dirichlet={"left": p_l, "right": p_r},
        single_phase=True,
        dt_init=1.0,
        dt_max=1.0,
    )
    p = traj.states[-1].pressure
    x = grid.cell_centers()[:, 0]
    L = grid.size_m()[0]
    analytical = p_l + (p_r - p_l) * (x / L)
    # cell-center exact for linear solution on uniform grid with face Dirichlet
    assert np.max(np.abs(p - analytical)) / (p_l - p_r) < 0.05


def test_control_observation_split_rejected() -> None:
    from reservoir_backend.exceptions import InvalidControl
    from reservoir_backend.ports.flow import FlowPort, validate_port_controls

    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    port = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.05, 0.15, 0.15))
    try:
        validate_port_controls([port], {"INJ": {"rate", "pressure"}})
        raise AssertionError("expected InvalidControl")
    except InvalidControl:
        pass
