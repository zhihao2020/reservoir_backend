import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.impes import simulate


def _welge_shock(relperm: CoreyTwoPhase, n: int = 400) -> tuple[float, float]:
    s = np.linspace(relperm.swi + 1.0e-4, 1.0 - relperm.sor, n)
    fw = relperm.fractional_flow(s)
    # tangent from (swi, 0): maximize fw / (s - swi)
    slope = fw / np.maximum(s - relperm.swi, 1.0e-12)
    i = int(np.argmax(slope))
    return float(s[i]), float(fw[i])


def test_buckley_leverett_front() -> None:
    nx = 60
    dx = 0.005
    grid = CartesianGrid(nx=nx, ny=1, nz=1, dx=np.full(nx, dx), dy=np.array([0.01]), dz=np.array([0.01]))
    phi = 0.20
    rock = Rock.uniform(grid.n_cells, k=1.0e-11, phi=phi)
    relperm = CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3, nw=2.0, no=2.0)
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.5 * dx, 0.005, 0.005), sw_inj=1.0 - relperm.sor)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", ((nx - 0.5) * dx, 0.005, 0.005))
    q = 2.0e-8
    t_end = 80.0
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([q, q])),
        ControlSeries("INJ", "composition", times, np.array([1.0 - relperm.sor, 1.0 - relperm.sor])),
        ControlSeries("PROD", "pressure", times, np.array([1.0e5, 1.0e5])),
    ]
    state0 = State(
        pressure=np.full(grid.n_cells, 1.05e5),
        sw=np.full(grid.n_cells, relperm.swi),
    )
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        controls,
        state0,
        t_end,
        capillary=NoCapillary(),
        dt_init=0.5,
        dt_max=2.0,
        max_cfl=0.4,
        max_ds=0.08,
    )
    sw = traj.states[-1].sw
    s_star, fw_star = _welge_shock(relperm)
    area = float(grid.dy[0] * grid.dz[0])
    x_shock = (q * t_end / (phi * area)) * (fw_star / (s_star - relperm.swi))
    x = grid.cell_centers()[:, 0]
    # numerical front: first cell from the right still near connate
    mid = 0.5 * (s_star + relperm.swi)
    crossed = x[sw >= mid]
    x_num = float(crossed.max()) if crossed.size else 0.0
    assert abs(x_num - x_shock) / max(x_shock, dx) < 0.25
    assert 0.0 <= float(np.min(sw)) and float(np.max(sw)) <= 1.0
    assert float(sw[0]) > float(sw[-1])
