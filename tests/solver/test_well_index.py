import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort, geometric_wi, half_cell_wi, peaceman_wi
from reservoir_backend.solver.impes import simulate


def test_pressure_port_is_not_cell_dirichlet() -> None:
    grid = CartesianGrid.uniform((0.3, 0.1, 0.1), (0.05, 0.1, 0.1))
    rock = Rock.uniform(grid.n_cells, k=1.0e-12, phi=0.2)
    inj = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.025, 0.05, 0.05), use_productivity=True)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.275, 0.05, 0.05), use_productivity=True)
    p_inj, p_prod = 2.0e5, 1.0e5
    times = np.array([0.0, 2.0])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, p_inj)),
            ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
            ControlSeries("PROD", "pressure", times, np.full(2, p_prod)),
        ],
        State(pressure=np.full(grid.n_cells, 1.5e5), sw=np.full(grid.n_cells, 0.2)),
        t_end=2.0,
        single_phase=True,
        dt_init=0.5,
        dt_max=0.5,
    )
    p = traj.states[-1].pressure
    inj_cell = int(inj.cell_ids[0])
    prod_cell = int(prod.cell_ids[0])
    assert p[inj_cell] < p_inj
    assert p[prod_cell] > p_prod
    assert p[inj_cell] > p[prod_cell]
    q = traj.port_rates[-1]
    assert q["INJ"] > 0.0
    assert q["PROD"] < 0.0


def test_all_rate_closed_system_has_finite_pressure() -> None:
    grid = CartesianGrid.uniform((0.20, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20)
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=0.80)
    prod = FlowPort.at_point(grid, "PROD", "producer", "rate", (0.18, 0.04, 0.04))
    q = 8.0e-9
    t_end = 5.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [inj, prod],
        [
            ControlSeries("INJ", "rate", times, np.full(2, q)),
            ControlSeries("INJ", "composition", times, np.full(2, 0.80)),
            ControlSeries("PROD", "rate", times, np.full(2, -q)),
        ],
        State(pressure=np.full(grid.n_cells, 1.2e5), sw=np.full(grid.n_cells, 0.20)),
        t_end=t_end,
        dt_init=0.5,
        dt_max=1.0,
    )
    p = traj.states[-1].pressure
    assert np.all(np.isfinite(p))
    assert float(np.max(p)) > float(np.min(p))


def test_geometric_wi_positive() -> None:
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    wi = geometric_wi(grid, 0, 1.0e-12)
    assert wi > 0.0


def test_peaceman_wi_uses_deck_radius() -> None:
    grid = CartesianGrid.uniform((2.4, 1.6, 0.9), (0.6, 0.4, 0.45))
    k = 50.0 * 9.869233e-16
    wi = peaceman_wi(grid, 0, k, 0.20 * 0.3048, geofac=0.34)
    assert wi > 0.0
    wi_thin = peaceman_wi(grid, 0, k, 0.10 * 0.3048, geofac=0.34)
    assert wi > wi_thin
    wi_i = peaceman_wi(grid, 0, k, 0.10, geofac=0.37, axis="i")
    wi_k = peaceman_wi(grid, 0, k, 0.10, geofac=0.37, axis="k")
    # Horizontal I-well length is dx; vertical uses dz. dx > dz on this grid.
    assert wi_i > wi_k


def test_half_cell_wi_stronger_than_hole() -> None:
    grid = CartesianGrid.uniform((0.3, 0.3, 0.3), 0.1)
    hole = geometric_wi(grid, 0, 1.0e-12)
    face = half_cell_wi(grid, 0, 1.0e-12)
    assert face > hole
    assert face == half_cell_wi(grid, 0, 2.0e-12) / 2.0


def test_pressure_injector_raises_sw() -> None:
    """BHP water injector must flood. fw(Swi)≈0 is not the wellbore composition."""
    grid = CartesianGrid.uniform((0.30, 0.10, 0.10), (0.05, 0.10, 0.10))
    rock = Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20)
    relperm = CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3)
    inj = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.025, 0.05, 0.05), sw_inj=0.85)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.275, 0.05, 0.05))
    assert inj.use_productivity is False
    p_inj, p_prod = 2.4e5, 1.0e5
    t_end = 25.0
    times = np.array([0.0, t_end])
    sw0 = 0.20
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, p_inj)),
            ControlSeries("INJ", "composition", times, np.full(2, 0.85)),
            ControlSeries("PROD", "pressure", times, np.full(2, p_prod)),
        ],
        State(pressure=np.full(grid.n_cells, 1.6e5), sw=np.full(grid.n_cells, sw0)),
        t_end=t_end,
        dt_init=0.5,
        dt_max=2.0,
        max_cfl=0.45,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    assert float(np.mean(sw)) > sw0 + 0.04
    assert float(np.max(sw)) > sw0 + 0.15
    assert traj.port_rates[-1]["INJ"] > 0.0
    assert np.all(sw >= -1.0e-12) and np.all(sw <= 1.0 + 1.0e-12)


def test_pressure_producer_does_not_trap_water() -> None:
    """Dirichlet producer must extract arriving water, not pile up past Sw=1."""
    grid = CartesianGrid.uniform((0.24, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=3.0e-12, phi=0.20)
    relperm = CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3)
    inj = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.02, 0.04, 0.04), sw_inj=0.85)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.22, 0.04, 0.04))
    t_end = 80.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, 2.4e5)),
            ControlSeries("INJ", "composition", times, np.full(2, 0.85)),
            ControlSeries("PROD", "pressure", times, np.full(2, 1.0e5)),
        ],
        State(pressure=np.full(grid.n_cells, 1.6e5), sw=np.full(grid.n_cells, 0.20)),
        t_end=t_end,
        dt_init=1.0,
        dt_max=4.0,
        max_cfl=0.45,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    assert float(np.mean(sw)) > 0.24
    assert float(np.max(sw)) <= 1.0 + 1.0e-8
    assert traj.port_rates[-1]["PROD"] <= 0.0


def test_productivity_injector_raises_sw() -> None:
    grid = CartesianGrid.uniform((0.30, 0.10, 0.10), (0.05, 0.10, 0.10))
    rock = Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20)
    relperm = CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3)
    inj = FlowPort.at_point(
        grid, "INJ", "injector", "pressure", (0.025, 0.05, 0.05), sw_inj=0.85, use_productivity=True
    )
    prod = FlowPort.at_point(
        grid, "PROD", "producer", "pressure", (0.275, 0.05, 0.05), use_productivity=True
    )
    t_end = 20.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, 2.4e5)),
            ControlSeries("INJ", "composition", times, np.full(2, 0.85)),
            ControlSeries("PROD", "pressure", times, np.full(2, 1.0e5)),
        ],
        State(pressure=np.full(grid.n_cells, 1.6e5), sw=np.full(grid.n_cells, 0.20)),
        t_end=t_end,
        dt_init=0.5,
        dt_max=2.0,
        max_cfl=0.45,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    p = traj.states[-1].pressure
    assert float(np.mean(sw)) > 0.24
    assert p[int(inj.cell_ids[0])] < 2.4e5
    assert p[int(prod.cell_ids[0])] > 1.0e5
    assert np.all(sw >= -1.0e-12) and np.all(sw <= 1.0 + 1.0e-12)


def test_column_wi_floods_high_k_first() -> None:
    """A layered K with the same BHP on both layers must wet the high-K layer more."""
    nx, ny, nz = 8, 1, 2
    grid = CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, 0.04),
        dy=np.array([0.04]),
        dz=np.array([0.04, 0.04]),
    )
    k = np.array([1.0e-13 if grid.ijk(c)[2] == 0 else 1.0e-12 for c in range(grid.n_cells)])
    rock = Rock(k, np.full(grid.n_cells, 0.20))
    inj = FlowPort.column(grid, "INJ", "injector", "pressure", 0.02, 0.02, sw_inj=0.85, use_productivity=True)
    prod = FlowPort.column(grid, "PROD", "producer", "pressure", 0.30, 0.02, use_productivity=True)
    t_end = 40.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, 2.2e5)),
            ControlSeries("INJ", "composition", times, np.full(2, 0.85)),
            ControlSeries("PROD", "pressure", times, np.full(2, 1.0e5)),
        ],
        State(pressure=np.full(grid.n_cells, 1.5e5), sw=np.full(grid.n_cells, 0.20)),
        t_end=t_end,
        dt_init=0.5,
        dt_max=2.0,
        max_cfl=0.45,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    low = np.array([sw[c] for c in range(grid.n_cells) if grid.ijk(c)[2] == 0])
    high = np.array([sw[c] for c in range(grid.n_cells) if grid.ijk(c)[2] == 1])
    front = lambda s: int(np.max(np.where(s > 0.25)[0])) if np.any(s > 0.25) else 0
    assert front(high) >= front(low)
    assert float(high[min(front(high), high.size - 1)]) >= 0.25
    assert float(np.max(sw)) <= 1.0 + 1.0e-8


def test_wellbore_head_injects_more_into_deeper_layer() -> None:
    """IMEX *K well: BHP at the top connection, hydrostatic head down the hole."""
    nx, nz = 6, 4
    grid = CartesianGrid(
        nx=nx,
        ny=1,
        nz=nz,
        dx=np.full(nx, 0.20),
        dy=np.array([0.20]),
        dz=np.full(nz, 1.0),
    )
    rock = Rock.uniform(grid.n_cells, k=3.0e-12, phi=0.20)
    inj = FlowPort.column(
        grid, "INJ", "injector", "pressure", 0.10, 0.10, sw_inj=1.0, use_productivity=True, rw_m=0.02
    )
    prod = FlowPort.column(
        grid, "PROD", "producer", "pressure", 1.10, 0.10, use_productivity=True, rw_m=0.02
    )
    t_end = 12.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [inj, prod],
        [
            ControlSeries("INJ", "pressure", times, np.full(2, 2.4e5)),
            ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
            ControlSeries("PROD", "pressure", times, np.full(2, 1.2e5)),
        ],
        State(pressure=np.full(grid.n_cells, 1.6e5), sw=np.full(grid.n_cells, 0.25)),
        t_end=t_end,
        gravity=9.81,
        dt_init=0.5,
        dt_max=1.0,
        max_cfl=0.45,
        max_ds=0.12,
    )
    sw = traj.states[-1].sw
    bot = np.array([sw[c] for c in range(grid.n_cells) if grid.ijk(c)[2] == 0])
    top = np.array([sw[c] for c in range(grid.n_cells) if grid.ijk(c)[2] == nz - 1])
    assert float(bot[0]) > float(top[0]) + 0.008


def test_connection_bhp_increases_down_the_well() -> None:
    from reservoir_backend.solver.impes import _connection_bhp

    grid = CartesianGrid(
        nx=1, ny=1, nz=3, dx=np.array([1.0]), dy=np.array([1.0]), dz=np.full(3, 2.0)
    )
    cells = np.array([0, 1, 2], dtype=np.int64)
    p = _connection_bhp(grid, cells, 1.0e5, 1000.0, 9.81)
    z = grid.cell_centers()[:, 2]
    assert p[2] == 1.0e5
    assert abs(p[0] - p[2] - 1000.0 * 9.81 * (z[2] - z[0])) < 1.0e-6


def test_rate_producer_min_bhp_floor() -> None:
    """Aggressive rate + min_bhp switches to pressure floor (IMEX *MIN *BHP)."""
    from reservoir_backend.physics.pvt import BlackOilPVT

    grid = CartesianGrid.uniform((60.0, 40.0, 20.0), (20.0, 20.0, 10.0))
    rock = Rock.uniform(grid.n_cells, k=1.0e-15, phi=0.08)
    cell = int(grid.locate_cell(30.0, 20.0, 10.0))
    min_bhp = 1.5e5
    prod = FlowPort(
        name="PROD",
        role="producer",
        control="rate",
        cell_ids=np.array([cell], dtype=np.int64),
        use_productivity=True,
        rw_m=0.08,
        geofac=0.34,
        axis="k",
        min_bhp_Pa=min_bhp,
    )
    # Huge offtake so unconstrained BHP would crash below floor
    q = -5.0e-4
    t_end = 2.0
    times = np.array([0.0, t_end])
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3),
        [prod],
        [ControlSeries("PROD", "rate", times, np.full(2, q))],
        State(pressure=np.full(grid.n_cells, 3.0e5), sw=np.full(grid.n_cells, 0.2)),
        t_end=t_end,
        pvt=BlackOilPVT.slightly_compressible(1.0e-9),
        dt_init=0.25,
        dt_max=0.5,
        single_phase=True,
        mu_single=1.0e-3,
    )
    p_conn = float(traj.states[-1].pressure[cell])
    assert p_conn >= min_bhp - 5.0e3
    # Without floor the same setup would go far below; with floor rates should be finite
    assert np.isfinite(traj.port_rates[-1]["PROD"])
