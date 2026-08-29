"""Rate and BHP wells on a tiny compositional row."""

import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi_comp import initialize_state, simulate_comp


def test_rate_and_bhp_modes_conserve_moles() -> None:
    grid = CartesianGrid(
        nx=4, ny=1, nz=1, dx=np.full(4, 1.0), dy=np.array([1.0]), dz=np.array([1.0])
    )
    rock = Rock.uniform(grid.n_cells, k=5.0e-13, phi=0.20)
    spec = fluid_from_name("example", temperature_k=350.0, z_inj=np.array([0.95, 0.05]))
    inj = FlowPort("INJ", "injector", "rate", np.array([0], dtype=np.int64))
    prod = FlowPort("PROD", "producer", "pressure", np.array([3], dtype=np.int64))
    t_end = 20.0
    controls = [
        ControlSeries("INJ", "rate", np.array([0.0, t_end]), np.array([0.02, 0.02])),
        ControlSeries("PROD", "pressure", np.array([0.0, t_end]), np.array([1.1e7, 1.1e7])),
    ]
    st0 = initialize_state(grid, rock, spec, 1.2e7)
    traj = simulate_comp(
        grid, rock, spec, [inj, prod], controls, st0, t_end, dt_init=2.0, dt_max=10.0, max_steps=80
    )
    assert traj.reports
    mb = traj.reports[-1].mass
    assert mb.relative_balance_error < 1.0e-4
    assert mb.injected_mass > 0.0
    assert mb.produced_mass > 0.0
    assert np.all(np.isfinite(traj.states[-1].pressure))
    assert traj.port_bhp
    assert traj.port_bhp[-1]["INJ"] > traj.port_bhp[-1]["PROD"]
    last_q = traj.port_rates[-1]
    assert last_q["INJ:q_inj"] > 0.0
    assert last_q["PROD:q_oil"] + last_q["PROD:q_gas"] >= 0.0


def test_injector_bhp_falls_when_k_rises() -> None:
    grid = CartesianGrid(
        nx=4, ny=1, nz=1, dx=np.full(4, 1.0), dy=np.array([1.0]), dz=np.array([1.0])
    )
    spec = fluid_from_name("example", temperature_k=350.0, z_inj=np.array([0.95, 0.05]))
    inj = FlowPort("INJ", "injector", "rate", np.array([0], dtype=np.int64))
    prod = FlowPort("PROD", "producer", "pressure", np.array([3], dtype=np.int64))
    t_end = 8.0
    controls = [
        ControlSeries("INJ", "rate", np.array([0.0, t_end]), np.array([0.05, 0.05])),
        ControlSeries("PROD", "pressure", np.array([0.0, t_end]), np.array([1.1e7, 1.1e7])),
    ]
    st0_lo = initialize_state(grid, Rock.uniform(grid.n_cells, k=2.0e-13, phi=0.20), spec, 1.2e7)
    st0_hi = initialize_state(grid, Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20), spec, 1.2e7)
    lo = simulate_comp(grid, Rock.uniform(grid.n_cells, k=2.0e-13, phi=0.20), spec, [inj, prod], controls, st0_lo, t_end, dt_init=2.0, dt_max=8.0)
    hi = simulate_comp(grid, Rock.uniform(grid.n_cells, k=2.0e-12, phi=0.20), spec, [inj, prod], controls, st0_hi, t_end, dt_init=2.0, dt_max=8.0)
    assert lo.port_bhp[-1]["INJ"] > hi.port_bhp[-1]["INJ"] + 1.0e4


def test_rates_and_bhp_use_value_at_or_before_t() -> None:
    from reservoir_backend.solver.impes import Trajectory
    from reservoir_backend.domain.types import State

    times = np.array([0.0, 10.0, 20.0])
    states = [
        State(pressure=np.array([1.0]), sw=np.array([0.2]), time_s=float(t)) for t in times
    ]
    rates = [{"INJ": 0.0}, {"INJ": 1.0}, {"INJ": 0.0}]
    bhp = [{"INJ": 4.7e7}, {"INJ": 5.0e7}, {"INJ": 4.7e7}]
    traj = Trajectory(times_s=times, states=states, reports=[], port_rates=rates, port_bhp=bhp)
    r, p = traj.rates_and_bhp_at(10.0)
    assert p["INJ"] == 5.0e7
    r, p = traj.rates_and_bhp_at(10.5)
    assert p["INJ"] == 5.0e7
    r, p = traj.rates_and_bhp_at(9.9)
    assert p["INJ"] == 4.7e7


def test_closed_row_no_wells_keeps_moles() -> None:
    grid = CartesianGrid(
        nx=4, ny=1, nz=1, dx=np.full(4, 1.0), dy=np.array([1.0]), dz=np.array([1.0])
    )
    rock = Rock.uniform(grid.n_cells, k=1.0e-13, phi=0.20)
    spec = fluid_from_name("example")
    st0 = initialize_state(grid, rock, spec, 1.2e7)
    n0 = st0.moles.copy()
    traj = simulate_comp(grid, rock, spec, [], [], st0, 5.0, dt_init=1.0, dt_max=5.0, max_steps=20)
    np.testing.assert_allclose(traj.states[-1].moles, n0, rtol=1.0e-8)
    assert traj.reports[-1].mass.relative_balance_error < 1.0e-8
