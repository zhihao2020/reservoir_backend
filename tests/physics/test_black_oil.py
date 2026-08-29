"""Black-oil F: surface-volume conservation and net-injection pressure rise."""

import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.pvt import BlackOilPVT, PSI
from reservoir_backend.physics.relperm import CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.impes import simulate, surface_water


def _box():
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.20)
    relperm = CoreyTwoPhase(mu_w=1.0e-3, mu_o=1.0e-3)
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=1.0)
    prod = FlowPort.at_point(grid, "PROD", "producer", "rate", (0.14, 0.04, 0.04))
    return grid, rock, relperm, inj, prod


def test_pvt_cmg_seawater_shrinkage_at_init() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    p = 3000.0 * PSI
    bw = float(np.asarray(pvt.b_w(p)))
    bo = float(np.asarray(pvt.b_o(p)))
    assert 0.90 < 1.0 / bw < 1.04
    assert 1.40 < 1.0 / bo < 1.52
    assert pvt.has_storage()
    assert not BlackOilPVT.incompressible().has_storage()


def test_surface_water_conserved_with_fvf() -> None:
    grid, rock, relperm, inj, prod = _box()
    pvt = BlackOilPVT(bw_ref=1.04, bo_ref=1.20, cw=0.0, co=0.0, cr=0.0, mu_w=1e-3, mu_o=1e-3)
    q_in, q_out = 8.0e-9, -5.0e-9
    t_end = 40.0
    times = np.array([0.0, t_end])
    p0 = 2.0e6
    state0 = State(pressure=np.full(grid.n_cells, p0), sw=np.full(grid.n_cells, 0.20))
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "rate", times, np.full(2, q_in)),
            ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
            ControlSeries("PROD", "rate", times, np.full(2, q_out)),
        ],
        state0,
        t_end,
        pvt=pvt,
        dt_init=1.0,
        dt_max=2.0,
        max_cfl=0.45,
    )
    mb = traj.reports[-1].mass
    assert mb.relative_balance_error < 5.0e-2
    last = traj.states[-1]
    assert abs(surface_water(grid, rock, last.sw, pressure=last.pressure, pvt=pvt) - mb.final_mass) < 1e-12
    assert np.all(last.sw >= -1e-12) and np.all(last.sw <= 1.0 + 1e-12)


def test_net_surface_injection_raises_pressure() -> None:
    """Unbalanced surface voidage must raise mean p. Datum pin would hide this."""
    grid, rock, relperm, inj, prod = _box()
    ct = 2.0e-9
    pvt = BlackOilPVT.slightly_compressible(ct, pref=2.0e6, mu_w=1e-3, mu_o=1e-3)
    q_in, q_out = 1.0e-8, -4.0e-9
    t_end = 80.0
    times = np.array([0.0, t_end])
    p0 = 2.0e6
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "rate", times, np.full(2, q_in)),
            ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
            ControlSeries("PROD", "rate", times, np.full(2, q_out)),
        ],
        State(pressure=np.full(grid.n_cells, p0), sw=np.full(grid.n_cells, 0.20)),
        t_end,
        pvt=pvt,
        dt_init=2.0,
        dt_max=4.0,
        max_cfl=0.45,
    )
    p_mean = float(np.mean(traj.states[-1].pressure))
    vp = float(np.sum(rock.porosity * grid.cell_volumes()))
    expected = (q_in + q_out) * t_end / max(vp * ct, 1.0e-30)
    assert p_mean > p0 + 0.4 * expected
    assert abs((p_mean - p0) - expected) / max(expected, 1.0) < 0.50
    assert np.all(np.isfinite(traj.states[-1].pressure))


def test_t_end_is_a_hard_cap() -> None:
    grid, rock, relperm, inj, prod = _box()
    t_end = 10.0
    times = np.array([0.0, 100.0])
    traj = simulate(
        grid,
        rock,
        relperm,
        [inj, prod],
        [
            ControlSeries("INJ", "rate", times, np.full(2, 5.0e-9)),
            ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
            ControlSeries("PROD", "rate", times, np.full(2, -3.0e-9)),
        ],
        State(pressure=np.full(grid.n_cells, 1.2e5), sw=np.full(grid.n_cells, 0.20)),
        t_end,
        report_times=np.array([5.0, 80.0, 100.0]),
        dt_init=1.0,
        dt_max=2.0,
    )
    assert float(traj.times_s[-1]) <= t_end + 1.0e-9
    assert 80.0 not in set(float(t) for t in traj.times_s)


def test_swt_table_kro_dies_before_corey() -> None:
    from reservoir_backend.physics.relperm import TableTwoPhase

    tab = TableTwoPhase.cmg_seawater()
    assert float(tab.kr(0.60)[1]) < 0.02
    assert float(tab.kr(0.20)[0]) == 0.0
    assert 0.0 < float(tab.fractional_flow(0.45)) < 1.0


def test_fault_transi_kills_x_face() -> None:
    from reservoir_backend.discretization.tpfa import geometric_transmissibility
    from reservoir_backend.grid.cartesian import CartesianGrid

    grid = CartesianGrid.uniform((0.9, 0.9, 0.4), 0.1)
    k = np.full(grid.n_cells, 1.0e-12)
    mx = np.ones((grid.nz, grid.ny, grid.nx - 1))
    mx[:, :, 4] = 0.0
    tx, _ty, _tz = geometric_transmissibility(grid, k, mult_x=mx)
    assert float(np.max(tx[:, :, 4])) == 0.0
    assert float(np.min(tx[:, :, 3])) > 0.0


def test_implicit_transport_runs_when_dt_below_thirty() -> None:
    from reservoir_backend.solver.impes import simulate as run

    grid, rock, relperm, inj, prod = _box()
    q_in, q_out = 6.0e-9, -3.0e-9
    t_end = 12.0
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.full(2, q_in)),
        ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
        ControlSeries("PROD", "rate", times, np.full(2, q_out)),
    ]
    state0 = State(pressure=np.full(grid.n_cells, 1.5e5), sw=np.full(grid.n_cells, 0.20))
    impl = run(
        grid, rock, relperm, [inj, prod], controls, state0, t_end,
        dt_init=4.0, dt_max=8.0, max_cfl=0.45, implicit=True,
    )
    assert any(getattr(r, "notes", None) is not None for r in impl.reports)
    assert impl.reports[-1].mass.relative_balance_error < 0.08
    assert float(np.mean(impl.states[-1].sw)) > 0.20


def test_implicit_transport_allows_large_dt() -> None:
    from reservoir_backend.solver.impes import simulate as run

    grid, rock, relperm, inj, prod = _box()
    q_in, q_out = 6.0e-9, -3.0e-9
    t_end = 40.0
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.full(2, q_in)),
        ControlSeries("INJ", "composition", times, np.full(2, 1.0)),
        ControlSeries("PROD", "rate", times, np.full(2, q_out)),
    ]
    state0 = State(pressure=np.full(grid.n_cells, 1.5e5), sw=np.full(grid.n_cells, 0.20))
    expl = run(
        grid, rock, relperm, [inj, prod], controls, state0, t_end,
        dt_init=1.0, dt_max=2.0, max_cfl=0.45, implicit=False,
    )
    impl = run(
        grid, rock, relperm, [inj, prod], controls, state0, t_end,
        dt_init=10.0, dt_max=20.0, max_cfl=0.45, implicit=True,
    )
    assert impl.reports[-1].mass.relative_balance_error < 0.08
    assert float(np.mean(impl.states[-1].sw)) > 0.20
    assert abs(float(np.mean(impl.states[-1].sw)) - float(np.mean(expl.states[-1].sw))) < 0.08
    assert any(r.notes == [] or "cfl" not in r.notes for r in impl.reports) or len(impl.reports) < len(expl.reports)


def test_cmg_seawater_b_converts_surface_rate() -> None:
    """Injecting 1 sm3 water occupies 1/bW reservoir m3, not 1 m3."""
    pvt = BlackOilPVT.cmg_seawater()
    p = 3000.0 * PSI
    bw = float(np.asarray(pvt.b_w(p)))
    bo = float(np.asarray(pvt.b_o(p)))
    q_w_s = 5000.0 * 0.158987 / 86400.0
    q_o_s = 2500.0 * 0.158987 / 86400.0
    net_res = q_w_s / bw - q_o_s / bo
    net_naive = q_w_s - q_o_s
    assert net_res < net_naive
    assert net_res > 0.0
