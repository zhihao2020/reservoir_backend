import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.pvt import PSI, BlackOilPVT
from reservoir_backend.physics.relperm import CoreyTwoPhase, TableThreePhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.impes import simulate, surface_gas


def test_cmg_seawater_tables_cap_rs_above_pb() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    pb = 2500.0 * PSI
    rs_hi = float(pvt.rs(3000.0 * PSI))
    rs_pb = float(pvt.rs(pb))
    assert rs_hi <= rs_pb + 1.0e-12
    assert rs_pb > 100.0
    bo_unsat = 1.0 / float(pvt.b_o(3000.0 * PSI))
    bo_pb = 1.0 / float(pvt.b_o(pb))
    assert bo_unsat < bo_pb
    bg_hi = 1.0 / float(pvt.b_g(2500.0 * PSI))
    bg_lo = 1.0 / float(pvt.b_g(14.7 * PSI))
    assert bg_hi < bg_lo
    # IMEX Eg is scf/RB; b_g must be SI (Eg * scf/stb) so Rs/Eg stay consistent.
    eg_si = float(pvt.b_g(2514.7 * PSI))
    assert 120.0 < eg_si < 160.0
    rho_g = float(pvt.rho_g_sc) * eg_si
    assert 80.0 < rho_g < 220.0


def test_b_o_depends_on_rs_when_undersaturated() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    p = np.array([1800.0 * PSI, 1800.0 * PSI])
    rs_sat = pvt.rs(p)
    rs_lo = 0.4 * rs_sat
    b_sat = pvt.b_o(p, rs=rs_sat)
    b_lo = pvt.b_o(p, rs=rs_lo)
    assert float(b_lo[0]) > float(b_sat[0])
    b_old = pvt.b_o(p)
    assert np.allclose(b_old, b_sat, rtol=1.0e-8)
    p_hi = np.array([3000.0 * PSI])
    b_hi = 1.0 / float(np.asarray(pvt.b_o(p_hi, rs=pvt.rs(p_hi))).ravel()[0])
    b_pb = 1.0 / float(np.asarray(pvt.b_o(np.array([2500.0 * PSI]))).ravel()[0])
    assert b_hi < b_pb


def test_density_o_includes_dissolved_gas() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    p = np.array([2500.0 * PSI])
    rs = pvt.rs(p)
    bo = pvt.b_o(p, rs=rs)
    rho = pvt.density_o(p, rs=rs, bo=bo)
    assert float(rho[0]) > float(pvt.rho_o_sc * bo[0])


def test_flash_is_noop_above_bubble_point() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    sw = np.full(4, 0.20)
    sg = np.full(4, 0.10)
    p = np.full(4, 3000.0 * PSI)
    out = pvt.flash_sg(sw, sg, p, p)
    assert np.allclose(out, sg)


def test_flash_liberates_gas_when_pressure_drops() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    sw = np.array([0.20, 0.20])
    sg = np.array([0.00, 0.05])
    p_old = np.full(2, 2500.0 * PSI)
    p_new = np.full(2, 1500.0 * PSI)
    out = pvt.flash_sg(sw, sg, p_new, p_old)
    assert np.all(out >= sg - 1.0e-12)
    assert float(out[0]) > 0.0
    assert np.all(sw + out <= 1.0 + 1.0e-12)
    g = pvt.surface_gas_holdup(np.array([0.20]), np.array([0.0]), np.array([3000.0 * PSI]))
    sg_mid = float(np.asarray(pvt.flash_from_total(np.array([0.20]), g, np.array([1920.0 * PSI]))).ravel()[0])
    assert sg_mid > 0.06


def test_drs_dp_zero_above_bubble_point() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    pb = 2500.0 * PSI
    assert float(pvt.drs_dp(3000.0 * PSI)) == 0.0
    assert float(pvt.drs_dp(1500.0 * PSI)) > 0.0
    ct_hi = float(pvt.ct(0.20, 0.80, 0.0, p=3000.0 * PSI))
    ct_lo = float(pvt.ct(0.20, 0.80, 0.0, p=1500.0 * PSI))
    assert ct_lo > ct_hi * 5.0
    mu_hi = float(pvt.viscosity_o(3000.0 * PSI))
    mu_pb = float(pvt.viscosity_o(pb))
    assert mu_hi < mu_pb
    cg_lo = float(pvt.cg_of(1500.0 * PSI))
    cg_hi = float(pvt.cg_of(5000.0 * PSI))
    assert cg_lo > cg_hi
    ct_gas = float(pvt.ct(0.20, 0.50, 0.30, p=1500.0 * PSI))
    ct_nogas = float(pvt.ct(0.20, 0.80, 0.0, p=1500.0 * PSI))
    assert ct_gas > ct_nogas


def test_flash_from_total_identity_at_same_p() -> None:
    pvt = BlackOilPVT.cmg_seawater()
    sw = np.array([0.20, 0.30])
    sg = np.array([0.05, 0.00])
    p = np.full(2, 1800.0 * PSI)
    g = pvt.surface_gas_holdup(sw, sg, p)
    out = pvt.flash_from_total(sw, g, p)
    assert np.allclose(out, sg, atol=1.0e-10)


def test_live_oil_depressurization_liberates_and_conserves() -> None:
    """Closed liquid production below pb must raise Sg and keep surface gas."""
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.20)
    pvt = BlackOilPVT.cmg_seawater()
    rp = TableThreePhase.cmg_seawater(mu_w=pvt.mu_w, mu_o=pvt.mu_o, mu_g=pvt.mu_g)
    prod = FlowPort.at_point(grid, "PROD", "producer", "rate", (0.14, 0.04, 0.04))
    q = -5.0e-8
    t_end = 120.0
    times = np.array([0.0, t_end])
    p0 = 2500.0 * PSI
    state0 = State(
        pressure=np.full(grid.n_cells, p0),
        sw=np.full(grid.n_cells, 0.20),
        sg=np.zeros(grid.n_cells),
    )
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(mu_w=pvt.mu_w, mu_o=pvt.mu_o),
        [prod],
        [ControlSeries("PROD", "rate", times, np.full(2, q))],
        state0,
        t_end,
        pvt=pvt,
        three_phase=rp,
        dt_init=2.0,
        dt_max=4.0,
        max_cfl=0.40,
        max_ds=0.12,
    )
    last = traj.states[-1]
    assert last.sg is not None
    assert float(np.mean(last.pressure)) < 2400.0 * PSI
    assert float(np.mean(last.sg)) > 0.002
    mb = traj.reports[-1].mass
    assert mb.gas_relative_balance_error < 0.06
    g0 = surface_gas(grid, rock, state0.sg, pressure=state0.pressure, pvt=pvt, sw=state0.sw)
    g1 = surface_gas(grid, rock, last.sg, pressure=last.pressure, pvt=pvt, sw=last.sw)
    assert g0 > 0.0
    assert g1 < g0


def test_vo_status_switches_rs_and_sg() -> None:
    pvt = BlackOilPVT.cmg_seawater(p_init=3000.0 * 6894.757293168)
    p = np.full(3, 2000.0 * 6894.757293168)
    rs_sat = pvt.rs(p)
    bo = pvt.b_o(p)
    bg = pvt.b_g(p)
    sw = np.array([0.2, 0.2, 0.2])
    # undersaturated: x = Rs < RsSat → Sg=0
    unsat = np.array([True, False, True])
    x = np.array([0.5 * float(rs_sat[0]), 0.05, 1.5 * float(rs_sat[2])])
    sg, rs, uns = pvt.vo_decode(x, sw, rs_sat, bo, bg, unsat)
    assert sg[0] == 0.0
    assert rs[0] < float(rs_sat[0])
    assert sg[1] == 0.05
    assert abs(float(rs[1]) - float(rs_sat[1])) < 1.0e-12
    assert uns[2] is False or sg[2] > 0.0


def test_three_phase_mobility_includes_gas() -> None:
    from reservoir_backend.solver.impes import _cell_mobility

    pvt = BlackOilPVT.incompressible()
    rp = TableThreePhase.cmg_seawater()
    relperm = CoreyTwoPhase(mu_w=rp.mu_w, mu_o=rp.mu_o)
    sw = np.array([0.25])
    p = np.array([2.0e5])
    _lw0, _lo0, lg0, m0 = _cell_mobility(
        relperm, rp, sw, np.array([0.0]), pvt, p, single_phase=False, mu_single=1.0e-3
    )
    _lw1, _lo1, lg1, m1 = _cell_mobility(
        relperm, rp, sw, np.array([0.25]), pvt, p, single_phase=False, mu_single=1.0e-3
    )
    assert lg0 is not None and float(lg0[0]) == 0.0
    assert lg1 is not None and float(lg1[0]) > 0.0
    assert float(m1[0]) > float(m0[0])
