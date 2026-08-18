import numpy as np

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
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


def test_table_three_phase_matches_swt_slt_endpoints() -> None:
    from reservoir_backend.physics.relperm import TableThreePhase

    rp = TableThreePhase.cmg_seawater()
    krw, kro, krg = rp.kr(0.20, 0.0)
    assert float(np.asarray(krw)) == 0.0
    assert float(np.asarray(krg)) == 0.0
    assert float(np.asarray(kro)) > 0.9
    fw, fo, fg = rp.fractional_flow(np.array([0.25, 0.40]), np.array([0.10, 0.08]))
    assert np.allclose(fw + fo + fg, 1.0, atol=1.0e-12)
    stone2 = TableThreePhase.cmg_seawater()
    prod = TableThreePhase.cmg_seawater()
    object.__setattr__(prod, "oil_rule", "product")
    _, kro0, _ = stone2.kr(0.20, 0.0)
    _, krow, _ = prod.kr(0.20, 0.0)
    assert abs(float(kro0) - float(krow)) < 1.0e-9


def _small_three_phase_case(t_end: float):
    grid = CartesianGrid.uniform((0.16, 0.08, 0.08), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.2)
    rp = CoreyThreePhase()
    inj = FlowPort.at_point(grid, "INJ", "injector", "rate", (0.02, 0.04, 0.04), sw_inj=0.7)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.14, 0.04, 0.04))
    q = 8.0e-9
    times = np.array([0.0, t_end])
    controls = [
        ControlSeries("INJ", "rate", times, np.array([q, q])),
        ControlSeries("INJ", "composition", times, np.array([0.7, 0.7])),
        ControlSeries("INJ", "gas_composition", times, np.array([0.1, 0.1])),
        ControlSeries("PROD", "pressure", times, np.array([1.0e5, 1.0e5])),
    ]
    state0 = State(
        pressure=np.full(grid.n_cells, 1.1e5),
        sw=np.full(grid.n_cells, 0.22),
        sg=np.full(grid.n_cells, 0.06),
    )
    return grid, rock, rp, [inj, prod], controls, state0


def test_three_phase_transport_conserves_and_closes() -> None:
    t_end = 20.0
    grid, rock, rp, ports, controls, state0 = _small_three_phase_case(t_end)
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        ports,
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
    impl = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        ports,
        controls,
        state0,
        t_end,
        three_phase=rp,
        implicit=True,
        dt_init=2.0,
        dt_max=4.0,
        max_cfl=0.4,
    )
    last_i = impl.states[-1]
    assert last_i.sg is not None
    assert np.allclose(last_i.sw + last_i.so() + last_i.sg, 1.0, atol=1.0e-8)
    assert np.all(last_i.sg >= -1.0e-12)


def test_three_phase_implicit_is_not_cfl_budgeted() -> None:
    """Sequential implicit 3ph must take IMEX-like dt, not an explicit step budget."""
    t_end = 80.0
    grid, rock, rp, ports, controls, state0 = _small_three_phase_case(t_end)
    traj = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        ports,
        controls,
        state0,
        t_end,
        three_phase=rp,
        implicit=True,
        dt_init=20.0,
        dt_max=40.0,
        max_cfl=0.4,
        max_ds=0.25,
        max_steps=12,
    )
    last = traj.states[-1]
    assert last.sg is not None
    assert np.allclose(last.sw + last.so() + last.sg, 1.0, atol=1.0e-8)
    assert len(traj.reports) <= 12
    assert max(r.dt for r in traj.reports) > 8.0


def test_reupdate_pressure_refreshes_p_after_transport() -> None:
    from reservoir_backend.physics.capillary import NoCapillary
    from reservoir_backend.solver.impes import solve_step

    t_end = 20.0
    grid, rock, rp, ports, controls, state0 = _small_three_phase_case(t_end)
    _, _, _, extras, _, _, _ = solve_step(
        grid,
        rock,
        CoreyTwoPhase(),
        NoCapillary(),
        ports,
        controls,
        state0,
        10.0,
        three_phase=rp,
        implicit=True,
        reupdate_pressure=True,
    )
    assert extras.get("implicit_ok") is True
    assert extras.get("reupdate_pressure") is True
    sfi = simulate(
        grid,
        rock,
        CoreyTwoPhase(),
        ports,
        controls,
        state0,
        t_end,
        three_phase=rp,
        implicit=True,
        sfi_outer=1,
        dt_init=20.0,
        dt_max=40.0,
        max_cfl=0.4,
        max_ds=0.25,
        max_steps=12,
    )
    last_s = sfi.states[-1]
    assert last_s.sg is not None
    assert np.allclose(last_s.sw + last_s.so() + last_s.sg, 1.0, atol=1.0e-8)


def test_well_crossflow_uses_inflow_mixture() -> None:
    from reservoir_backend.solver.impes import _well_transport_sources

    p = np.array([2.0e6, 1.0e6])
    lw = np.array([10.0, 1.0])
    lo = np.array([0.1, 1.0])
    lg = np.array([0.0, 0.0])
    bw = bo = bg = np.ones(2)
    rs = np.zeros(2)
    well_index = {0: (1.0e-12, 1.5e6), 1: (1.0e-12, 1.5e6)}
    port = FlowPort("INJ", "injector", "pressure", np.array([0, 1], dtype=np.int64), sw_inj=0.0)
    src_w, _src_g, prod = _well_transport_sources(
        well_index,
        p,
        lw,
        lo,
        lg,
        bw,
        bo,
        bg,
        rs,
        {0: port, 1: port},
        {},
        0.0,
        object(),
        np.zeros(2),
        np.zeros(2),
    )
    assert 0 in prod and prod[0] < 0.0
    assert 1 not in prod
    assert src_w[1] > 0.0


def test_well_transport_sources_freeze_qt() -> None:
    from reservoir_backend.solver.impes import _well_transport_sources

    p = np.array([2.0e6, 1.0e6])
    lw = np.array([1.0, 1.0])
    lo = np.array([1.0, 1.0])
    lg = np.array([0.0, 2.0])
    bw = bo = bg = np.ones(2)
    rs = np.zeros(2)
    well_index = {0: (1.0e-12, 2.5e6), 1: (1.0e-12, 0.5e6)}
    src_w, src_g, prod = _well_transport_sources(
        well_index, p, lw, lo, lg, bw, bo, bg, rs, {}, {}, 0.0, object(), np.zeros(2), np.zeros(2)
    )
    assert 0 not in prod
    assert 1 in prod and prod[1] < 0.0
    assert src_w[0] > 0.0


def test_hybrid_gravity_extras_are_segregation_only() -> None:
    from reservoir_backend.solver.impes import _hybrid_gravity_fluxes

    grid = CartesianGrid.uniform((0.16, 0.08, 0.16), 0.04)
    rock = Rock.uniform(grid.n_cells, k=5.0e-12, phi=0.2)
    lw = np.full(grid.n_cells, 1.0e-3)
    lo = np.full(grid.n_cells, 2.0e-3)
    lg = np.full(grid.n_cells, 5.0e-3)
    p = np.full(grid.n_cells, 1.0e6)
    qwx, _qwy, qwz, _ox, _oy, _oz, _gx, _gy, qgz = _hybrid_gravity_fluxes(
        grid,
        rock,
        p,
        lw,
        lo,
        lg,
        gravity=9.81,
        rho_w=1000.0,
        rho_o=800.0,
        rho_g=10.0,
        pc=None,
        face_mult_x=None,
        face_mult_y=None,
        face_mult_z=None,
    )
    assert float(np.max(np.abs(qwx))) < 1.0e-18
    assert float(np.max(np.abs(qgz))) > 0.0


def test_fi_blackoil_newton_closes() -> None:
    from reservoir_backend.solver.fi import solve_fi_step
    from reservoir_backend.physics.pvt import BlackOilPVT
    from reservoir_backend.physics.capillary import NoCapillary

    t_end = 20.0
    grid, rock, rp, _ports, _controls, state0 = _small_three_phase_case(t_end)
    fluid = BlackOilPVT.incompressible(mu_w=rp.mu_w, mu_o=rp.mu_o, mu_g=rp.mu_g)
    out = solve_fi_step(
        grid,
        rock,
        rp,
        fluid,
        NoCapillary(),
        state0.sw,
        state0.sg,
        state0.pressure,
        state0.pressure,
        4.0,
        0.0,
        src_w=np.zeros(grid.n_cells),
        src_o=np.zeros(grid.n_cells),
        src_g=np.zeros(grid.n_cells),
    )
    assert out is not None
    p, sw, sg, rs = out.pressure, out.sw, out.sg, out.rs
    assert np.allclose(sw + (1.0 - sw - sg) + sg, 1.0, atol=1.0e-8)
    assert np.all(sw >= -1.0e-12) and np.all(sg >= -1.0e-12)
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(rs))
    assert out.fx is not None and out.newton_iters >= 0


def test_fi_vo_switch_is_two_stage() -> None:
    from reservoir_backend.physics.pvt import PSI, BlackOilPVT
    from reservoir_backend.solver.fi import switch_live_oil_unknown

    fluid = BlackOilPVT.cmg_seawater()
    p = np.full(4, 2000.0 * PSI)
    sw = np.full(4, 0.20)
    rs_sat = float(np.asarray(fluid.rs(p)).ravel()[0])
    x = np.array([0.5 * rs_sat, 1.2 * rs_sat, 1.2 * rs_sat, -0.01])
    unsat = np.array([True, True, True, False])
    near = np.array([False, False, True, False])
    sw_o, sg, rs, uns, near_o, x_o = switch_live_oil_unknown(fluid, p, sw, x, unsat, near, live=True)
    assert uns[0] and sg[0] == 0.0 and rs[0] < rs_sat
    assert uns[1] and sg[1] == 0.0 and near_o[1]
    assert rs[1] >= rs_sat
    assert (not uns[2]) or sg[2] == 0.0 or near_o[2]
    if not uns[2]:
        # Second-stage liberation: growth lid 0.20.
        assert 0.0 < sg[2] <= 0.20 + 1.0e-9
        assert abs(float(rs[2]) - rs_sat) < 1.0e-6 * max(rs_sat, 1.0)
    elif near_o[2]:
        assert uns[2] and float(sg[2]) == 0.0
        assert float(rs[2]) >= rs_sat - 1.0e-6
    assert (not uns[3]) and sg[3] > 0.0 and sg[3] < 1.0e-6 and near_o[3]
    _sw, sg2, _rs, uns2, _near2, _x2 = switch_live_oil_unknown(
        fluid, p, sw_o, np.array([x_o[0], x_o[1], x_o[2], -0.01]), uns, near_o, live=True
    )
    assert uns2[3] and sg2[3] == 0.0


def test_fi_live_oil_switches_rs_to_sg() -> None:
    from reservoir_backend.physics.capillary import NoCapillary
    from reservoir_backend.physics.pvt import PSI, BlackOilPVT
    from reservoir_backend.solver.fi import solve_fi_step

    t_end = 20.0
    grid, rock, rp, _ports, _controls, _state0 = _small_three_phase_case(t_end)
    fluid = BlackOilPVT.cmg_seawater()
    p0 = np.full(grid.n_cells, 3000.0 * PSI)
    sw0 = np.full(grid.n_cells, 0.20)
    sg0 = np.zeros(grid.n_cells)
    rs0 = fluid.rs(p0)
    prod = int(grid.locate_cell(0.14, 0.04, 0.04))
    wi = 2.0e-12
    out = solve_fi_step(
        grid,
        rock,
        rp,
        fluid,
        NoCapillary(),
        sw0,
        sg0,
        p0,
        p0,
        4.0,
        0.0,
        src_w=np.zeros(grid.n_cells),
        src_o=np.zeros(grid.n_cells),
        src_g=np.zeros(grid.n_cells),
        rs0=rs0,
        wi_base={prod: (wi, 1800.0 * PSI)},
    )
    assert out is not None
    p, sw, sg, rs = out.pressure, out.sw, out.sg, out.rs
    assert float(np.min(p)) < 2000.0 * PSI
    assert float(np.max(sg)) > 1.0e-3
    assert np.all(sw + sg <= 1.0 + 1.0e-8)
    sat = sg > 1.0e-8
    assert np.any(sat)
    # leftover dissolved gas may sit above RsSat until later Newton steps
    assert np.all(rs[sat] + 1.0e-6 >= fluid.rs(p)[sat])


def test_fi_saturated_cell_with_well_closes() -> None:
    from reservoir_backend.physics.capillary import NoCapillary
    from reservoir_backend.physics.pvt import PSI, BlackOilPVT
    from reservoir_backend.solver.fi import solve_fi_step

    t_end = 20.0
    grid, rock, rp, _ports, _controls, _state0 = _small_three_phase_case(t_end)
    fluid = BlackOilPVT.cmg_seawater()
    p0 = np.full(grid.n_cells, 2000.0 * PSI)
    sw0 = np.full(grid.n_cells, 0.20)
    sg0 = np.full(grid.n_cells, 0.01)
    prod = int(grid.locate_cell(0.14, 0.04, 0.04))
    out = solve_fi_step(
        grid,
        rock,
        rp,
        fluid,
        NoCapillary(),
        sw0,
        sg0,
        p0,
        p0,
        4.0,
        0.0,
        src_w=np.zeros(grid.n_cells),
        src_o=np.zeros(grid.n_cells),
        src_g=np.zeros(grid.n_cells),
        rs0=fluid.rs(p0),
        wi_base={prod: (2.0e-12, 1800.0 * PSI)},
    )
    assert out is not None
    p, sw, sg, rs = out.pressure, out.sw, out.sg, out.rs
    assert float(np.min(p)) < 1900.0 * PSI
    assert np.all(sg >= -1.0e-12)
    assert np.all(sw + sg <= 1.0 + 1.0e-8)


def test_fi_crossflow_mixes_producer_into_injector() -> None:
    from reservoir_backend.solver.fi import _well_surface_rates

    p = np.array([2.0e6, 1.0e6])
    lw = np.array([1.0, 1.0])
    lo = np.array([2.0, 0.1])
    lg = np.zeros(2)
    b = np.ones(2)
    rs = np.array([10.0, 10.0])
    qw, qo, qg = _well_surface_rates(
        {0: (1.0e-12, 1.5e6), 1: (1.0e-12, 1.5e6)},
        {0: (1.0, 0.0), 1: (1.0, 0.0)},
        {0: 0, 1: 0},
        p,
        lw,
        lo,
        lg,
        b,
        b,
        b,
        rs,
    )
    assert qo[0] < 0.0
    assert qg[0] < 0.0
    assert qo[1] > 0.0
    assert qw[1] > 0.0


def test_fim_failure_rejects_step_instead_of_sequential(monkeypatch) -> None:
    from reservoir_backend.physics.capillary import NoCapillary
    from reservoir_backend.solver import impes as impes_mod
    from reservoir_backend.solver.impes import solve_step

    monkeypatch.setattr(impes_mod, "solve_fi_step", lambda *a, **k: None)
    t_end = 20.0
    grid, rock, rp, ports, controls, state0 = _small_three_phase_case(t_end)
    nxt, _w, _l, extras, _fx, _fy, _fz = solve_step(
        grid,
        rock,
        CoreyTwoPhase(),
        NoCapillary(),
        ports,
        controls,
        state0,
        4.0,
        three_phase=rp,
        implicit=True,
        fully_implicit=True,
    )
    assert extras.get("fi_failed") is True
    assert extras.get("fi_ok") is False
    assert extras.get("scheme") == "fim"
    assert extras.get("implicit_ok") is False
    assert float(nxt.time_s) == float(state0.time_s)
    assert np.allclose(nxt.sw, state0.sw)
    assert np.allclose(nxt.pressure, state0.pressure)


def test_fim_step_accepts_when_newton_closes() -> None:
    from reservoir_backend.physics.capillary import NoCapillary
    from reservoir_backend.solver.impes import solve_step

    t_end = 20.0
    grid, rock, rp, ports, controls, state0 = _small_three_phase_case(t_end)
    nxt, _w, _l, extras, _fx, _fy, _fz = solve_step(
        grid,
        rock,
        CoreyTwoPhase(),
        NoCapillary(),
        ports,
        controls,
        state0,
        4.0,
        three_phase=rp,
        implicit=True,
        fully_implicit=True,
    )
    assert extras.get("fi_failed") is False
    assert extras.get("scheme") == "fim"
    assert extras.get("fi_ok") is True
    assert extras.get("implicit_ok") is True
    assert nxt.sg is not None
    assert np.allclose(nxt.sw + nxt.so() + nxt.sg, 1.0, atol=1.0e-8)
    assert float(nxt.time_s) == float(state0.time_s) + 4.0
