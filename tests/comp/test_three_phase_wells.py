"""Aqueous + HC Peaceman wells: rate vs BHP controls. EXAMPLE fluid only."""

import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.comp.residual import coupled_residual
from reservoir_backend.comp.wells import well_molar_sources
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi_comp import initialize_state, solve_comp_step


def _spec(**kw):
    return fluid_from_name(
        "example",
        has_water=True,
        sw_init=0.25,
        temperature_k=350.0,
        z_inj=np.array([0.95, 0.05]),
        **kw,
    )


def _row():
    grid = CartesianGrid(
        nx=3, ny=1, nz=1, dx=np.full(3, 1.0), dy=np.array([1.0]), dz=np.array([1.0])
    )
    rock = Rock.uniform(grid.n_cells, k=5.0e-13, phi=0.20)
    return grid, rock, _spec()


def test_three_phase_saturations_sum_to_one() -> None:
    grid, rock, spec = _row()
    st = initialize_state(grid, rock, spec, 1.2e7)
    props = flash_state(spec, st.pressure, st.moles)
    s = props.sl + props.sv + props.sw
    np.testing.assert_allclose(s, 1.0, atol=1.0e-8)
    assert np.all((props.sw >= 0.0) & (props.sw <= 1.0))


def test_rate_peaceman_injector_honors_q_and_splits_water() -> None:
    grid, rock, spec = _row()
    port = FlowPort(
        "INJ",
        "injector",
        "rate",
        np.array([0], dtype=np.int64),
        sw_inj=0.40,
        use_productivity=True,
        rw_m=0.10,
        geofac=0.198,
    )
    p = np.full(grid.n_cells, 1.2e7)
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    props = flash_state(spec, p, moles)
    q_spec = 0.05
    controls = {("INJ", "rate"): ControlSeries("INJ", "rate", np.array([0.0, 10.0]), np.array([q_spec, q_spec]))}
    q, rates, bhp = well_molar_sources(grid, rock, [port], controls, p, props, spec, 1.0)
    assert rates["INJ"] == q_spec
    np.testing.assert_allclose(float(np.sum(q[0])), q_spec, rtol=1.0e-12)
    fw = 0.40
    assert abs(float(q[0, spec.n_hc]) - q_spec * fw) < 1.0e-12
    np.testing.assert_allclose(q[0, : spec.n_hc] / (q_spec * (1.0 - fw)), spec.z_inj, atol=1.0e-12)
    assert bhp["INJ"] > float(p[0])
    assert rates["INJ:q_water"] > 0.0


def test_bhp_peaceman_injector_matches_rate_implied_bhp() -> None:
    """Rate-implied p_wf reused as Dirichlet BHP: molar sources stay consistent."""
    grid, rock, spec = _row()
    cells = np.array([0], dtype=np.int64)
    rate_port = FlowPort(
        "INJ", "injector", "rate", cells, sw_inj=0.35, use_productivity=True, rw_m=0.10, geofac=0.198
    )
    p = np.full(grid.n_cells, 1.2e7)
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    props = flash_state(spec, p, moles)
    q_spec = 0.04
    rate_ctrl = {("INJ", "rate"): ControlSeries("INJ", "rate", np.array([0.0, 10.0]), np.array([q_spec, q_spec]))}
    q_rate, rates_r, bhp_r = well_molar_sources(grid, rock, [rate_port], rate_ctrl, p, props, spec, 1.0)
    p_wf = float(bhp_r["INJ"])
    bhp_port = FlowPort(
        "INJ", "injector", "pressure", cells, sw_inj=0.35, use_productivity=True, rw_m=0.10, geofac=0.198
    )
    bhp_ctrl = {("INJ", "pressure"): ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([p_wf, p_wf]))}
    q_bhp, rates_b, bhp_b = well_molar_sources(grid, rock, [bhp_port], bhp_ctrl, p, props, spec, 1.0)
    assert bhp_b["INJ"] == p_wf
    np.testing.assert_allclose(q_bhp[0], q_rate[0], rtol=0.08, atol=1.0e-8)
    assert rates_b["INJ:q_water"] > 0.0
    assert abs(rates_b["INJ"] - rates_r["INJ"]) / max(q_spec, 1.0e-12) < 0.08


def test_bhp_producer_is_dirichlet_and_residual_drops() -> None:
    grid, rock, spec = _row()
    prod = FlowPort(
        "PROD",
        "producer",
        "pressure",
        np.array([2], dtype=np.int64),
        use_productivity=True,
        rw_m=0.10,
        geofac=0.198,
    )
    p_wf = 1.1e7
    controls = {("PROD", "pressure"): ControlSeries("PROD", "pressure", np.array([0.0, 5.0]), np.array([p_wf, p_wf]))}
    st = initialize_state(grid, rock, spec, 1.2e7)
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q0, _, bhp0 = well_molar_sources(grid, rock, [prod], controls, st.pressure, flash_state(spec, st.pressure, st.moles), spec, 1.0)
    assert bhp0["PROD"] == p_wf
    assert float(np.sum(q0[2])) < 0.0
    res0, _ = coupled_residual(grid, rock, spec, st.moles, st.pressure, st.moles, 1.0, q0, t_geom)
    r0 = float(np.linalg.norm(res0))
    out = solve_comp_step(grid, rock, spec, [prod], controls, st.moles, st.pressure, dt=1.0, t=0.0, max_newton=16, tol=1.0e-8)
    assert out is not None
    assert out.port_bhp["PROD"] == p_wf
    q1, _, _ = well_molar_sources(
        grid, rock, [prod], controls, out.pressure, flash_state(spec, out.pressure, out.moles), spec, 1.0
    )
    res1, props = coupled_residual(grid, rock, spec, out.moles, out.pressure, st.moles, 1.0, q1, t_geom)
    r1 = float(np.linalg.norm(res1))
    assert r0 > 0.0
    assert r1 / r0 < 1.0e-2 or r1 < 1.0e-6
    np.testing.assert_allclose(props.sl + props.sv + props.sw, 1.0, atol=1.0e-8)


def test_rate_injector_newton_residual_drops() -> None:
    grid, rock, spec = _row()
    inj = FlowPort(
        "INJ",
        "injector",
        "rate",
        np.array([0], dtype=np.int64),
        sw_inj=1.0,
        use_productivity=True,
        rw_m=0.10,
        geofac=0.198,
    )
    q_spec = 0.03
    controls = {("INJ", "rate"): ControlSeries("INJ", "rate", np.array([0.0, 5.0]), np.array([q_spec, q_spec]))}
    st = initialize_state(grid, rock, spec, 1.2e7)
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q0, rates0, _ = well_molar_sources(
        grid, rock, [inj], controls, st.pressure, flash_state(spec, st.pressure, st.moles), spec, 1.0
    )
    assert rates0["INJ"] == q_spec
    res0, _ = coupled_residual(grid, rock, spec, st.moles, st.pressure, st.moles, 1.0, q0, t_geom)
    r0 = float(np.linalg.norm(res0))
    out = solve_comp_step(grid, rock, spec, [inj], controls, st.moles, st.pressure, dt=1.0, t=0.0, max_newton=16, tol=1.0e-8)
    assert out is not None
    assert abs(out.port_rates["INJ"] - q_spec) < 1.0e-12
    assert out.port_bhp["INJ"] > 0.0
    res1, props = coupled_residual(grid, rock, spec, out.moles, out.pressure, st.moles, 1.0, out.q_src, t_geom)
    r1 = float(np.linalg.norm(res1))
    assert r0 > 0.0
    assert r1 / r0 < 5.0e-2 or r1 < 1.0e-6
    assert float(np.mean(props.sw)) >= 0.25 - 1.0e-3
    np.testing.assert_allclose(props.sl + props.sv + props.sw, 1.0, atol=1.0e-8)
