"""Immiscible water on the EXAMPLE compositional kernel. Not a GEM card."""

import numpy as np

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.comp.residual import coupled_residual, volume_residual
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.fi_comp import initialize_state, simulate_comp, solve_comp_step


def _spec_water(**kw):
    return fluid_from_name("example", has_water=True, sw_init=0.25, temperature_k=350.0, **kw)


def test_water_init_volume_and_sw() -> None:
    grid = CartesianGrid(nx=1, ny=1, nz=1, dx=np.array([1.0]), dy=np.array([1.0]), dz=np.array([1.0]))
    rock = Rock.uniform(1, k=1.0e-13, phi=0.20)
    spec = _spec_water()
    assert spec.nc == 3
    p = np.array([1.2e7])
    pv = rock.porosity * grid.cell_volumes()
    moles = moles_from_z(spec, p, spec.z_init, pv)
    props = flash_state(spec, p, moles)
    vol = volume_residual(moles, props, pv, spec.n_hc)
    assert abs(float(vol[0])) / float(pv[0]) < 2.0e-2
    assert 0.15 < float(props.sw[0]) < 0.35
    assert abs(float(props.sw[0] + props.sl[0] + props.sv[0]) - 1.0) < 1.0e-6


def test_one_cell_water_newton_closes() -> None:
    grid = CartesianGrid(nx=1, ny=1, nz=1, dx=np.array([1.0]), dy=np.array([1.0]), dz=np.array([1.0]))
    rock = Rock.uniform(1, k=1.0e-13, phi=0.20)
    spec = _spec_water()
    p = np.array([1.2e7])
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q = np.zeros_like(moles)
    res0, _ = coupled_residual(grid, rock, spec, moles, p * 1.06, moles, 1.0, q, t_geom)
    r0 = float(np.linalg.norm(res0))
    out = solve_comp_step(grid, rock, spec, [], {}, moles, p * 1.06, dt=1.0, t=0.0, max_newton=12, tol=1.0e-8)
    assert out is not None
    res1, _ = coupled_residual(grid, rock, spec, out.moles, out.pressure, moles, 1.0, q, t_geom)
    assert float(np.linalg.norm(res1)) / max(r0, 1.0e-18) < 1.0e-4


def test_water_injection_raises_sw_and_conserves() -> None:
    grid = CartesianGrid(nx=4, ny=1, nz=1, dx=np.full(4, 1.0), dy=np.array([1.0]), dz=np.array([1.0]))
    rock = Rock.uniform(grid.n_cells, k=5.0e-13, phi=0.20)
    spec = _spec_water()
    inj = FlowPort("INJ", "injector", "rate", np.array([0], dtype=np.int64), sw_inj=1.0)
    prod = FlowPort("PROD", "producer", "pressure", np.array([3], dtype=np.int64), sw_inj=0.0)
    t_end = 15.0
    controls = [
        ControlSeries("INJ", "rate", np.array([0.0, t_end]), np.array([0.05, 0.05])),
        ControlSeries("PROD", "pressure", np.array([0.0, t_end]), np.array([1.1e7, 1.1e7])),
    ]
    st0 = initialize_state(grid, rock, spec, 1.2e7)
    assert st0.moles.shape[1] == 3
    sw0 = float(np.mean(st0.sw))
    traj = simulate_comp(grid, rock, spec, [inj, prod], controls, st0, t_end, dt_init=2.0, dt_max=8.0, max_steps=80)
    assert traj.reports[-1].mass.relative_balance_error < 1.0e-4
    assert float(np.mean(traj.states[-1].sw)) > sw0
    assert float(traj.states[-1].sw[0]) >= float(traj.states[-1].sw[-1])


def test_water_twin_yaml_and_forward() -> None:
    from reservoir_backend.cli.main import main
    from reservoir_backend.io.case import load_case
    from reservoir_backend.physics.rock import Rock
    from reservoir_backend.synthetic import make_two_layer_compositional

    code = main(["validate", "examples/compositional/comp_example_water.yaml"])
    assert code == 0
    twin = load_case("examples/compositional/comp_example_water.yaml")
    assert twin.physics.fluid is not None
    assert twin.physics.fluid.has_water
    rock = Rock.uniform(twin.grid.n_cells, k=5.0e-13, phi=0.20)
    traj = twin.simulate(rock)
    assert traj.states[-1].moles.shape[1] == 3
    assert float(np.max(traj.states[-1].sw)) > float(np.min(traj.states[-1].sw)) - 1.0e-12

    case = make_two_layer_compositional(
        n=(4, 3, 1),
        size_m=(4.0, 3.0, 1.0),
        n_times=2,
        t_end=10.0,
        has_water=True,
        sw_init=0.25,
        sw_inj=1.0,
        seed=1,
    )
    assert case.twin.physics.fluid.has_water
    assert case.twin.initial_state().moles.shape[1] == 3
    assert any(s.kind == "saturation" for s in case.twin.experiment.sensors)


def test_lm_recovers_layer_with_immiscible_water() -> None:
    """Water is in F and H; θ is still 2-region log K. Not a GEM card."""
    from reservoir_backend.synthetic import evaluate_synthetic, make_two_layer_compositional

    case = make_two_layer_compositional(
        n=(4, 3, 1),
        size_m=(4.0, 3.0, 1.0),
        n_times=2,
        t_end=12.0,
        has_water=True,
        sw_init=0.25,
        sw_inj=1.0,
        seed=3,
        history_frac=0.85,
        q_inj=0.08,
    )
    assert case.twin.physics.fluid is not None and case.twin.physics.fluid.has_water
    assert any(o.kind == "saturation" for o in case.twin.experiment.observations)
    assert any(o.kind == "bhp" for o in case.twin.experiment.observations)
    post = case.twin.calibrate()
    metrics = evaluate_synthetic(case, post)
    assert metrics["posterior_data_nrmse"] < metrics["prior_data_nrmse"], metrics
    assert metrics["posterior_logk_rmse"] < metrics["prior_logk_rmse"], metrics
    assert metrics["contrast_post"] > 1.5, metrics
    assert post.history.reports[-1].mass.relative_balance_error < 0.08
    sw = np.asarray(post.history.states[-1].sw, dtype=float)
    inj = case.twin.ports[0].cell_ids
    assert np.all(np.isfinite(sw))
    assert np.all((sw >= 0.0) & (sw <= 1.0))
    # Molar rate is small vs pore volume; Sw barely moves. Injector cells must
    # not dry out, and mean Sw stays at or above the initial connate-ish fill.
    assert float(np.mean(sw[inj])) >= 0.25 - 1.0e-4
    assert float(np.mean(sw)) >= 0.25 - 1.0e-3
