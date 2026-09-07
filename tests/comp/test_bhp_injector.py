import numpy as np
import pytest

from reservoir_backend.comp.properties import flash_state
from reservoir_backend.comp.wells import well_molar_sources
from reservoir_backend.domain.types import ControlSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.io.case import load_case
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort


def test_pressure_injector_adds_z_inj_not_cell_oil() -> None:
    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    spec = twin.physics.fluid
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    rock = Rock.uniform(1, k=1.776e-17, phi=0.0367)
    port = FlowPort(
        "INJ",
        "injector",
        "pressure",
        np.array([0], dtype=np.int64),
        use_productivity=True,
        rw_m=0.003,
        geofac=0.34,
    )
    p = np.array([4.99e7])
    moles = np.broadcast_to(spec.z_init, (1, spec.nc)).copy()
    props = flash_state(spec, p, moles)
    controls = {("INJ", "pressure"): ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([5.0e7, 5.0e7]))}
    q, rates, _ = well_molar_sources(grid, rock, [port], controls, p, props, spec, 1.0)
    tot = float(np.sum(q[0]))
    assert tot > 0.0
    np.testing.assert_allclose(q[0] / tot, spec.z_inj, atol=1.0e-12)
    assert rates["INJ:q_inj"] > 0.0


def test_oil_block_injector_uses_cell_mobility_not_co2() -> None:
    from reservoir_backend.comp.wells import _injectate_xi_lam, _wi, perforation_pressures

    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    spec = twin.physics.fluid
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    rock = Rock.uniform(1, k=1.776e-17, phi=0.0367)
    port = FlowPort(
        "INJ", "injector", "pressure", np.array([0], dtype=np.int64),
        use_productivity=True, rw_m=0.003, geofac=0.198,
    )
    p = np.array([4.99e7])
    props = flash_state(spec, p, np.broadcast_to(spec.z_init, (1, spec.nc)).copy())
    controls = {("INJ", "pressure"): ControlSeries("INJ", "pressure", np.array([0.0, 10.0]), np.array([5.0e7, 5.0e7]))}
    q, _, _ = well_molar_sources(grid, rock, [port], controls, p, props, spec, 1.0)
    xi, lam_inj = _injectate_xi_lam(spec, 5.0e7)
    pw = perforation_pressures(grid, port, spec, props, 5.0e7)
    wi = _wi(grid, rock, port, 0)
    q_oil = wi * float(props.lam_l[0]) * float(pw[0] - p[0]) * xi
    q_gas = wi * lam_inj * float(pw[0] - p[0]) * xi
    tot = float(np.sum(q[0]))
    np.testing.assert_allclose(tot, q_oil, rtol=0.05)
    assert tot < 0.5 * q_gas


def test_physical_3d_peaceman_matches_gem_connection_pi() -> None:
    from reservoir_backend.comp.wells import _wi
    from reservoir_backend.ports.flow import peaceman_wi

    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    inj = next(p for p in twin.ports if p.name == "INJ")
    assert inj.geofac == pytest.approx(0.198, rel=1e-3)
    rock = twin.rock_from_theta(np.zeros(twin.parameterization.n_params))
    wi = _wi(twin.grid, rock, inj, int(inj.cell_ids[0]))
    # GEM INJ k=11: q=7.358e-8 m3/d, dp=126.78 Pa, oil λ≈915 → WI≈7.3e-18 m3.
    assert 6.5e-18 < wi < 9.0e-18
    wi14 = peaceman_wi(twin.grid, int(inj.cell_ids[0]), float(rock.permeability[0]), 0.003, geofac=0.0)
    np.testing.assert_allclose(wi, wi14, rtol=0.02)


def test_co2_injectate_uses_dense_viscosity() -> None:
    from reservoir_backend.comp.wells import _injectate_xi_lam

    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    spec = twin.physics.fluid
    _, lam = _injectate_xi_lam(spec, 5.0e7)
    # Ideal-gas mu_vapor=2e-5 would give lam=5e4. Dense CO2 LBC is ~0.06 cP.
    assert lam < 2.5e4
    assert lam > 5.0e2
