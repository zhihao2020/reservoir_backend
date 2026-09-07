"""YAML geomech switch and Cartesian linear elasticity."""

import numpy as np
import pytest

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import moles_from_z
from reservoir_backend.comp.residual import coupled_residual
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.io.case import load_case
from reservoir_backend.physics.geomech import CartesianElasticity, GeomechSpec, geomech_from_cfg
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.fi_comp import solve_comp_step


def test_geomech_from_cfg_missing_is_off() -> None:
    spec = geomech_from_cfg({}, p_ref=5.0e7)
    assert spec.enabled is False
    assert spec.p_ref == pytest.approx(5.0e7)


def test_geomech_true_without_moduli_raises() -> None:
    with pytest.raises(ValueError, match="E_GPa"):
        geomech_from_cfg({"geomech": True})


def test_nocouperm_false_is_rejected() -> None:
    with pytest.raises(ValueError, match="nocouperm"):
        GeomechSpec(enabled=True, nocouperm=False, E=20.0e9, nu=0.22)


def test_uniform_pressure_unconstrained_matches_bulk_strain() -> None:
    grid = CartesianGrid.uniform((0.08, 0.08, 0.08), 0.02)
    spec = GeomechSpec(enabled=True, E=20.0e9, nu=0.22, biot=1.0, p_ref=5.0e7, boundary="unconstrained")
    el = CartesianElasticity(grid, spec)
    p = np.full(grid.n_cells, 5.1e7)
    theta = el.volumetric_strain(p)
    expected = spec.biot * 1.0e6 / spec.K_dr
    np.testing.assert_allclose(theta, expected, rtol=2.0e-4)
    k0 = np.full(grid.n_cells, 1.776e-17)
    rock = Rock(k0.copy(), np.full(grid.n_cells, 0.0367), biot=1.0, k_dry=0.0)
    _ = el.volumetric_strain(p + 1.0)
    np.testing.assert_array_equal(rock.permeability, k0)


def test_confined_one_cell_has_zero_strain() -> None:
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    spec = GeomechSpec(enabled=True, E=20.0e9, nu=0.22, biot=1.0, p_ref=0.0, boundary="confined")
    el = CartesianElasticity(grid, spec)
    theta = el.volumetric_strain(np.array([1.0e6]))
    np.testing.assert_allclose(theta, 0.0, atol=1.0e-18)


def test_vol_strain_pore_volume_uses_alpha_over_phi() -> None:
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    rock = Rock.uniform(1, k=1.0e-13, phi=0.0367)
    rock.cpor = 0.0
    rock.prpor = 5.0e7
    rock.biot = 1.0
    p = np.array([5.0e7])
    theta = np.array([1.0e-6])
    pv0 = rock.pore_volume(grid.cell_volumes(), p, vol_strain=np.zeros(1))
    pv = rock.pore_volume(grid.cell_volumes(), p, vol_strain=theta)
    np.testing.assert_allclose(pv[0] / pv0[0], 1.0 + rock.biot * theta[0] / 0.0367, rtol=1e-12)


def test_vol_strain_skips_scalar_biot_storage() -> None:
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    rock = Rock.uniform(1, k=1.0e-13, phi=0.20)
    rock.cpor = 1.2e-9
    rock.prpor = 5.0e7
    rock.biot = 1.0
    rock.k_dry = 11.9e9
    p = np.array([5.1e7])
    pv_scalar = rock.pore_volume(grid.cell_volumes(), p)
    pv_cpor = rock.pore_volume(grid.cell_volumes(), p, vol_strain=np.zeros(1))
    assert float(pv_scalar[0]) > float(pv_cpor[0])
    np.testing.assert_allclose(pv_cpor[0] / (0.20 * grid.cell_volumes()[0]), np.exp(1.2e-9 * 1.0e6), rtol=1e-12)


def test_disabled_geomech_residual_matches_cpor_only() -> None:
    grid = CartesianGrid.uniform((0.04, 0.04, 0.04), 0.02)
    rock = Rock.uniform(grid.n_cells, k=1.0e-13, phi=0.20)
    rock.cpor = 1.2e-9
    rock.prpor = 1.2e7
    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.full(grid.n_cells, 1.2e7)
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q = np.zeros_like(moles)
    r0, _ = coupled_residual(grid, rock, spec, moles, p, moles, 1.0, q, t_geom)
    r1, _ = coupled_residual(grid, rock, spec, moles, p, moles, 1.0, q, t_geom, vol_strain=None)
    np.testing.assert_allclose(r0, r1)


def test_one_cell_newton_with_elasticity_drops_residual() -> None:
    grid = CartesianGrid.uniform((0.02, 0.02, 0.02), 0.02)
    rock = Rock.uniform(1, k=1.0e-13, phi=0.20)
    gm = GeomechSpec(enabled=True, E=20.0e9, nu=0.22, biot=1.0, p_ref=1.2e7)
    el = CartesianElasticity(grid, gm)
    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.array([1.2e7])
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    p_bad = p * 1.08
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q = np.zeros_like(moles)
    theta0 = el.volumetric_strain(p_bad)
    res0, _ = coupled_residual(grid, rock, spec, moles, p_bad, moles, 1.0, q, t_geom, vol_strain=theta0)
    r0 = float(np.linalg.norm(res0))
    out = solve_comp_step(
        grid, rock, spec, [], {}, moles, p_bad, dt=1.0, t=0.0, max_newton=12, tol=1.0e-8, elasticity=el
    )
    assert out is not None
    theta1 = el.volumetric_strain(out.pressure)
    res1, _ = coupled_residual(grid, rock, spec, out.moles, out.pressure, moles, 1.0, q, t_geom, vol_strain=theta1)
    r1 = float(np.linalg.norm(res1))
    assert r1 / r0 < 1.0e-4
    np.testing.assert_array_equal(rock.permeability, np.full(1, 1.0e-13))


def test_yaml_youngs_modulus_sets_stiffness() -> None:
    soft = geomech_from_cfg({"geomech": {"enabled": True, "E_GPa": 5.0, "nu": 0.22}}, p_ref=0.0)
    stiff = geomech_from_cfg({"geomech": {"enabled": True, "E_GPa": 20.0, "nu": 0.22}}, p_ref=0.0)
    assert soft.E == pytest.approx(5.0e9)
    assert stiff.E == pytest.approx(20.0e9)
    grid = CartesianGrid.uniform((0.08, 0.08, 0.08), 0.02)
    p = np.full(grid.n_cells, 1.0e6)
    th_soft = CartesianElasticity(grid, soft).volumetric_strain(p)
    th_stiff = CartesianElasticity(grid, stiff).volumetric_strain(p)
    np.testing.assert_allclose(float(np.mean(th_soft)) / float(np.mean(th_stiff)), 4.0, rtol=2.0e-3)


def test_physical_3d_yaml_enables_geomech_product_stays_off() -> None:
    twin = load_case("examples/lab_v1/cmg_gem/physical_3d/case.yaml")
    gm = twin.physics.geomech
    assert gm.enabled
    assert gm.boundary == "unconstrained"
    assert gm.nocouperm
    assert gm.biot == pytest.approx(1.0)
    assert gm.E == pytest.approx(20.0e9)
    assert twin.physics.k_dry == pytest.approx(0.0)
    rock = twin.rock_from_theta(np.zeros(twin.parameterization.n_params))
    assert rock.k_dry == pytest.approx(0.0)
    assert rock.biot == pytest.approx(1.0)
    prod = load_case("examples/lab_v1/case.yaml")
    assert prod.physics.geomech.enabled is False
    assert prod.physics.geomech.E == pytest.approx(20.0e9)
    assert prod.physics.geomech.nu == pytest.approx(0.22)
    assert prod.uses_dpdp()
