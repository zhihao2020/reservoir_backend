"""One-cell / small-grid compositional Newton. ||R|| drop is the gate."""

import numpy as np
import pytest

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.comp.residual import coupled_residual, volume_residual
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.fi_comp import solve_comp_step


def _one_cell():
    grid = CartesianGrid(nx=1, ny=1, nz=1, dx=np.array([1.0]), dy=np.array([1.0]), dz=np.array([1.0]))
    rock = Rock.uniform(1, k=1.0e-13, phi=0.20)
    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.array([1.2e7])
    pv = rock.porosity * grid.cell_volumes()
    moles = moles_from_z(spec, p, spec.z_init, pv)
    return grid, rock, spec, moles, p


def test_init_moles_satisfy_volume() -> None:
    grid, rock, spec, moles, p = _one_cell()
    props = flash_state(spec, p, moles)
    pv = rock.porosity * grid.cell_volumes()
    vol = volume_residual(moles, props, pv, spec.n_hc)
    assert abs(float(vol[0])) / float(pv[0]) < 1.0e-8


def test_one_cell_volume_newton_drops_residual() -> None:
    """Closed cell: perturb p, Newton restores n_tot * v_mix = V_pore. ||R|| drop >= 4 decades."""
    grid, rock, spec, moles, p = _one_cell()
    p_bad = p * 1.08
    t_geom = geometric_transmissibility(grid, rock.permeability)
    q = np.zeros_like(moles)
    res0, _ = coupled_residual(grid, rock, spec, moles, p_bad, moles, 1.0, q, t_geom)
    r0 = float(np.linalg.norm(res0))
    out = solve_comp_step(grid, rock, spec, [], {}, moles, p_bad, dt=1.0, t=0.0, max_newton=12, tol=1.0e-8)
    assert out is not None
    res1, _ = coupled_residual(grid, rock, spec, out.moles, out.pressure, moles, 1.0, q, t_geom)
    r1 = float(np.linalg.norm(res1))
    assert r0 > 0.0
    assert r1 / r0 < 1.0e-4
    np.testing.assert_allclose(out.moles, moles, rtol=1.0e-6)


def test_flash_inner_not_rs_switch() -> None:
    grid, rock, spec, moles, p = _one_cell()
    props = flash_state(spec, p, moles)
    assert props.sv.shape == (1,)
    assert 0.0 <= float(props.sv[0]) <= 1.0
    assert abs(float(props.sl[0] + props.sv[0]) - 1.0) < 1.0e-12


def test_biot_storage_adds_alpha_squared_over_k_dry() -> None:
    grid, rock, spec, moles, p = _one_cell()
    rock.cpor = 1.2e-9
    rock.prpor = float(p[0])
    rock.biot = 1.0
    rock.k_dry = 11.9e9
    assert rock.storage_1_per_pa() == pytest.approx(1.2e-9 + 1.0 / 11.9e9)
    pv_cpor = np.exp(1.2e-9 * 1.0e6)
    pv_both = rock.pore_volume(grid.cell_volumes(), p + 1.0e6)[0] / rock.pore_volume(grid.cell_volumes(), p)[0]
    assert float(pv_both) > float(pv_cpor)
    np.testing.assert_allclose(pv_both, np.exp(rock.storage_1_per_pa() * 1.0e6), rtol=1e-12)


def test_cpor_increases_pore_volume_with_pressure() -> None:
    grid, rock, spec, moles, p = _one_cell()
    rock.cpor = 1.2e-9
    rock.prpor = float(p[0])
    pv0 = rock.pore_volume(grid.cell_volumes(), p)
    pv1 = rock.pore_volume(grid.cell_volumes(), p * 1.01)
    assert float(pv1[0]) > float(pv0[0])
    np.testing.assert_allclose(pv1[0] / pv0[0], np.exp(rock.cpor * 0.01 * p[0]), rtol=1e-12)
    q = np.zeros_like(moles)
    t_geom = geometric_transmissibility(grid, rock.permeability)
    res, _ = coupled_residual(grid, rock, spec, moles, p, moles, 1.0, q, t_geom)
    # At p = prpor the pore volume matches the uncompressed init moles.
    vol = res.reshape(1, spec.nc + 1)[0, spec.nc]
    assert abs(float(vol)) / float(pv0[0]) < 1.0e-8
