"""Single-porosity compositional Jacobian: local-flash FD + analytic TPFA."""

import numpy as np
import pytest

from reservoir_backend.comp.fluid import fluid_from_name
from reservoir_backend.comp.properties import flash_state, moles_from_z
from reservoir_backend.comp.residual import coupled_residual
from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.dpdp_blocks import assemble_single_jacobian
from reservoir_backend.solver.fi_comp import solve_comp_step
from reservoir_backend.solver.linear import _pressure_dofs, solve_newton_system


def _tiny():
    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), 0.1)
    rock = Rock.uniform(grid.n_cells, k=1.0e-13, phi=0.20)
    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.full(grid.n_cells, 1.2e7)
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    return grid, rock, spec, moles, p


def test_parallel_thermo_fd_matches_serial() -> None:
    import os

    from reservoir_backend.solver.dpdp_blocks import cell_thermo_fd

    grid = CartesianGrid.uniform((0.4, 0.4, 0.2), 0.1)
    rock = Rock.uniform(grid.n_cells, k=1.0e-13, phi=0.20)
    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.full(grid.n_cells, 1.2e7)
    moles = moles_from_z(spec, p, spec.z_init, rock.porosity * grid.cell_volumes())
    props = flash_state(spec, p, moles)
    n_scale = max(float(np.mean(np.sum(moles, axis=1))), 1.0)
    prev = os.environ.get("RESERVOIR_JAC_THREADS")
    try:
        os.environ["RESERVOIR_JAC_THREADS"] = "1"
        a, _ = cell_thermo_fd(spec, p, moles, props, n_scale, 1.2e7)
        os.environ["RESERVOIR_JAC_THREADS"] = "8"
        b, _ = cell_thermo_fd(spec, p, moles, props, n_scale, 1.2e7)
    finally:
        if prev is None:
            os.environ.pop("RESERVOIR_JAC_THREADS", None)
        else:
            os.environ["RESERVOIR_JAC_THREADS"] = prev
    np.testing.assert_allclose(a.dv_mix, b.dv_mix, rtol=1.0e-10, atol=1.0e-16)
    np.testing.assert_allclose(a.dlam_l, b.dlam_l, rtol=1.0e-10, atol=1.0e-16)


@pytest.mark.parametrize("pressure_offset", [0.0, 2.0e5, -2.0e5])
def test_single_jacobian_matches_column_fd(pressure_offset) -> None:
    grid, rock, spec, moles, p = _tiny()
    p = p.copy()
    p[0] += pressure_offset
    dt = 1.0
    t_geom = geometric_transmissibility(grid, rock.permeability, kz=rock.kz)
    props = flash_state(spec, p, moles)
    q = np.zeros_like(moles)
    res0, _ = coupled_residual(grid, rock, spec, moles, p, moles, dt, q, t_geom, props=props)
    n_scale = max(float(np.mean(np.sum(moles, axis=1))), 1.0)
    p_scale = max(float(np.mean(np.abs(p))), 1.0e5)
    jac, _ = assemble_single_jacobian(grid, spec, moles, p, props, dt, t_geom, n_scale, p_scale)
    js = np.asarray(jac.todense())
    n_cells, nc = moles.shape
    nu = nc + 1
    jd = np.zeros_like(js)
    for col in range(n_cells * nu):
        c = col // nu
        slot = col % nu
        eps = 1.0e-8 * n_scale if slot < nc else 1.0e-8 * p_scale
        n2 = moles.copy()
        p2 = p.copy()
        if slot < nc:
            n2[c, slot] = n2[c, slot] + eps
        else:
            p2[c] = p2[c] + eps
        r2, _ = coupled_residual(grid, rock, spec, n2, p2, moles, dt, q, t_geom)
        jd[:, col] = (r2 - res0) / eps
    scale = np.maximum(np.max(np.abs(jd), axis=0), 1.0)
    err = np.max(np.abs(js - jd) / scale[None, :])
    # Physical-unit column FD; 0.5% allows local flash FD/upwind truncation.
    print(f"column FD offset={pressure_offset:g} Pa error={err:.9g} tolerance=0.005")
    assert err < 5.0e-3


def test_residual_ok_accepts_twenty_x_drop() -> None:
    from reservoir_backend.solver.fi_comp import (
        _clip_comp_step,
        _residual_floor,
        _residual_ok,
        _residual_tight,
    )

    assert _residual_ok(0.0036, 0.232, 1.0e-6)
    assert not _residual_tight(0.0036, 0.232, 1.0e-6)
    assert not _residual_ok(0.23, 0.46, 1.0e-6)
    assert _residual_ok(1.0e-9, 1.0, 1.0e-6)
    assert _residual_tight(1.0e-9, 1.0, 1.0e-6)
    assert _residual_floor(1.236e-8, 1.318e-8)
    assert _residual_ok(1.236e-8, 1.318e-8, 1.0e-6)
    moles = np.ones((2, 2))
    du = np.zeros(2 * 3)
    du[0] = 5.0
    du[2] = 2.0e7
    w = _clip_comp_step(du, moles, 2, 2).reshape(2, 3)
    assert abs(w[0, 2]) <= 5.0e6
    assert abs(w[0, 0]) <= 0.25 * 2.0


def test_single_newton_uses_compiled_sparse_solve() -> None:
    grid, rock, spec, moles, p = _tiny()
    p_bad = p * 1.05
    out = solve_comp_step(grid, rock, spec, [], {}, moles, p_bad, dt=1.0, t=0.0, max_newton=12, tol=1.0e-8)
    assert out is not None
    np.testing.assert_allclose(out.moles, moles, rtol=1.0e-5)


def test_pressure_dofs_single_continuum() -> None:
    nc = 2
    n_cells = 4
    nu = nc + 1
    n = n_cells * nu
    pd = _pressure_dofs(n, nc, continua=1)
    assert pd.tolist() == [2, 5, 8, 11]


def test_direct_solver_on_single_jacobian() -> None:
    grid, rock, spec, moles, p = _tiny()
    dt = 1.0
    t_geom = geometric_transmissibility(grid, rock.permeability, kz=rock.kz)
    props = flash_state(spec, p, moles)
    n_scale = max(float(np.mean(np.sum(moles, axis=1))), 1.0)
    jac, _ = assemble_single_jacobian(grid, spec, moles, p, props, dt, t_geom, n_scale, 1.2e7)
    n = jac.shape[0]
    rhs = np.ones(n)
    lin = solve_newton_system(jac + 1.0e-6 * sparse_eye(n), rhs, n_comp=spec.nc, continua=1, backend="direct")
    assert lin.x.size == n
    assert np.all(np.isfinite(lin.x))
    assert lin.method == "spsolve"


def sparse_eye(n: int):
    from scipy import sparse

    return sparse.eye(n, format="csc")
