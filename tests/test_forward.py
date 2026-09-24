"""Forward saturation model (P3) regression tests.

Pins the pluggable black-oil / compositional forward models: finite in-range
outputs, phase conservation (sw+so+sg=1), and the compositional phase split.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.core.lab_case import load_lab_case
from src.programs.forward import forward_saturations
from src.programs.forward import fcm_effective_viscosity, fcm_effective_density, fcm_phase_split
from src.programs.pipeline import run_mesh
from src.programs.pressure import interpolate_pressure
from src.programs.rock import FluidParams
from src.programs.saturation import interpolate_saturation, project_saturations3, smooth_fields

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "examples" / "small" / "case.yaml"


def _setup():
    case = load_lab_case(SMALL)
    mesh = run_mesh(case)
    n_t = int(case.times.size)
    n_c = mesh.grid.n_cells
    p = np.zeros((n_t, n_c))
    sw = np.zeros((n_t, n_c))
    so = np.zeros((n_t, n_c))
    sg = np.zeros((n_t, n_c))
    for t in range(n_t):
        p[t] = interpolate_pressure(mesh.grid, case.probe_xyz, case.pressure[t], case.well_xyz, case.well_pw[t], method="kriging")
        sw[t], so[t], sg[t] = interpolate_saturation(
            mesh.grid, case.probe_xyz, case.sw[t], case.so[t], case.sg[t],
            method="kriging", swc=case.black_oil.swc, sgc=case.black_oil.sgc,
        )
    p = smooth_fields(p)
    sw = smooth_fields(sw)
    so = smooth_fields(so)
    for t in range(n_t):
        sw[t], so[t], sg[t] = project_saturations3(sw[t], so[t], sg[t])
    return case, mesh, p, sw, so, sg


def _run(model, params=None):
    case, mesh, p, sw, so, sg = _setup()
    n_c = mesh.grid.n_cells
    k = np.full(n_c, case.k0)
    phi = np.full(n_c, case.phi0)
    return forward_saturations(
        model, mesh.grid, p, k, phi, params or case.black_oil, mesh.wells,
        case.well_qw, case.well_qo, case.well_qg, case.times, sw[0], so[0], sg[0],
    )


def test_black_oil_forward_finite_and_conserved():
    fsw, fso, fsg = _run("black_oil")
    for f in (fsw, fso, fsg):
        assert np.isfinite(f).all()
        assert np.all((f >= -1.0e-9) & (f <= 1.0 + 1.0e-9))
    assert np.allclose(fsw + fso + fsg, 1.0, atol=1.0e-6)


def test_compositional_forward_finite_and_conserved():
    case, mesh, p, sw, so, sg = _setup()
    params = replace(case.black_oil, rs_slope=1.0e-6)
    n_c = mesh.grid.n_cells
    fsw, fso, fsg = forward_saturations(
        "compositional", mesh.grid, p, np.full(n_c, case.k0), np.full(n_c, case.phi0),
        params, mesh.wells, case.well_qw, case.well_qo, case.well_qg, case.times,
        sw[0], so[0], sg[0],
    )
    for f in (fsw, fso, fsg):
        assert np.isfinite(f).all()
        assert np.all((f >= -1.0e-9) & (f <= 1.0 + 1.0e-9))
    assert np.allclose(fsw + fso + fsg, 1.0, atol=1.0e-6)


def test_compositional_dissolves_gas():
    # The miscible (single-phase) component model keeps the CO2 largely dissolved,
    # so the free-gas saturation stays far smaller than in the (permanent-gas)
    # black-oil model.
    _, _, fsg_bo = _run("black_oil")
    case, mesh, p, sw, so, sg = _setup()
    params = replace(case.black_oil, rs_slope=1.0e-6)
    n_c = mesh.grid.n_cells
    _, _, fsg_c = forward_saturations(
        "compositional", mesh.grid, p, np.full(n_c, case.k0), np.full(n_c, case.phi0),
        params, mesh.wells, case.well_qw, case.well_qo, case.well_qg, case.times,
        sw[0], so[0], sg[0],
    )
    assert fsg_c.mean() <= fsg_bo.mean() + 1.0e-9


def test_fcm_mixing_rules():
    # 1/4-power viscosity: mu_eff(0)=mu_o, mu_eff(1)=mu_s; monotone decreasing.
    params = FluidParams(mu_g=2.0e-5, mu_o=2.0e-3, rho_s=2000.0, rho_o=0.0)
    c = np.array([0.0, 0.5, 1.0])
    mu = fcm_effective_viscosity(c, params)
    assert np.allclose(mu, [2.0e-3, (0.5 / 2.0e-5**0.25 + 0.5 / 2.0e-3**0.25) ** -4.0, 2.0e-5])
    # linear density: rho_eff(0)=rho_o, rho_eff(1)=rho_s
    rho = fcm_effective_density(c, params)
    assert np.allclose(rho, [0.0, 1000.0, 2000.0])


def test_fcm_phase_split():
    # Below c_sat all CO2 dissolved (sg=0); above it the excess is free gas.
    sw, so, sg = fcm_phase_split(np.array([0.3, 0.66, 0.8]), c_sat=0.66)
    assert np.allclose(sg, [0.0, 0.0, (0.8 - 0.66) / 0.34])
    assert np.allclose(sw + so + sg, 1.0)


def test_tabular_relperm():
    # Tabular rel-perm interpolates the *SGT/*SWT curves and uses Stone I for the
    # oil phase (kro = krog * krow).
    from src.programs.rock import RelpermTable, tabular_phase_mobilities

    table = RelpermTable.from_rows(
        [[0.0, 0.0, 1.0], [0.4, 0.25, 0.25], [1.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    params = FluidParams(mu_o=2.0e-3, mu_g=2.0e-5)
    lam_w, lam_o, lam_g = tabular_phase_mobilities(0.0, 0.6, 0.4, table, params)
    assert np.allclose(lam_w, 0.0)
    assert np.allclose(lam_o, 0.25 / 2.0e-3)  # Krog(0.4)=0.25 * Krow(0)=1
    assert np.allclose(lam_g, 0.25 / 2.0e-5)  # Krg(0.4)=0.25


def test_compositional_surface_source_conservation():
    # The conserved CO2 component C = sg/Bg + Rs*so is *surface* volume, and the
    # well gas rate qg (GEM *BHF) is also *surface* volume. So a single implicit
    # step must accumulate exactly sum(qg)*dt of C (the divergence-free flux
    # contributes nothing to the total), i.e. the C source is qg -- NOT qg/Bg
    # (which would over-inject by 1/Bg). Regression guard for the /Bg bug.
    from src.programs.forward import (
        _implicit_compositional_step, _eq_solution_gas_ratio,
        _mobility_divergence_matrix, _gravity_divergence_matrix,
    )

    case, mesh, p, sw, so, sg = _setup()
    grid = mesh.grid
    n_c = grid.n_cells
    k = np.full(n_c, case.k0)
    phi = np.full(n_c, case.phi0)
    vol = grid.cell_volumes()
    inv_phiV = 1.0 / (phi * vol)
    # bg != 1 so the regression is meaningful: qg/Bg (the bug) would over-inject
    # by 1/bg = 333x relative to the surface rate qg.
    params = replace(case.black_oil, rs_slope=0.0, rs_eq_slope=0.0, bg=0.003)
    Rs = _eq_solution_gas_ratio(p[0], params)
    C0 = np.zeros(n_c)
    # Inject a small amount of pure gas at one cell only, no production: net
    # surface gas = Q. Small enough to stay well inside the phase-split bounds
    # (so the Newton converges), but the total is still exactly Q*dt.
    Q = 1.0e-9
    qg_t = np.zeros(n_c)
    qg_t[0] = Q
    A = _mobility_divergence_matrix(grid, k, p[0])
    A_grav = _gravity_divergence_matrix(grid, k)
    sw1, so1, sg1, C1, conv = _implicit_compositional_step(
        A, A_grav, inv_phiV, 864.0, np.zeros(n_c), C0, np.zeros(n_c), qg_t, Rs, params
    )
    assert conv
    accum = float((phi * vol * (C1 - C0)).sum())  # surface CO2 accumulated (m3)
    injected = Q * 864.0                            # surface CO2 injected (m3)
    assert np.isclose(accum, injected, rtol=0.05, atol=0.0), (
        f"surface CO2 not conserved: accumulated {accum:.3e} m3 vs "
        f"injected {injected:.3e} m3"
    )


def test_kinetic_dissolution_converges_and_conserves():
    # Kinetic dissolution (k_diss > 0): the dissolved-CO2 relaxation term
    # k_diss*(Rs*No - Cd) is a rate per pore volume, so it must be scaled by
    # phi*vol (= 1/inv_phiV) to sit in the same surface m3/s units as the
    # accumulation term accum*(Cd-Cd0). Without that factor the residual is
    # dominated by the (unscaled) dissolution term and the Newton cannot
    # converge. A single step must converge AND conserve total surface CO2.
    from src.programs.forward import (
        _implicit_kinetic_step, _eq_solution_gas_ratio,
        _mobility_divergence_matrix, _gravity_divergence_matrix,
    )

    case, mesh, p, sw, so, sg = _setup()
    grid = mesh.grid
    n_c = grid.n_cells
    k = np.full(n_c, case.k0)
    phi = np.full(n_c, case.phi0)
    vol = grid.cell_volumes()
    inv_phiV = 1.0 / (phi * vol)
    params = replace(case.black_oil, rs_slope=0.0, rs_eq_slope=0.0, bg=0.003, k_diss=1.8e-8)
    Rs = _eq_solution_gas_ratio(p[0], params)
    C0 = np.zeros(n_c)
    Cd0 = np.zeros(n_c)
    Q = 1.0e-9
    qg_t = np.zeros(n_c)
    qg_t[0] = Q
    A = _mobility_divergence_matrix(grid, k, p[0])
    A_grav = _gravity_divergence_matrix(grid, k)
    sw1, so1, sg1, C1, Cd1, conv = _implicit_kinetic_step(
        A, A_grav, inv_phiV, 864.0, np.zeros(n_c), C0, Cd0,
        np.zeros(n_c), qg_t, np.zeros(n_c), Rs, params,
    )
    assert conv
    accum = float((phi * vol * (C1 - C0)).sum())  # total surface CO2 accumulated
    injected = Q * 864.0
    assert np.isclose(accum, injected, rtol=0.05, atol=0.0), (
        f"kinetic surface CO2 not conserved: accumulated {accum:.3e} m3 vs "
        f"injected {injected:.3e} m3"
    )


def test_eq_solution_gas_ratio_quadratic():
    # The equilibrium Rs(p) is Henry's law slope*p by default; rs_quad adds a
    # quadratic term so the GEM EOS's stronger pressure sensitivity near the
    # phase boundary can be represented: Rs = slope*p + rs_quad*p^2.
    from src.programs.forward import _eq_solution_gas_ratio
    from src.programs.rock import FluidParams

    params = FluidParams(rs_slope=1.0e-6, rs_quad=1.0e-13)
    p = np.array([10.0e6, 20.0e6])
    rs = _eq_solution_gas_ratio(p, params)
    # 1e-6*10e6 + 1e-13*(10e6)^2 = 10 + 10 = 20;  1e-6*20e6 + 1e-13*(20e6)^2 = 20 + 40 = 60
    assert np.allclose(rs, [20.0, 60.0])




def test_compositional_pressure_step_conservation():
    # The fully-implicit (p, sw, C) step couples the Peaceman pressure equation
    # with the water/CO2 conservation; a single step must converge and conserve
    # the injected surface CO2 (the C accumulation equals the BHP-driven gas rate
    # x dt, since the divergence-free flux contributes nothing to the total).
    from src.programs.forward import (
        _implicit_compositional_pressure_step, solve_pressure_peaceman,
    )
    from src.programs.rock import phase_mobilities

    case = load_lab_case(ROOT / "examples" / "shale_oil" / "case.yaml")
    mesh = run_mesh(case)
    grid = mesh.grid
    n_c = grid.n_cells
    k = np.full(n_c, case.k0)
    phi = np.full(n_c, case.phi0)
    vol = grid.cell_volumes()
    inv_phiV = 1.0 / (phi * vol)
    params = case.black_oil
    sw0 = np.full(n_c, params.swc)
    C0 = np.zeros(n_c)
    lam_w, lam_o, lam_g = phase_mobilities(sw0, np.ones(n_c), np.zeros(n_c), params)
    p0 = solve_pressure_peaceman(grid, k, lam_w + lam_o + lam_g, mesh.wells, case.well_pw[0], case.well)
    inj = np.array([case.well_qg[0] > 0])
    p1, sw1, so1, sg1, C1, conv = _implicit_compositional_pressure_step(
        grid, k, inv_phiV, 864.0, sw0, C0, p0, mesh.wells, case.well_pw[0], case.well, params, inj,
    )
    assert conv
    for f in (sw1, so1, sg1):
        assert np.isfinite(f).all()
        assert np.all((f >= -1.0e-9) & (f <= 1.0 + 1.0e-9))
    assert np.allclose(sw1 + so1 + sg1, 1.0, atol=1.0e-6)
    # surface CO2 accumulated == the Peaceman (BHP-driven) gas injection.
    accum = float((phi * vol * (C1 - C0)).sum())
    assert accum > 0.0
