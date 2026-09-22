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
from src.programs.rock import BlackOilParams
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
    params = BlackOilParams(mu_g=2.0e-5, mu_o=2.0e-3, rho_s=2000.0, rho_o=0.0)
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
    params = BlackOilParams(mu_o=2.0e-3, mu_g=2.0e-5)
    lam_w, lam_o, lam_g = tabular_phase_mobilities(0.0, 0.6, 0.4, table, params)
    assert np.allclose(lam_w, 0.0)
    assert np.allclose(lam_o, 0.25 / 2.0e-3)  # Krog(0.4)=0.25 * Krow(0)=1
    assert np.allclose(lam_g, 0.25 / 2.0e-5)  # Krg(0.4)=0.25
