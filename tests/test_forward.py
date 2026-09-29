"""Forward saturation model (P3) regression tests.

Pins the pluggable black-oil / compositional forward models: finite in-range
outputs, phase conservation (sw+so+sg=1), and the compositional phase split.
"""

from pathlib import Path

import numpy as np
import pytest

from src.core.lab_case import load_lab_case
from src.programs.forward import forward_saturations
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
    from src.core.cartesian import CartesianGrid
    from src.programs.mesh import WellMap

    case, _, _, sw, so, sg = _setup()
    grid = CartesianGrid(nx=2, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    wells = WellMap(ids=("INJ",), xyz=grid.cell_centers()[:1], cells=(np.array([0]),))
    p = np.full((2, 2), 19.0e6)
    qg = np.full((2, 1), 1.0e-9)
    z = np.zeros((2, 1))
    s0 = np.full(2, float(sw[0, 0]))
    fsw, fso, fsg = forward_saturations(
        "compositional", grid, p, np.full(2, case.k0), np.full(2, case.phi0),
        case.black_oil, wells, z, z, qg, np.array([0.0, 30.0]),
        s0, np.full(2, float(so[0, 0])), np.full(2, float(sg[0, 0])),
        well_bhp=np.full((2, 1), 19.5e6), well_params=case.well,
    )
    for f in (fsw, fso, fsg):
        assert np.isfinite(f).all()
        assert np.all((f >= -1.0e-9) & (f <= 1.0 + 1.0e-9))
    assert np.allclose(fsw + fso + fsg, 1.0, atol=1.0e-6)


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


def test_land_free_gas_trapping():
    # Land's residual trapping: all gas free at the reversal (sg = sg_max), all
    # trapped at the Land residual (sg = sgr), monotonic in between, and C=0
    # disables trapping entirely (unlike the old linear Carlson model).
    from src.programs.rock import land_free_gas

    C = 1.0
    sg_max = 0.5
    sgr = sg_max / (1.0 + C * sg_max)  # 1/3
    assert land_free_gas(np.array([sg_max]), np.array([sg_max]), C)[0] == pytest.approx(sg_max)
    assert land_free_gas(np.array([sgr]), np.array([sg_max]), C)[0] == pytest.approx(0.0, abs=1.0e-9)
    sgs = np.linspace(sgr, sg_max, 25)
    sgf = land_free_gas(sgs, np.full_like(sgs, sg_max), C)
    assert np.all(np.diff(sgf) >= 0.0)  # monotonic free gas
    # C = 0 -> no trapping (sgf = sg)
    assert np.allclose(land_free_gas(sgs, np.full_like(sgs, sg_max), 0.0), sgs)
