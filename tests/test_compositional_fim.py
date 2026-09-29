"""Compositional forward model (FIM) regression tests.

Pins the 14-component molar forward model (``_implicit_compositional_full_step``)
and the shared EOS / operator helpers it is built on:

- molar-balance pressure equation (non-CO2 hydrocarbon conserved).
- bubble-point crossing convergence (no Newton stall).
- EOS phase density gravity sign (CO2 denser than oil -> sinks).
- EOS gravity head (per-face density difference) and its scalar degeneration.
- molar-consistent well terms (injector / producer moles on the EOS molar volumes).

The slow 30-day plume test is marked ``@pytest.mark.slow``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.core.cartesian import CartesianGrid
from src.core.lab_case import load_lab_case
from src.core.pr_eos import (
    _CO2_IDX, _V_CO2_STD, _Z_OIL_DEAD, co2_molar_volume, flash_direct_full,
    flash_direct_volumes, phase_molar_volumes,
)
from src.programs.mesh import WellMap
from src.programs.pipeline import run_mesh
from src.programs.forward import (
    _GRAVITY,
    _eos_gas_head,
    _implicit_compositional_full_adaptive,
    _implicit_compositional_full_step,
    _mobility_divergence_matrix,
    _peaceman_well_data,
    _phase_divergence_matrix,
    _split_full_N,
    solve_pressure_peaceman,
)
from src.programs.rock import phase_mobilities

CASE = "examples/shale_oil/case.yaml"


def _setup():
    case = load_lab_case(CASE)
    mesh = run_mesh(case)
    grid = mesh.grid
    n_c = grid.n_cells
    k = np.full(n_c, case.k0)
    phi = np.full(n_c, case.phi0)
    vol = grid.cell_volumes()
    inv_phiV = 1.0 / (phi * vol)
    params = case.black_oil
    sw = np.full(n_c, params.swc)
    z = np.broadcast_to(_Z_OIL_DEAD, (n_c, _Z_OIL_DEAD.size)).copy()
    lam_w, lam_l, lam_g = phase_mobilities(sw, np.ones(n_c), np.zeros(n_c), params)
    p = solve_pressure_peaceman(grid, k, lam_w + lam_l + lam_g, mesh.wells, case.well_pw[0], case.well)
    injects_gas = (case.well_qg > 0.0).any(axis=0)
    inj_idx = int(np.flatnonzero(injects_gas)[0])
    inj_cells = mesh.wells.cells[inj_idx]
    qg_fixed = np.zeros(n_c)
    qg_fixed[inj_cells] += case.well_qg[0, inj_idx] / inj_cells.size
    return case, mesh, grid, k, phi, vol, inv_phiV, params, sw, z, p, injects_gas, qg_fixed


def _step(grid, k, inv_phiV, params, sw, z, p, injects_gas, qg_fixed, wells, well_pw, well_params, dt=864.0):
    return _implicit_compositional_full_step(
        grid, k, inv_phiV, dt, sw, z, p, wells, well_pw, well_params,
        params, injects_gas, well_qg_fixed=qg_fixed)


def _small_setup(nx=3, inject_qg=1.0e-7, producer=True):
    """Small injector(/producer) column for fast full-model physics tests."""
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=nx, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    n = grid.n_cells
    k = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    vol = grid.cell_volumes()
    inv_phiV = 1.0 / (phi * vol)
    params = case.black_oil
    sw = np.full(n, params.swc)
    z = np.broadcast_to(_Z_OIL_DEAD, (n, _Z_OIL_DEAD.size)).copy()
    if producer:
        wells = WellMap(
            ids=("INJ", "PROD"),
            xyz=grid.cell_centers()[[0, n - 1]],
            cells=(np.array([0]), np.array([n - 1])),
        )
        bhp = np.array([19.5e6, 19.0e6])
        injects_gas = np.array([True, False])
    else:
        wells = WellMap(ids=("INJ",), xyz=grid.cell_centers()[[0]], cells=(np.array([0]),))
        bhp = np.array([19.5e6])
        injects_gas = np.array([True])
    qg_fixed = np.zeros(n)
    qg_fixed[0] += inject_qg
    p = np.full(n, 19.0e6)
    return case, grid, k, phi, vol, inv_phiV, params, sw, z, p, wells, bhp, injects_gas, qg_fixed


def _oil_moles(grid, phi, vol, sw, z, p, params):
    """Total non-CO2 hydrocarbon moles (the molar pressure equation conserves these)."""
    N = _split_full_N(sw, z, p, params)
    return float(((1.0 - z[:, _CO2_IDX]) * N * phi * vol).sum())


def test_molar_pressure_conserves_oil():
    """The molar-balance pressure equation must conserve the non-CO2 hydrocarbon."""
    (case, grid, k, phi, vol, inv_phiV, params,
     sw, z, p, wells, bhp, injects_gas, qg_fixed) = _small_setup(producer=False, inject_qg=1.0e-9)
    oil0 = _oil_moles(grid, phi, vol, sw, z, p, params)
    for _ in range(3):
        p, sw, sl, sg, z, conv = _step(
            grid, k, inv_phiV, params, sw, z, p, injects_gas, qg_fixed,
            wells, bhp, case.well)
        assert conv
    oil1 = _oil_moles(grid, phi, vol, sw, z, p, params)
    # 3 substeps of injection must not leak oil (the old volume balance did).
    assert abs(oil1 - oil0) / oil0 < 1.0e-2


def test_free_gas_forms_at_bubble_point():
    """CO2 injected past the bubble point must exsolve free gas (no over-dissolution)."""
    (case, grid, k, phi, vol, inv_phiV, params,
     sw, z, p, wells, bhp, injects_gas, qg_fixed) = _small_setup()
    sg_max = 0.0
    for _ in range(15):
        p, sw, sl, sg, z, conv = _step(
            grid, k, inv_phiV, params, sw, z, p, injects_gas, qg_fixed,
            wells, bhp, case.well)
        assert conv
        sg_max = max(sg_max, float(sg.max()))
    assert sg_max > 1.0e-3  # free gas appears once z crosses the bubble point


def test_bubble_point_crossing_converges():
    """The Newton must converge through the bubble point (no stall)."""
    (case, grid, k, phi, vol, inv_phiV, params,
     sw, z, p, wells, bhp, injects_gas, qg_fixed) = _small_setup()
    crossed = False
    for _ in range(20):
        p, sw, sl, sg, z, conv = _step(
            grid, k, inv_phiV, params, sw, z, p, injects_gas, qg_fixed,
            wells, bhp, case.well)
        assert conv
        if z[:, _CO2_IDX].max() > 0.36:  # bubble point (dead-oil base)
            crossed = True
    assert crossed


def test_eos_co2_denser_than_oil():
    """The Peneloux-shifted EOS must give pure CO2 denser than pure oil (gravity sinks)."""
    p = np.array([20.0e6])
    # pure CO2 gas density (z -> 1)
    rho_co2 = flash_direct_volumes(p, np.array([0.999]), with_density=True)[6][0]
    # pure (dead) oil liquid density (z = 0)
    rho_oil = flash_direct_volumes(p, np.array([0.0]), with_density=True)[5][0]
    assert rho_co2 > rho_oil  # CO2 sinks (was backwards before the volume shift)


def test_near_pure_co2_flash_is_vapor():
    """z ≳ 0.995 must not keep the trivial x=y root (arbitrary V)."""
    p = np.array([20.0e6])
    V, x_co2, y_co2, v_l, v_g = flash_direct_volumes(p, np.array([0.999]))
    assert float(V[0]) > 0.99
    assert abs(float(y_co2[0]) - 0.999) < 1.0e-6
    # A real two-phase state still splits (x ≠ y).
    V2, x2, y2, _, _ = flash_direct_volumes(p, np.array([0.6]))
    assert 0.05 < float(V2[0]) < 0.95
    assert abs(float(y2[0]) - float(x2[0])) > 0.05


def test_rock_compressibility_lowers_pressure_rise():
    """The same injected moles raise pressure less when the pore volume can grow."""
    case, grid, k, phi, inv_phiV, wells = _single_cell()
    sw = np.full(1, case.black_oil.swc)
    z = _Z_OIL_DEAD[None, :].copy()
    p0 = np.full(1, 19.0e6)
    bhp = np.array([20.0e6])
    q = np.full(1, 2.0e-10)
    kw = dict(well_qg_fixed=q, tol=1.0e-3, max_iter=20)

    def rise(ct):
        params = replace(case.black_oil, ct=ct)
        p1, _, _, _, _, conv = _implicit_compositional_full_step(
            grid, k, inv_phiV, 30.0, sw, z, p0, wells, bhp, case.well, params,
            np.array([True]), **kw)
        assert conv
        return float(p1[0] - p0[0])

    assert rise(1.0e-6) < rise(0.0)


def test_eos_gas_density_rises_with_pressure():
    """Supercritical CO2 density is strongly pressure dependent; a scalar head is not."""
    p = np.array([15.0e6, 20.0e6, 22.0e6])
    rho_g = flash_direct_volumes(p, np.full(3, 0.9), with_density=True)[6]
    assert np.all(np.diff(rho_g) > 0.0)
    assert rho_g[2] > 1.5 * rho_g[0]
    rho_co2_20 = 0.04401 / co2_molar_volume(20.0e6)[0]
    assert abs(rho_co2_20 - 590.0) < 5.0  # Peneloux shift calibrated to NIST


def _column(nz=3, nx=2):
    return CartesianGrid(nx=nx, ny=1, nz=nz, dx=0.1, dy=0.1, dz=0.1)


def test_eos_gas_head_reduces_to_scalar_when_densities_constant():
    grid = _column()
    n = grid.n_cells
    rho_gas, rho_liq = 590.0, 512.6
    head = _eos_gas_head(grid, np.full(n, 0.6), np.full(n, 0.3),
                         np.full(n, rho_liq), np.full(n, rho_gas), fallback=-1.0)
    scalar = _GRAVITY * (rho_gas - rho_liq)
    assert np.allclose(head, scalar)
    k = np.full(n, 1.0e-15)
    p = 20.0e6 + np.random.default_rng(0).normal(0.0, 50.0, n)
    A_face = _phase_divergence_matrix(grid, k, p, head)
    A_scalar = _phase_divergence_matrix(grid, k, p, scalar)
    assert np.allclose(A_face.toarray(), A_scalar.toarray(), rtol=1e-12, atol=0.0)
    # no gas anywhere: every face keeps the scalar fallback head
    head0 = _eos_gas_head(grid, np.full(n, 0.9), np.zeros(n),
                          np.full(n, rho_liq), np.zeros(n), fallback=759.0)
    assert np.allclose(head0, 759.0)


def test_eos_gas_head_takes_present_phase_density():
    """An absent phase must not dilute the face density (saturation weighting)."""
    grid = _column(nz=2, nx=1)  # cell 0 bottom, cell 1 top
    sl = np.array([1.0, 0.5])
    sg = np.array([0.0, 0.5])
    rho_l = np.array([530.0, 540.0])
    rho_g = np.array([0.0, 600.0])  # no gas in the bottom cell
    head = _eos_gas_head(grid, sl, sg, rho_l, rho_g, fallback=0.0)
    rho_l_face = (1.0 * 530.0 + 0.5 * 540.0) / 1.5
    assert np.isclose(float(head.ravel()[0]), _GRAVITY * (600.0 - rho_l_face))


def test_scalar_head_matches_phase_potential_operator():
    """The z-face head path equals upwinding on the cell potential ``p + rho_g z``."""
    grid = CartesianGrid(nx=2, ny=2, nz=3, dx=0.1, dy=0.1, dz=0.1)
    n = grid.n_cells
    k = np.full(n, 1.0e-15)
    p = 20.0e6 + np.random.default_rng(1).normal(0.0, 100.0, n)
    rho = 759.0
    z = grid.cell_centers()[:, 2]
    A_pot = _mobility_divergence_matrix(grid, k, p + rho * z)
    A_head = _phase_divergence_matrix(grid, k, p, rho)
    # same upwind pattern; values differ only by the cancellation in (p + rho z) differences
    assert np.array_equal(A_pot.toarray() != 0.0, A_head.toarray() != 0.0)
    assert np.allclose(A_pot.toarray(), A_head.toarray(), rtol=1e-7, atol=0.0)


def _single_cell():
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=1, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    k = np.full(1, case.k0)
    phi = np.full(1, case.phi0)
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    wells = WellMap(ids=("w",), xyz=grid.cell_centers()[:1], cells=(np.array([0]),))
    return case, grid, k, phi, inv_phiV, wells


def _full_state(sw, z, p, params):
    N = _split_full_N(sw, z, p, params)
    return N, z[:, _CO2_IDX] * N * _V_CO2_STD


def test_injector_moles_match_surface_target():
    """Rate-controlled injector: accumulated moles == target surface rate / V_CO2_STD.

    The initial BHP guess injects ~300x the target, so this also pins the BHP
    Schur coupling (the full-compositional step read the rate residual from the
    wrong index before the fix).
    """
    case, grid, k, phi, inv_phiV, wells = _single_cell()
    params = case.black_oil
    sw = np.full(1, params.swc)
    p = np.full(1, 19.0e6)
    q_surf = 1.0e-9  # m3/s surface CO2
    dt = 60.0
    bhp = np.array([19.5e6])
    kw = dict(well_qg_fixed=np.full(1, q_surf), tol=1.0e-4)
    inj = np.array([True])
    z = _Z_OIL_DEAD[None, :].copy()
    N0, C0 = _full_state(sw, z, p, params)
    p1, sw1, _, _, z1, conv = _implicit_compositional_full_step(
        grid, k, inv_phiV, dt, sw, z, p, wells, bhp, case.well, params, inj, **kw)
    N1, C1 = _full_state(sw1, z1, p1, params)
    assert conv
    pv = 1.0 / inv_phiV
    # Pore growth φ·ct·Δp stores part of the injected moles; N and C are per
    # reference pore volume, so the balance includes that term.
    rock = params.ct * (p1 - p)
    target_mol = q_surf * dt / _V_CO2_STD
    assert float((((N1 - N0) + rock * N1) * pv).sum()) == pytest.approx(target_mol, rel=5.0e-3)
    assert float((((C1 - C0) + rock * C1) * pv).sum()) == pytest.approx(q_surf * dt, rel=5.0e-3)
    assert p1[0] > p[0]  # closed cell pressurizes


def test_producer_gas_moles_use_eos_molar_volume():
    """BHP producer: produced moles are WI·(λg/v_g + λl/v_l)·(p−bhp), not λg/(Bg·V_CO2_STD)."""
    case, grid, k, phi, inv_phiV, wells = _single_cell()
    params = replace(case.black_oil, bg=0.003)
    sw = np.full(1, params.swc)
    p = np.full(1, 16.0e6)  # low pressure: EOS Bg of the CO2-rich gas >> 0.003
    z = (1.0 - 0.8) * _Z_OIL_DEAD
    z[_CO2_IDX] += 0.8
    z = z[None, :].copy()  # overall CO2 mole fraction 0.8, rest dead oil
    bhp = np.array([15.5e6])
    inj = np.array([False])
    dt = 60.0
    N0 = _split_full_N(sw, z, p, params)
    p1, sw1, sl1, sg1, z1, conv = _implicit_compositional_full_step(
        grid, k, inv_phiV, dt, sw, z, p, wells, bhp, case.well, params, inj, tol=1.0e-3)
    assert conv
    V, x, y = flash_direct_full(z1, p1)
    v_l, v_g = phase_molar_volumes(x, y, p1)
    lam_w, lam_l, lam_g = phase_mobilities(sw1, sl1, sg1, params)
    assert float(sg1[0]) > 0.05 and float(lam_g[0]) > 0.0
    _, wi, _, _ = _peaceman_well_data(grid, k, wells, bhp, case.well, inj)
    dp = float(bhp[0] - p1[0])
    q_mol_eos = float(wi[0] * (lam_g[0] / v_g[0] + lam_l[0] / v_l[0]) * dp)
    q_mol_bg = float(wi[0] * (lam_g[0] / (params.bg * _V_CO2_STD) + lam_l[0] / v_l[0]) * dp)
    pv = 1.0 / inv_phiV
    rock = params.ct * (p1 - p)
    N1 = _split_full_N(sw1, z1, p1, params)
    dN = float((((N1 - N0) + rock * N1) * pv).sum())
    assert dN == pytest.approx(q_mol_eos * dt, rel=1.0e-2)
    assert abs(q_mol_bg - q_mol_eos) > 0.1 * abs(q_mol_eos)  # the scalar Bg would be off


@pytest.mark.slow
def test_bottom_plume_sinks():
    """30-day plume: the free gas must concentrate toward the bottom (gravity)."""
    (case, mesh, grid, k, phi, vol, inv_phiV, params,
     sw, z, p, injects_gas, qg_fixed) = _setup()
    z_c = grid.cell_centers()[:, 2]
    bottom = z_c <= 0.05
    top = z_c >= 0.27
    target = 30.0 * 86400.0
    t = 0.0
    dt_sub0 = None
    while t < target - 1.0:
        dt = min(86400.0, target - t)
        p, sw, sl, sg, z, conv, last_dt = _implicit_compositional_full_adaptive(
            grid, k, inv_phiV, dt, sw, z, p, mesh.wells, case.well_pw[0], case.well,
            params, injects_gas, well_qg_fixed=qg_fixed, dt_sub0=dt_sub0, dt_max=864.0, dt_min=1.0)
        dt_sub0 = last_dt
        assert conv, f"full 30-day plume did not converge at t={t / 86400.0:.1f} d"
        t += dt
    # The plume should be bottom-weighted (truth bottom k0-2 mean = 0.456).
    assert sg[bottom].mean() > sg[top].mean()
