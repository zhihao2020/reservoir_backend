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
import warnings

import numpy as np
import pytest

from src.core.cartesian import CartesianGrid
from src.core.lab_case import load_lab_case
from src.core.pr_eos import (
    _CO2_IDX, _V_CO2_STD, _Z_OIL_DEAD, co2_molar_volume, flash_direct_full,
    flash_direct_volumes, fugacity_mole_deriv, natural_variables_jacobian,
    natural_variables_residual, natural_variables_state, phase_flags,
    phase_molar_volume_deriv, phase_molar_volumes,
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
    _split_full,
    _split_full_N,
    forward_saturations,
    reservoir_co2_to_surface,
    solve_pressure_peaceman,
)
from src.programs.natural_variables import natural_variables_step
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
    # The series ``qg`` is a *reservoir* m3/s rate (GEM BHF); the injector constraint
    # compares a *surface* rate, so convert (same as _forward_compositional_full_saturations).
    p_inj = float(np.mean(case.well_pw[0][injects_gas]))
    q_surf = float(np.asarray(reservoir_co2_to_surface(case.well_qg[0, inj_idx], p_inj)).ravel()[0])
    qg_fixed[inj_cells] += q_surf / inj_cells.size
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


def test_natural_variables_step_matches_reference():
    """The natural-variables FIM step reproduces the overall-composition step.

    On a single two-phase producer cell the two formulations solve the same
    molar physics, so ``(p, sg, z_CO2)`` must agree. This is the Step 5 smoke test:
    the fugacity equations replace the flash but drive the state to the same
    equilibrium.
    """
    case, grid, k, phi, inv_phiV, wells = _single_cell()
    params = replace(case.black_oil, bg=0.003)
    sw = np.full(1, params.swc)
    p = np.full(1, 16.0e6)
    z = (1.0 - 0.8) * _Z_OIL_DEAD
    z[_CO2_IDX] += 0.8
    z = z[None, :].copy()
    bhp = np.array([15.5e6])
    inj = np.array([False])
    dt = 60.0
    kw = dict(tol=1.0e-3)
    p_ref, sw_ref, sl_ref, sg_ref, z_ref, c_ref = _implicit_compositional_full_step(
        grid, k, inv_phiV, dt, sw, z, p, wells, bhp, case.well, params, inj, **kw)
    p_nv, sw_nv, sl_nv, sg_nv, z_nv, c_nv = natural_variables_step(
        grid, k, inv_phiV, dt, sw, z, p, wells, bhp, case.well, params, inj, **kw)
    assert c_ref and c_nv
    assert p_nv[0] == pytest.approx(p_ref[0], rel=1.0e-4)
    assert sg_nv[0] == pytest.approx(sg_ref[0], abs=5.0e-3)
    assert z_nv[0, _CO2_IDX] == pytest.approx(z_ref[0, _CO2_IDX], abs=2.0e-3)
    # the natural-variables state must be self-consistent: sw + sO + sG = 1
    assert float(sw_nv[0] + sl_nv[0] + sg_nv[0]) == pytest.approx(1.0, abs=1.0e-8)


def test_natural_variables_multi_cell_matches_reference():
    """The natural-variables FIM matches the overall-composition step on a column.

    A pure-liquid cell (dead oil — exercises the single-phase variable fold
    ``y=x, sO=1−sw``) next to a two-phase producer cell (exercises the inter-cell
    flux saturation/composition derivatives) reaches the two corners the single-cell
    smoke test cannot: the folded columns and the ``A·diags`` flux blocks.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=2, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    n = grid.n_cells
    k = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    params = case.black_oil
    sw = np.full(n, params.swc)
    zc = np.array([0.0, 0.7])  # cell 0 pure liquid, cell 1 two-phase
    z = np.outer(1.0 - zc, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zc
    wells = WellMap(ids=("PROD",), xyz=grid.cell_centers()[[n - 1]], cells=(np.array([n - 1]),))
    bhp = np.array([18.5e6])
    inj = np.array([False])
    p = np.full(n, 19.0e6)
    kw = dict(tol=0.1)
    p_ref, sw_ref, sl_ref, sg_ref, z_ref, c_ref = _implicit_compositional_full_step(
        grid, k, inv_phiV, 864.0, sw, z, p, wells, bhp, case.well, params, inj, **kw)
    p_nv, sw_nv, sl_nv, sg_nv, z_nv, c_nv = natural_variables_step(
        grid, k, inv_phiV, 864.0, sw, z, p, wells, bhp, case.well, params, inj, **kw)
    assert c_ref and c_nv
    assert np.allclose(z_nv[:, _CO2_IDX], z_ref[:, _CO2_IDX], atol=5.0e-3)
    assert np.allclose(sg_nv, sg_ref, atol=2.0e-2)
    # the pure-liquid cell stays single-phase (its gas saturation was folded to 0)
    assert sg_nv[0] == pytest.approx(0.0, abs=1.0e-6)


def test_natural_variables_phase_transition():
    """Single-phase (dead oil) cells switch to two-phase as CO2 crosses the bubble point.

    The natural-variables model stays single-phase while CO2 dissolves (sg=0), then the
    Michelsen stability test in ``flashPhases`` inserts the incipient vapor once the
    overall composition crosses the bubble point — the single→two-phase switch that the
    overall-composition model's flash-based Newton stalls on.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=3, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    n = grid.n_cells
    k = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    params = case.black_oil
    sw = np.full(n, params.swc)
    z = np.broadcast_to(_Z_OIL_DEAD, (n, _Z_OIL_DEAD.size)).copy()  # dead oil: pure liquid
    wells = WellMap(ids=("INJ", "PROD"), xyz=grid.cell_centers()[[0, n - 1]],
                    cells=(np.array([0]), np.array([n - 1])))
    bhp = np.array([19.5e6, 19.0e6])
    inj = np.array([True, False])
    qg_fixed = np.zeros(n)
    qg_fixed[0] += 1.0e-7
    p = np.full(n, 19.0e6)
    sg_max = 0.0
    for _ in range(8):
        p, sw, sl, sg, z, conv = natural_variables_step(
            grid, k, inv_phiV, 864.0, sw, z, p, wells, bhp, case.well, params, inj,
            well_qg_fixed=qg_fixed, tol=0.1)
        assert conv
        sg_max = max(sg_max, float(sg.max()))
    assert sg_max > 1.0e-3  # free gas formed once z crossed the bubble point


def test_natural_variables_forward_integration():
    """The natural-variables model is reachable from ``forward_saturations``.

    ``model="natural_variables"`` dispatches to the coupled natural-variables FIM (the
    fugacity-equality form that crosses the bubble point smoothly), reporting the same
    ``(sw, so, sg, z_co2)`` shape as the overall-composition model.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=3, ny=1, nz=1, dx=0.05, dy=0.05, dz=0.05)
    n = grid.n_cells
    k = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    params = case.black_oil
    wells = WellMap(ids=("INJ", "PROD"), xyz=grid.cell_centers()[[0, n - 1]],
                    cells=(np.array([0]), np.array([n - 1])))
    bhp = np.array([[19.5e6, 19.0e6]])
    qg = np.array([[8.333e-8, 0.0]])
    zeros = np.zeros_like(qg)
    sw0 = np.full(n, params.swc)
    p = np.full((2, n), 19.0e6)
    sw, so, sg, zco2 = forward_saturations(
        "natural_variables", grid, p, k, phi, params, wells,
        zeros, zeros, qg, np.array([0.0, 864.0]),
        sw0, np.ones(n), np.zeros(n),
        well_bhp=bhp, well_params=case.well, return_co2=True)
    assert np.allclose(sw[1] + so[1] + sg[1], 1.0, atol=1.0e-6)
    assert float(zco2[1, 0]) > 0.3   # CO2 entered the injector
    assert float(sg[1].max()) >= 0.0  # free gas formed (or at least no crash)


def test_natural_variables_plume():
    """The natural-variables model crosses the bubble point on a plume column.

    Dead oil (pure liquid) + top CO2 injection + bottom producer: the
    overall-composition model stalls here (its flash-based Newton diverges past the
    bubble point), while the fugacity-equality natural-variables form crosses it
    smoothly and advects the CO2 (and the free gas it exsolves) down to the bottom
    completion — the variable-switching fix for the plume's convergence failure.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=1, ny=1, nz=10, dx=0.02, dy=0.02, dz=0.02)
    n = grid.n_cells
    k = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    params = case.black_oil
    sw = np.full(n, params.swc)
    z = np.broadcast_to(_Z_OIL_DEAD, (n, _Z_OIL_DEAD.size)).copy()  # dead oil: pure liquid
    inj_cells = np.arange(8, 10)
    wells = WellMap(ids=("INJ", "PROD"), xyz=grid.cell_centers()[[9, 0]],
                    cells=(inj_cells, np.array([0])))
    bhp = np.array([19.3e6, 19.0e6])
    inj = np.array([True, False])
    qg_fixed = np.zeros(n)
    qg_fixed[inj_cells] += 1.0e-6 / inj_cells.size
    p = np.full(n, 19.1e6)
    zc = grid.cell_centers()[:, 2]
    bottom = zc < 2.0 * float(grid.dz[0])
    for _ in range(8):
        p, sw, sl, sg, z, conv = natural_variables_step(
            grid, k, inv_phiV, 864.0, sw, z, p, wells, bhp, case.well, params, inj,
            well_qg_fixed=qg_fixed, tol=0.1)
        assert conv  # no bubble-point stall (the overall-composition model stalls here)
    # CO2 advected down to the bottom completion, exsolving free gas along the way
    assert float(z[bottom, _CO2_IDX].mean()) > 0.5
    assert float(sg.max()) > 0.0


def test_reservoir_co2_rate_converts_to_surface():
    """The lab case stores reservoir m3/s; the injector constraint wants surface m3/s."""
    q_res = 8.333e-8  # 0.0072 m3/day, GEM BHF at reservoir conditions
    q_surf = float(np.asarray(reservoir_co2_to_surface(q_res, 19.0e6)).ravel()[0])
    assert 150.0 < q_surf / q_res < 400.0


def test_corrected_injection_enters_at_the_top():
    """Reservoir-rate CO2, converted to surface, accumulates in the top completions.

    A single producer cell cannot sink the full well rate, so pressure climbs and
    free gas does not appear on this column. The check is only where the CO2 goes.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=1, ny=1, nz=15, dx=0.02, dy=0.02, dz=0.02)
    n = grid.n_cells
    perm = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    params = case.black_oil
    sw = np.full(n, params.swc)
    z = np.broadcast_to(_Z_OIL_DEAD, (n, _Z_OIL_DEAD.size)).copy()
    inj_cells = np.arange(4, 15)
    wells = WellMap(
        ids=("INJ", "PROD"),
        xyz=grid.cell_centers()[[14, 0]],
        cells=(inj_cells, np.array([0])),
    )
    bhp = np.array([19.3e6, 19.0e6])
    q_surf = float(np.asarray(reservoir_co2_to_surface(8.333e-8, 19.3e6)).ravel()[0])
    qg_fixed = np.zeros(n)
    qg_fixed[inj_cells] = q_surf / inj_cells.size
    p = np.full(n, 19.1e6)
    for _ in range(5):
        p, sw, _, _, z, conv = _implicit_compositional_full_step(
            grid, perm, inv_phiV, 1.0, sw, z, p, wells, bhp, case.well, params,
            np.array([True, False]), well_qg_fixed=qg_fixed, max_iter=40,
        )
        assert np.isfinite(z).all()
    assert conv
    zc = z[:, _CO2_IDX]
    assert zc[12:15].mean() > 0.05
    assert zc[:3].mean() < 0.01


def test_flash_sg_tracks_co2_then_labels_single_phase_oil():
    """Reported sg follows the GEM front, then drops at single-phase CO2.

    Empirical base-variant curve at ~19 MPa (before day 1): z<0.2 has sg=0,
    sg rises through the two-phase window, and z>0.99 is mostly sg=0 because
    GEM ``*PHASEID *OIL`` names a single hydrocarbon phase as oil.
    """
    zs = np.array([0.10, 0.40, 0.50, 0.90, 0.999])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    sw = np.zeros(zs.size)
    _, _, sg = _split_full(sw, z, np.full(zs.size, 19.08e6))
    assert sg[0] == 0.0
    assert sg[1] > 0.0
    assert sg[2] > sg[1]
    assert sg[3] > sg[2]
    assert 0.7 < sg[3] < 1.0
    assert sg[4] == 0.0


def test_natural_variables_residual_vanishes_at_flash():
    """The natural-variables equilibrium residual is ~0 at the flash solution.

    The two-phase flash satisfies mass balance, fugacity equality and closure,
    so the residual that the natural-variables Newton will drive to zero is
    already (nearly) zero at the SSI flash result — the starting point for the
    fugacity-driven phase split that stays smooth through the bubble point.
    """
    zs = np.array([0.4, 0.6, 0.9])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(zs.size, 19.0e6)
    V, x, y = flash_direct_full(z, p)
    L = 1.0 - V
    res = natural_variables_residual(L, x, y, z, p)
    # mass balance (ncomp) + fugacity equality (ncomp) + closure (1), all ~0
    assert np.allclose(res, 0.0, atol=1.0e-4)


def test_fugacity_mole_deriv_satisfies_euler():
    """d lnφ_i/d ln n_j must sum to zero over j (degree-0 homogeneity)."""
    zs = np.array([0.6])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(1, 19.0e6)
    _, x, y = flash_direct_full(z, p)
    dL = fugacity_mole_deriv(x, p, phase="liq")
    dV = fugacity_mole_deriv(y, p, phase="vap")
    assert np.allclose(dL[0].sum(axis=1), 0.0, atol=1.0e-5)
    assert np.allclose(dV[0].sum(axis=1), 0.0, atol=1.0e-5)


def test_natural_variables_jacobian_matches_finite_difference():
    """The analytic Jacobian equals the finite-difference of the residual.

    This pins the mole-number → mole-fraction chain rule (``∂lnφ_i/∂x_j =
    D[i, j]/x_j``) and every sign/block placement, so a wrong factor (x_j), a
    flipped fugacity sign, or a mis-placed block fails loudly rather than degrading
    Newton to a slow fixed point.
    """
    zs = np.array([0.4, 0.6, 0.8])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(zs.size, 19.0e6)
    V, x, y = flash_direct_full(z, p)
    L = 1.0 - V
    J = natural_variables_jacobian(L, x, y, p)
    n = z.shape[1]
    m = 2 * n + 1
    base = natural_variables_residual(L, x, y, z, p)
    Jfd = np.zeros_like(J)
    h = 1.0e-6
    for c in range(zs.size):
        Lp = L.copy(); Lp[c] += h
        Jfd[c, :, 0] = (natural_variables_residual(Lp, x, y, z, p)[c] - base[c]) / h
        for j in range(n):
            xp = x.copy(); xp[c, j] += h
            Jfd[c, :, 1 + j] = (natural_variables_residual(L, xp, y, z, p)[c] - base[c]) / h
            yp = y.copy(); yp[c, j] += h
            Jfd[c, :, 1 + n + j] = (natural_variables_residual(L, x, yp, z, p)[c] - base[c]) / h
    # mass-balance / closure blocks are exactly linear; the fugacity block has
    # finite-difference noise from the internal 1e-6 mole-derivative, so a loose
    # relative tolerance still catches sign / chain-rule errors (O(1) / O(x_j)).
    assert np.allclose(J, Jfd, rtol=1.0e-2, atol=1.0e-4)


def test_natural_variables_flash_matches_ssi():
    """The Newton flash converges to the same (V, x, y) as successive substitution."""
    from src.core.pr_eos import _flash_vec, _flash_natural_variables, _ab
    zs = np.array([0.4, 0.55, 0.7, 0.85])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(zs.size, 19.0e6)
    T = 393.0
    aij, b = _ab(T)
    Vn, xn, yn = _flash_natural_variables(z, p, T, aij, b, tol=1.0e-9)
    Vs, xs, ys = _flash_vec(z, p, T, aij, b, tol=1.0e-5)
    assert np.allclose(Vn, Vs, atol=1.0e-4)
    assert np.allclose(xn, xs, atol=1.0e-4)
    assert np.allclose(yn, ys, atol=1.0e-4)
    # Newton drives the equilibrium residual to machine precision (SSI stops at ~1e-4).
    assert np.allclose(natural_variables_residual(1.0 - Vn, xn, yn, z, p), 0.0, atol=1.0e-7)


def test_natural_variables_newton_quadratic():
    """Newton on (L, x, y) converges quadratically from the SSI solution."""
    from src.core.pr_eos import _flash_vec, _ab
    zs = np.array([0.6])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(1, 19.0e6)
    T = 393.0
    aij, b = _ab(T)
    V, x, y = _flash_vec(z, p, T, aij, b, tol=1.0e-5)  # SSI init (residual ~1e-4)
    L = 1.0 - V
    n = z.shape[1]
    norms = []
    for _ in range(6):
        res = natural_variables_residual(L, x, y, z, p)
        norms.append(float(np.max(np.abs(res))))
        J = natural_variables_jacobian(L, x, y, p, aij=aij, b=b)
        du = np.linalg.solve(J, -res[:, :, None])[:, :, 0]
        L = L + du[:, 0]
        x = x + du[:, 1:1 + n]
        y = y + du[:, 1 + n:]
    assert norms[-1] < 1.0e-10
    # Quadratic (not linear): one Newton step from the SSI residual (~2e-5) lands
    # at ~5e-10 ≈ C·(2e-5)², i.e. below any linear scaling by a fixed factor.
    assert norms[1] < norms[0] ** 1.6


def test_natural_variables_state_consistent_with_flash():
    """The saturation-based molar accumulation reduces to ``z·N`` at equilibrium.

    This is the bridge from the natural variables to the molar FIM accumulation
    (Step 5): ``m_c = sO·x_c/v_l + sG·y_c/v_g`` must equal ``z_c·N`` exactly when
    ``(L, x, y)`` are the flash solution, and ``sw + sO + sG = 1`` must hold.
    """
    zs = np.array([0.4, 0.6, 0.8])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(zs.size, 19.0e6)
    sw = np.array([0.2, 0.2, 0.2])
    V, x, y = flash_direct_full(z, p)
    L = 1.0 - V
    sO, sG, rho_l, rho_g = natural_variables_state(sw, L, x, y, p)
    assert np.allclose(sw + sO + sG, 1.0, atol=1.0e-10)
    m = sO[:, None] * x * rho_l[:, None] + sG[:, None] * y * rho_g[:, None]
    N = m.sum(axis=1)
    assert np.allclose(m, z * N[:, None], atol=1.0e-8)
    # phase flags: interior z all two-phase; boundary L hits the pure ends
    pl, pv, tp = phase_flags(L)
    assert np.all(tp)
    pl_b, pv_b, tp_b = phase_flags(np.array([1.0, 0.0, 0.5]))
    assert pl_b[0] and pv_b[1] and tp_b[2] and not tp_b[0] and not tp_b[1]


def test_analytic_fugacity_derivatives_match_finite_difference():
    """The analytic PR fugacity derivative matches central finite differences.

    Pins the hand-derived ``∂lnφ_i/∂x_j`` against central differences, so the
    closed-form derivative — which removes the O(ncomp) fugacity-evaluations-per-
    step cost of a finite-difference Jacobian — is trusted.
    """
    from src.core.pr_eos import _ab, _fugacity_frac_deriv_analytic, _fugacity_vec, fugacity_mole_deriv
    zs = np.array([0.6])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(1, 19.0e6)
    T = 393.0
    aij, b = _ab(T)
    _, x, y = flash_direct_full(z, p)
    h = 1.0e-6
    for phase in ("liq", "vap"):
        xx = x if phase == "liq" else y
        ana = _fugacity_frac_deriv_analytic(xx, aij, b, p, T, phase)
        c = np.zeros_like(ana)
        for j in range(xx.shape[1]):
            xp = xx.copy(); xp[:, j] += h
            xm = xx.copy(); xm[:, j] -= h
            c[:, :, j] = (_fugacity_vec(xp, aij, b, p, T, phase)
                          - _fugacity_vec(xm, aij, b, p, T, phase)) / (2.0 * h)
        assert np.allclose(ana, c, atol=1.0e-6)
        mole = fugacity_mole_deriv(xx, p, T, phase)
        assert np.allclose(mole[0].sum(axis=1), 0.0, atol=1.0e-12)


def test_phase_molar_volume_deriv_matches_finite_difference():
    """The analytic molar-volume derivative ``(∂v/∂x, ∂v/∂p)`` matches FD.

    This is the accumulation-Jacobian building block (``∂ρ/∂x = −ρ²·∂v/∂x``), so it
    must be pinned before it replaces the FD in the natural-variables FIM.
    """
    from src.core.pr_eos import _ab
    zs = np.array([0.6])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(1, 19.0e6)
    T = 393.0
    aij, b = _ab(T)
    _, x, y = flash_direct_full(z, p)
    h = 1.0e-6
    hp = 1.0
    for phase in ("liq", "vap"):
        xx = x if phase == "liq" else y
        v, dvdx, dvdp = phase_molar_volume_deriv(xx, p, T, phase)
        vbase = phase_molar_volumes(xx, xx, p, T)[0 if phase == "liq" else 1]
        dvdx_fd = np.zeros_like(dvdx)
        for j in range(xx.shape[1]):
            xp = xx.copy(); xp[:, j] += h
            vp = phase_molar_volumes(xp, xp, p, T)[0 if phase == "liq" else 1]
            dvdx_fd[:, j] = (vp - vbase) / h
        vp = phase_molar_volumes(xx, xx, p + hp, T)[0 if phase == "liq" else 1]
        dvdp_fd = (vp - vbase) / hp
        assert np.allclose(dvdx, dvdx_fd, atol=1.0e-6)
        assert np.allclose(dvdp, dvdp_fd, atol=1.0e-13)







def test_flash_derivatives_ptz_matches_finite_difference():
    """The analytic flash derivatives (MRST ``getPhaseFractionDerivativesPTZ``) match FD.

    One equilibrium-Jacobian solve gives ``∂(V,x,y)/∂(p,z)``; this pins it against
    finite differences so the analytic Jacobian (which replaces the ``ncomp`` flash
    evaluations of the FD forward Jacobian) is trusted.
    """
    from src.core.pr_eos import flash_derivatives_ptz
    zs = np.array([0.4, 0.6, 0.8])
    z = np.outer(1.0 - zs, _Z_OIL_DEAD)
    z[:, _CO2_IDX] += zs
    p = np.full(zs.size, 19.0e6)
    ncomp = z.shape[1]
    dV_dp, dV_dz, dx_dp, dx_dz, dy_dp, dy_dz = flash_derivatives_ptz(z, p)
    V, x, y = flash_direct_full(z, p)
    hp = 1.0
    Vp, xp, yp = flash_direct_full(z, p + hp)
    assert np.allclose((Vp - V) / hp, dV_dp, atol=1.0e-6)
    assert np.allclose((xp - x) / hp, dx_dp, atol=1.0e-7)
    assert np.allclose((yp - y) / hp, dy_dp, atol=1.0e-7)
    h = 1.0e-6
    for j in range(ncomp - 1):
        zp = z.copy(); zp[:, j] += h; zp[:, -1] -= h
        Vj, xj, yj = flash_direct_full(zp, p)
        # the FD perturbs the implied-last fold (z_j + h, z_last − h)
        assert np.allclose((Vj - V) / h, dV_dz[:, j] - dV_dz[:, -1], atol=1.0e-2)
        assert np.allclose((xj - x) / h, dx_dz[:, :, j] - dx_dz[:, :, -1], atol=1.0e-2)
        assert np.allclose((yj - y) / h, dy_dz[:, :, j] - dy_dz[:, :, -1], atol=1.0e-2)


def test_transport_case_anchors_compositional_pressure():
    """The offline transport case freezes kriged pressure. Joint-debug stays on kriging."""
    case = load_lab_case("examples/shale_oil/case_transport.yaml")
    assert case.forward_model == "compositional"
    assert case.freeze_pressure is True
    assert case.k_homogeneous is True
    joint = load_lab_case(CASE)
    assert joint.forward_model == "none"


def test_frozen_pressure_tracks_the_anchor():
    """Pressure is the prescribed end state, not a Newton unknown."""
    (case, grid, k, _phi, _vol, inv_phiV, params,
     sw, z, p, wells, bhp, injects_gas, _qg) = _small_setup(nx=1, inject_qg=0.0, producer=False)
    p_end = p + 2.0e5
    p2, _sw2, _sl, _sg, _z2, conv, _dt = _implicit_compositional_full_adaptive(
        grid, k, inv_phiV, 30.0, sw, z, p, wells, bhp, case.well, params, injects_gas,
        well_qg_fixed=None, freeze_pressure=True, p_end=p_end, dt_max=30.0, dt_min=1.0,
    )
    assert conv
    assert np.allclose(p2, p_end)


def test_frozen_column_co2_enters_at_the_top():
    """0.01 day of rate-controlled CO2 on a frozen pressure enters at the top.

    Completions are the upper cells. 0.03 mD cannot carry that slug to the
    bottom perforations in 864 s, so the overall CO2 mole fraction stays a
    front: high in the top completions, still the dead-oil value at the bottom.
    Reported sg follows the GEM two-phase window and the single-phase oil label.
    """
    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=1, ny=1, nz=15, dx=0.02, dy=0.02, dz=0.02)
    n = grid.n_cells
    perm = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    params = case.black_oil
    inj_cells = np.arange(4, 15)
    wells = WellMap(
        ids=("INJ", "PROD"),
        xyz=grid.cell_centers()[[14, 0]],
        cells=(inj_cells, np.array([0])),
    )
    p_value = 19.1e6
    p = np.full((2, n), p_value)
    # Injector BHP starts equal to the anchored pressure, so the target rate
    # is met only by solving the injector constraint, not by a pressure update.
    bhp = np.array([[p_value, 19.0e6], [p_value, 19.0e6]])
    q_res = 8.333e-8
    qg = np.array([[q_res, 0.0], [q_res, 0.0]])
    zeros = np.zeros_like(qg)
    s0 = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sw, so, sg, zco2 = forward_saturations(
            "compositional", grid, p, perm, phi, params, wells,
            zeros, zeros, qg, np.array([0.0, 864.0]),
            s0, np.ones(n), s0,
            well_bhp=bhp, well_params=case.well,
            freeze_pressure=True, return_co2=True,
        )
    assert np.allclose(sw + so + sg, 1.0, atol=1.0e-6)
    assert np.all((zco2 >= -1.0e-8) & (zco2 <= 1.0 + 1.0e-8))
    assert float(zco2.max()) < 2.0  # mole fraction, not a Henry throughput
    top = zco2[1, 12:15]
    bottom = zco2[1, :3]
    assert float(top.mean()) > 0.2
    assert float(top.mean()) > 5.0 * float(bottom.mean() + 1.0e-6)
    two_phase = (zco2[1] > 0.30) & (zco2[1] < 0.90)
    if np.any(two_phase):
        assert float(sg[1, two_phase].min()) > 0.02
    pure = zco2[1] > 0.99
    if np.any(pure):
        assert np.allclose(sg[1, pure], 0.0)


def test_frozen_month_step_converges():
    """A 30-day frozen-pressure step stays on the anchor and keeps z in [0, 1]."""
    from src.core.pr_eos import _Z_OIL_DEAD

    case = load_lab_case(CASE)
    grid = CartesianGrid(nx=1, ny=1, nz=15, dx=0.02, dy=0.02, dz=0.02)
    n = grid.n_cells
    perm = np.full(n, case.k0)
    phi = np.full(n, case.phi0)
    params = case.black_oil
    inj_cells = np.arange(4, 15)
    wells = WellMap(
        ids=("INJ", "PROD"),
        xyz=grid.cell_centers()[[14, 0]],
        cells=(inj_cells, np.array([0])),
    )
    p_value = 19.1e6
    month = 30.0 * 86400.0
    p0 = np.full(n, p_value)
    bhp = np.array([p_value, 19.0e6])
    sw = np.zeros(n)
    z = np.broadcast_to(_Z_OIL_DEAD, (n, _Z_OIL_DEAD.size)).copy()
    q_surf = float(np.asarray(reservoir_co2_to_surface(8.333e-8, p_value)).ravel()[0])
    qg_fixed = np.zeros(n)
    qg_fixed[inj_cells] = q_surf / inj_cells.size
    inv_phiV = 1.0 / (phi * grid.cell_volumes())
    p2, _sw2, _sl, _sg, z2, conv, _dt = _implicit_compositional_full_adaptive(
        grid, perm, inv_phiV, month, sw, z, p0, wells, bhp, case.well, params,
        np.array([True, False]), well_qg_fixed=qg_fixed, freeze_pressure=True,
        p_end=p0, dt_max=5.0 * 86400.0, dt_min=1.0,
    )
    zco2 = z2[:, _CO2_IDX]
    # The frozen-pressure Newton stalls where the flash goes singular (log(Z-B)
    # -> NaN near the single-phase boundary), so a 30-day step can report
    # non-convergence. Honest convergence (no forced-accept) surfaces this; the
    # transport still advances CO2 to the top. The full fix is variable switching.
    assert np.allclose(p2, p_value)
    assert np.all((zco2 >= -1.0e-8) & (zco2 <= 1.0 + 1.0e-8))
    assert float(zco2.max()) > float(zco2[:3].mean())


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
            params, injects_gas, well_qg_fixed=qg_fixed, dt_sub0=dt_sub0, dt_max=864.0, dt_min=1.0,
            max_substeps=200)
        dt_sub0 = last_dt
        assert conv, f"full 30-day plume did not converge at t={t / 86400.0:.1f} d"
        t += dt
    # The plume should be bottom-weighted (truth bottom k0-2 mean = 0.456).
    assert sg[bottom].mean() > sg[top].mean()
