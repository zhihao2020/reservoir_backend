"""Sequential black-oil helpers: well mixture, CNV, Newton, timestep."""

import numpy as np

from reservoir_backend.solver.seqtools import (
    NewtonRelaxation,
    cell_status_vo,
    cnv_mb,
    compute_flash_blackoil,
    cross_flow_mixture,
    hybrid_upwind_flags,
    iteration_count_timestep,
    limit_update_abs,
    state_change_timestep,
    multiphase_upwind_indices,
    outer_converged,
    saturation_increment,
    volume_discrepancy,
)


def test_cross_flow_mixture_mixes_inflow_with_topside() -> None:
    # One well, two perfs: cell 0 produces water, cell 1 injects.
    flux = np.array(
        [
            [-2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    compi = np.array([[0.0, 1.0, 0.0]])
    out = cross_flow_mixture(flux, compi, np.array([0, 0]), 1)
    # Inflow is all water; net injection is 0 so mixture is water.
    assert out[0, 0] > 0.9
    assert out[0, 1] < 0.1


def test_cross_flow_mixture_keeps_topside_without_inflow() -> None:
    flux = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    compi = np.array([[0.2, 0.8, 0.0]])
    out = cross_flow_mixture(flux, compi, np.array([0, 0]), 1)
    assert np.allclose(out, compi)


def test_cnv_mb_zero_residual_converges() -> None:
    n = 8
    pv = np.full(n, 1.0)
    r = np.zeros(n)
    b = np.ones(n)
    cnv, mb, ok = cnv_mb([r, r], pv, [b, b], 10.0)
    assert ok
    assert np.allclose(cnv, 0.0)
    assert np.allclose(mb, 0.0)


def test_cnv_mb_flags_local_mass_error() -> None:
    n = 4
    pv = np.full(n, 1.0)
    r = np.array([0.0, 0.0, 0.05, 0.0])
    b = np.ones(n)
    cnv, mb, ok = cnv_mb([r], pv, [b], 1.0, tol_cnv=1.0e-3, tol_mb=1.0e-7)
    assert cnv[0] > 1.0e-3
    assert not ok
    assert mb[0] > 0.0


def test_limit_update_abs_scales_peak() -> None:
    dx = np.array([0.5, -0.1, 0.0])
    out = limit_update_abs(dx, 0.20)
    assert float(np.max(np.abs(out))) == 0.20
    assert np.sign(out[0]) == 1.0
    small = np.array([0.01, -0.02])
    assert np.allclose(limit_update_abs(small, 0.20), small)


def test_newton_relaxation_dampens_on_oscillation() -> None:
    nls = NewtonRelaxation()
    hist = np.array(
        [
            [1.0, 0.8],
            [0.4, 0.9],
            [0.9, 0.4],
        ]
    )
    w = nls.update(hist, np.array([False, False]))
    assert w < 1.0
    dx = np.array([0.2, -0.2])
    damped = nls.apply(dx)
    assert float(np.max(np.abs(damped))) < float(np.max(np.abs(dx)))


def test_critical_point_chop_stops_at_band() -> None:
    from reservoir_backend.solver.seqtools import critical_point_chop

    x0 = np.array([0.10, 0.25, 0.30])
    x1 = np.array([0.30, 0.10, 0.31])
    out = critical_point_chop(x0, x1, 0.20, eps=1.0e-2)
    assert float(out[0]) < 0.20
    assert float(out[1]) > 0.10
    assert float(out[2]) == 0.31


def test_state_change_timestep_shrinks_on_large_ds() -> None:
    grown = state_change_timestep(10.0, 0.05, 0.05, target_ds=0.15, target_dp_rel=0.20)
    assert grown > 10.0
    shrunk = state_change_timestep(10.0, 0.40, 0.05, target_ds=0.15, target_dp_rel=0.20)
    assert shrunk < 10.0
    clamped = state_change_timestep(10.0, 1.0, 1.0, target_ds=0.15, max_rel=2.0, min_rel=0.5)
    assert clamped == 5.0


def test_iteration_count_grows_when_easy() -> None:
    dt = iteration_count_timestep(10.0, 1, dt_max=100.0)
    assert dt > 10.0
    hard = iteration_count_timestep(10.0, 12, dt_max=100.0)
    assert hard < 10.0
    two = iteration_count_timestep(10.0, 4, dt0=8.0, its0=6, dt_max=100.0)
    assert two > 0.0


def test_cell_status_vo_disgas_flags() -> None:
    so = np.array([0.7, 0.0, 0.5])
    sw = np.array([0.3, 0.2, 0.3])
    sg = np.array([0.0, 0.8, 0.2])
    st1, st2, st3 = cell_status_vo(so, sw, sg, disgas=True, vapoil=False)
    assert bool(st1[0])
    assert not bool(st1[1])
    assert bool(st3[2])
    assert not bool(np.any(st2))


def test_compute_flash_chops_negative_gas() -> None:
    sw = np.array([0.30])
    so = np.array([0.75])
    sg = np.array([-0.05])
    rs = np.array([10.0])
    rs_sat = np.array([20.0])
    sw2, so2, sg2, rs2, status = compute_flash_blackoil(
        sw,
        so,
        sg,
        rs,
        rs_sat,
        np.array([0.30]),
        np.array([0.70]),
        np.array([0.00]),
        np.array([10.0]),
        np.array([20.0]),
        disgas=True,
    )
    assert float(sg2[0]) == 0.0
    assert float(sw2[0] + so2[0] + sg2[0]) == 1.0 or abs(float(sw2[0] + so2[0] + sg2[0]) - 1.0) < 1.0e-12
    assert float(rs2[0]) <= 20.0 + 1.0e-12
    assert int(status[0]) >= 1


def test_sequential_phase_fluxes_potential_sums_to_vt() -> None:
    from reservoir_backend.solver.seqtools import sequential_phase_fluxes

    vt = np.array([2.0, -1.0])
    t = np.ones(2)
    pot = np.array([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
    mob = np.ones((2, 3))
    q = sequential_phase_fluxes(vt, t, pot, mob, mob, upwind="potential")
    assert q.shape == (2, 3)
    assert np.allclose(np.sum(q, axis=1), vt)
    qh = sequential_phase_fluxes(vt, t, pot, mob, mob, upwind="hybrid")
    assert np.allclose(np.sum(qh, axis=1), vt)


def test_sequential_gravity_face_is_segregation() -> None:
    from reservoir_backend.solver.seqtools import sequential_gravity_face

    t = np.ones(2)
    pot = np.array([[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]])
    mob = np.ones((2, 3))
    q = sequential_gravity_face(t, pot, mob, mob)
    assert q.shape == (2, 3)
    assert float(np.max(np.abs(q[1]))) < 1.0e-15
    assert float(np.max(np.abs(q[0]))) > 0.0
    assert abs(float(np.sum(q[0]))) < 1.0e-12


def test_hybrid_and_brenie_upwind() -> None:
    flag_v, flag_g = hybrid_upwind_flags(np.array([1.0, -2.0]), nph=3)
    assert bool(flag_v[0, 0])
    assert not bool(flag_v[1, 0])
    assert flag_g.shape == (2, 3)
    pot = np.array([[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]])
    mob_l = np.ones((2, 3))
    mob_r = np.ones((2, 3))
    up = multiphase_upwind_indices(pot, np.array([0.0, 0.0]), np.ones(2), mob_l, mob_r)
    assert up.shape == (2, 3)
    assert up.dtype == bool


def test_outer_convergence_volume_and_increment() -> None:
    sw = np.array([0.2, 0.3])
    sg = np.array([0.1, 0.1])
    assert volume_discrepancy(sw, sg) < 1.0e-15
    assert saturation_increment(sw, sw, sg, sg) == 0.0
    assert outer_converged(sw, sw, sg, sg)
    assert not outer_converged(sw, sw + 0.05, sg, sg)
