"""Natural-variables compositional FIM (MRST ``NaturalVariablesCompositionalModel``).

Primary variables ``(p, sw, sO, sG, x, y)`` replace ``(p, sw, z)`` + flash, and the
fugacity equality ``ln(y·φᵛ) − ln(x·φˡ) = 0`` *replaces* the flash. Single-phase
cells switch to a *reduced* variable set — pure liquid ``(p, sw, x)``, pure vapor
``(p, sw, y)`` — so the absent phase's variables are removed from the system
rather than weakly coupled (MRST ``ReducedLinearizedSystem``). The global Jacobian
is assembled over only the active variables/equations via a per-cell mask.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.pr_eos import (
    _CO2_IDX,
    _V_CO2_STD,
    _ab,
    _fugacity_frac_deriv_analytic,
    _fugacity_vec,
    co2_molar_volume,
    flash_direct_full,
    natural_variables_state,
    phase_mass_densities,
    phase_molar_volume_deriv,
    phase_molar_volumes,
    phase_stability_test,
)
from .forward import (
    _EOS_NEWTON_RTOL,
    _eos_gas_head,
    _injector_inv_molar_volume,
    _mobility_derivs_full,
    _mobility_divergence_matrix,
    _peaceman_well_data,
    _phase_divergence_matrix,
    _split_full_N,
    _upwind_tpfa_matrix_vec,
    phase_mobilities,
)

_T = 393.0
_DS_MAX = 0.2   # max saturation update per Newton step (MRST dsMaxAbs)
_DX_MAX = 0.1   # max composition update per Newton step (MRST dzMaxAbs)
_DP_MAX = 2.0e6 # max pressure update per Newton step (MRST dpMaxAbs), Pa
_NV_DEBUG = False
_AB_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _eos_ab():
    global _AB_CACHE
    if _AB_CACHE is None:
        _AB_CACHE = _ab(_T)
    return _AB_CACHE


def natural_variables_step(
    grid,
    permeability: NDArray[np.float64],
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    z0: NDArray[np.float64],
    p0: NDArray[np.float64],
    wells,
    well_bhp: NDArray[np.float64],
    well_params,
    params,
    injects_gas: NDArray[np.bool_],
    *,
    well_qg_fixed: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    tol: float = _EOS_NEWTON_RTOL,
) -> tuple[NDArray[np.float64], ...]:
    """One fully-implicit natural-variables step → ``(p, sw, sl, sg, z, converged)``."""
    ncomp = z0.shape[1]
    n = grid.n_cells
    sw = np.asarray(sw0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    accum = 1.0 / (dt * inv_phiV)
    aij, b = _eos_ab()
    k1 = (ncomp - 1) * n
    n_full = 4 * n + 2 * k1  # p, sw, sO, sG, x_ind, y_ind

    cells, wi, bhp_w, is_inj = _peaceman_well_data(
        grid, permeability, wells, well_bhp, well_params, injects_gas)
    rate_controlled = well_qg_fixed is not None
    qg_fixed = np.asarray(well_qg_fixed, dtype=float).ravel() if rate_controlled else np.zeros(n)
    qg_target = float(qg_fixed.sum())
    bhp_full = np.zeros(n)
    for idx in range(cells.size):
        bhp_full[cells[idx]] = bhp_w[idx]
    inj_cell = np.zeros(n, dtype=bool)
    for idx in range(cells.size):
        inj_cell[cells[idx]] = is_inj[idx]
    bhp_inj = float(np.mean(bhp_w[is_inj])) if is_inj.any() else 0.0
    inj_cells = cells[is_inj]
    eos_gravity = params.gravity == "eos"
    has_rate = rate_controlled and is_inj.any()

    # Initial full state from the flash of the incoming overall state.
    V0, x0, y0 = flash_direct_full(z0, p0)
    x0 = np.where(x0 > 0.0, x0, y0)
    y0 = np.where(y0 > 0.0, y0, x0)
    x0 = x0 / np.maximum(x0.sum(axis=1, keepdims=True), 1.0e-30)
    y0 = y0 / np.maximum(y0.sum(axis=1, keepdims=True), 1.0e-30)
    sO0, sG0, _, _ = natural_variables_state(sw0, 1.0 - V0, x0, y0, p0)
    N0 = _split_full_N(sw0, z0, p0, params)
    m_c0 = z0 * N0[:, None]

    sO = np.clip(sO0.copy(), 0.0, 1.0 - sw0)
    sG = np.clip(sG0.copy(), 0.0, 1.0 - sw0)
    x_ind = x0[:, : ncomp - 1].copy()
    y_ind = y0[:, : ncomp - 1].copy()

    def pack_full(p_, sw_, sO_, sG_, x_ind_, y_ind_):
        return np.concatenate([p_, sw_, sO_, sG_, x_ind_.ravel(), y_ind_.ravel()])

    def unpack_full(u_):
        p_ = u_[:n]
        sw_ = u_[n:2 * n]
        sO_ = u_[2 * n:3 * n]
        sG_ = u_[3 * n:4 * n]
        # x_ind/y_ind are stored cell-major by ``pack_full`` (``.ravel()``), so the
        # inverse is ``(n, ncomp-1)`` — ``(ncomp-1, n).T`` would scramble the
        # composition across cells (only equal when ``n == 1``).
        x_ind_ = u_[4 * n:4 * n + k1].reshape(n, ncomp - 1)
        y_ind_ = u_[4 * n + k1:4 * n + 2 * k1].reshape(n, ncomp - 1)
        return p_, sw_, sO_, sG_, x_ind_, y_ind_

    def flags_from_state(sO_, sG_):
        tol = 1.0e-12
        pl = sG_ <= tol
        pv = (sO_ <= tol) & ~pl  # mutually exclusive with pure liquid
        return pl, pv, ~(pl | pv)

    def var_mask(pl, pv, two):
        m = np.zeros(n_full, dtype=bool)
        m[:2 * n] = True                                    # p, sw
        m[2 * n:3 * n] = two                                 # sO (two-phase only)
        m[3 * n:4 * n] = two                                 # sG (two-phase only)
        m[4 * n:4 * n + k1] = np.repeat(~pv, ncomp - 1)      # x (not pure vapor)
        m[4 * n + k1:4 * n + 2 * k1] = np.repeat(~pl, ncomp - 1)  # y (not pure liquid)
        return m

    def eq_mask(two):
        m = np.zeros(n_full, dtype=bool)
        m[:15 * n] = True                                    # component + water
        m[15 * n:29 * n] = np.repeat(two, ncomp)             # fugacity
        m[29 * n:30 * n] = two                               # closure
        return m

    def keep_var_mask(pl, pv, two):
        """Variables kept after the Schur reduction (MRST ``keepNum``).

        Keeps ``p, sw``, the oil saturation ``sO`` (two-phase only), and liquid
        compositions ``x[1..12]``; eliminates ``sG``, ``x[0]`` and the whole vapor
        composition ``y``. Moving one liquid component (``x[0]``) into the eliminated
        block is what makes the fugacity+closure block non-singular. Single-phase cells
        keep their full composition (``x`` for pure liquid, ``y`` for pure vapor) — they
        have no fugacity/closure to eliminate, so their 15×15 block stays intact.
        """
        m = np.zeros(n_full, dtype=bool)
        m[:n] = True                              # p
        m[n:2 * n] = True                         # sw
        m[2 * n:3 * n] = two                      # sO (two-phase only)
        comp = np.arange(k1) % (ncomp - 1)        # component index per x slot
        cell = np.arange(k1) // (ncomp - 1)       # cell index per x slot
        m[4 * n:4 * n + k1] = ((comp >= 1) & two[cell]) | pl[cell]
        m[4 * n + k1:4 * n + 2 * k1] = pv[cell]   # y kept only for pure vapor
        return m

    def keep_eq_mask():
        """Equations kept after the Schur reduction: component balance + water."""
        m = np.zeros(n_full, dtype=bool)
        m[:15 * n] = True
        return m

    def fill_inactive(u_full_, pl, pv):
        p_, sw_, sO_, sG_, x_ind_, y_ind_ = unpack_full(u_full_)
        sG_[pl] = 0.0
        sO_[pl] = 1.0 - sw_[pl]
        y_ind_[pl] = x_ind_[pl]
        sO_[pv] = 0.0
        sG_[pv] = 1.0 - sw_[pv]
        x_ind_[pv] = y_ind_[pv]
        return pack_full(p_, sw_, sO_, sG_, x_ind_, y_ind_)

    def full_comps(x_ind_, y_ind_):
        x_ = np.concatenate([x_ind_, 1.0 - x_ind_.sum(axis=1, keepdims=True)], axis=1)
        y_ = np.concatenate([y_ind_, 1.0 - y_ind_.sum(axis=1, keepdims=True)], axis=1)
        return x_, y_

    def flash_phases(p_, sw_, sO_, sG_, x_, y_):
        """Phase transition (MRST ``flashPhases``).

        Single-phase cells run the Michelsen stability test: an unstable pure-liquid
        cell gains an ``saturationEpsilon`` of the incipient vapor, an unstable
        pure-vapor cell an ``saturationEpsilon`` of the incipient liquid. Two-phase
        cells where a phase vanished (``sO<=0``/``sG<=0``) collapse to single-phase.
        ``x_``/``y_`` are full ``(n, ncomp)`` compositions; returns updated
        ``(sO, sG, x, y)``.
        """
        pl, pv, two = flags_from_state(sO_, sG_)
        sO_n = sO_.copy()
        sG_n = sG_.copy()
        x_n = x_.copy()
        y_n = y_.copy()
        smax = 1.0 - sw_
        eps = 1.0e-6
        # pure liquid -> two-phase: insert incipient vapor
        if pl.any():
            stable, _, y_inc = phase_stability_test(x_[pl], p_[pl], aij=aij, b=b)
            c = pl.copy()
            c[pl] = ~stable
            if c.any():
                sG_n[c] = eps * smax[c]
                sO_n[c] = smax[c] - sG_n[c]
                y_n[c] = y_inc[~stable]
        # pure vapor -> two-phase: insert incipient liquid
        if pv.any():
            stable, x_inc, _ = phase_stability_test(y_[pv], p_[pv], aij=aij, b=b)
            c = pv.copy()
            c[pv] = ~stable
            if c.any():
                sO_n[c] = eps * smax[c]
                sG_n[c] = smax[c] - sO_n[c]
                x_n[c] = x_inc[~stable]
        # two-phase -> single-phase (a phase vanished)
        to_liq = two & (sG_n <= 0.0)
        to_vap = two & (sO_n <= 0.0)
        if to_liq.any():
            sO_n[to_liq] = smax[to_liq]
            sG_n[to_liq] = 0.0
            y_n[to_liq] = x_n[to_liq]
        if to_vap.any():
            sG_n[to_vap] = smax[to_vap]
            sO_n[to_vap] = 0.0
            x_n[to_vap] = y_n[to_vap]
        return sO_n, sG_n, x_n, y_n

    def state_qty(u_full_, bhp_inj_):
        """(rho_l, rho_g, lam_w, lam_l, lam_g, lam_t, qg_inj) from the full state."""
        p_, sw_, sO_, sG_, x_ind_, y_ind_ = unpack_full(u_full_)
        x_, y_ = full_comps(x_ind_, y_ind_)
        sO_ = np.clip(sO_, 0.0, 1.0 - sw_)
        sG_ = np.clip(sG_, 0.0, 1.0 - sw_)
        v_l_, v_g_ = phase_molar_volumes(x_, y_, p_, _T)
        rho_l_ = np.nan_to_num(1.0 / np.maximum(v_l_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        rho_g_ = np.nan_to_num(1.0 / np.maximum(v_g_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        lam_w_, lam_l_, lam_g_ = phase_mobilities(sw_, sO_, sG_, params)
        lam_t_ = lam_w_ + lam_l_ + lam_g_
        qg_ = np.zeros(n)
        for idx in range(cells.size):
            if is_inj[idx]:
                c = cells[idx]
                v_inj = float(np.asarray(co2_molar_volume(np.array([p_[c]]), _T)).ravel()[0])
                qg_[c] += wi[idx] * lam_t_[c] * (bhp_inj_ - p_[c]) / np.maximum(v_inj, 1.0e-30)
        return rho_l_, rho_g_, lam_w_, lam_l_, lam_g_, lam_t_, qg_

    def residual_full(u_full_, bhp_inj_):
        p_, sw_, sO_, sG_, x_ind_, y_ind_ = unpack_full(u_full_)
        x_, y_ = full_comps(x_ind_, y_ind_)
        sO_ = np.clip(sO_, 0.0, 1.0 - sw_)
        sG_ = np.clip(sG_, 0.0, 1.0 - sw_)
        v_l_, v_g_ = phase_molar_volumes(x_, y_, p_, _T)
        rho_l_ = np.nan_to_num(1.0 / np.maximum(v_l_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        rho_g_ = np.nan_to_num(1.0 / np.maximum(v_g_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        m_c_ = sO_[:, None] * x_ * rho_l_[:, None] + sG_[:, None] * y_ * rho_g_[:, None]
        lam_w_, lam_l_, lam_g_ = phase_mobilities(sw_, sO_, sG_, params)
        lam_t_ = lam_w_ + lam_l_ + lam_g_
        if eos_gravity:
            rho_l_m, rho_g_m = phase_mass_densities(x_, y_, v_l_, v_g_)
            head_ = _eos_gas_head(grid, sO_, sG_, rho_l_m, rho_g_m, params.rho_g)
        else:
            head_ = params.rho_g
        qw_ = np.zeros(n)
        qo_ = np.zeros(n)
        qg_ = np.zeros(n)
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and has_rate) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            if is_inj[idx]:
                v_inj = float(np.asarray(co2_molar_volume(np.array([p_[c]]), _T)).ravel()[0])
                qg_[c] += w * lam_t_[c] * dp / np.maximum(v_inj, 1.0e-30)
            else:
                qw_[c] += w * lam_w_[c] * dp
                qo_[c] += w * lam_l_[c] * dp * rho_l_[c]
                qg_[c] += w * lam_g_[c] * dp * rho_g_[c]
        A_ = _mobility_divergence_matrix(grid, permeability, p_)
        A_g_ = _phase_divergence_matrix(grid, permeability, p_, head_)
        rock_ = params.ct * (p_ - p0)
        rc_ = []
        for c in range(ncomp):
            q_c_ = np.where(inj_cell, (qg_ if c == _CO2_IDX else 0.0),
                            y_[:, c] * qg_ + x_[:, c] * qo_)
            r_c_ = (accum * ((m_c_[:, c] - m_c0[:, c]) + rock_ * m_c_[:, c]) - q_c_
                    + A_g_ @ (y_[:, c] * lam_g_ * rho_g_)
                    + A_ @ (x_[:, c] * lam_l_ * rho_l_))
            rc_.append(r_c_)
        r_w_ = accum * ((sw_ - sw0) + rock_ * sw_) - qw_ + A_ @ lam_w_
        fL_ = _fugacity_vec(x_, aij, b, p_, _T, "liq")
        fV_ = _fugacity_vec(y_, aij, b, p_, _T, "vap")
        # Linear fugacity (MRST ``f_L - f_V``): x·φL − y·φV — avoids the log form's
        # 1/y blow-up for the tiny vapor components.
        r_fug_ = x_ * np.exp(fL_) - y_ * np.exp(fV_)
        r_close_ = sw_ + sO_ + sG_ - 1.0
        return np.concatenate(rc_ + [r_w_, r_fug_.ravel(), r_close_])

    def rate_residual(u_full_, bhp_inj_):
        qg_ = state_qty(u_full_, bhp_inj_)[6]
        return params.bg * (_V_CO2_STD * float(np.sum(qg_[inj_cells])) - qg_target)

    def sparse_jacobian_full(u_full_, bhp_inj_):
        """Full analytic sparse block Jacobian (MRST ``equationsNaturalVariables``).

        Assembles every block of ``∂residual_full/∂u_full`` exactly: flux-p via the
        frozen upwind Laplacian + density p-derivative, accumulation/fugacity/closure
        block-diagonal, flux-s/x/y via ``A·diags`` with the mobility (3×3) and
        density-composition derivatives, and the well terms via the Peaceman product
        rule. The implied single-phase variables (``y=x``/``sO=1−sw`` for pure liquid,
        ``x=y``/``sG=1−sw`` for pure vapor) are folded into the kept columns (MRST's
        AD chain rule) before the ``em``/``vm`` masking. Returns an
        ``(n_full + has_rate, n_full + has_rate)`` CSR; with ``has_rate`` the last
        row/column are the injector surface-rate residual and ``bhp_inj``.
        """
        from scipy.sparse import coo_matrix, identity
        p_, sw_, sO_raw, sG_raw, x_ind_, y_ind_ = unpack_full(u_full_)
        pl, pv, _ = flags_from_state(sO_raw, sG_raw)
        x_, y_ = full_comps(x_ind_, y_ind_)
        sO_ = np.clip(sO_raw, 0.0, 1.0 - sw_)
        sG_ = np.clip(sG_raw, 0.0, 1.0 - sw_)
        v_l_, v_g_ = phase_molar_volumes(x_, y_, p_, _T)
        rho_l_ = np.nan_to_num(1.0 / np.maximum(v_l_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        rho_g_ = np.nan_to_num(1.0 / np.maximum(v_g_, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
        lam_w_, lam_l_, lam_g_, dlam = _mobility_derivs_full(sw_, sO_, sG_, params)
        lam_t_ = lam_w_ + lam_l_ + lam_g_
        if eos_gravity:
            rho_l_m, rho_g_m = phase_mass_densities(x_, y_, v_l_, v_g_)
            head_ = _eos_gas_head(grid, sO_, sG_, rho_l_m, rho_g_m, params.rho_g)
        else:
            head_ = params.rho_g
        A_ = _mobility_divergence_matrix(grid, permeability, p_)
        A_g_ = _phase_divergence_matrix(grid, permeability, p_, head_)
        _, dvdx_l, dvdp_l = phase_molar_volume_deriv(x_, p_, _T, "liq", aij=aij, b=b)
        _, dvdx_g, dvdp_g = phase_molar_volume_deriv(y_, p_, _T, "vap", aij=aij, b=b)
        drhol_dx = -rho_l_[:, None] ** 2 * dvdx_l      # (n, ncomp)
        drhog_dy = -rho_g_[:, None] ** 2 * dvdx_g
        drhol_dp = -rho_l_ ** 2 * dvdp_l
        drhog_dp = -rho_g_ ** 2 * dvdp_g
        dL_ = _fugacity_frac_deriv_analytic(x_, aij, b, p_, _T, "liq")
        dV_ = _fugacity_frac_deriv_analytic(y_, aij, b, p_, _T, "vap")
        phiL_ = np.exp(_fugacity_vec(x_, aij, b, p_, _T, "liq"))
        phiV_ = np.exp(_fugacity_vec(y_, aij, b, p_, _T, "vap"))
        dpf = 1.0
        dlnphiL_dp = (_fugacity_vec(x_, aij, b, p_ + dpf, _T, "liq")
                      - _fugacity_vec(x_, aij, b, p_ - dpf, _T, "liq")) / (2.0 * dpf)
        dlnphiV_dp = (_fugacity_vec(y_, aij, b, p_ + dpf, _T, "vap")
                      - _fugacity_vec(y_, aij, b, p_ - dpf, _T, "vap")) / (2.0 * dpf)
        rock1_ = 1.0 + params.ct * (p_ - p0)
        mc_ = sO_[:, None] * x_ * rho_l_[:, None] + sG_[:, None] * y_ * rho_g_[:, None]  # (n, ncomp)
        inv_vi, dinv_vi = _injector_inv_molar_volume(n, inj_cells, p_)
        drhol_ind = drhol_dx[:, : ncomp - 1] - drhol_dx[:, ncomp - 1: ncomp]   # ∂ρl/∂x_ind
        drhog_ind = drhog_dy[:, : ncomp - 1] - drhog_dy[:, ncomp - 1: ncomp]
        cx = 4 * n + np.arange(n) * (ncomp - 1)           # x_ind[j] base column per cell
        cy = 4 * n + k1 + np.arange(n) * (ncomp - 1)      # y_ind[j] base column per cell

        n_aug = n_full + (1 if has_rate else 0)
        rows = []
        cols = []
        vals = []

        def add_diag(r0, c0, d):
            rows.append(r0 + np.arange(n))
            cols.append(c0 + np.arange(n))
            vals.append(np.asarray(d, dtype=float))

        def add_sparse(r0, c0, S):
            S = S.tocoo()
            rows.append(r0 + S.row)
            cols.append(c0 + S.col)
            vals.append(S.data)

        def add_scatter(r0, colmap, S):
            S = S.tocoo()
            rows.append(r0 + S.row)
            cols.append(colmap[S.col])
            vals.append(S.data)

        # --- component balance rows (component-major: row = c*n + cell) ---
        for c in range(ncomp):
            r0 = c * n
            # flux-p (frozen upwind Laplacian + density p-derivative)
            add_sparse(r0, 0,
                       _upwind_tpfa_matrix_vec(grid, permeability, y_[:, c] * lam_g_ * rho_g_, p_, head_)
                       + _upwind_tpfa_matrix_vec(grid, permeability, x_[:, c] * lam_l_ * rho_l_, p_)
                       + A_g_.multiply((y_[:, c] * lam_g_ * drhog_dp)[None, :])
                       + A_.multiply((x_[:, c] * lam_l_ * drhol_dp)[None, :]))
            # accumulation-p / sO / sG (diag)
            dmc_dp = sO_ * x_[:, c] * drhol_dp + sG_ * y_[:, c] * drhog_dp
            add_diag(r0, 0, accum * (rock1_ * dmc_dp + params.ct * mc_[:, c]))
            add_diag(r0, 2 * n, accum * rock1_ * x_[:, c] * rho_l_)
            add_diag(r0, 3 * n, accum * rock1_ * y_[:, c] * rho_g_)
            # flux-s (mobility saturation derivatives): A_g·diag(y_c ρg ∂λg/∂s_k) + A·diag(x_c ρl ∂λl/∂s_k)
            for k, ck in ((0, n), (1, 2 * n), (2, 3 * n)):
                add_sparse(r0, ck,
                           A_g_.multiply((y_[:, c] * rho_g_ * dlam[:, 2, k])[None, :])
                           + A_.multiply((x_[:, c] * rho_l_ * dlam[:, 1, k])[None, :]))
            # accumulation-x/y + flux-x/y (composition columns, cell-major)
            for j in range(ncomp - 1):
                dj = (1.0 if c == j else 0.0) - (1.0 if c == ncomp - 1 else 0.0)
                acc_x = accum * rock1_ * sO_ * (dj * rho_l_ + x_[:, c] * drhol_ind[:, j])
                dmob_x = dj * lam_l_ * rho_l_ + x_[:, c] * lam_l_ * drhol_ind[:, j]
                rows.append(r0 + np.arange(n)); cols.append(cx + j); vals.append(acc_x)
                add_scatter(r0, cx + j, A_.multiply(dmob_x[None, :]))
                acc_y = accum * rock1_ * sG_ * (dj * rho_g_ + y_[:, c] * drhog_ind[:, j])
                dmob_y = dj * lam_g_ * rho_g_ + y_[:, c] * lam_g_ * drhog_ind[:, j]
                rows.append(r0 + np.arange(n)); cols.append(cy + j); vals.append(acc_y)
                add_scatter(r0, cy + j, A_g_.multiply(dmob_y[None, :]))

        # --- well sources and their derivatives (Peaceman product rule) ---
        qo_ = np.zeros(n)
        qg_ = np.zeros(n)
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and has_rate) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            if is_inj[idx]:
                qg_[c] += w * lam_t_[c] * dp * inv_vi[c]
            else:
                qo_[c] += w * lam_l_[c] * dp * rho_l_[c]
                qg_[c] += w * lam_g_[c] * dp * rho_g_[c]
        dq_dp = np.zeros((ncomp, n))
        dq_dsw = np.zeros((ncomp, n))
        dq_dso = np.zeros((ncomp, n))
        dq_dsg = np.zeros((ncomp, n))
        dq_dx = np.zeros((ncomp, n, ncomp - 1))
        dq_dy = np.zeros((ncomp, n, ncomp - 1))
        dqw_dp = np.zeros(n)
        dqw_dsw = np.zeros(n)
        dlamt = dlam[:, 0] + dlam[:, 1] + dlam[:, 2]   # (n, 3): ∂λt/∂(sw,so,sg)
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and has_rate) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            if is_inj[idx]:
                dq_dp[_CO2_IDX, c] += w * (-lam_t_[c] * inv_vi[c] + lam_t_[c] * dp * dinv_vi[c])
                dq_dsw[_CO2_IDX, c] += w * dp * inv_vi[c] * dlamt[c, 0]
                dq_dso[_CO2_IDX, c] += w * dp * inv_vi[c] * dlamt[c, 1]
                dq_dsg[_CO2_IDX, c] += w * dp * inv_vi[c] * dlamt[c, 2]
            else:
                lamw = lam_w_[c]; laml = lam_l_[c]; lamg = lam_g_[c]
                rl = rho_l_[c]; rg = rho_g_[c]
                dqw_dp[c] += -w * lamw
                dqw_dsw[c] += w * dp * dlam[c, 0, 0]
                dqo_dp = -w * laml * rl + w * laml * dp * drhol_dp[c]
                dqo_dsw = w * dp * rl * dlam[c, 1, 0]
                dqo_dso = w * dp * rl * dlam[c, 1, 1]
                dqo_dsg = w * dp * rl * dlam[c, 1, 2]
                dqo_dx = w * laml * dp * drhol_dx[c, :]
                dqg_dp = -w * lamg * rg + w * lamg * dp * drhog_dp[c]
                dqg_dsw = w * dp * rg * dlam[c, 2, 0]
                dqg_dso = w * dp * rg * dlam[c, 2, 1]
                dqg_dsg = w * dp * rg * dlam[c, 2, 2]
                dqg_dy = w * lamg * dp * drhog_dy[c, :]
                for cc in range(ncomp):
                    yc = y_[c, cc]; xc = x_[c, cc]
                    dq_dp[cc, c] += yc * dqg_dp + xc * dqo_dp
                    dq_dsw[cc, c] += yc * dqg_dsw + xc * dqo_dsw
                    dq_dso[cc, c] += yc * dqg_dso + xc * dqo_dso
                    dq_dsg[cc, c] += yc * dqg_dsg + xc * dqo_dsg
                    for j in range(ncomp - 1):
                        dq_dx[cc, c, j] += ((cc == j) - (cc == ncomp - 1)) * qo_[c] + xc * (dqo_dx[j] - dqo_dx[ncomp - 1])
                        dq_dy[cc, c, j] += ((cc == j) - (cc == ncomp - 1)) * qg_[c] + yc * (dqg_dy[j] - dqg_dy[ncomp - 1])
        for c in range(ncomp):
            r0 = c * n
            add_diag(r0, 0, -dq_dp[c])
            add_diag(r0, n, -dq_dsw[c])
            add_diag(r0, 2 * n, -dq_dso[c])
            add_diag(r0, 3 * n, -dq_dsg[c])
            for j in range(ncomp - 1):
                rows.append(r0 + np.arange(n)); cols.append(cx + j); vals.append(-dq_dx[c, :, j])
                rows.append(r0 + np.arange(n)); cols.append(cy + j); vals.append(-dq_dy[c, :, j])

        # --- water row (14n .. 15n) ---
        r0 = 14 * n
        add_sparse(r0, 0, _upwind_tpfa_matrix_vec(grid, permeability, lam_w_, p_))
        add_diag(r0, 0, accum * params.ct * sw_)
        add_diag(r0, n, accum * rock1_)
        add_sparse(r0, n, A_.multiply(dlam[:, 0, 0][None, :]))
        add_diag(r0, 0, -dqw_dp)
        add_diag(r0, n, -dqw_dsw)

        # --- fugacity rows (15n .. 29n, cell-major: row = 15n + cell*ncomp + i) ---
        dfug_dx = phiL_[:, :, None] * (np.eye(ncomp)[None] + x_[:, :, None] * dL_)
        dfug_dy = -phiV_[:, :, None] * (np.eye(ncomp)[None] + y_[:, :, None] * dV_)
        dfug_dx_ind = dfug_dx[:, :, : ncomp - 1] - dfug_dx[:, :, ncomp - 1: ncomp]
        dfug_dy_ind = dfug_dy[:, :, : ncomp - 1] - dfug_dy[:, :, ncomp - 1: ncomp]
        rowf = 15 * n + np.arange(n) * ncomp
        for i in range(ncomp):
            rows.append(rowf + i); cols.append(np.arange(n))
            vals.append(x_[:, i] * phiL_[:, i] * dlnphiL_dp[:, i] - y_[:, i] * phiV_[:, i] * dlnphiV_dp[:, i])
            for j in range(ncomp - 1):
                rows.append(rowf + i); cols.append(cx + j); vals.append(dfug_dx_ind[:, i, j])
                rows.append(rowf + i); cols.append(cy + j); vals.append(dfug_dy_ind[:, i, j])

        # --- closure rows (29n .. 30n) ---
        r0 = 29 * n
        add_diag(r0, n, np.ones(n))
        add_diag(r0, 2 * n, np.ones(n))
        add_diag(r0, 3 * n, np.ones(n))

        if has_rate:
            WI_inj = np.zeros(n)
            for idx in range(cells.size):
                if is_inj[idx]:
                    WI_inj[cells[idx]] += wi[idx]
            g_inj = WI_inj * lam_t_ * inv_vi
            C_p = params.bg * _V_CO2_STD * WI_inj * lam_t_ * (-inv_vi + (bhp_inj_ - p_) * dinv_vi)
            C_s = params.bg * _V_CO2_STD * WI_inj * (bhp_inj_ - p_) * inv_vi
            for off, vec in ((0, C_p), (n, C_s * dlamt[:, 0]), (2 * n, C_s * dlamt[:, 1]), (3 * n, C_s * dlamt[:, 2])):
                nz = np.nonzero(vec)[0]
                if nz.size:
                    rows.append(np.full(nz.size, n_full))
                    cols.append(off + nz)
                    vals.append(vec[nz])
            rows.append(np.array([n_full])); cols.append(np.array([n_full]))
            vals.append(np.array([params.bg * _V_CO2_STD * float(g_inj.sum())]))
            nz = np.nonzero(g_inj)[0]
            rows.append(_CO2_IDX * n + nz)
            cols.append(np.full(nz.size, n_full))
            vals.append(-g_inj[nz])

        J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_aug, n_aug)).tocsr()
        # implied-variable chain rule (MRST AD): fold the absent-phase variables
        # into the kept columns before masking.
        F = identity(n_aug, format="lil")
        for cell in range(n):
            if pl[cell]:
                F[2 * n + cell, 2 * n + cell] = 0
                F[3 * n + cell, 3 * n + cell] = 0
                F[2 * n + cell, n + cell] = -1
                for j in range(ncomp - 1):
                    F[4 * n + k1 + cell * (ncomp - 1) + j, 4 * n + k1 + cell * (ncomp - 1) + j] = 0
                    F[4 * n + k1 + cell * (ncomp - 1) + j, 4 * n + cell * (ncomp - 1) + j] = 1
            elif pv[cell]:
                F[2 * n + cell, 2 * n + cell] = 0
                F[3 * n + cell, 3 * n + cell] = 0
                F[3 * n + cell, n + cell] = -1
                for j in range(ncomp - 1):
                    F[4 * n + cell * (ncomp - 1) + j, 4 * n + cell * (ncomp - 1) + j] = 0
                    F[4 * n + cell * (ncomp - 1) + j, 4 * n + k1 + cell * (ncomp - 1) + j] = 1
        return (J @ F.tocsr()).tocsr()

    u_full = pack_full(p, sw, sO, sG, x_ind, y_ind)
    p_i, sw_i, sO_i, sG_i, _, _ = unpack_full(u_full)
    pl, pv, two = flags_from_state(sO_i, sG_i)

    def eval_active(u_a, pl_, pv_, two_, vm_, em_):
        """Active residual ``r_full[em_]`` (+ rate row), evaluated at the filled state."""
        n_bhp = 1 if has_rate else 0
        n_state = u_a.size - n_bhp
        bhp_inj_ = u_a[-1] if has_rate else bhp_inj
        u_f = u_full.copy()
        u_f[vm_] = u_a[:n_state]
        u_f = fill_inactive(u_f, pl_, pv_)
        r_a = residual_full(u_f, bhp_inj_)[em_]
        if has_rate:
            r_a = np.concatenate([r_a, np.array([rate_residual(u_f, bhp_inj_)])])
        return r_a

    vm = var_mask(pl, pv, two)
    em = eq_mask(two)
    u_active = u_full[vm]
    if has_rate:
        u_active = np.concatenate([u_active, np.array([bhp_inj])])

    r_active = eval_active(u_active, pl, pv, two, vm, em)
    r0 = float(np.linalg.norm(r_active))
    converged = False

    def schur_solve(J_active, r_active_, pl_, pv_, two_):
        """Schur-reduce the fugacity+closure block (MRST ``ReducedLinearizedSystem``).

        Keeps component+water (+ the rate row/col); eliminates fugacity+closure and
        ``(sG, x[0], y)``. The eliminated block ``E`` is cell-local (block-diagonal per
        two-phase cell), so its LU is cheap. Returns the full recovered increment.
        """
        from scipy.sparse.linalg import splu as _splu
        n_aug = J_active.shape[0]
        em_aug = np.flatnonzero(em)
        vm_aug = np.flatnonzero(vm)
        keep_eq = keep_eq_mask()
        keep_var = keep_var_mask(pl_, pv_, two_)
        if has_rate:
            em_aug = np.concatenate([em_aug, np.array([n_full])])
            vm_aug = np.concatenate([vm_aug, np.array([n_full])])
            keep_eq = np.concatenate([keep_eq, np.array([True])])
            keep_var = np.concatenate([keep_var, np.array([True])])
        kk = np.flatnonzero(keep_eq[em_aug])   # kept equations (active space)
        ke = np.flatnonzero(~keep_eq[em_aug])  # eliminated equations
        vk = np.flatnonzero(keep_var[vm_aug])  # kept variables
        ve = np.flatnonzero(~keep_var[vm_aug])  # eliminated variables
        if ke.size == 0:  # no two-phase cells: nothing to eliminate
            return _splu(J_active.tocsc()).solve(-r_active_)
        B = J_active[kk][:, vk].toarray()
        C = J_active[kk][:, ve].toarray()
        D = J_active[ke][:, vk].toarray()
        E = J_active[ke][:, ve].tocsc()
        f = r_active_[kk]
        h = r_active_[ke]
        Elu = _splu(E)
        E_inv_D = Elu.solve(D)          # E^-1 D  (n_elim × n_keep)
        E_inv_h = Elu.solve(h)
        A_red = B - C @ E_inv_D         # Schur complement (dense, small-grid)
        # [[B, C], [D, E]] @ [du_k; du_e] = [-f; -h]  ->  du_k = A_red^-1 (-f + C E^-1 h)
        du_keep = np.linalg.solve(A_red, -f + C @ E_inv_h)
        du_elim = Elu.solve(-h - D @ du_keep)
        du = np.zeros(n_aug)
        du[vk] = du_keep
        du[ve] = du_elim
        return du

    for _ in range(max_iter):
        p_i, sw_i, sO_i, sG_i, _, _ = unpack_full(u_full)
        pl, pv, two = flags_from_state(sO_i, sG_i)
        vm = var_mask(pl, pv, two)
        em = eq_mask(two)
        u_active = u_full[vm]
        if has_rate:
            u_active = np.concatenate([u_active, np.array([bhp_inj])])
        n_active = u_active.size
        r_active = eval_active(u_active, pl, pv, two, vm, em)
        if not np.isfinite(r_active).all():
            break
        J_aug = sparse_jacobian_full(u_full, bhp_inj)
        if has_rate:
            em_aug = np.concatenate([np.flatnonzero(em), np.array([n_full])])
            vm_aug = np.concatenate([np.flatnonzero(vm), np.array([n_full])])
            J_active = J_aug[np.ix_(em_aug, vm_aug)]
        else:
            J_active = J_aug[np.ix_(np.flatnonzero(em), np.flatnonzero(vm))]
        try:
            du = schur_solve(J_active, r_active, pl, pv, two)
        except (np.linalg.LinAlgError, RuntimeError):
            break
        if _NV_DEBUG:
            du_full_diag = np.zeros(n_full)
            du_full_diag[vm] = du[: n_active - (1 if has_rate else 0)]
            print(f"    du_p (before damp) = {du_full_diag[:n]}  r_active[:n] = {r_active[:n]}")
        # Saturation chopping (MRST ``updateState``): relax the saturation and
        # composition updates to ``dsMaxAbs``/``dzMaxAbs`` so a Newton step cannot
        # overshoot a phase boundary; pressure / BHP stay full-step.
        n_bhp = 1 if has_rate else 0
        du_state = du[: n_active - n_bhp]
        du_full = np.zeros(n_full)
        du_full[vm] = du_state
        # Pressure damping (MRST ``dpMaxAbs``): the tiny shale transmissibility makes
        # the flux-p Jacobian tiny, so an undamped pressure step overshoots. Clamp it
        # to ``dpMaxAbs`` per cell, then the per-cell saturation/composition chopping.
        w_p = np.minimum(1.0, _DP_MAX / np.maximum(np.abs(du_full[:n]), 1.0e-30))
        du_full[:n] *= w_p
        # Per-cell saturation chopping (MRST ``updateState``): each cell's own
        # relaxation ``w = min(dsMax/max|ds|, dzMax/max|dz|, 1)``, not a global
        # factor — a global factor over-damps the well cells and breaks the Newton.
        ds_c = np.maximum(np.abs(du_full[2 * n:3 * n]), np.abs(du_full[3 * n:4 * n]))
        du_x = du_full[4 * n:4 * n + k1].reshape(n, ncomp - 1)
        du_y = du_full[4 * n + k1:4 * n + 2 * k1].reshape(n, ncomp - 1)
        dx_c = np.maximum(np.abs(du_x).max(axis=1), np.abs(du_y).max(axis=1))
        w = np.minimum(1.0, np.minimum(_DS_MAX / np.maximum(ds_c, 1.0e-30),
                                       _DX_MAX / np.maximum(dx_c, 1.0e-30)))
        du_full[2 * n:3 * n] *= w
        du_full[3 * n:4 * n] *= w
        du_full[4 * n:4 * n + k1] *= np.repeat(w, ncomp - 1)
        du_full[4 * n + k1:4 * n + 2 * k1] *= np.repeat(w, ncomp - 1)
        du_tail = du[-n_bhp:] if n_bhp else np.zeros(0)
        du = np.concatenate([du_full[vm], du_tail])
        # MRST applies the (chopped) increments directly, no line search — the
        # per-cell chopping above is the step limiter. A line search here re-damps
        # the already-chopped step and breaks the pressure↔saturation coupling.
        u_active = u_active + du
        u_full[vm] = u_active[: u_active.size - (1 if has_rate else 0)]
        if has_rate:
            bhp_inj = u_active[-1]
        u_full = fill_inactive(u_full, pl, pv)
        # phase transition (MRST flashPhases): single <-> two-phase switching via the
        # stability test / phase vanishing, picked up by the next Newton iteration.
        p_i, sw_i, sO_i, sG_i, x_ind_i, y_ind_i = unpack_full(u_full)
        x_i, y_i = full_comps(x_ind_i, y_ind_i)
        sO_i, sG_i, x_i, y_i = flash_phases(p_i, sw_i, sO_i, sG_i, x_i, y_i)
        u_full = pack_full(p_i, sw_i, sO_i, sG_i, x_i[:, : ncomp - 1], y_i[:, : ncomp - 1])
        r_now = float(np.linalg.norm(eval_active(u_active, pl, pv, two, vm, em)))
        if _NV_DEBUG:
            print(f"  it: |r|={r_now:.3e} npl={int(pl.sum())} npv={int(pv.sum())} "
                  f"n2ph={int(two.sum())} bhp={bhp_inj:.3e}")
        if r_now < tol * max(r0, 1.0e-12):
            converged = True
            break

    p, sw, sO, sG, x_ind, y_ind = unpack_full(u_full)
    x, y = full_comps(x_ind, y_ind)
    sO = np.clip(sO, 0.0, 1.0 - sw)
    sG = np.clip(sG, 0.0, 1.0 - sw)
    v_l, v_g = phase_molar_volumes(x, y, p, _T)
    rho_l = np.nan_to_num(1.0 / np.maximum(v_l, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
    rho_g = np.nan_to_num(1.0 / np.maximum(v_g, 1.0e-30), nan=0.0, posinf=0.0, neginf=0.0)
    m_c = sO[:, None] * x * rho_l[:, None] + sG[:, None] * y * rho_g[:, None]
    N = m_c.sum(axis=1)
    z_out = m_c / np.maximum(N[:, None], 1.0e-30)
    return p, sw, sO, sG, z_out, converged
