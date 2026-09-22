"""Minimal single-pressure forward model and a synthetic validation harness.

The forward model solves the same total-mobility mass balance that
``programs.rock.invert_rock`` inverts, so ``validate_inversion`` can generate
synthetic observations from a known ``k``/``phi`` field, run the inversion, and
report how well the true field is recovered.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.cartesian import CartesianGrid
from .mesh import PointMap, WellMap
from .rock import (
    FluidParams,
    _face_cell_pairs,
    _face_geometry,
    _harmonic_mean,
    phase_mobilities,
    corey_total_mobility,
    darcy_divergence,
    invert_rock,
    solution_gas_ratio,
    well_cell_rates,
)


def _harmonic(k1: float, k2: float) -> float:
    """Harmonic mean of two non-negative permeabilities; 0 if both are 0.

    No additive ``1e-30`` epsilon: it would bias the mean by ~33% at the
    permeability clamp floor, and the denominator is positive whenever either
    face permeability is positive.
    """
    den = k1 + k2
    return 2.0 * k1 * k2 / den if den > 0.0 else 0.0


def tpfa_matrix(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    mobility: NDArray[np.float64],
):
    """Sparse graph Laplacian ``L`` (CSR) such that ``L @ p == div(-k lambda grad p)``.

    Face transmissibilities use the harmonic mean of permeability and the
    arithmetic mean of mobility, with no-flow (Neumann-zero) outer faces.
    """
    from scipy.sparse import coo_matrix

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    n = grid.n_cells
    k3 = np.asarray(permeability, dtype=float).reshape((nz, ny, nx))
    lam3 = np.asarray(mobility, dtype=float).reshape((nz, ny, nx))
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    diag = np.zeros(n, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = grid.index(i, j, k)
                if i + 1 < nx:
                    nb = grid.index(i + 1, j, k)
                    kh = _harmonic(k3[k, j, i], k3[k, j, i + 1])
                    lamf = 0.5 * (lam3[k, j, i] + lam3[k, j, i + 1])
                    t = kh * lamf * float(dy[j] * dz[k]) / (0.5 * float(dx[i] + dx[i + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if j + 1 < ny:
                    nb = grid.index(i, j + 1, k)
                    kh = _harmonic(k3[k, j, i], k3[k, j + 1, i])
                    lamf = 0.5 * (lam3[k, j, i] + lam3[k, j + 1, i])
                    t = kh * lamf * float(dx[i] * dz[k]) / (0.5 * float(dy[j] + dy[j + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if k + 1 < nz:
                    nb = grid.index(i, j, k + 1)
                    kh = _harmonic(k3[k, j, i], k3[k + 1, j, i])
                    lamf = 0.5 * (lam3[k, j, i] + lam3[k + 1, j, i])
                    t = kh * lamf * float(dx[i] * dy[j]) / (0.5 * float(dz[k] + dz[k + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
    rows += list(range(n))
    cols += list(range(n))
    vals += diag.tolist()
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def forward_pressure(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    phi: NDArray[np.float64],
    mobility: NDArray[np.float64],
    sources_per_step: NDArray[np.float64],
    *,
    ct: float,
    p_init: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    """Implicit single-pressure forward solve with source wells and no-flow BC.

    Solves ``phi*ct*(p - p0)/dt = div(k lambda grad p) + sources`` at each step.
    Returns a ``(n_steps + 1, n_cells)`` pressure history.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    L = tpfa_matrix(grid, permeability, mobility)
    volumes = grid.cell_volumes()
    accum = np.asarray(phi, dtype=float) * ct * volumes / dt
    A = diags(accum) + L
    p = np.asarray(p_init, dtype=float).copy()
    out = [p.copy()]
    for step in range(sources_per_step.shape[0]):
        rhs = accum * p + np.asarray(sources_per_step[step], dtype=float)
        p = spsolve(A, rhs)
        out.append(p.copy())
    return np.asarray(out)


def validate_inversion(
    grid: CartesianGrid,
    k_true: NDArray[np.float64],
    phi_true: NDArray[np.float64],
    probes: PointMap,
    wells: WellMap,
    well_rate: NDArray[np.float64],
    times: NDArray[np.float64],
    p_init: NDArray[np.float64],
    *,
    params: FluidParams | None = None,
    sw: NDArray[np.float64] | None = None,
    so: NDArray[np.float64] | None = None,
    sg: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Forward-simulate observations from true k/phi, invert, and score the fit.

    Saturations are prescribed (constant 0.4/0.4/0.2 unless given); the pressure
    history comes from the forward solve. The inversion is run against the exact
    fields, so the reported errors isolate the inversion's reconstruction power.
    """
    oil = params or FluidParams()
    n_c = grid.n_cells
    n_t = int(times.size)
    if sw is None:
        sw = np.full((n_t, n_c), 0.4)
        so = np.full((n_t, n_c), 0.4)
        sg = np.full((n_t, n_c), 0.2)
    else:
        sw = np.asarray(sw, dtype=float)
        so = np.asarray(so, dtype=float)
        sg = np.asarray(sg, dtype=float)
        if sw.ndim == 1:
            sw = sw[None, :]
            so = so[None, :]
            sg = sg[None, :]

    mobility = np.stack([corey_total_mobility(sw[t], so[t], sg[t], oil) for t in range(n_t)])
    sources = np.stack([well_cell_rates(grid, wells, well_rate[t]) for t in range(n_t)])
    dt = float(times[1] - times[0]) if n_t > 1 else 1.0
    p_true = forward_pressure(
        grid, k_true, phi_true, mobility[0], sources[:-1] if n_t > 1 else sources,
        ct=oil.ct, p_init=p_init, dt=dt,
    )
    if p_true.shape[0] < n_t:
        p_true = np.vstack([p_true, np.repeat(p_true[-1:], n_t - p_true.shape[0], axis=0)])

    phi_rec, k_rec, diag = invert_rock(
        grid,
        p_true,
        sw,
        so,
        sg,
        times,
        probes,
        wells,
        well_rate,
        phi0=float(np.mean(phi_true)),
        k0=float(np.mean(k_true)),
        params=oil,
    )

    logk_true = np.log(np.maximum(k_true, 1.0e-30))
    logk_rec = np.log(np.maximum(k_rec, 1.0e-30))
    return {
        "logk_rmse": float(np.sqrt(np.mean((logk_rec - logk_true) ** 2))),
        "phi_rmse": float(np.sqrt(np.mean((phi_rec - phi_true) ** 2))),
        "logk_corr": _corr(logk_rec, logk_true),
        "phi_corr": _corr(phi_rec, phi_true),
        "mass_residual_rel": diag.mass_residual_rel,
    }


def _corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    if float(np.std(a)) < 1.0e-15 or float(np.std(b)) < 1.0e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _solve_pressure_with_source(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    mobility: NDArray[np.float64],
    source: NDArray[np.float64],
    ref_cell: int,
    ref_p: float,
) -> NDArray[np.float64]:
    """Solve ``div(-k lambda grad p) = source`` with p[ref_cell] = ref_p."""
    from scipy.sparse.linalg import spsolve

    L = tpfa_matrix(grid, permeability, mobility)
    n = grid.n_cells
    ref = int(ref_cell)
    free = np.array([i for i in range(n) if i != ref], dtype=np.int64)
    rhs = np.asarray(source, dtype=float).copy()
    l_fc = L[free][:, [ref]].toarray().ravel()
    rhs[free] = rhs[free] - l_fc * float(ref_p)
    p = np.zeros(n, dtype=float)
    p[ref] = float(ref_p)
    p[free] = spsolve(L[free][:, free], rhs[free])
    return p


def _project_three(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sw_p = np.clip(np.asarray(sw, dtype=float), 0.0, None)
    so_p = np.clip(np.asarray(so, dtype=float), 0.0, None)
    sg_p = np.clip(np.asarray(sg, dtype=float), 0.0, None)
    total = sw_p + so_p + sg_p
    total = np.where(total > 0.0, total, 1.0)
    return sw_p / total, so_p / total, sg_p / total


def black_oil_forward(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    phi: NDArray[np.float64],
    params: FluidParams,
    wells: WellMap,
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
    times: NDArray[np.float64],
    *,
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    ref_cell: int,
    ref_p: float,
    max_ds: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Minimal IMPES three-phase black-oil forward model (adaptive substepping).

    Pressure is solved implicitly from the total-mobility equation (with a
    reference pressure at ``ref_cell``). Saturations are advanced explicitly, but
    each report interval is sub-stepped so the per-step saturation change is
    bounded by ``max_ds``, keeping the explicit update inside its stability (CFL)
    limit. Returns ``(p, sw, so, sg)`` histories, each of shape
    ``(n_times, n_cells)``.
    """
    n_c = grid.n_cells
    times_a = np.asarray(times, dtype=float)
    n_t = int(times_a.size)
    phi_a = np.asarray(phi, dtype=float)
    vol = grid.cell_volumes()
    pw = np.asarray(well_qw, dtype=float)
    po = np.asarray(well_qo, dtype=float)
    pg = np.asarray(well_qg, dtype=float)
    p_hist = np.zeros((n_t, n_c))
    sw_hist = np.zeros((n_t, n_c))
    so_hist = np.zeros((n_t, n_c))
    sg_hist = np.zeros((n_t, n_c))
    sw = np.asarray(sw0, dtype=float).copy()
    so = np.asarray(so0, dtype=float).copy()
    sg = np.asarray(sg0, dtype=float).copy()

    for t in range(n_t):
        lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
        qw_t = well_cell_rates(grid, wells, pw[t])
        qo_t = well_cell_rates(grid, wells, po[t])
        qg_t = well_cell_rates(grid, wells, pg[t])
        p = _solve_pressure_with_source(grid, permeability, lam_w + lam_o + lam_g, qw_t + qo_t + qg_t, ref_cell, ref_p)
        p_hist[t] = p
        sw_hist[t] = sw.copy()
        so_hist[t] = so.copy()
        sg_hist[t] = sg.copy()

        if t < n_t - 1:
            remaining = float(times_a[t + 1] - times_a[t])
            n_sub = 0
            while remaining > 1.0e-12 and n_sub < _MAX_SUBSTEPS:
                div_w = darcy_divergence(grid, permeability, lam_w, p)
                div_o = darcy_divergence(grid, permeability, lam_o, p)
                div_g = darcy_divergence(grid, permeability, lam_g, p)
                rate = max(
                    float(np.max(np.abs(qw_t - div_w) / (phi_a * vol))),
                    float(np.max(np.abs(qo_t - div_o) / (phi_a * vol))),
                    float(np.max(np.abs(qg_t - div_g) / (phi_a * vol))),
                )
                dt_sub = min(remaining, max_ds / max(rate, 1.0e-12))
                sw = sw + (dt_sub / (phi_a * vol)) * (qw_t - div_w)
                so = so + (dt_sub / (phi_a * vol)) * (qo_t - div_o)
                sg = sg + (dt_sub / (phi_a * vol)) * (qg_t - div_g)
                sw, so, sg = _project_three(sw, so, sg)
                remaining -= dt_sub
                n_sub += 1
                lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
                p = _solve_pressure_with_source(grid, permeability, lam_w + lam_o + lam_g, qw_t + qo_t + qg_t, ref_cell, ref_p)
    return p_hist, sw_hist, so_hist, sg_hist


# ---------------------------------------------------------------------------
# P3: pluggable forward saturation models (black-oil / compositional).
#
# These take the *known* pressure field (the kriged reconstruction, already
# accurate to ~0.1%) as the flow field and only advance the phase saturations,
# so the result satisfies mass conservation rather than being a pure
# interpolation. ``black_oil`` keeps CO2 as a permanent free-gas phase;
# ``compositional`` lets CO2 dissolve into the oil (solution gas), so injected
# CO2 largely travels with the oil instead of accumulating as gas.
# ---------------------------------------------------------------------------


def forward_saturations(
    model: str,
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    permeability: NDArray[np.float64],
    phi: NDArray[np.float64],
    params: FluidParams,
    wells: WellMap,
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
    times: NDArray[np.float64],
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    *,
    max_ds: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Forward-simulate the saturation history from a given pressure field.

    Returns ``(sw, so, sg)`` histories, each of shape ``(n_times, n_cells)``.
    ``model`` is ``"black_oil"`` or ``"compositional"`` (alias ``solution_gas`` /
    ``co2``). The pressure ``pressure`` is ``(n_times, n_cells)``; only the
    saturations are advanced.
    """
    name = str(model).strip().lower()
    if name == "black_oil":
        return _forward_black_oil_saturations(
            grid, pressure, permeability, phi, params, wells,
            well_qw, well_qo, well_qg, times, sw0, so0, sg0, max_ds=max_ds,
        )
    # ``fcm`` is an alias kept for backward compatibility: it is the miscible
    # (single-phase) limit of the unified compositional model.
    if name in ("compositional", "solution_gas", "co2", "fcm"):
        return _forward_compositional_saturations(
            grid, pressure, permeability, phi, params, wells,
            well_qw, well_qo, well_qg, times, sw0, so0, sg0, max_ds=max_ds,
        )
    raise ValueError(f"unknown forward model {model!r}")


def _mobility_divergence_matrix(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    pressure: NDArray[np.float64],
):
    """Sparse matrix ``A`` such that ``A @ lam == div(-k lam grad p)`` (upwind).

    The divergence is *linear* in the cell mobility field ``lam`` (the upwind
    face mobility enters linearly), so ``A`` is a fixed sparse operator for a
    given permeability and pressure. This lets the fully-implicit Newton Jacobian
    be assembled as ``I + (dt/phiV) A diag(dlam/ds)`` without a numerical loop.
    """
    from scipy.sparse import coo_matrix

    n = grid.n_cells
    k3 = np.asarray(permeability, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    coef_x, up_x, coef_y, up_y, coef_z, up_z = _face_geometry(grid, pressure)
    (c1x, c2x), (c1y, c2y), (c1z, c2z) = _face_cell_pairs(grid.nx, grid.ny, grid.nz)
    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    vals: list[NDArray[np.float64]] = []

    def add(lo, hi, up, g):
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        up = np.asarray(up, dtype=bool)
        g = np.asarray(g, dtype=float)
        # low cell is upstream (flow low -> high)
        ulo = lo[up]
        glo = g[up]
        rows.append(ulo); cols.append(ulo); vals.append(glo)          # A[lo, lo] += g
        rows.append(hi[up]); cols.append(ulo); vals.append(-glo)      # A[hi, lo] -= g
        # high cell is upstream
        uhi = hi[~up]
        ghi = g[~up]
        rows.append(lo[~up]); cols.append(uhi); vals.append(ghi)      # A[lo, hi] += g
        rows.append(uhi); cols.append(uhi); vals.append(-ghi)         # A[hi, hi] -= g

    if coef_x is not None:
        kh = _harmonic_mean(k3[:, :, :-1], k3[:, :, 1:]).ravel()
        add(c1x, c2x, up_x.ravel(), kh * coef_x.ravel())
    if coef_y is not None:
        kh = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :]).ravel()
        add(c1y, c2y, up_y.ravel(), kh * coef_y.ravel())
    if coef_z is not None:
        kh = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :]).ravel()
        add(c1z, c2z, up_z.ravel(), kh * coef_z.ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# Upper bound on a forward sub-step. The GEM deck resolves the shale-oil injection
# with ``*DTMAX 0.01`` days (14.4 min); a 30-day report step is far too coarse for
# the lab-scale injection (fills the model ~4.5 h), so sub-steps are capped here.
_MAX_DT = 864.0


def _balance_well_rates(
    qw: NDArray[np.float64],
    qo: NDArray[np.float64],
    qg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scale producing wells so total production matches total injection.

    The incompressible forward model needs a divergence-free total source; real
    well-rate exports can carry a net injection/production imbalance (the
    compressible reservoir is being filled or depleted). We scale the producing
    wells uniformly to close the total, keeping the per-well phase split intact.
    """
    qw = np.array(qw, dtype=float, copy=True)
    qo = np.array(qo, dtype=float, copy=True)
    qg = np.array(qg, dtype=float, copy=True)
    if qw.ndim == 1:
        qw, qo, qg = qw[None, :], qo[None, :], qg[None, :]
    q = qw + qo + qg
    for t in range(q.shape[0]):
        qt = q[t]
        inj = float(qt[qt > 0.0].sum())
        prod = float(-qt[qt < 0.0].sum())
        if inj > 0.0 and prod > 0.0:
            scale = inj / prod
            mask = qt < 0.0
            qw[t, mask] *= scale
            qo[t, mask] *= scale
            qg[t, mask] *= scale
    return qw, qo, qg


def _corey_mobilities_derivs(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    params: FluidParams,
) -> tuple[NDArray[np.float64], ...]:
    """Corey phase mobilities and their own-saturation derivatives (numerical)."""
    h = 1.0e-6
    lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
    lw_p, _, _ = phase_mobilities(sw + h, so, sg, params)
    _, lo_p, _ = phase_mobilities(sw, so + h, sg, params)
    _, _, lg_p = phase_mobilities(sw, so, sg + h, params)
    return lam_w, lam_o, lam_g, (lw_p - lam_w) / h, (lo_p - lam_o) / h, (lg_p - lam_g) / h


def _gravity_divergence_matrix(grid, permeability):
    """Sparse matrix ``G`` such that ``G @ lam = div(k lam ∇(-z))``.

    This is the z-divergence operator (unit *downward* potential gradient) used
    to add buoyancy: ``div_grav = -rho_g * (G @ lam)``. Using ``pressure = -z``
    in the ordinary divergence operator gives exactly this, so the gravity
    correction is just ``-rho_g * (_mobility_divergence_matrix(grid, k, -z) @ lam)``.
    """
    z = np.asarray(grid.cell_centers()[:, 2], dtype=float)
    return _mobility_divergence_matrix(grid, permeability, -z)


def _solve_linear(A, b):
    """Solve ``A x = b``: fast GMRES first, AMG-preconditioned GMRES for the
    non-symmetric advection blocks GMRES can't crack, then a direct LU fallback."""
    from scipy.sparse.linalg import gmres, spsolve

    x, info = gmres(A, b, rtol=1.0e-6, atol=1.0e-10, maxiter=50)
    if info == 0:
        return x
    # GMRES stalled (the saturated advection block is non-symmetric): use an
    # algebraic multigrid (smoothed aggregation) preconditioner.
    try:
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(A.tocsr())
        M = ml.aspreconditioner(cycle="V")
        x, info = gmres(A, b, M=M, rtol=1.0e-6, atol=1.0e-10, maxiter=100)
        if info == 0:
            return x
    except Exception:
        pass
    return spsolve(A, b)


def _implicit_black_oil_step(
    A,
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    qw: NDArray[np.float64],
    qo: NDArray[np.float64],
    params: FluidParams,
    *,
    max_iter: int = 30,
    tol: float = 1.0e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """One fully-implicit (backward-Euler) black-oil saturation step (Newton)."""
    from scipy.sparse import bmat, diags, identity
    from scipy.sparse.linalg import spsolve

    n = int(sw0.size)
    I = identity(n, format="csr")
    sw = np.asarray(sw0, dtype=float).copy()
    so = np.asarray(so0, dtype=float).copy()
    converged = False
    for _ in range(max_iter):
        sg = 1.0 - sw - so
        lam_w, lam_o, lam_g, dlw, dlo, _dlg = _corey_mobilities_derivs(sw, so, sg, params)
        r_sw = sw - sw0 - dt * inv_phiV * (qw - A @ lam_w)
        r_so = so - so0 - dt * inv_phiV * (qo - A @ lam_o)
        J_ww = I + A @ diags(dt * inv_phiV * dlw)
        J_ss = I + A @ diags(dt * inv_phiV * dlo)
        J = bmat([[J_ww, None], [None, J_ss]], format="csr")
        delta = _solve_linear(J, -np.concatenate([r_sw, r_so]))
        sw_new = np.clip(sw + 0.5 * delta[:n], 0.0, 1.0)
        so_new = np.clip(so + 0.5 * delta[n:], 0.0, 1.0 - sw_new)
        norm_d = float(np.linalg.norm(np.concatenate([sw_new - sw, so_new - so])))
        sw, so = sw_new, so_new
        norm_x = float(np.linalg.norm(np.concatenate([sw, so])))
        if norm_d < tol * max(1.0, norm_x):
            converged = True
            break
    sg = 1.0 - sw - so
    return sw, so, sg, converged


def _implicit_compositional_step(
    A,
    A_grav,
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    C0: NDArray[np.float64],
    qw: NDArray[np.float64],
    q_co2: NDArray[np.float64],
    Rs: NDArray[np.float64],
    params: FluidParams,
    *,
    max_iter: int = 30,
    tol: float = 1.0e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
    """One fully-implicit solution-gas (CO2 component) step (Newton).

    Tracks water saturation ``sw`` and the surface-volume CO2 component
    ``C = sg/Bg + Rs*so``; the phase split is recomputed inside the Newton
    iteration from ``Rs`` and ``Bg``. The free gas carries a buoyancy term
    ``rho_g`` (the gas sinks when ``rho_g > 0``); the dissolved CO2 moves with
    the oil and has no separate gravity.
    """
    from scipy.sparse import diags

    Bg = float(params.bg)
    inv_Bg = 1.0 / max(Bg, 1.0e-12)
    # Gas Darcy velocity v = -k lam (grad p + rho_g grad z); its divergence is
    # A@lam - rho_g*A_grav@lam (A_grav@lam = div(k lam grad z)). rho_g > 0 sinks.
    A_g = A - params.rho_g * A_grav
    n = int(sw0.size)
    sw = np.asarray(sw0, dtype=float).copy()
    C = np.asarray(C0, dtype=float).copy()
    h = 1.0e-6
    # Rescale to the flux form ``phi*vol/dt*(x-x0) - (source - flux) = 0``: the
    # raw form multiplies the flux by ``dt*inv_phiV`` (~1e9), which makes the
    # Jacobian stiff and GMRES crawl. Dividing through by ``dt*inv_phiV`` keeps
    # the accumulation diagonal ``phi*vol/dt`` and the flux operator O(1).
    accum = 1.0 / (dt * inv_phiV)
    I = diags(accum)

    def residual(sw_, C_):
        """Residual of the (sw, C) system at a trial state (flux form)."""
        _, so_, sg_ = _split_compositional(sw_, C_, Rs, Bg, params.bo_slope)
        Rs_act_ = np.clip(np.where(so_ > 1.0e-12, (C_ - sg_ / Bg) / np.maximum(so_, 1.0e-12), 0.0), 0.0, Rs)
        lam_w_, lam_o_, lam_g_ = phase_mobilities(sw_, so_, sg_, params)
        r_sw_ = accum * (sw_ - sw0) - (qw - A @ lam_w_)
        flux_dissolved_ = A @ (Rs_act_ * lam_o_)  # upwind dissolved-CO2 flux (A upwinds Rs*lam)
        r_c_ = accum * (C_ - C0) - (q_co2 - (inv_Bg * A_g @ lam_g_ + flux_dissolved_))
        return np.concatenate([r_sw_, r_c_])

    r0_norm = float(np.linalg.norm(residual(sw, C)))
    converged = False
    for _ in range(max_iter):
        _, so, sg = _split_compositional(sw, C, Rs, Bg, params.bo_slope)
        # actual dissolved ratio (<= equilibrium Rs): undersaturated oil carries
        # less than the equilibrium bound, so the flux/source use the real value.
        Rs_act = np.clip(np.where(so > 1.0e-12, (C - sg / Bg) / np.maximum(so, 1.0e-12), 0.0), 0.0, Rs)
        lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
        # dissolved CO2 flux is the *upwind* Rs*lam_o carried by the oil:
        # A @ (Rs_act * lam_o). Using the centred Rs_act * (A @ lam_o) instead
        # breaks CO2 conservation where Rs_act jumps (e.g. the plume front), so
        # the dissolved CO2 never reaches cells with lower Rs.
        r = np.concatenate([
            accum * (sw - sw0) - (qw - A @ lam_w),
            accum * (C - C0) - (q_co2 - (inv_Bg * A_g @ lam_g + A @ (Rs_act * lam_o))),
        ])
        # derivatives (local, numerical through the phase split, per phase + ratio)
        _, so_p, sg_p = _split_compositional(sw + h, C, Rs, Bg, params.bo_slope)
        Rs_sw = np.clip(np.where(so_p > 1.0e-12, (C - sg_p / Bg) / np.maximum(so_p, 1.0e-12), 0.0), 0.0, Rs)
        lw_p, lo_p, lg_p = phase_mobilities(sw + h, so_p, sg_p, params)
        dlw_dsw = (lw_p - lam_w) / h
        dlg_dsw = (lg_p - lam_g) / h
        dlo_dsw = (lo_p - lam_o) / h
        dRs_dsw = (Rs_sw - Rs_act) / h
        _, so_c, sg_c = _split_compositional(sw, C + h, Rs, Bg, params.bo_slope)
        Rs_c = np.clip(np.where(so_c > 1.0e-12, (C + h - sg_c / Bg) / np.maximum(so_c, 1.0e-12), 0.0), 0.0, Rs)
        _lw_c, lo_c, lg_c = phase_mobilities(sw, so_c, sg_c, params)
        dlg_dC = (lg_c - lam_g) / h
        dlo_dC = (lo_c - lam_o) / h
        dRs_dC = (Rs_c - Rs_act) / h
        J_ww = I + A @ diags(dlw_dsw)
        # Derivative of the upwind dissolved-CO2 flux A@(Rs_act*lam_o): one
        # combined column-scaling term d(Rs_act*lam_o)/d(sw,C) = dRs*lam_o + Rs*dlo.
        J_cw = (inv_Bg * A_g @ diags(dlg_dsw)
                + A @ diags(dRs_dsw * lam_o + Rs_act * dlo_dsw))
        J_cc = (I + inv_Bg * A_g @ diags(dlg_dC)
                + A @ diags(dRs_dC * lam_o + Rs_act * dlo_dC))
        # Block forward substitution (the Jacobian is lower block-triangular):
        # two n×n solves instead of one 2n×2n — cheaper and better-conditioned.
        r = -r
        delta_sw = _solve_linear(J_ww, r[:n])
        delta_c = _solve_linear(J_cc, r[n:] - J_cw @ delta_sw)
        delta = np.concatenate([delta_sw, delta_c])
        # Backtracking line search (Armijo): the phase split has a kink at the
        # saturation point, so a full Newton step can overshoot into the wrong
        # region; shrink the step until the residual actually decreases.
        alpha = 1.0
        r_norm = float(np.linalg.norm(r))
        sw_new = sw.copy()
        C_new = C.copy()
        r_new = r
        for _ in range(12):
            sw_new = np.clip(sw + alpha * delta[:n], 0.0, 1.0)
            C_new = np.clip(C + alpha * delta[n:], 0.0, (Rs + inv_Bg) * (1.0 - sw_new))
            r_new = residual(sw_new, C_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([sw_new - sw, C_new - C])))
        sw, C = sw_new, C_new
        norm_x = float(np.linalg.norm(np.concatenate([sw, C])))
        # converge on the *residual* (relative to the initial source), not the
        # change: the ill-conditioned Jacobian can freeze the step while the
        # residual is still large. Relative to r0_norm (with a tiny floor, not a
        # clamp to 1 — the rescaled residual has norm < 1).
        if float(np.linalg.norm(r_new)) < max(tol, 0.05) * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    _, so, sg = _split_compositional(sw, C, Rs, Bg, params.bo_slope)
    return sw, so, sg, C, converged


def _split_kinetic(
    sw: NDArray[np.float64],
    C: NDArray[np.float64],
    Cd: NDArray[np.float64],
    Rs: NDArray[np.float64],
    Bg: float,
    bo_slope: float,
) -> tuple[NDArray[np.float64], ...]:
    """Kinetic-dissolution phase split.

    ``Cd`` is the dissolved CO2 (a state, not an equilibrium-derived value); the
    free gas is the excess ``sg = (C - Cd)*Bg`` and the oil fills the rest.
    ``Rs_act = Cd/No`` is the actual dissolved ratio (bounded by ``Rs``).
    Returns ``(sw, so, sg, No, Rs_act)``.
    """
    sw_p = np.clip(np.asarray(sw, dtype=float), 0.0, None)
    C_p = np.clip(np.asarray(C, dtype=float), 0.0, None)
    Cd_p = np.clip(np.asarray(Cd, dtype=float), 0.0, None)
    Rs_p = np.clip(np.asarray(Rs, dtype=float), 0.0, None)
    nw = 1.0 - sw_p
    Bo = 1.0 + bo_slope * Rs_p
    sg = np.clip((C_p - Cd_p) * float(Bg), 0.0, nw)
    so = nw - sg
    No = so / Bo
    Rs_act = np.clip(Cd_p / np.maximum(No, 1.0e-12), 0.0, Rs_p)
    return sw_p, so, sg, No, Rs_act


def _implicit_kinetic_step(
    A,
    A_grav,
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    C0: NDArray[np.float64],
    Cd0: NDArray[np.float64],
    qw: NDArray[np.float64],
    qg: NDArray[np.float64],
    qo: NDArray[np.float64],
    Rs: NDArray[np.float64],
    params: FluidParams,
    *,
    max_iter: int = 30,
    tol: float = 1.0e-3,
) -> tuple[NDArray[np.float64], ...]:
    """One fully-implicit kinetic-dissolution step (Newton) for (sw, C, Cd).

    ``Cd`` (dissolved CO2) relaxes to the equilibrium ``Rs*No`` at rate
    ``k_diss``; the free gas is the excess. With ``k_diss -> inf`` this reduces
    to the equilibrium phase split of :func:`_implicit_compositional_step`.
    """
    from scipy.sparse import diags

    Bg = float(params.bg)
    inv_Bg = 1.0 / max(Bg, 1.0e-12)
    A_g = A - params.rho_g * A_grav
    k_diss = float(params.k_diss)
    n = int(sw0.size)
    sw = np.asarray(sw0, dtype=float).copy()
    C = np.asarray(C0, dtype=float).copy()
    Cd = np.asarray(Cd0, dtype=float).copy()
    h = 1.0e-6
    accum = 1.0 / (dt * inv_phiV)
    phiV = 1.0 / inv_phiV  # pore volume per cell: scales k_diss (a rate per PV) to m3/s
    I = diags(accum)

    def residual(sw_, C_, Cd_):
        _, so_, sg_, No_, Rs_act_ = _split_kinetic(sw_, C_, Cd_, Rs, Bg, params.bo_slope)
        lam_w_, lam_o_, lam_g_ = phase_mobilities(sw_, so_, sg_, params)
        flux_dissolved_ = A @ (Rs_act_ * lam_o_)  # upwind dissolved-CO2 flux
        r_sw_ = accum * (sw_ - sw0) - (qw - A @ lam_w_)
        r_C_ = accum * (C_ - C0) - (qg + Rs_act_ * qo - (inv_Bg * A_g @ lam_g_ + flux_dissolved_))
        r_Cd_ = accum * (Cd_ - Cd0) - k_diss * (Rs * No_ - Cd_) * phiV - Rs_act_ * qo + flux_dissolved_
        return np.concatenate([r_sw_, r_C_, r_Cd_])

    r0_norm = float(np.linalg.norm(residual(sw, C, Cd)))
    converged = False
    for _ in range(max_iter):
        _, so, sg, No, Rs_act = _split_kinetic(sw, C, Cd, Rs, Bg, params.bo_slope)
        lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
        r = residual(sw, C, Cd)
        # numerical derivatives w.r.t. sw / C / Cd
        _, so_s, sg_s, No_s, Rs_s = _split_kinetic(sw + h, C, Cd, Rs, Bg, params.bo_slope)
        lw_s, lo_s, lg_s = phase_mobilities(sw + h, so_s, sg_s, params)
        dlw_dsw = (lw_s - lam_w) / h
        dlo_dsw = (lo_s - lam_o) / h
        dlg_dsw = (lg_s - lam_g) / h
        dRs_dsw = (Rs_s - Rs_act) / h
        dNo_dsw = (No_s - No) / h
        _, so_c, sg_c, No_c, Rs_c = _split_kinetic(sw, C + h, Cd, Rs, Bg, params.bo_slope)
        lw_c, lo_c, lg_c = phase_mobilities(sw, so_c, sg_c, params)
        dlw_dC = (lw_c - lam_w) / h
        dlo_dC = (lo_c - lam_o) / h
        dlg_dC = (lg_c - lam_g) / h
        dRs_dC = (Rs_c - Rs_act) / h
        dNo_dC = (No_c - No) / h
        _, so_d, sg_d, No_d, Rs_d = _split_kinetic(sw, C, Cd + h, Rs, Bg, params.bo_slope)
        lw_d, lo_d, lg_d = phase_mobilities(sw, so_d, sg_d, params)
        dlw_dCd = (lw_d - lam_w) / h
        dlo_dCd = (lo_d - lam_o) / h
        dlg_dCd = (lg_d - lam_g) / h
        dRs_dCd = (Rs_d - Rs_act) / h
        dNo_dCd = (No_d - No) / h
        # Jacobian blocks (flux form). The dissolved-CO2 flux is the upwind
        # A@(Rs_act*lam_o), so its d/d(sw,C,Cd) is A @ diags(dRs*lam_o + Rs*dlo);
        # the produced dissolved gas Rs_act*qo contributes -diags(dRs*qo).
        dF_sw = dRs_dsw * lam_o + Rs_act * dlo_dsw
        dF_C = dRs_dC * lam_o + Rs_act * dlo_dC
        dF_Cd = dRs_dCd * lam_o + Rs_act * dlo_dCd
        J_ww = I + A @ diags(dlw_dsw)
        J_wC = A @ diags(dlw_dC)
        J_wCd = A @ diags(dlw_dCd)
        J_Cw = inv_Bg * A_g @ diags(dlg_dsw) + A @ diags(dF_sw) - diags(dRs_dsw * qo)
        J_CC = I + inv_Bg * A_g @ diags(dlg_dC) + A @ diags(dF_C) - diags(dRs_dC * qo)
        J_CCd = inv_Bg * A_g @ diags(dlg_dCd) + A @ diags(dF_Cd) - diags(dRs_dCd * qo)
        J_Cdw = -k_diss * diags(Rs * dNo_dsw * phiV) + A @ diags(dF_sw) - diags(dRs_dsw * qo)
        J_CdC = -k_diss * diags(Rs * dNo_dC * phiV) + A @ diags(dF_C) - diags(dRs_dC * qo)
        J_CdCd = I + k_diss * diags((1.0 - Rs * dNo_dCd) * phiV) + A @ diags(dF_Cd) - diags(dRs_dCd * qo)
        # block forward substitution (lower block-triangular)
        r = -r
        delta_sw = _solve_linear(J_ww, r[:n])
        delta_c = _solve_linear(J_CC, r[n:2 * n] - J_Cw @ delta_sw)
        delta_cd = _solve_linear(J_CdCd, r[2 * n:] - J_Cdw @ delta_sw - J_CdC @ delta_c)
        delta = np.concatenate([delta_sw, delta_c, delta_cd])
        # backtracking line search
        alpha = 1.0
        r_norm = float(np.linalg.norm(r))
        sw_new = sw.copy()
        C_new = C.copy()
        Cd_new = Cd.copy()
        r_new = -r
        for _ in range(12):
            sw_new = np.clip(sw + alpha * delta[:n], 0.0, 1.0)
            C_new = np.clip(C + alpha * delta[n:2 * n], 0.0, (Rs + inv_Bg) * (1.0 - sw_new))
            Cd_new = np.clip(Cd + alpha * delta[2 * n:], 0.0, Rs * (1.0 - sw_new))
            r_new = residual(sw_new, C_new, Cd_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([sw_new - sw, C_new - C, Cd_new - Cd])))
        sw, C, Cd = sw_new, C_new, Cd_new
        norm_x = float(np.linalg.norm(np.concatenate([sw, C, Cd])))
        if float(np.linalg.norm(r_new)) < max(tol, 0.05) * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    _, so, sg, _No, _Rs_act = _split_kinetic(sw, C, Cd, Rs, Bg, params.bo_slope)
    return sw, so, sg, C, Cd, converged


def _forward_black_oil_saturations(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    permeability: NDArray[np.float64],
    phi: NDArray[np.float64],
    params: FluidParams,
    wells: WellMap,
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
    times: NDArray[np.float64],
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    *,
    max_ds: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Three-phase black-oil saturation transport.

    The flow pressure is solved implicitly from total mobility + total source at
    every (sub)step (anchored at the kriged pressure of the reference cell), so
    the divergence matches the source and the explicit saturation update stays
    stable. The kriged ``pressure`` is used only for the reference level.
    """
    p_in = np.asarray(pressure, dtype=float)
    if p_in.ndim == 1:
        p_in = p_in[None, :]
    n_t = int(p_in.shape[0])
    n_c = grid.n_cells
    k = np.asarray(permeability, dtype=float)
    phi_a = np.asarray(phi, dtype=float)
    vol = grid.cell_volumes()
    qw = np.asarray(well_qw, dtype=float)
    qo = np.asarray(well_qo, dtype=float)
    qg = np.asarray(well_qg, dtype=float)
    if qw.ndim == 1:
        qw, qo, qg = qw[None, :], qo[None, :], qg[None, :]
    qw, qo, qg = _balance_well_rates(qw, qo, qg)
    times_a = np.asarray(times, dtype=float)
    ref_cell = 0
    inv_phiV = 1.0 / (phi_a * vol)
    sw = np.asarray(sw0, dtype=float).copy()
    so = np.asarray(so0, dtype=float).copy()
    sg = np.asarray(sg0, dtype=float).copy()
    sw_hist = np.zeros((n_t, n_c))
    so_hist = np.zeros((n_t, n_c))
    sg_hist = np.zeros((n_t, n_c))
    for t in range(n_t):
        lam_w, lam_o, lam_g = phase_mobilities(sw, so, sg, params)
        qw_t = well_cell_rates(grid, wells, qw[t])
        qo_t = well_cell_rates(grid, wells, qo[t])
        qg_t = well_cell_rates(grid, wells, qg[t])
        ref_p = float(p_in[t, ref_cell])
        p = _solve_pressure_with_source(grid, k, lam_w + lam_o + lam_g, qw_t + qo_t + qg_t, ref_cell, ref_p)
        sw_hist[t] = sw.copy()
        so_hist[t] = so.copy()
        sg_hist[t] = sg.copy()
        if t < n_t - 1:
            remaining = float(times_a[t + 1] - times_a[t])
            A = _mobility_divergence_matrix(grid, k, p)
            n_halve = 0
            while remaining > 1.0e-12 and n_halve < 20:
                sw_new, so_new, sg_new, conv = _implicit_black_oil_step(
                    A, inv_phiV, remaining, sw, so, qw_t, qo_t, params
                )
                if conv:
                    sw, so, sg = sw_new, so_new, sg_new
                    break
                remaining *= 0.5
                n_halve += 1
            sw, so, sg = _project_three(sw, so, sg)
    return sw_hist, so_hist, sg_hist


def _tpfa_matrix_vec(grid, permeability, mobility):
    """Vectorized TPFA Laplacian: ``L @ p == div(-k mobility grad p)``.

    Face mobility is the arithmetic mean of the adjacent cells (no upwind), the
    same discretisation as ``tpfa_matrix`` but without the Python triple loop.
    """
    from scipy.sparse import coo_matrix

    n = grid.n_cells
    k3 = np.asarray(permeability, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    lam3 = np.asarray(mobility, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    (c1x, c2x), (c1y, c2y), (c1z, c2z) = _face_cell_pairs(grid.nx, grid.ny, grid.nz)
    rows: list = []
    cols: list = []
    vals: list = []

    def add(c1, c2, t):
        t = np.asarray(t, dtype=float).ravel()
        rows.append(c1); cols.append(c1); vals.append(t)
        rows.append(c2); cols.append(c2); vals.append(t)
        rows.append(c1); cols.append(c2); vals.append(-t)
        rows.append(c2); cols.append(c1); vals.append(-t)

    if grid.nx > 1:
        kh = _harmonic_mean(k3[:, :, :-1], k3[:, :, 1:])
        lamf = 0.5 * (lam3[:, :, :-1] + lam3[:, :, 1:])
        area = dy[None, :, None] * dz[:, None, None]  # (nz, ny, 1)
        dist = 0.5 * (dx[:-1] + dx[1:])
        add(c1x, c2x, kh * lamf * area / dist[None, None, :])
    if grid.ny > 1:
        kh = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :])
        lamf = 0.5 * (lam3[:, :-1, :] + lam3[:, 1:, :])
        area = dz[:, None, None] * dx[None, None, :]  # (nz, 1, nx)
        dist = 0.5 * (dy[:-1] + dy[1:])
        add(c1y, c2y, kh * lamf * area / dist[None, :, None])
    if grid.nz > 1:
        kh = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :])
        lamf = 0.5 * (lam3[:-1, :, :] + lam3[1:, :, :])
        area = dx[None, None, :] * dy[None, :, None]  # (1, ny, nx)
        dist = 0.5 * (dz[:-1] + dz[1:])
        add(c1z, c2z, kh * lamf * area / dist[:, None, None])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def _eq_solution_gas_ratio(
    pressure: NDArray[np.float64],
    params: FluidParams,
) -> NDArray[np.float64]:
    """Equilibrium solution gas-oil ratio ``Rs`` (surface gas / surface oil).

    Uses the *equilibrium* solubility slope ``rs_eq_slope`` (falling back to
    ``rs_slope`` when unset). This is the surface-volume solubility bound used in
    the forward model's phase split, distinct from the output ``rs`` metric.
    """
    slope = params.rs_eq_slope if params.rs_eq_slope > 0.0 else params.rs_slope
    return np.maximum(slope * np.asarray(pressure, dtype=float), 0.0)


def fcm_effective_viscosity(
    c: NDArray[np.float64],
    params: FluidParams,
) -> NDArray[np.float64]:
    """FCM single-phase effective viscosity (1/4-power mixing of solvent and oil).

    ``c`` is the solvent (CO2) volume fraction. ``mu_s`` is ``params.mu_g`` and
    ``mu_o`` is ``params.mu_o``.
    """
    mu_s = max(float(params.mu_g), 1.0e-12)
    mu_o = max(float(params.mu_o), 1.0e-12)
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return (c / mu_s**0.25 + (1.0 - c) / mu_o**0.25) ** (-4.0)


def fcm_effective_density(
    c: NDArray[np.float64],
    params: FluidParams,
) -> NDArray[np.float64]:
    """FCM single-phase effective density head ``rho*g`` (Pa/m), linear mixing.

    ``rho_s > rho_o`` means the CO2-rich mixture is denser and sinks.
    """
    rho_s = float(params.rho_s)
    rho_o = float(params.rho_o)
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    return c * rho_s + (1.0 - c) * rho_o


def _fcm_advection_matrix(grid, k, lam, rho_eff, p):
    """Advection operator ``C``: ``C @ c = div(c v)`` for the FCM velocity
    ``v = -k lam (grad p + rho_eff grad z)``.

    The face velocity is the total Darcy flux (pressure-driven, with the
    density-driven gravity head); the upwind value is the advected concentration.
    """
    from scipy.sparse import coo_matrix

    n = grid.n_cells
    k3 = np.asarray(k, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    lam3 = np.asarray(lam, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    rho3 = np.asarray(rho_eff, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    coef_x, up_x, coef_y, up_y, coef_z, up_z = _face_geometry(grid, p)
    (c1x, c2x), (c1y, c2y), (c1z, c2z) = _face_cell_pairs(grid.nx, grid.ny, grid.nz)
    rows: list = []
    cols: list = []
    vals: list = []

    def add(lo, hi, up, vel):
        lo = np.asarray(lo)
        hi = np.asarray(hi)
        up = np.asarray(up, dtype=bool)
        vel = np.asarray(vel, dtype=float)
        ulo = lo[up]
        vlo = vel[up]
        rows.append(ulo); cols.append(ulo); vals.append(vlo)
        rows.append(hi[up]); cols.append(ulo); vals.append(-vlo)
        uhi = hi[~up]
        vhi = vel[~up]
        rows.append(lo[~up]); cols.append(uhi); vals.append(vhi)
        rows.append(uhi); cols.append(uhi); vals.append(-vhi)

    if coef_x is not None:
        kh = _harmonic_mean(k3[:, :, :-1], k3[:, :, 1:]).ravel()
        lam_f = np.where(up_x.ravel(), lam3[:, :, :-1].ravel(), lam3[:, :, 1:].ravel())
        add(c1x, c2x, up_x.ravel(), kh * lam_f * coef_x.ravel())
    if coef_y is not None:
        kh = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :]).ravel()
        lam_f = np.where(up_y.ravel(), lam3[:, :-1, :].ravel(), lam3[:, 1:, :].ravel())
        add(c1y, c2y, up_y.ravel(), kh * lam_f * coef_y.ravel())
    if coef_z is not None:
        kh = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :]).ravel()
        lam_f = np.where(up_z.ravel(), lam3[:-1, :, :].ravel(), lam3[1:, :, :].ravel())
        rho_f = 0.5 * (rho3[:-1, :, :].ravel() + rho3[1:, :, :].ravel())
        area = grid.face_area_z()[1:-1, :, :].ravel()
        add(c1z, c2z, up_z.ravel(), kh * lam_f * (coef_z.ravel() - area * rho_f))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def fcm_phase_split(c, *, c_sat, swc=0.0):
    """Map the FCM solvent fraction ``c`` to phase saturations (sw, so, sg).

    Below the solubility ``c_sat`` all CO2 is dissolved (sg=0); above it the
    excess is free gas.
    """
    c = np.clip(np.asarray(c, dtype=float), 0.0, 1.0)
    nw = 1.0 - swc
    sg = np.where(c <= c_sat, 0.0, (c - c_sat) / max(1.0 - c_sat, 1.0e-12) * nw)
    sg = np.clip(sg, 0.0, nw)
    so = nw - sg
    return np.full_like(sg, swc), so, sg


def _dissolved_fraction(
    pressure: NDArray[np.float64],
    params: FluidParams,
) -> NDArray[np.float64]:
    """Equilibrium dissolved-CO2 fraction of the oil phase ``x_d = Rs/(1+Rs)``.

    Uses the *equilibrium* solubility slope ``rs_eq_slope`` (falling back to
    ``rs_slope`` when unset), which bounds how much CO2 the oil can hold at a
    given pressure — distinct from the output ``rs`` throughput metric.
    """
    slope = params.rs_eq_slope if params.rs_eq_slope > 0.0 else params.rs_slope
    rs = np.maximum(slope * np.asarray(pressure, dtype=float), 0.0)
    x_d = rs / (1.0 + rs)
    return np.clip(x_d, 0.0, 0.999999)


def _smooth_relu(x: NDArray[np.float64], delta: float) -> NDArray[np.float64]:
    """Smooth ``max(x, 0)`` (quadratic spline, C¹-continuous).

    Quadratic on ``(-delta, delta)``, 0 below and linear above. Removes the kink
    in the phase split at the saturation point so the Newton Jacobian is
    continuous (the practical equivalent of variable switching).
    """
    x = np.asarray(x, dtype=float)
    d = np.maximum(np.asarray(delta, dtype=float), 1.0e-30)
    return np.where(x >= d, x, np.where(x <= -d, 0.0, (x + d) ** 2 / (4.0 * d)))


def _split_compositional(
    sw: NDArray[np.float64],
    C: NDArray[np.float64],
    Rs: NDArray[np.float64],
    Bg: float,
    bo_slope: float = 0.0,
    smooth: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Phase split of a conserved CO2 component ``C = sg/Bg + Rs*No`` (surface).

    ``sw`` is the water saturation, ``Rs`` the equilibrium solution gas-oil ratio
    (surface gas / surface oil), ``Bg`` the gas formation-volume factor, and
    ``bo_slope`` the linear oil-swelling slope (``Bo = 1 + bo_slope*Rs``, reservoir
    oil / surface oil). With swelling, the oil occupies ``so = No*Bo`` reservoir
    volume, so the dissolved capacity per reservoir volume shrinks by ``Bo`` and
    the free gas appears earlier. ``smooth`` is the relative transition width of
    the free-gas onset (fraction of the dissolved capacity) used to keep the
    split C¹-continuous. Returns ``(sw, so, sg)`` with ``sw+so+sg=1``.
    """
    sw_p = np.clip(np.asarray(sw, dtype=float), 0.0, None)
    C_p = np.clip(np.asarray(C, dtype=float), 0.0, None)
    Rs_p = np.clip(np.asarray(Rs, dtype=float), 0.0, None)
    nw = 1.0 - sw_p
    inv_Bg = 1.0 / max(float(Bg), 1.0e-12)
    Bo = 1.0 + bo_slope * Rs_p  # reservoir oil / surface oil (swelling)
    dissolved_cap = Rs_p * nw / Bo
    denom = inv_Bg - Rs_p / Bo
    excess = C_p - dissolved_cap
    delta = smooth * np.maximum(dissolved_cap, 1.0e-6)
    sg = _smooth_relu(excess, delta) / np.maximum(denom, 1.0e-12)
    sg = np.clip(sg, 0.0, nw)
    so = nw - sg
    return sw_p, so, sg


def _forward_compositional_saturations(
    grid: CartesianGrid,
    pressure: NDArray[np.float64],
    permeability: NDArray[np.float64],
    phi: NDArray[np.float64],
    params: FluidParams,
    wells: WellMap,
    well_qw: NDArray[np.float64],
    well_qo: NDArray[np.float64],
    well_qg: NDArray[np.float64],
    times: NDArray[np.float64],
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    *,
    max_ds: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solution-gas (CO2-in-oil) saturation transport (GEM-like component model).

    Tracks the water saturation ``sw`` and the conserved surface-volume CO2
    component ``C = sg/Bg + Rs*so``; the oil/gas phase split is recomputed at
    every substep from the equilibrium dissolved fraction. The CO2 component flux
    is the free-gas Darcy flux plus the dissolved CO2 carried by the oil phase,
    so injected CO2 largely dissolves and moves with the oil. Fully implicit
    (backward Euler + Newton) with adaptive sub-stepping; the flow pressure is
    the given (kriged) ``pressure`` field, not re-solved (see ``forward_saturations``).
    """
    p_in = np.asarray(pressure, dtype=float)
    if p_in.ndim == 1:
        p_in = p_in[None, :]
    n_t = int(p_in.shape[0])
    n_c = grid.n_cells
    k = np.asarray(permeability, dtype=float)
    phi_a = np.asarray(phi, dtype=float)
    vol = grid.cell_volumes()
    qw = np.asarray(well_qw, dtype=float)
    qo = np.asarray(well_qo, dtype=float)
    qg = np.asarray(well_qg, dtype=float)
    if qw.ndim == 1:
        qw, qo, qg = qw[None, :], qo[None, :], qg[None, :]
    qw, qo, qg = _balance_well_rates(qw, qo, qg)
    times_a = np.asarray(times, dtype=float)
    inv_phiV = 1.0 / (phi_a * vol)
    sw = np.asarray(sw0, dtype=float).copy()
    sg0_a = np.asarray(sg0, dtype=float)
    # Initial surface-volume CO2 component: assume the initial oil is CO2-free,
    # so only the free gas (sg0/Bg) contributes. The injected CO2 then dissolves
    # as it moves.
    C = sg0_a.copy() / float(params.bg)
    kinetic = float(params.k_diss) > 0.0
    Cd = np.zeros(n_c)  # dissolved CO2 (kinetic state); 0 initially (oil CO2-free)
    sw_hist = np.zeros((n_t, n_c))
    so_hist = np.zeros((n_t, n_c))
    sg_hist = np.zeros((n_t, n_c))
    for t in range(n_t):
        Rs_t = _eq_solution_gas_ratio(p_in[t], params)
        Bg = float(params.bg)
        if kinetic:
            sw, so, sg, _No, Rs_act = _split_kinetic(sw, C, Cd, Rs_t, Bg, params.bo_slope)
        else:
            sw, so, sg = _split_compositional(sw, C, Rs_t, Bg, params.bo_slope)
            Rs_act = np.clip(np.where(so > 1.0e-12, (C - sg / Bg) / np.maximum(so, 1.0e-12), 0.0), 0.0, Rs_t)
        qw_t = well_cell_rates(grid, wells, qw[t])
        qo_t = well_cell_rates(grid, wells, qo[t])
        qg_t = well_cell_rates(grid, wells, qg[t])
        # qg_t is already surface-volume (GEM *BHF is a surface rate), so the
        # surface-volume CO2 component source is qg_t + Rs*qo (NOT qg_t/Bg).
        q_c = qg_t + Rs_act * qo_t
        # Use the kriged pressure field directly: the incompressible total-mobility
        # re-solve treats the surface gas rate as a reservoir rate and over-drives
        # the top->bottom gradient (2 MPa vs the true ~0.06 MPa), which is wrong
        # for a compressible (Bg<<1) injection. The kriged field is already ~0.1%.
        p = p_in[t]
        sw_hist[t] = sw.copy()
        so_hist[t] = so.copy()
        sg_hist[t] = sg.copy()
        if t < n_t - 1:
            remaining = float(times_a[t + 1] - times_a[t])
            A = _mobility_divergence_matrix(grid, k, p)
            A_grav = _gravity_divergence_matrix(grid, k)
            dt = remaining
            while remaining > 1.0e-12 and dt > 1.0e-3:
                if kinetic:
                    sw_new, so_new, sg_new, C_new, Cd_new, conv = _implicit_kinetic_step(
                        A, A_grav, inv_phiV, dt, sw, C, Cd, qw_t, qg_t, qo_t, Rs_t, params
                    )
                else:
                    sw_new, so_new, sg_new, C_new, conv = _implicit_compositional_step(
                        A, A_grav, inv_phiV, dt, sw, C, qw_t, q_c, Rs_t, params
                    )
                    Cd_new = Cd
                if conv:
                    sw, so, sg, C, Cd = sw_new, so_new, sg_new, C_new, Cd_new
                    remaining -= dt
                    dt = min(remaining, max(dt * 1.5, 1.0))
                else:
                    dt *= 0.5
    return sw_hist, so_hist, sg_hist
