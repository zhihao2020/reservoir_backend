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
    WellModelParams,
    _face_cell_pairs,
    _face_geometry,
    _harmonic_mean,
    phase_mobilities,
    corey_total_mobility,
    darcy_divergence,
    invert_rock,
    peaceman_wi,
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

def solve_pressure_peaceman(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    lam_total: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
) -> NDArray[np.float64]:
    """Solve ``div(-k·λ_total·∇p) = Σ WI·λ_total·(bhp − p)`` (Peaceman well model).

    Every well is a BHP-driven source ``q = WI·λ_total·(bhp − p_cell)``. Because the
    source is *linear* in ``p``, the ``−WI·λ_total·p`` part moves to the diagonal and
    acts as a relaxation toward the BHP: the matrix stays well-posed and the
    reservoir pressure at each well is ``bhp − drawdown`` (with ``drawdown =
    q/(WI·λ_total)``), instead of the raw wellbore BHP. The diagonal also keeps the
    pressure bounded for a compressible (Bg ≪ 1) gas injector, where a fixed
    reservoir-volume source cannot be balanced by the incompressible equation.
    """
    from scipy.sparse.linalg import spsolve

    n = grid.n_cells
    L = _tpfa_matrix_vec(grid, permeability, lam_total).tolil()
    rhs = np.zeros(n)
    bhp = np.asarray(well_bhp, dtype=float).ravel()
    directions = wells.directions if wells.directions else [np.zeros(3)] * len(wells.cells)
    for i, cells in enumerate(wells.cells):
        cells = np.asarray(cells, dtype=np.int64).ravel()
        if cells.size == 0 or i >= bhp.size or not np.isfinite(bhp[i]):
            continue
        direction = directions[i] if i < len(directions) else np.zeros(3)
        wi = peaceman_wi(
            grid, permeability, cells, direction,
            rw=wells.rw_of(i, well_params.rw), skin=wells.skin_of(i, well_params.skin),
            kv_kh=wells.kv_kh_of(i, well_params.kv_kh),
            geofac=wells.geofac_of(i, well_params.geofac),
        )
        tw = wi * lam_total[cells]
        for j, c in enumerate(cells):
            L[c, c] += tw[j]
            rhs[c] += tw[j] * float(bhp[i])
    return spsolve(L.tocsr(), rhs)

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
    well_bhp: NDArray[np.float64] | None = None,
    well_params: WellModelParams | None = None,
    freeze_pressure: bool = False,
    return_co2: bool = False,
) -> tuple[NDArray[np.float64], ...]:
    """Forward-simulate the saturation history.

    Returns ``(sw, so, sg)`` histories, each of shape ``(n_times, n_cells)``.
    ``model`` is ``"black_oil"`` or ``"compositional"``. By default the pressure
    ``pressure`` is used as the given flow field and the well rates are the fixed
    ``well_qw/qo/qg``. When ``well_bhp`` and ``well_params`` are supplied, the
    pressure is instead *solved* with the Peaceman well model (BHP-driven wells)
    and the well rates are derived from the solved pressure.

    ``freeze_pressure`` (compositional only) keeps each report pressure at the
    supplied ``pressure[t]`` and transports ``sw`` and composition. With
    ``return_co2``, the compositional model also returns the overall CO2 mole
    fraction, shape ``(n_times, n_cells)``.
    """
    name = str(model).strip().lower()
    if name == "black_oil":
        return _forward_black_oil_saturations(
            grid, pressure, permeability, phi, params, wells,
            well_qw, well_qo, well_qg, times, sw0, so0, sg0, max_ds=max_ds,
        )
    if name == "compositional":
        sw, so, sg, z_co2 = _forward_compositional_full_saturations(
            grid, pressure, permeability, phi, params, wells,
            well_qw, well_qo, well_qg, times, sw0, so0, sg0, max_ds=max_ds,
            well_bhp=well_bhp, well_params=well_params,
            freeze_pressure=freeze_pressure,
        )
        if return_co2:
            return sw, so, sg, z_co2
        return sw, so, sg
    raise ValueError(f"unknown forward model {model!r}")

def _mobility_divergence_matrix(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    pressure: NDArray[np.float64],
    z_head: float | NDArray[np.float64] | None = None,
):
    """Sparse matrix ``A`` such that ``A @ lam == div(-k lam grad p)`` (upwind).

    The divergence is *linear* in the cell mobility field ``lam`` (the upwind
    face mobility enters linearly), so ``A`` is a fixed sparse operator for a
    given permeability and pressure. This lets the fully-implicit Newton Jacobian
    be assembled as ``I + (dt/phiV) A diag(dlam/ds)`` without a numerical loop.
    ``z_head`` adds a gravity head to the z-face potential drop (see
    :func:`_face_geometry`).
    """
    from scipy.sparse import coo_matrix

    n = grid.n_cells
    k3 = np.asarray(permeability, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    coef_x, up_x, coef_y, up_y, coef_z, up_z = _face_geometry(grid, pressure, z_head)
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
    if not rows:  # single cell: no interior faces
        return coo_matrix((n, n)).tocsr()
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

# Upper bound on a forward sub-step. The GEM deck resolves the shale-oil injection
# with ``*DTMAX 0.01`` days (14.4 min); a 30-day report step is far too coarse for
# the lab-scale injection (fills the model ~4.5 h), so sub-steps are capped here.
_MAX_DT = 864.0
# Frozen pressure is not a Newton unknown, so the transport step can take a
# longer sub-step. Five days lets a 30-day month finish inside the adaptive
# step budget. The coupled FIM still uses ``_MAX_DT``.
_FROZEN_DT_MAX = 5.0 * 86400.0

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

def _phase_divergence_matrix(grid, permeability, pressure, rho_g):
    """Sparse ``A_g`` such that ``A_g @ lam == div(-k lam^up grad(p + rho_g z))``.

    The face mobility is taken upstream of the *phase potential*, not of ``p`` and
    ``z`` separately: subtracting a ``-z``-upwinded gravity operator from the
    pressure operator picks the lower cell's mobility for a sinking phase
    (``rho_g > 0``), i.e. the downstream one, so gas above a gas-free cell could
    never sink into it. ``rho_g`` is the gas head relative to the liquid (Pa/m):
    a scalar, or one value per z-face from :func:`_eos_gas_head`.
    """
    return _mobility_divergence_matrix(grid, permeability, pressure, rho_g)

_GRAVITY = 9.80665  # m/s2
# Pressure finite-difference step (Pa) for the EOS Jacobians: the 1e-6 step used
# for saturations / mole fractions is below double-precision round-off relative to
# ~2e7 Pa, so dN/dp would be noise.
_FD_STEP_P = 1.0
# Default Newton convergence of the EOS steps: residual norm < _EOS_NEWTON_RTOL·|r0|.
# Tighter values (e.g. 1e-3) conserve CO2 to ~0.3% per step but do not converge
# when a single 864 s step crosses the bubble point.
_EOS_NEWTON_RTOL = 0.1

def _eos_gas_head(grid, sl, sg, rho_l, rho_g, fallback):
    """Per-z-face gas head ``g*(rho_g - rho_l)`` (Pa/m) from EOS phase densities.

    Each face density is the saturation-weighted mean of its two cells (the
    OPM/MRST face density), so a phase present on one side only takes that side's
    density. A face with no liquid on either side uses the domain liquid density
    (saturation-weighted mean); a face with no gas on either side carries no gas
    flux and keeps the scalar ``fallback`` head. Returns shape ``(nz-1, ny, nx)``.
    """
    shape = (grid.nz, grid.ny, grid.nx)
    if grid.nz < 2:
        return np.zeros((0, grid.ny, grid.nx))
    sl3 = np.clip(np.asarray(sl, dtype=float), 0.0, None).reshape(shape)
    sg3 = np.clip(np.asarray(sg, dtype=float), 0.0, None).reshape(shape)
    rl3 = np.asarray(rho_l, dtype=float).reshape(shape)
    rg3 = np.asarray(rho_g, dtype=float).reshape(shape)
    sl_tot = float(sl3.sum())
    rho_l_ref = float((sl3 * rl3).sum() / sl_tot) if sl_tot > 0.0 else 0.0
    wl = sl3[:-1] + sl3[1:]
    wg = sg3[:-1] + sg3[1:]
    rl_f = np.divide(sl3[:-1] * rl3[:-1] + sl3[1:] * rl3[1:], wl,
                     out=np.full(wl.shape, rho_l_ref), where=wl > 1.0e-12)
    rg_f = np.divide(sg3[:-1] * rg3[:-1] + sg3[1:] * rg3[1:], wg,
                     out=np.zeros(wg.shape), where=wg > 1.0e-12)
    return np.where(wg > 1.0e-12, _GRAVITY * (rg_f - rl_f), float(fallback))

def _injector_inv_molar_volume(n_cells, inj_cells, pressure):
    """``(1/v_CO2, d(1/v_CO2)/dp)`` of the injected pure CO2 at ``inj_cells`` (0 elsewhere).

    ``v_CO2`` is the EOS reservoir molar volume at the cell pressure, so the
    injector's reservoir rate ``WI·λt·(bhp−p)`` converts to moles consistently with
    the inter-cell fluxes (the scalar ``Bg·V_CO2_STD`` it replaces is exact only at
    one pressure).
    """
    from ..core.pr_eos import co2_molar_volume

    inv = np.zeros(n_cells)
    dinv = np.zeros(n_cells)
    if inj_cells.size:
        pc = np.asarray(pressure, dtype=float)[inj_cells]
        dp = 1.0e-7 * np.maximum(np.abs(pc), 1.0)
        inv_c = 1.0 / co2_molar_volume(pc)
        inv[inj_cells] = inv_c
        dinv[inj_cells] = (1.0 / co2_molar_volume(pc + dp) - inv_c) / dp
    return inv, dinv

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
    from scipy.sparse import diags, identity

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
        # Water and oil are decoupled (the Jacobian is block-diagonal), so solve the
        # two n×n systems separately instead of one 2n×2n bmat — cheaper and avoids
        # the ill-conditioned zero blocks that stall GMRES on the larger system.
        delta_sw = _solve_linear(J_ww, -r_sw)
        delta_so = _solve_linear(J_ss, -r_so)
        delta = np.concatenate([delta_sw, delta_so])
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

def _peaceman_well_data(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
    injects_gas: NDArray[np.bool_],
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Flatten every well completion cell to ``(cells, wi, bhp, is_inj)`` arrays.

    ``wi`` is the anisotropic Peaceman well index per completion cell (geometric,
    independent of pressure/saturation); ``bhp`` the well bottom-hole pressure;
    ``is_inj`` whether the well injects gas (its total rate goes to the gas phase).
    """
    bhp = np.asarray(well_bhp, dtype=float).ravel()
    inj = np.asarray(injects_gas, dtype=bool).ravel()
    directions = wells.directions if wells.directions else [np.zeros(3)] * len(wells.cells)
    cells_l, wi_l, bhp_l, inj_l = [], [], [], []
    for i, cells in enumerate(wells.cells):
        cells = np.asarray(cells, dtype=np.int64).ravel()
        if cells.size == 0 or i >= bhp.size or not np.isfinite(bhp[i]):
            continue
        direction = directions[i] if i < len(directions) else np.zeros(3)
        wi = peaceman_wi(
            grid, permeability, cells, direction,
            rw=wells.rw_of(i, well_params.rw), skin=wells.skin_of(i, well_params.skin),
            kv_kh=wells.kv_kh_of(i, well_params.kv_kh),
            geofac=wells.geofac_of(i, well_params.geofac),
        )
        is_inj = bool(inj[i]) if i < inj.size else False
        for j, c in enumerate(cells):
            cells_l.append(c)
            wi_l.append(float(wi[j]))
            bhp_l.append(float(bhp[i]))
            inj_l.append(is_inj)
    return (
        np.asarray(cells_l, dtype=np.int64),
        np.asarray(wi_l, dtype=float),
        np.asarray(bhp_l, dtype=float),
        np.asarray(inj_l, dtype=bool),
    )

def _implicit_compositional_full_step(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    z0: NDArray[np.float64],
    p0: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
    params: FluidParams,
    injects_gas: NDArray[np.bool_],
    *,
    well_qg_fixed: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    tol: float = _EOS_NEWTON_RTOL,
    freeze_pressure: bool = False,
    p_fixed: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """One fully-implicit full-compositional (14-component) step (Newton).

    Primary variables ``(p, sw, z_1..z_13)`` (``z`` is the 14-component overall
    mole fraction, the last component implied). Each component is conserved as
    *moles* (``z_c·N``) with the molar flux ``y_c·λg/v_g + x_c·λl/v_l``; the
    pressure is the total molar balance and the gas/liquid split is the full PR
    flash. Well terms use the same EOS molar volumes; ``params.gravity == "eos"``
    switches the gas head to the per-face EOS density difference.

    ``freeze_pressure`` drops ``p`` from the unknowns and holds it at ``p_fixed``
    (or at ``p0`` when ``p_fixed`` is omitted). ``p0`` stays the accumulation
    reference at the start of the step. The injector rate constraint and the
    producer BHP terms are unchanged.
    Returns ``(p, sw, sl, sg, z, converged)``.
    """
    from scipy.sparse import bmat as _bmat
    from scipy.sparse import diags
    from scipy.sparse.linalg import gmres as _gmres
    from scipy.sparse.linalg import splu as _splu

    from ..core.pr_eos import (_Z_OIL, _CO2_IDX, _V_CO2_STD,
                               flash_direct_full, phase_mass_densities, phase_molar_volumes)

    ncomp = _Z_OIL.size  # 14
    # Only scales the rate-constraint row (reservoir-volume units).
    Bg = float(params.bg)
    n = grid.n_cells
    sw = np.asarray(sw0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()  # (n, 14)
    if freeze_pressure:
        p = np.asarray(p0 if p_fixed is None else p_fixed, dtype=float).copy()
    else:
        p = np.asarray(p0, dtype=float).copy()
    h = 1.0e-6
    hp = _FD_STEP_P
    accum = 1.0 / (dt * inv_phiV)
    I = diags(accum)
    cells, wi, bhp_w, is_inj = _peaceman_well_data(
        grid, permeability, wells, well_bhp, well_params, injects_gas,
    )
    rate_controlled = well_qg_fixed is not None
    qg_fixed = np.zeros(n, dtype=float)
    if rate_controlled:
        qg_fixed = np.asarray(well_qg_fixed, dtype=float).ravel()
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

    def state(p_, sw_, z_):
        V_, x_, y_ = flash_direct_full(z_, p_)
        # EOS (Peneloux-shifted) reservoir phase molar volumes (per cell).
        v_l_, v_g_ = phase_molar_volumes(x_, y_, p_)
        denom = V_ * v_g_ + (1.0 - V_) * v_l_
        N_ = (1.0 - sw_) / np.maximum(denom, 1.0e-30)
        sg_ = V_ * N_ * v_g_
        sl_ = (1.0 - V_) * N_ * v_l_
        lam_w_, lam_l_, lam_g_ = phase_mobilities(sw_, sl_, sg_, params)
        if eos_gravity:
            rho_l_, rho_g_ = phase_mass_densities(x_, y_, v_l_, v_g_)
            head_ = _eos_gas_head(grid, sl_, sg_, rho_l_, rho_g_, params.rho_g)
        else:
            head_ = params.rho_g
        return sl_, sg_, V_, x_, y_, N_, lam_w_, lam_l_, lam_g_, v_l_, v_g_, head_

    def inj_inv_v(p_):
        return _injector_inv_molar_volume(n, inj_cells, p_)[0]

    def well_diag_rates(p_, lam_w_, lam_l_, lam_g_, v_l_, v_g_, bhp_inj_):
        """Well diagonal ``D`` and rates: water (reservoir m3/s), liquid / gas (mol/s)."""
        D_ = np.zeros(n)
        qw_ = np.zeros(n); qo_ = np.zeros(n); qg_ = np.zeros(n)
        lam_t_ = lam_w_ + lam_l_ + lam_g_
        inv_v_l_ = 1.0 / np.maximum(v_l_, 1.0e-30)
        inv_v_g_ = 1.0 / np.maximum(v_g_, 1.0e-30)
        inv_v_inj_ = inj_inv_v(p_)
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and rate_controlled) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            D_[c] += w * lam_t_[c]
            if is_inj[idx]:
                if not (freeze_pressure and rate_controlled):
                    qg_[c] += w * lam_t_[c] * dp * inv_v_inj_[c]
            else:
                qw_[c] += w * lam_w_[c] * dp
                qo_[c] += w * lam_l_[c] * dp * inv_v_l_[c]
                qg_[c] += w * lam_g_[c] * dp * inv_v_g_[c]
        if freeze_pressure and rate_controlled:
            # Prescribed surface rate, already converted. With pressure held
            # fixed, Peaceman would demand an unphysical drawdown at shale
            # permeability; the injector source is the target itself.
            qg_ += qg_fixed / _V_CO2_STD
        return D_, qw_, qo_, qg_

    def inj_diag(lam_t_):
        D_inj_ = np.zeros(n)
        for idx in range(cells.size):
            if is_inj[idx]:
                D_inj_[cells[idx]] += wi[idx] * lam_t_[cells[idx]]
        return D_inj_

    def rate_residual(p_, D_inj_, bhp_inj_):
        """Injector surface-rate constraint, scaled by ``Bg`` to reservoir-volume units."""
        if not rate_controlled or freeze_pressure:
            return 0.0
        q_surf = _V_CO2_STD * float(D_inj_ @ ((bhp_inj_ - p_) * inj_inv_v(p_)))
        return Bg * (q_surf - qg_target)

    def residual(p_, sw_, z_, bhp_inj_):
        (sl_, sg_, V_, x_, y_, N_, lam_w_, lam_l_, lam_g_,
         v_l_, v_g_, head_) = state(p_, sw_, z_)
        lam_t_ = lam_w_ + lam_l_ + lam_g_
        # molar well rates (injector pure CO2; producer gas + liquid split)
        D_, qw_, qo_mol_, qg_mol_ = well_diag_rates(p_, lam_w_, lam_l_, lam_g_, v_l_, v_g_, bhp_inj_)
        if freeze_pressure:
            A_ = flux_ops["A"]
            A_g_ = flux_ops["Ag"]
        else:
            A_ = _mobility_divergence_matrix(grid, permeability, p_)
            A_g_ = _phase_divergence_matrix(grid, permeability, p_, head_)
        r_p_ = (accum * ((N_ - N0) + params.ct * N_ * (p_ - p0)) - (qg_mol_ + qo_mol_)
                + A_g_ @ (lam_g_ / np.maximum(v_g_, 1.0e-30))
                + A_ @ (lam_l_ / np.maximum(v_l_, 1.0e-30)))
        rock_p_ = params.ct * (p_ - p0)
        r_sw_ = accum * ((sw_ - sw0) + rock_p_ * sw_) - qw_ + A_ @ lam_w_
        # component residuals (moles)
        rc = []
        for c in range(ncomp):
            q_c_ = np.where(inj_cell, (qg_mol_ if c == _CO2_IDX else 0.0),
                            y_[:, c] * qg_mol_ + x_[:, c] * qo_mol_)
            held_ = z_[:, c] * N_
            r_c_ = (accum * ((held_ - z0[:, c] * N0) + rock_p_ * held_)
                    - q_c_ + A_g_ @ (y_[:, c] * lam_g_ / v_g_)
                    + A_ @ (x_[:, c] * lam_l_ / v_l_))
            rc.append(r_c_)
        D_inj_ = inj_diag(lam_t_)
        r_rate_ = rate_residual(p_, D_inj_, bhp_inj_)
        return np.concatenate([r_p_, r_sw_] + rc + [np.array([r_rate_])])

    N0 = _split_full_N(sw0, z0, p0, params)
    # Pressure is fixed for this sub-step, so the upwind divergence operators do
    # not change between Newton iterations or line-search residual evaluations.
    flux_ops: dict[str, object] = {}
    if freeze_pressure:
        head0 = state(p, sw, z)[-1]
        flux_ops["A"] = _mobility_divergence_matrix(grid, permeability, p)
        flux_ops["Ag"] = _phase_divergence_matrix(grid, permeability, p, head0)

    def newton_vector(full: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rows the convergence test uses.

        Frozen pressure drops the molar-pressure residual (it is not an
        unknown). The implied last component stays out of the Jacobian; the
        rate scalar stays in the norm.
        """
        if not freeze_pressure:
            return full
        return np.concatenate([full[n:(ncomp + 1) * n], full[-1:]])

    r0_norm = float(np.linalg.norm(newton_vector(residual(p, sw, z, bhp_inj))))
    converged = False
    for _ in range(max_iter):
        sl, sg, V, x, y, N, lam_w, lam_l, lam_g, v_l, v_g, head = state(p, sw, z)
        m_Ng = lam_g / np.maximum(v_g, 1.0e-30)  # gas molar mobility (gas potential)
        m_Nl = lam_l / np.maximum(v_l, 1.0e-30)  # liquid molar mobility (p)
        lam_t = lam_w + lam_l + lam_g
        D, qw, qo, qg = well_diag_rates(p, lam_w, lam_l, lam_g, v_l, v_g, bhp_inj)
        D_inj = inj_diag(lam_t)
        inv_vi, dinv_vi = _injector_inv_molar_volume(n, inj_cells, p)
        dp_inj = np.where(inj_cell, (bhp_inj if rate_controlled else bhp_full) - p, 0.0)
        g_inj_p = D * (inv_vi - dp_inj * dinv_vi)  # −∂(injector mol/s)/∂p
        if freeze_pressure:
            A = flux_ops["A"]
            A_g = flux_ops["Ag"]
        else:
            A = _mobility_divergence_matrix(grid, permeability, p)
            A_g = _phase_divergence_matrix(grid, permeability, p, head)
        r_full = residual(p, sw, z, bhp_inj)
        # numerical derivatives of the split (N, x, y, λ) wrt sw/z, and wrt p
        # only when pressure is an unknown.
        _, _, _, _, _, N_s, lw_s, ll_s, lg_s, _, _, _ = state(p, sw + h, z)
        dN_dsw = (N_s - N) / h
        dlw_dsw = (lw_s - lam_w) / h
        dz = {}
        for c in range(ncomp - 1):
            zc = z.copy(); zc[:, c] += h; zc[:, -1] -= h
            zc = zc / zc.sum(axis=1, keepdims=True)
            _, _, _, x_c, y_c, N_c, lw_c, ll_c, lg_c, _, _, _ = state(p, sw, zc)
            dz[c] = (x_c, y_c, N_c, lw_c, ll_c, lg_c)
        J_ww = I + A @ diags(dlw_dsw)
        if not freeze_pressure:
            _, _, _, _, _, N_p, lw_p, ll_p, lg_p, _, _, _ = state(p + hp, sw, z)
            dN_dp = (N_p - N) / hp
            frac_t = D / np.maximum(lam_t, 1.0e-12)
            dlam_t_dp = (lw_p + ll_p + lg_p - lam_t) / hp  # approx for the flux p-derivative (secondary)
            D_molar = np.where(inj_cell, g_inj_p, frac_t * (m_Nl + m_Ng))
            rock_ct = accum * params.ct * (N + (p - p0) * dN_dp)
            J_pp = (diags(dN_dp * accum + rock_ct) + A @ diags(dlam_t_dp)
                    + _upwind_tpfa_matrix_vec(grid, permeability, m_Ng, p, head)
                    + _upwind_tpfa_matrix_vec(grid, permeability, m_Nl, p)
                    + diags(D_molar)).tocsr()
            J_pw = diags(dN_dsw * accum)
            J_wp = (diags(np.where(inj_cell, 0.0, frac_t * lam_w))
                    + _upwind_tpfa_matrix_vec(grid, permeability, lam_w, p)).tocsr()
        # component blocks (molar formulation)
        row_blocks = []
        for c in range(ncomp - 1):
            dFg_dsw = y[:, c] * (lg_s - lam_g) / h / v_g
            dFl_dsw = x[:, c] * (ll_s - lam_l) / h / v_l
            dq_c_dsw = np.where(inj_cell, 0.0,
                                y[:, c] * (qg * 0.0) + x[:, c] * 0.0)  # lag well sw-deriv
            J_cw = (diags((dN_dsw * z[:, c]) * accum) + A_g @ diags(dFg_dsw)
                    + A @ diags(dFl_dsw) - diags(dq_c_dsw))
            col_blocks = []
            for j in range(ncomp - 1):
                x_j, y_j, N_j, lw_j, ll_j, lg_j = dz[j]
                dN_dzj = (N_j - N) / h
                dy_dzj = (y_j[:, c] - y[:, c]) / h
                dx_dzj = (x_j[:, c] - x[:, c]) / h
                dlamg_dzj = (lg_j - lam_g) / h
                dlaml_dzj = (ll_j - lam_l) / h
                J_cc = (diags((dN_dzj * z[:, c] + (1.0 if j == c else 0.0) * N) * accum)
                        + A_g @ diags(dy_dzj * lam_g / v_g + y[:, c] * dlamg_dzj / v_g)
                        + A @ diags(dx_dzj * lam_l / v_l + x[:, c] * dlaml_dzj / v_l))
                col_blocks.append(J_cc)
            if freeze_pressure:
                row_blocks.append([J_cw] + col_blocks)
            else:
                # ∂r_c/∂p (well p-derivative + flux ∇p-derivative)
                J_cp = (diags(np.where(inj_cell, (g_inj_p if c == _CO2_IDX else 0.0),
                                frac_t * (y[:, c] * m_Ng + x[:, c] * m_Nl)))
                        + _upwind_tpfa_matrix_vec(grid, permeability, y[:, c] * lam_g / v_g, p, head)
                        + _upwind_tpfa_matrix_vec(grid, permeability, x[:, c] * lam_l / v_l, p)).tocsr()
                row_blocks.append([J_cp, J_cw] + col_blocks)
        r = -r_full
        if freeze_pressure:
            # Unknowns are (sw, z_1..z_13). Pressure is the kriged field.
            blocks = [[J_ww] + [None] * (ncomp - 1)] + row_blocks
            rhs = r[n:(ncomp + 1) * n].copy()
        else:
            top = [[J_pp, J_pw] + [None] * (ncomp - 1)]
            wrow = [[J_wp, J_ww] + [None] * (ncomp - 1)]
            blocks = top + wrow + [[b if b is not None else None for b in row] for row in row_blocks]
            rhs = r[:(ncomp + 1) * n].copy()
        J_blk = _bmat(blocks, format="csr")
        delta_bhp = 0.0
        if rate_controlled and not freeze_pressure:
            # The injector is pure CO2 (_CO2_IDX) at WI·λt·(bhp−p)/v_CO2 mol/s.
            # ∂/∂bhp_inj = −D_inj/v_CO2 on the molar-balance r_p row (coupled mode)
            # and on the CO2 component row. r_sw has no injector dependence.
            g_inj = D_inj * inv_vi
            B_c = np.concatenate([-g_inj, np.zeros(n)]
                                 + [np.zeros(n)] * _CO2_IDX
                                 + [-g_inj]
                                 + [np.zeros(n)] * (ncomp - 2 - _CO2_IDX))
            C_b = np.concatenate([-Bg * _V_CO2_STD * np.where(inj_cell, g_inj_p, 0.0), np.zeros(n)]
                                 + [np.zeros(n)] * (ncomp - 1))
            d_rr = float(Bg * _V_CO2_STD * g_inj.sum())
            # Factor J_blk once and reuse it for both the B_c and rhs solves
            # (the two direct solves previously factored the same matrix twice).
            try:
                lu = _splu(J_blk.tocsc())
            except RuntimeError:
                break  # singular at a phase boundary; caller cuts the step
            u = lu.solve(B_c)
            delta_x0 = lu.solve(rhs)
            denom = d_rr - float(C_b @ u)
            # r_rate is the last entry: the residual also carries the 14th (implied)
            # component block, so it sits at (ncomp + 2) * n, not (ncomp + 1) * n.
            delta_bhp = ((r[-1] - float(C_b @ delta_x0)) / denom
                         if abs(denom) > 1.0e-30 else 0.0)
            delta = delta_x0 - delta_bhp * u
        else:
            try:
                if freeze_pressure:
                    # SuperLU fill-in on the 15^3 transport Jacobian exhausts memory.
                    # An iterative solve stays within RAM and is only used here.
                    delta, info = _gmres(J_blk, rhs, rtol=1.0e-5, restart=30, maxiter=60)
                    if info != 0 or not np.isfinite(delta).all():
                        break
                else:
                    delta = _splu(J_blk.tocsc()).solve(rhs)
            except (RuntimeError, MemoryError):
                break
        if freeze_pressure:
            delta_p = np.zeros(n)
            delta_sw = delta[:n]
            delta_z = delta[n:ncomp * n].reshape(ncomp - 1, n).T
        else:
            delta_p = delta[:n]
            delta_sw = delta[n:2 * n]
            delta_z = delta[2 * n: (ncomp + 1) * n].reshape(ncomp - 1, n).T
        alpha = 1.0
        r_norm = float(np.linalg.norm(newton_vector(r_full)))
        p_new = p.copy(); sw_new = sw.copy(); z_new = z.copy(); bhp_new = bhp_inj
        r_new = r_full
        for _ in range(12):
            p_new = p if freeze_pressure else p + alpha * delta_p
            sw_new = np.clip(sw + alpha * delta_sw, 0.0, 1.0)
            z_new = z.copy()
            for c in range(ncomp - 1):
                z_new[:, c] += alpha * delta_z[:, c]
                if freeze_pressure:
                    # Match the mole-fraction finite difference: the dependent
                    # component gives up what the independent ones gain.
                    z_new[:, -1] -= alpha * delta_z[:, c]
            z_new = np.clip(z_new, 0.0, 1.0)
            z_new = z_new / np.maximum(z_new.sum(axis=1, keepdims=True), 1.0e-30)
            bhp_new = bhp_inj + alpha * delta_bhp
            if freeze_pressure and alpha == 1.0 and np.isfinite(z_new).all() and np.isfinite(sw_new).all():
                r_new = residual(p_new, sw_new, z_new, bhp_new)
                break
            r_new = residual(p_new, sw_new, z_new, bhp_new)
            r_new_n = newton_vector(r_new)
            if np.isfinite(r_new_n).all() and float(np.linalg.norm(r_new_n)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate(
            [p_new - p, sw_new - sw, (z_new - z).ravel()])))
        z_old = z
        p, sw, z, bhp_inj = p_new, sw_new, z_new, bhp_new
        r_now = float(np.linalg.norm(newton_vector(r_new))) if np.isfinite(r_new).all() else np.inf
        co2_gain = float(np.max(z_new[:, _CO2_IDX] - z_old[:, _CO2_IDX]))
        co2_drop = float(np.min(z_new[:, _CO2_IDX] - z_old[:, _CO2_IDX]))
        already_rich = float(np.max(z_old[:, _CO2_IDX])) > 0.5
        finite_state = np.isfinite(p_new).all() and np.isfinite(sw_new).all() and np.isfinite(z_new).all()
        if freeze_pressure and finite_state and (
            co2_gain > 1.0e-3 or (already_rich and co2_drop > -1.0e-3) or r_now < r0_norm
        ):
            # Keep a finite CO2 update. Extra iterates on a cell that is already
            # full make the flash singular at fixed pressure.
            converged = True
            break
        norm_x = float(np.linalg.norm(np.concatenate([p, sw, z.ravel()])))
        if r_now < tol * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    sl, sg = state(p, sw, z)[:2]
    return p, sw, sl, sg, z, converged

# GEM ``*PHASEID *OIL``: a single hydrocarbon phase is reported as oil.
# The two-phase branch keeps the volumetric split. Near-pure CO2 (V = 1) is
# therefore sg = 0, not a free-gas saturation of 1.
_SINGLE_PHASE = 1.0e-8


def _hydrocarbon_saturations(sw, V, v_l, v_g):
    """Liquid/gas saturations from a flash, with single-phase labeled oil."""
    sw_p = np.asarray(sw, dtype=float)
    V_p = np.asarray(V, dtype=float)
    denom = V_p * v_g + (1.0 - V_p) * v_l
    N = (1.0 - sw_p) / np.maximum(denom, 1.0e-30)
    sg = V_p * N * v_g
    sl = (1.0 - V_p) * N * v_l
    single = (V_p <= _SINGLE_PHASE) | (V_p >= 1.0 - _SINGLE_PHASE)
    sg = np.where(single, 0.0, sg)
    sl = np.where(single, 1.0 - sw_p, sl)
    return sl, sg


def reservoir_co2_to_surface(q_reservoir, pressure):
    """Reservoir CO2 rate (m3/s) → surface rate for the injector constraint."""
    from ..core.pr_eos import _V_CO2_STD, co2_molar_volume

    v = float(np.asarray(co2_molar_volume(pressure), dtype=float).ravel()[0])
    return np.asarray(q_reservoir, dtype=float) * (_V_CO2_STD / max(v, 1.0e-30))


def _split_full(sw, z, p):
    """14-component flash split → ``(sw, sl, sg)``."""
    from ..core.pr_eos import flash_direct_full, phase_molar_volumes
    sw_p = np.asarray(sw, dtype=float)
    V, x, y = flash_direct_full(z, p)
    v_l, v_g = phase_molar_volumes(x, y, p)
    sl, sg = _hydrocarbon_saturations(sw_p, V, v_l, v_g)
    return sw_p, sl, sg

def _split_full_N(sw, z, p, params):
    """Total hydrocarbon moles N for the full-compositional model."""
    from ..core.pr_eos import flash_direct_full, phase_molar_volumes
    V, x, y = flash_direct_full(z, p)
    v_l, v_g = phase_molar_volumes(x, y, p)
    denom = V * v_g + (1.0 - V) * v_l
    return (1.0 - np.asarray(sw, dtype=float)) / np.maximum(denom, 1.0e-30)

def _implicit_compositional_full_adaptive(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    z0: NDArray[np.float64],
    p0: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
    params: FluidParams,
    injects_gas: NDArray[np.bool_],
    *,
    well_qg_fixed: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    tol: float = _EOS_NEWTON_RTOL,
    dt_sub0: float | None = None,
    dt_min: float | None = None,
    dt_max: float | None = None,
    dt_growth: float = 2.0,
    max_substeps: int = 64,
    freeze_pressure: bool = False,
    p_end: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Adaptive sub-stepping around :func:`_implicit_compositional_full_step`.

    With ``freeze_pressure``, ``p`` is linearly interpolated from ``p0`` to
    ``p_end`` across the interval and is not a Newton unknown.
    """
    dt_min = (dt / 64.0) if dt_min is None else dt_min
    dt_max = dt if dt_max is None else dt_max
    p_start = np.asarray(p0, dtype=float).copy()
    p_target = p_start if p_end is None else np.asarray(p_end, dtype=float)
    p = p_start.copy()
    sw = np.asarray(sw0, dtype=float).copy()
    z = np.asarray(z0, dtype=float).copy()
    _, sl, sg = _split_full(sw, z, p)
    remaining = float(dt)
    elapsed = 0.0
    span = float(dt)
    dt_sub = min(dt_sub0 if dt_sub0 is not None else dt_max, remaining, dt_max)
    last_dt = dt_sub
    failed_above = None
    for _ in range(max_substeps):
        if remaining <= 1.0e-12:
            break
        if freeze_pressure and span > 0.0:
            a0 = elapsed / span
            a1 = min(elapsed + dt_sub, span) / span
            p_sub0 = (1.0 - a0) * p_start + a0 * p_target
            p_sub1 = (1.0 - a1) * p_start + a1 * p_target
        else:
            p_sub0 = p
            p_sub1 = None
        p_new, sw_new, sl_new, sg_new, z_new, conv = _implicit_compositional_full_step(
            grid, permeability, inv_phiV, dt_sub, sw, z, p_sub0,
            wells, well_bhp, well_params, params, injects_gas,
            well_qg_fixed=well_qg_fixed, max_iter=max_iter, tol=tol,
            freeze_pressure=freeze_pressure, p_fixed=p_sub1,
        )
        if conv:
            p, sw, z = p_new, sw_new, z_new
            sl, sg = sl_new, sg_new
            remaining -= dt_sub
            elapsed += dt_sub
            last_dt = dt_sub
            grown = min(dt_growth * dt_sub, dt_max, remaining)
            if failed_above is not None and grown >= failed_above * (1.0 - 1.0e-12):
                grown = min(dt_sub, remaining)
            dt_sub = grown
        else:
            failed_above = dt_sub
            dt_sub *= 0.5
            if dt_sub < dt_min:
                break
    return p, sw, sl, sg, z, remaining <= 1.0e-12, last_dt

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

def _upwind_tpfa_matrix_vec(grid, permeability, mobility, pressure, z_head=None):
    """Upwind TPFA Laplacian: ``L @ p == div(-k mobility^up grad p)``.

    The flux-p-derivative of the residual's upwind divergence ``A @ m`` (see
    :func:`_mobility_divergence_matrix`) is the TPFA Laplacian with the *upwind*
    face mobility, not the arithmetic-mean face mobility used by
    :func:`_tpfa_matrix_vec`. The two agree only for a uniform mobility field, so
    the fully-implicit Jacobian's flux-p block was wrong where the phase mobilities
    jump (the injector's gas front), which stalled the Newton. ``z_head`` selects
    the upwind side on the phase potential (as in :func:`_phase_divergence_matrix`).
    """
    from scipy.sparse import coo_matrix

    n = grid.n_cells
    k3 = np.asarray(permeability, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    lam3 = np.asarray(mobility, dtype=float).reshape((grid.nz, grid.ny, grid.nx))
    _, up_x, _, up_y, _, up_z = _face_geometry(grid, pressure, z_head)
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
        lamf = np.where(up_x, lam3[:, :, :-1], lam3[:, :, 1:])
        area = dy[None, :, None] * dz[:, None, None]
        dist = 0.5 * (dx[:-1] + dx[1:])
        add(c1x, c2x, kh * lamf * area / dist[None, None, :])
    if grid.ny > 1:
        kh = _harmonic_mean(k3[:, :-1, :], k3[:, 1:, :])
        lamf = np.where(up_y, lam3[:, :-1, :], lam3[:, 1:, :])
        area = dz[:, None, None] * dx[None, None, :]
        dist = 0.5 * (dy[:-1] + dy[1:])
        add(c1y, c2y, kh * lamf * area / dist[None, :, None])
    if grid.nz > 1:
        kh = _harmonic_mean(k3[:-1, :, :], k3[1:, :, :])
        lamf = np.where(up_z, lam3[:-1, :, :], lam3[1:, :, :])
        area = dx[None, None, :] * dy[None, :, None]
        dist = 0.5 * (dz[:-1] + dz[1:])
        add(c1z, c2z, kh * lamf * area / dist[:, None, None])
    if not rows:  # single cell: no interior faces
        return coo_matrix((n, n)).tocsr()
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

def _forward_compositional_full_saturations(
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
    well_bhp: NDArray[np.float64] | None = None,
    well_params: WellModelParams | None = None,
    freeze_pressure: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """14-component molar compositional forward (full PR flash).

    Tracks ``(p, sw, z)`` with ``z`` the 14-component overall mole fraction.
    Returns ``(sw, so, sg, z_co2)`` where ``so`` is the liquid saturation and
    ``z_co2`` is the overall CO2 mole fraction. With ``freeze_pressure``, each
    report step uses ``pressure[t]`` and interpolates it toward ``pressure[t+1]``
    instead of re-solving the pressure.
    """
    from ..core.pr_eos import _CO2_IDX, _Z_OIL_DEAD

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
    if well_bhp is None or well_params is None:
        raise ValueError(
            "compositional forward model requires well_bhp + well_params"
        )
    bhp = np.asarray(well_bhp, dtype=float)
    if bhp.ndim == 1:
        bhp = bhp[None, :]
    injects_gas = (qg > 0.0).any(axis=0)
    times_a = np.asarray(times, dtype=float)
    inv_phiV = 1.0 / (phi_a * vol)
    sw = np.asarray(sw0, dtype=float).copy()
    z = np.broadcast_to(_Z_OIL_DEAD, (n_c, _Z_OIL_DEAD.size)).copy()
    sw_hist = np.zeros((n_t, n_c))
    so_hist = np.zeros((n_t, n_c))
    sg_hist = np.zeros((n_t, n_c))
    z_hist = np.zeros((n_t, n_c))
    p = np.asarray(p_in[0], dtype=float).copy()
    dt_sub0: float | None = None
    for t in range(n_t):
        if freeze_pressure:
            p = np.asarray(p_in[t], dtype=float).copy()
        _, sl_r, sg_r = _split_full(sw, z, p)
        sw_hist[t] = sw.copy()
        so_hist[t] = sl_r.copy()
        sg_hist[t] = sg_r.copy()
        z_hist[t] = z[:, _CO2_IDX]
        if t >= n_t - 1:
            continue
        qg_fixed = np.zeros(n_c)
        # ``qg`` is a reservoir rate (m3/s). The injector constraint compares a
        # surface rate ``V_std * (moles/s)``; using the reservoir number there
        # under-injects by ``V_std / v_CO2`` (~300 at 19 MPa).
        p_inj = float(np.mean(bhp[t, injects_gas])) if np.any(injects_gas) else float(np.mean(p))
        for i in np.flatnonzero(injects_gas):
            cells_i = wells.cells[i]
            if cells_i.size > 0:
                q_surf = float(np.asarray(reservoir_co2_to_surface(qg[t, i], p_inj)).ravel()[0])
                qg_fixed[cells_i] += q_surf / cells_i.size
        dt = float(times_a[t + 1] - times_a[t])
        p_end = np.asarray(p_in[t + 1], dtype=float) if freeze_pressure else None
        dt_cap = _FROZEN_DT_MAX if freeze_pressure else _MAX_DT
        p, sw, sl, sg, z, conv, last_dt = _implicit_compositional_full_adaptive(
            grid, k, inv_phiV, dt, sw, z, p, wells, bhp[t], well_params, params, injects_gas,
            well_qg_fixed=qg_fixed, dt_sub0=dt_sub0, dt_max=dt_cap, dt_min=1.0,
            freeze_pressure=freeze_pressure, p_end=p_end,
            max_iter=4 if freeze_pressure else 30,
            max_substeps=48 if freeze_pressure else 64,
        )
        dt_sub0 = last_dt
        if not conv:
            import warnings

            warnings.warn(
                f"compositional forward step t={t} did not converge",
                RuntimeWarning,
            )
    return sw_hist, so_hist, sg_hist, z_hist

