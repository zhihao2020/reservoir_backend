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
            rw=well_params.rw, skin=well_params.skin, kv_kh=well_params.kv_kh,
        )
        tw = wi * lam_total[cells]
        for j, c in enumerate(cells):
            L[c, c] += tw[j]
            rhs[c] += tw[j] * float(bhp[i])
    return spsolve(L.tocsr(), rhs)


def well_cell_rates_peaceman(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    lam_w: NDArray[np.float64],
    lam_o: NDArray[np.float64],
    lam_g: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    pressure: NDArray[np.float64],
    well_params: WellModelParams,
    *,
    injects_gas: NDArray[np.bool_],
    bg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Peaceman well phase rates (surface volume) from the solved pressure.

    ``q_phase = WI·λ_phase·(bhp − p_cell)`` per completion cell, converted back to
    *surface* volume (gas ÷ Bg, oil ÷ Bo≈1) so it sits in the same surface-volume
    units as the CO₂ component source ``q_c = qg + Rs·qo``. A gas injector puts the
    whole total rate into the gas phase (the injected fluid is pure gas); a
    producer splits the total rate by phase mobility. Injection is positive.
    """
    n = grid.n_cells
    qw = np.zeros(n)
    qo = np.zeros(n)
    qg = np.zeros(n)
    bhp = np.asarray(well_bhp, dtype=float).ravel()
    p = np.asarray(pressure, dtype=float).ravel()
    directions = wells.directions if wells.directions else [np.zeros(3)] * len(wells.cells)
    inj = np.asarray(injects_gas, dtype=bool).ravel()
    for i, cells in enumerate(wells.cells):
        cells = np.asarray(cells, dtype=np.int64).ravel()
        if cells.size == 0 or i >= bhp.size or not np.isfinite(bhp[i]):
            continue
        direction = directions[i] if i < len(directions) else np.zeros(3)
        wi = peaceman_wi(
            grid, permeability, cells, direction,
            rw=well_params.rw, skin=well_params.skin, kv_kh=well_params.kv_kh,
        )
        dp = float(bhp[i]) - p[cells]
        if i < inj.size and inj[i]:
            # Gas injector: the injected fluid is pure gas, so the whole total
            # reservoir rate is gas; ÷ Bg converts to surface volume.
            qg[cells] += wi * (lam_w[cells] + lam_o[cells] + lam_g[cells]) * dp / bg
        else:
            qw[cells] += wi * lam_w[cells] * dp
            qo[cells] += wi * lam_o[cells] * dp
            qg[cells] += wi * lam_g[cells] * dp / bg
    return qw, qo, qg


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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Forward-simulate the saturation history.

    Returns ``(sw, so, sg)`` histories, each of shape ``(n_times, n_cells)``.
    ``model`` is ``"black_oil"`` or ``"compositional"`` (alias ``solution_gas`` /
    ``co2``). By default the pressure ``pressure`` is used as the given flow field
    and the well rates are the fixed ``well_qw/qo/qg``. When ``well_bhp`` and
    ``well_params`` are supplied, the pressure is instead *solved* with the
    Peaceman well model (BHP-driven wells) and the well rates are derived from the
    solved pressure — the compressible-injection-correct path.
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
            well_bhp=well_bhp, well_params=well_params,
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
# Fixed-point (Picard) iterations coupling the Peaceman pressure solve and the
# saturation transport: the pressure solve uses the *previous* mobility, so a few
# outer iterations let the mobility (which drops as free gas accumulates at the
# injector) feed back into the pressure/injection-rate — the "saturation ↔
# mobility ↔ injection rate" loop the fully-implicit well model closes.
_MAX_PICARD = 3


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
    ``C = sg/Bg + Rs*so`` (mass-conserving primary: ``C`` is *not* clipped to the
    per-cell maximum, so injected CO2 beyond the cell's free-gas capacity is
    carried in ``C`` and drained by the free-gas flux on the next substep). The
    phase split is recomputed inside the Newton from ``Rs`` and ``Bg``.
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
        # Backtracking line search (Armijo). C is *not* upper-clipped: the excess
        # injected CO2 is carried and drained by the free-gas flux next substep
        # (clipping it here silently destroys the plume mass).
        alpha = 1.0
        r_norm = float(np.linalg.norm(r))
        sw_new = sw.copy()
        C_new = C.copy()
        r_new = r
        for _ in range(12):
            sw_new = np.clip(sw + alpha * delta[:n], 0.0, 1.0)
            # C is the *conserved* CO2 component. Clipping it at the exact per-cell
            # maximum (1/Bg·(1-sw)) destroys the injected CO2 whenever the free-gas
            # flux cannot drain a cell fast enough (the plume under-production). Give
            # it a generous headroom (100x the per-cell bound) so the excess is
            # carried and drained by the flux, while keeping C bounded so the Newton
            # stays well-conditioned (the unbounded C stalls on a large excess).
            C_new = np.clip(C + alpha * delta[n:], 0.0, 100.0 * np.maximum(Rs, inv_Bg) * (1.0 - sw_new))
            r_new = residual(sw_new, C_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([sw_new - sw, C_new - C])))
        sw, C = sw_new, C_new
        norm_x = float(np.linalg.norm(np.concatenate([sw, C])))
        if float(np.linalg.norm(r_new)) < max(tol, 0.1) * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    _, so, sg = _split_compositional(sw, C, Rs, Bg, params.bo_slope)
    return sw, so, sg, C, converged


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
            rw=well_params.rw, skin=well_params.skin, kv_kh=well_params.kv_kh,
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


def _implicit_compositional_pressure_step(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    C0: NDArray[np.float64],
    p0: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
    params: FluidParams,
    injects_gas: NDArray[np.bool_],
    *,
    well_qg_fixed: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    tol: float = 1.0e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]:
    """One fully-implicit (p, sw, C) step: pressure + saturation + wells coupled.

    Solves the incompressible Peaceman pressure equation
    ``L(λ_total)·p + D(λ_total)·p = D(λ_total)·bhp + q_inj`` together with the water and
    surface-volume-CO2 conservation, with the well rates
    ``q_phase = WI·λ_phase·(bhp − p)`` and the EOS solubility ``Rs(p)``. The pressure
    is a primary, so the mobility feedback (free gas accumulates → λ_total drops →
    injection rate drops) is in the well coupling.

    ``well_qg_fixed`` (n_cells,) is the *surface* gas rate of the rate-controlled
    injector, placed on its completion cells (GEM ``OPERATE MAX BHF``). When given,
    the injector is a fixed reservoir source ``q_inj = well_qg_fixed·Bg`` rather than
    a BHP-driven well (its ``D`` entry is zero), so it injects at the specified rate
    instead of the mobility-scaled Peaceman rate. Producers stay BHP-driven.
    Returns ``(p, sw, so, sg, C, conv)``.
    """
    from scipy.sparse import diags

    Bg = float(params.bg)
    inv_Bg = 1.0 / max(Bg, 1.0e-12)
    n = grid.n_cells
    sw = np.asarray(sw0, dtype=float).copy()
    C = np.asarray(C0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    h = 1.0e-6
    accum = 1.0 / (dt * inv_phiV)  # phi*vol/dt (flux form)
    I = diags(accum)
    A_grav = _gravity_divergence_matrix(grid, permeability)
    cells, wi, bhp_w, is_inj = _peaceman_well_data(
        grid, permeability, wells, well_bhp, well_params, injects_gas,
    )
    # Fixed injector surface rate (rate-controlled); None keeps the BHP-driven model.
    rate_controlled = well_qg_fixed is not None
    qg_fixed = np.zeros(n, dtype=float)
    if rate_controlled:
        qg_fixed = np.asarray(well_qg_fixed, dtype=float).ravel()
    # Total surface gas rate target (m3/s surface) for the rate-controlled injector.
    qg_target = float(qg_fixed.sum())
    bhp_full = np.zeros(n)
    for idx in range(cells.size):
        bhp_full[cells[idx]] = bhp_w[idx]
    # per-cell gas-injector mask (for the well rate p-derivatives below).
    inj_cell = np.zeros(n, dtype=bool)
    for idx in range(cells.size):
        inj_cell[cells[idx]] = is_inj[idx]
    # Injector BHP is a *variable* under rate control (GEM OPERATE MAX BHF): solved
    # so the Peaceman rate matches qg_target, preserving the wellbore drawdown.
    bhp_inj = float(np.mean(bhp_w[is_inj])) if is_inj.any() else 0.0

    def well_diag_rates(p_, lam_w_, lam_o_, lam_g_, bhp_inj_):
        """Return the well diagonal ``D`` and the surface phase rates (qw, qo, qg)."""
        D_ = np.zeros(n)
        qw_ = np.zeros(n); qo_ = np.zeros(n); qg_ = np.zeros(n)
        lam_t_ = lam_w_ + lam_o_ + lam_g_
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and rate_controlled) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            D_[c] += w * lam_t_[c]
            if is_inj[idx]:
                qg_[c] += w * lam_t_[c] * dp / Bg
            else:
                qw_[c] += w * lam_w_[c] * dp
                qo_[c] += w * lam_o_[c] * dp
                qg_[c] += w * lam_g_[c] * dp / Bg
        return D_, qw_, qo_, qg_

    def inj_diag(lam_t_):
        """Injector well diagonal ``D_inj`` (n_cells,), non-zero on the injector cells."""
        D_inj_ = np.zeros(n)
        for idx in range(cells.size):
            if is_inj[idx]:
                D_inj_[cells[idx]] += wi[idx] * lam_t_[cells[idx]]
        return D_inj_

    def state(p_, sw_, C_):
        Rs_ = _eq_solution_gas_ratio(p_, params)
        _, so_, sg_ = _split_compositional(sw_, C_, Rs_, Bg, params.bo_slope)
        Rs_act_ = np.clip(np.where(so_ > 1.0e-12, (C_ - sg_ / Bg) / np.maximum(so_, 1.0e-12), 0.0), 0.0, Rs_)
        lam_w_, lam_o_, lam_g_ = phase_mobilities(sw_, so_, sg_, params)
        return Rs_, so_, sg_, Rs_act_, lam_w_, lam_o_, lam_g_

    def residual(p_, sw_, C_, bhp_inj_):
        Rs_, so_, sg_, Rs_act_, lam_w_, lam_o_, lam_g_ = state(p_, sw_, C_)
        lam_t_ = lam_w_ + lam_o_ + lam_g_
        L_ = _tpfa_matrix_vec(grid, permeability, lam_t_)
        D_, qw_, qo_, qg_ = well_diag_rates(p_, lam_w_, lam_o_, lam_g_, bhp_inj_)
        bhp_eff = bhp_full.copy()
        if rate_controlled:
            bhp_eff[inj_cell] = bhp_inj_
        r_p_ = L_ @ p_ + D_ * p_ - D_ * bhp_eff
        D_inj_ = inj_diag(lam_t_)
        r_rate_ = float(D_inj_ @ (bhp_inj_ - p_) - qg_target * Bg) if rate_controlled else 0.0
        A_ = _mobility_divergence_matrix(grid, permeability, p_)
        A_g_ = A_ - params.rho_g * A_grav
        r_sw_ = accum * (sw_ - sw0) - (qw_ - A_ @ lam_w_)
        q_c_ = qg_ + Rs_act_ * qo_
        r_c_ = accum * (C_ - C0) - (q_c_ - (inv_Bg * A_g_ @ lam_g_ + A_ @ (Rs_act_ * lam_o_)))
        return np.concatenate([r_p_, r_sw_, r_c_, np.array([r_rate_])])

    r0_norm = float(np.linalg.norm(residual(p, sw, C, bhp_inj)))
    converged = False
    for _ in range(max_iter):
        Rs, so, sg, Rs_act, lam_w, lam_o, lam_g = state(p, sw, C)
        lam_t = lam_w + lam_o + lam_g
        L = _tpfa_matrix_vec(grid, permeability, lam_t)
        D, qw, qo, qg = well_diag_rates(p, lam_w, lam_o, lam_g, bhp_inj)
        D_inj = inj_diag(lam_t)
        A = _mobility_divergence_matrix(grid, permeability, p)
        A_g = A - params.rho_g * A_grav
        r = residual(p, sw, C, bhp_inj)
        # numerical derivatives wrt sw and C (through the phase split + relperm).
        _, so_p, sg_p, Rs_sw, lw_p, lo_p, lg_p = state(p, sw + h, C)
        dlw_dsw = (lw_p - lam_w) / h
        dlg_dsw = (lg_p - lam_g) / h
        dlo_dsw = (lo_p - lam_o) / h
        dRs_dsw = (Rs_sw - Rs_act) / h
        _, so_c, sg_c, Rs_c, lw_c, lo_c, lg_c = state(p, sw, C + h)
        dlg_dC = (lg_c - lam_g) / h
        dlo_dC = (lo_c - lam_o) / h
        dRs_dC = (Rs_c - Rs_act) / h
        J_pp = (L + diags(D)).tocsr()
        # Well p-derivatives (diagonal). For a gas injector the total rate is gas
        # (qw=qo=0, qg=w·λt·dp/Bg), so ∂r_sw/∂p=0 and ∂r_c/∂p=w·λt/Bg; for a
        # producer the phases split by mobility.
        J_wp = diags(np.where(inj_cell, 0.0, D * (lam_w / np.maximum(lam_t, 1.0e-12))))
        J_cp = diags(np.where(inj_cell, D / Bg, D * ((lam_g / Bg + Rs_act * lam_o) / np.maximum(lam_t, 1.0e-12))))
        J_ww = I + A @ diags(dlw_dsw)
        J_cw = inv_Bg * A_g @ diags(dlg_dsw) + A @ diags(dRs_dsw * lam_o + Rs_act * dlo_dsw)
        J_cc = I + inv_Bg * A_g @ diags(dlg_dC) + A @ diags(dRs_dC * lam_o + Rs_act * dlo_dC)
        r = -r
        # Block forward substitution: δ(p, bhp_inj) → δsw → δC. The injector BHP is
        # eliminated by the Schur complement of J_pp (rank-1: column/row = ±D_inj).
        u = _solve_linear(J_pp, r[:n])
        if rate_controlled:
            w = _solve_linear(J_pp, D_inj)
            denom = float(D_inj.sum()) - float(D_inj @ w)
            delta_bhp = (r[3 * n] + float(D_inj @ u)) / denom if abs(denom) > 1.0e-30 else 0.0
            delta_p = u + delta_bhp * w
        else:
            delta_bhp = 0.0
            delta_p = u
        rhs_w = r[n:2 * n] - J_wp @ delta_p
        delta_sw = _solve_linear(J_ww, rhs_w)
        # Complete well Jacobian: the C equation couples to bhp_inj via the injector's
        # qg=w·λt·(bhp−p)/Bg, so ∂r_c/∂bhp_inj = −D_inj/Bg adds +D_inj/Bg·δbhp to rhs_c.
        rhs_c = r[2 * n:3 * n] - J_cw @ delta_sw - J_cp @ delta_p
        if rate_controlled:
            rhs_c = rhs_c + (D_inj / Bg) * delta_bhp
        delta_c = _solve_linear(J_cc, rhs_c)
        delta = np.concatenate([delta_p, delta_sw, delta_c])
        alpha = 1.0
        r_norm = float(np.linalg.norm(r))
        p_new = p.copy(); sw_new = sw.copy(); C_new = C.copy(); bhp_new = bhp_inj
        r_new = r
        for _ in range(12):
            p_new = p + alpha * delta[:n]
            sw_new = np.clip(sw + alpha * delta[n:2 * n], 0.0, 1.0)
            C_new = np.clip(C + alpha * delta[2 * n:3 * n], 0.0, 100.0 * np.maximum(Rs, inv_Bg) * (1.0 - sw_new))
            bhp_new = bhp_inj + alpha * delta_bhp
            r_new = residual(p_new, sw_new, C_new, bhp_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([p_new - p, sw_new - sw, C_new - C])))
        p, sw, C, bhp_inj = p_new, sw_new, C_new, bhp_new
        norm_x = float(np.linalg.norm(np.concatenate([p, sw, C])))
        if float(np.linalg.norm(r_new)) < max(tol, 0.1) * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    Rs, so, sg, Rs_act, lam_w, lam_o, lam_g = state(p, sw, C)
    return p, sw, so, sg, C, converged


def _implicit_compositional_pressure_kinetic_step(
    grid: CartesianGrid,
    permeability: NDArray[np.float64],
    inv_phiV: NDArray[np.float64],
    dt: float,
    sw0: NDArray[np.float64],
    C0: NDArray[np.float64],
    Cd0: NDArray[np.float64],
    p0: NDArray[np.float64],
    wells: WellMap,
    well_bhp: NDArray[np.float64],
    well_params: WellModelParams,
    params: FluidParams,
    injects_gas: NDArray[np.bool_],
    *,
    well_qg_fixed: NDArray[np.float64] | None = None,
    max_iter: int = 30,
    tol: float = 1.0e-3,
) -> tuple[NDArray[np.float64], ...]:
    """One fully-implicit kinetic-dissolution (p, sw, C, Cd) step (Newton).

    Extends :func:`_implicit_compositional_pressure_step` to a *kinetic* phase
    split: the dissolved CO2 ``Cd`` relaxes to the equilibrium ``Rs*No`` at rate
    ``k_diss``, so injected CO2 stays free gas at the injector (residence time <<
    1/k_diss) and only dissolves along the flow path. The pressure is still a
    primary, so the mobility feedback (free gas accumulates → λ_total drops →
    injection rate drops) closes inside one Newton. ``well_qg_fixed`` (see
    :func:`_implicit_compositional_pressure_step`) makes the injector a fixed
    rate-controlled source. Returns ``(p, sw, so, sg, C, Cd, converged)``.
    """
    from scipy.sparse import bmat, diags

    Bg = float(params.bg)
    inv_Bg = 1.0 / max(Bg, 1.0e-12)
    k_diss = float(params.k_diss)
    n = grid.n_cells
    sw = np.asarray(sw0, dtype=float).copy()
    C = np.asarray(C0, dtype=float).copy()
    Cd = np.asarray(Cd0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    h = 1.0e-6
    accum = 1.0 / (dt * inv_phiV)
    phiV = 1.0 / inv_phiV  # pore volume per cell: scales k_diss (a rate per PV) to m3/s
    I = diags(accum)
    A_grav = _gravity_divergence_matrix(grid, permeability)
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

    def well_diag_rates(p_, lam_w_, lam_o_, lam_g_, bhp_inj_):
        D_ = np.zeros(n)
        qw_ = np.zeros(n); qo_ = np.zeros(n); qg_ = np.zeros(n)
        lam_t_ = lam_w_ + lam_o_ + lam_g_
        for idx in range(cells.size):
            c = cells[idx]
            w = wi[idx]
            bhp_eff = bhp_inj_ if (is_inj[idx] and rate_controlled) else bhp_w[idx]
            dp = bhp_eff - p_[c]
            D_[c] += w * lam_t_[c]
            if is_inj[idx]:
                qg_[c] += w * lam_t_[c] * dp / Bg
            else:
                qw_[c] += w * lam_w_[c] * dp
                qo_[c] += w * lam_o_[c] * dp
                qg_[c] += w * lam_g_[c] * dp / Bg
        return D_, qw_, qo_, qg_

    def inj_diag(lam_t_):
        D_inj_ = np.zeros(n)
        for idx in range(cells.size):
            if is_inj[idx]:
                D_inj_[cells[idx]] += wi[idx] * lam_t_[cells[idx]]
        return D_inj_

    def state(p_, sw_, C_, Cd_):
        Rs_ = _eq_solution_gas_ratio(p_, params)
        _, so_, sg_, No_, Rs_act_ = _split_kinetic(sw_, C_, Cd_, Rs_, Bg, params.bo_slope)
        lam_w_, lam_o_, lam_g_ = phase_mobilities(sw_, so_, sg_, params)
        return Rs_, so_, sg_, No_, Rs_act_, lam_w_, lam_o_, lam_g_

    def residual(p_, sw_, C_, Cd_, bhp_inj_):
        Rs_, so_, sg_, No_, Rs_act_, lam_w_, lam_o_, lam_g_ = state(p_, sw_, C_, Cd_)
        lam_t_ = lam_w_ + lam_o_ + lam_g_
        L_ = _tpfa_matrix_vec(grid, permeability, lam_t_)
        D_, qw_, qo_, qg_ = well_diag_rates(p_, lam_w_, lam_o_, lam_g_, bhp_inj_)
        bhp_eff = bhp_full.copy()
        if rate_controlled:
            bhp_eff[inj_cell] = bhp_inj_
        r_p_ = L_ @ p_ + D_ * p_ - D_ * bhp_eff
        D_inj_ = inj_diag(lam_t_)
        r_rate_ = float(D_inj_ @ (bhp_inj_ - p_) - qg_target * Bg) if rate_controlled else 0.0
        A_ = _mobility_divergence_matrix(grid, permeability, p_)
        A_g_ = A_ - params.rho_g * A_grav
        flux_dissolved_ = A_ @ (Rs_act_ * lam_o_)
        r_sw_ = accum * (sw_ - sw0) - (qw_ - A_ @ lam_w_)
        r_C_ = accum * (C_ - C0) - (qg_ + Rs_act_ * qo_ - (inv_Bg * A_g_ @ lam_g_ + flux_dissolved_))
        r_Cd_ = (accum * (Cd_ - Cd0) - k_diss * (Rs_ * No_ - Cd_) * phiV
                 - Rs_act_ * qo_ + flux_dissolved_)
        return np.concatenate([r_p_, r_sw_, r_C_, r_Cd_, np.array([r_rate_])])

    r0_norm = float(np.linalg.norm(residual(p, sw, C, Cd, bhp_inj)))
    converged = False
    for _ in range(max_iter):
        Rs, so, sg, No, Rs_act, lam_w, lam_o, lam_g = state(p, sw, C, Cd)
        lam_t = lam_w + lam_o + lam_g
        L = _tpfa_matrix_vec(grid, permeability, lam_t)
        D, qw, qo, qg = well_diag_rates(p, lam_w, lam_o, lam_g, bhp_inj)
        D_inj = inj_diag(lam_t)
        A = _mobility_divergence_matrix(grid, permeability, p)
        A_g = A - params.rho_g * A_grav
        r = residual(p, sw, C, Cd, bhp_inj)
        # numerical derivatives wrt sw / C / Cd (through the kinetic split + relperm)
        _, so_s, sg_s, No_s, Rs_s, lw_s, lo_s, lg_s = state(p, sw + h, C, Cd)
        dlw_dsw = (lw_s - lam_w) / h
        dlo_dsw = (lo_s - lam_o) / h
        dlg_dsw = (lg_s - lam_g) / h
        dRs_dsw = (Rs_s - Rs_act) / h
        dNo_dsw = (No_s - No) / h
        _, so_c, sg_c, No_c, Rs_c, lw_c, lo_c, lg_c = state(p, sw, C + h, Cd)
        dlo_dC = (lo_c - lam_o) / h
        dlg_dC = (lg_c - lam_g) / h
        dRs_dC = (Rs_c - Rs_act) / h
        dNo_dC = (No_c - No) / h
        _, so_d, sg_d, No_d, Rs_d, lw_d, lo_d, lg_d = state(p, sw, C, Cd + h)
        dlo_dCd = (lo_d - lam_o) / h
        dlg_dCd = (lg_d - lam_g) / h
        dRs_dCd = (Rs_d - Rs_act) / h
        dNo_dCd = (No_d - No) / h
        dF_sw = dRs_dsw * lam_o + Rs_act * dlo_dsw
        dF_C = dRs_dC * lam_o + Rs_act * dlo_dC
        dF_Cd = dRs_dCd * lam_o + Rs_act * dlo_dCd
        # Pressure-equation block (well diagonal added to the tpfa Laplacian).
        J_pp = (L + diags(D)).tocsr()
        # Well p-derivatives (diagonal). For a gas injector qw=qo=0 and qg=w·λt·dp/Bg
        # so ∂r_sw/∂p=0 and ∂r_c/∂p=w·λt/Bg; a producer splits by mobility.
        J_wp = diags(np.where(inj_cell, 0.0, D * (lam_w / np.maximum(lam_t, 1.0e-12))))
        J_cp = diags(np.where(inj_cell, D / Bg, D * ((lam_g / Bg + Rs_act * lam_o) / np.maximum(lam_t, 1.0e-12))))
        # ∂r_Cd/∂p: the produced dissolved gas −Rs_act·qo (qo=w·λo·dp) is p-linear;
        # the dissolution source k_diss·(Rs(p)·No − Cd) depends on p through Rs(p)
        # and is *lagged* (Rs is re-evaluated each Newton iteration). Its p-derivative
        # k_diss·No·dRs/dp·φV is ~1e-17 for the calibrated k_diss ≲ 1e-5, ~10 orders
        # below the well term, so it is omitted from the diagonal.
        J_dp = diags(np.where(inj_cell, 0.0, D * (Rs_act * lam_o / np.maximum(lam_t, 1.0e-12))))
        # Water / CO2 / dissolved-CO2 blocks (flux form, from _implicit_kinetic_step).
        J_ww = I + A @ diags(dlw_dsw)
        J_Cw = inv_Bg * A_g @ diags(dlg_dsw) + A @ diags(dF_sw) - diags(dRs_dsw * qo)
        J_CC = I + inv_Bg * A_g @ diags(dlg_dC) + A @ diags(dF_C) - diags(dRs_dC * qo)
        J_CCd = inv_Bg * A_g @ diags(dlg_dCd) + A @ diags(dF_Cd) - diags(dRs_dCd * qo)
        J_Cdw = -k_diss * diags(Rs * dNo_dsw * phiV) + A @ diags(dF_sw) - diags(dRs_dsw * qo)
        J_CdC = -k_diss * diags(Rs * dNo_dC * phiV) + A @ diags(dF_C) - diags(dRs_dC * qo)
        J_CdCd = I + k_diss * diags((1.0 - Rs * dNo_dCd) * phiV) + A @ diags(dF_Cd) - diags(dRs_dCd * qo)
        r = -r
        # Block forward substitution: δ(p, bhp_inj) → δsw → (δC, δCd). The injector BHP
        # is eliminated by the Schur complement of J_pp (rank-1); the (C, Cd) block is
        # solved *together* (the free-gas split couples them at the same order).
        u = _solve_linear(J_pp, r[:n])
        if rate_controlled:
            w = _solve_linear(J_pp, D_inj)
            denom = float(D_inj.sum()) - float(D_inj @ w)
            delta_bhp = (r[4 * n] + float(D_inj @ u)) / denom if abs(denom) > 1.0e-30 else 0.0
            delta_p = u + delta_bhp * w
        else:
            delta_bhp = 0.0
            delta_p = u
        rhs_w = r[n:2 * n] - J_wp @ delta_p
        delta_sw = _solve_linear(J_ww, rhs_w)
        # Complete well Jacobian: ∂r_C/∂bhp_inj = −D_inj/Bg (the injector's qg), so
        # add +D_inj/Bg·δbhp to rhs_c (r_Cd has no direct bhp_inj dependence).
        rhs_c = r[2 * n:3 * n] - J_Cw @ delta_sw - J_cp @ delta_p
        if rate_controlled:
            rhs_c = rhs_c + (D_inj / Bg) * delta_bhp
        rhs_cd = r[3 * n:4 * n] - J_Cdw @ delta_sw - J_dp @ delta_p
        J_blk = bmat([[J_CC, J_CCd], [J_CdC, J_CdCd]], format="csr")
        d_ccd = _solve_linear(J_blk, np.concatenate([rhs_c, rhs_cd]))
        delta_c = d_ccd[:n]
        delta_cd = d_ccd[n:]
        delta = np.concatenate([delta_p, delta_sw, delta_c, delta_cd])
        alpha = 1.0
        r_norm = float(np.linalg.norm(r))
        p_new = p.copy(); sw_new = sw.copy(); C_new = C.copy(); Cd_new = Cd.copy(); bhp_new = bhp_inj
        r_new = r
        for _ in range(12):
            p_new = p + alpha * delta[:n]
            sw_new = np.clip(sw + alpha * delta[n:2 * n], 0.0, 1.0)
            C_new = np.clip(C + alpha * delta[2 * n:3 * n], 0.0, 100.0 * np.maximum(Rs, inv_Bg) * (1.0 - sw_new))
            Cd_new = np.clip(Cd + alpha * delta[3 * n:4 * n], 0.0, Rs * (1.0 - sw_new))
            bhp_new = bhp_inj + alpha * delta_bhp
            r_new = residual(p_new, sw_new, C_new, Cd_new, bhp_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([p_new - p, sw_new - sw, C_new - C, Cd_new - Cd])))
        p, sw, C, Cd, bhp_inj = p_new, sw_new, C_new, Cd_new, bhp_new
        norm_x = float(np.linalg.norm(np.concatenate([p, sw, C, Cd])))
        if float(np.linalg.norm(r_new)) < max(tol, 0.1) * max(r0_norm, 1.0e-12):
            converged = True
            break
        if norm_d < 1.0e-12 * max(1.0, norm_x):
            break  # stalled
    Rs, so, sg, No, Rs_act, lam_w, lam_o, lam_g = state(p, sw, C, Cd)
    return p, sw, so, sg, C, Cd, converged


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
    # Smooth saturation for the actual dissolved ratio Rs_act = Cd/No bounded by
    # Rs: a harmonic mean Rs*Cd/(Rs*No + Cd). It approaches Cd/No when the oil is
    # undersaturated (Cd << Rs*No) and Rs when saturated (Cd >> Rs*No), and is
    # smooth through No=0 (oil fully displaced), unlike the hard clip
    # clip(Cd/max(No,eps), 0, Rs) which jumps and stalls the Newton.
    Rs_act = Rs_p * Cd_p / (Rs_p * No + Cd_p + 1.0e-30)
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
        # Block solve. Water is decoupled (J_wC = J_wCd = 0), so solve delta_sw
        # first; but r_C couples to Cd via the free-gas split (J_CCd ~ -A_g dlam/dsg,
        # the same order as J_CC), so the (C, Cd) block must be solved *together*.
        # A lower-triangular pass that drops J_CCd stalls the Newton at ~6% residual.
        from scipy.sparse import bmat
        r = -r
        delta_sw = _solve_linear(J_ww, r[:n])
        rhs_c = r[n:2 * n] - J_Cw @ delta_sw
        rhs_cd = r[2 * n:] - J_Cdw @ delta_sw
        J_blk = bmat([[J_CC, J_CCd], [J_CdC, J_CdCd]], format="csr")
        d_ccd = _solve_linear(J_blk, np.concatenate([rhs_c, rhs_cd]))
        delta_c = d_ccd[:n]
        delta_cd = d_ccd[n:]
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
            C_new = np.clip(C + alpha * delta[n:2 * n], 0.0, np.maximum(Rs, inv_Bg) * (1.0 - sw_new))
            Cd_new = np.clip(Cd + alpha * delta[2 * n:], 0.0, Rs * (1.0 - sw_new))
            r_new = residual(sw_new, C_new, Cd_new)
            if np.isfinite(r_new).all() and float(np.linalg.norm(r_new)) < (1.0 - 1.0e-4 * alpha) * r_norm:
                break
            alpha *= 0.5
        norm_d = float(np.linalg.norm(np.concatenate([sw_new - sw, C_new - C, Cd_new - Cd])))
        sw, C, Cd = sw_new, C_new, Cd_new
        norm_x = float(np.linalg.norm(np.concatenate([sw, C, Cd])))
        if float(np.linalg.norm(r_new)) < max(tol, 0.1) * max(r0_norm, 1.0e-12):
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

    With ``rs_eos`` the bound is the Peng-Robinson bubble-point solubility
    ``Rs_sat(p)`` (``core.pr_eos.rs_sat_interp``), the thermodynamic bound GEM
    uses; otherwise it is Henry's law ``rs_slope*p`` (or ``rs_eq_slope`` /
    ``rs_quad``). This is the surface-volume solubility bound used in the forward
    model's phase split, distinct from the output ``rs`` metric.
    """
    p = np.asarray(pressure, dtype=float)
    if params.rs_eos:
        from ..core.pr_eos import rs_sat_interp
        return np.maximum(rs_sat_interp(p), 0.0)
    slope = params.rs_eq_slope if params.rs_eq_slope > 0.0 else params.rs_slope
    rs = slope * p + params.rs_quad * p * p
    return np.maximum(rs, 0.0)


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
    # Smooth cap at the fully-saturated boundary (sg -> nw): a hard clip(sg,0,nw)
    # has a vanishing derivative at sg=nw, so dsg/dC -> 0 once a cell is
    # gas-saturated and the Newton Jacobian becomes singular. Cap sg towards nw
    # with the same C¹-continuous _smooth_relu so dsg/dC stays finite.
    sg = nw - _smooth_relu(nw - sg, smooth * np.maximum(nw, 1.0e-6))
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
    well_bhp: NDArray[np.float64] | None = None,
    well_params: WellModelParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Solution-gas (CO2-in-oil) saturation transport (GEM-like component model).

    Tracks the water saturation ``sw`` and the conserved surface-volume CO2
    component ``C = sg/Bg + Rs*so``; the oil/gas phase split is recomputed at
    every substep from the equilibrium dissolved fraction. The CO2 component flux
    is the free-gas Darcy flux plus the dissolved CO2 carried by the oil phase,
    so injected CO2 largely dissolves and moves with the oil. Fully implicit
    (backward Euler + Newton) with adaptive sub-stepping.

    When ``well_bhp``/``well_params`` are given, the flow pressure is *solved*
    with the Peaceman well model (BHP-driven wells) and the well rates are derived
    from that pressure, so a compressible gas injector gets the correct drawdown
    and injection peak (and the plume can exsolve). Otherwise the given (kriged)
    ``pressure`` field and the fixed well rates are used.
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
    # BHP-driven mode: solve pressure with the Peaceman well model. The BHP is
    # the well control (injector 19.325 / producers 19.0 MPa); the well rates are
    # derived from the solved pressure, not the fixed series rates.
    bhp_mode = well_bhp is not None and well_params is not None
    if bhp_mode:
        bhp = np.asarray(well_bhp, dtype=float)
        if bhp.ndim == 1:
            bhp = bhp[None, :]
        # Gas injector mask: a *per-well* boolean (a well injects gas if any of its
        # time steps has a positive gas rate). The well type is fixed, so this is
        # computed once over the whole series, not per time step.
        injects_gas = (qg > 0.0).any(axis=0)
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
    # Flow pressure: in BHP mode this is the *solved* Peaceman pressure (carried
    # across time steps); the kriged field is only the initial guess.
    p = np.asarray(p_in[0], dtype=float).copy()
    sw_hist = np.zeros((n_t, n_c))
    so_hist = np.zeros((n_t, n_c))
    sg_hist = np.zeros((n_t, n_c))
    for t in range(n_t):
        Bg = float(params.bg)
        # Start-of-step phase split (for the t-value records), then advance C.
        Rs_t = _eq_solution_gas_ratio(p if bhp_mode else p_in[t], params)
        if kinetic:
            _sw_r, so_r, sg_r, _No, _ = _split_kinetic(sw, C, Cd, Rs_t, Bg, params.bo_slope)
        else:
            _sw_r, so_r, sg_r = _split_compositional(sw, C, Rs_t, Bg, params.bo_slope)
        sw_hist[t] = sw.copy()
        so_hist[t] = so_r.copy()
        sg_hist[t] = sg_r.copy()
        if t >= n_t - 1:
            continue
        if bhp_mode:
            # Fully-implicit coupled pressure + saturation + wells solved together,
            # so the mobility feedback (free gas accumulates → λ_total drops →
            # injection rate drops) closes inside one Newton instead of the Picard
            # loop below. The kinetic model (k_diss > 0) solves (p, sw, C, Cd) so
            # injected CO2 stays free gas at the injector and dissolves along the
            # path; the equilibrium model solves (p, sw, C).
            lam_w, lam_o, lam_g = phase_mobilities(sw, so_r, sg_r, params)
            p = solve_pressure_peaceman(
                grid, k, lam_w + lam_o + lam_g, wells, bhp[t], well_params,
            )
            # Rate-controlled injector (GEM ``OPERATE MAX BHF``): a fixed *surface*
            # gas rate distributed over the injector's completions, instead of the
            # BHP-driven Peaceman rate. Producers stay BHP-driven (MIN BHP 19 MPa).
            qg_fixed = np.zeros(n_c)
            for i in np.flatnonzero(injects_gas):
                cells_i = wells.cells[i]
                if cells_i.size > 0:
                    qg_fixed[cells_i] += qg[t, i] / cells_i.size
            remaining = float(times_a[t + 1] - times_a[t])
            dt = min(remaining, _MAX_DT)
            while remaining > 1.0e-12 and dt > 1.0e-3:
                if kinetic:
                    p, sw, so, sg, C, Cd, conv = _implicit_compositional_pressure_kinetic_step(
                        grid, k, inv_phiV, dt, sw, C, Cd, p, wells, bhp[t], well_params,
                        params, injects_gas, well_qg_fixed=qg_fixed,
                    )
                else:
                    p, sw, so, sg, C, conv = _implicit_compositional_pressure_step(
                        grid, k, inv_phiV, dt, sw, C, p, wells, bhp[t], well_params,
                        params, injects_gas, well_qg_fixed=qg_fixed,
                    )
                if conv:
                    remaining -= dt
                    dt = min(remaining, _MAX_DT, max(dt * 1.5, 1.0))
                else:
                    dt *= 0.5
            if remaining > 1.0e-12:
                # Newton stalled even at the minimum substep: accept the partial
                # advance and surface it (the unsolved tail's injection is lost).
                import warnings

                warnings.warn(
                    f"compositional forward step t={t} left {remaining:.3g} s unsolved "
                    f"(Newton did not converge at dt<=1e-3)",
                    RuntimeWarning,
                )
            continue
        sw_t, C_t, Cd_t = sw.copy(), C.copy(), Cd.copy()
        C_prev = None
        # Picard (fixed-point) loop (the non-BHP path): couple the Peaceman pressure
        # solve and the saturation transport. (The BHP path now always `continue`s
        # through the fully-implicit coupled step above, so this loop is non-BHP only.)
        for _picard in range(_MAX_PICARD):
            sw, C, Cd = sw_t.copy(), C_t.copy(), Cd_t.copy()
            # Phase split from the current CO2 component + solubility.
            if kinetic:
                sw, so, sg, _No, Rs_act = _split_kinetic(sw, C, Cd, Rs_t, Bg, params.bo_slope)
            else:
                sw, so, sg = _split_compositional(sw, C, Rs_t, Bg, params.bo_slope)
                Rs_act = np.clip(np.where(so > 1.0e-12, (C - sg / Bg) / np.maximum(so, 1.0e-12), 0.0), 0.0, Rs_t)
            qw_t = well_cell_rates(grid, wells, qw[t])
            qo_t = well_cell_rates(grid, wells, qo[t])
            qg_t = well_cell_rates(grid, wells, qg[t])
            p = p_in[t]
            # qg_t is already surface-volume (GEM *BHF is a surface rate), so the
            # surface-volume CO2 component source is qg_t + Rs*qo (NOT qg_t/Bg).
            q_c = qg_t + Rs_act * qo_t
            remaining = float(times_a[t + 1] - times_a[t])
            A = _mobility_divergence_matrix(grid, k, p)
            A_grav = _gravity_divergence_matrix(grid, k)
            dt = min(remaining, _MAX_DT)
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
                    dt = min(remaining, _MAX_DT, max(dt * 1.5, 1.0))
                else:
                    dt *= 0.5
            # Fixed-point convergence: the advanced C stopped changing (so the
            # mobility/pressure/injection-rate loop has closed).
            if C_prev is not None and float(np.linalg.norm(C - C_prev)) < 1.0e-5 * max(1.0, float(np.linalg.norm(C))):
                break
            C_prev = C.copy()
    return sw_hist, so_hist, sg_hist
