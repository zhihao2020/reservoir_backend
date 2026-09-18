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
    BlackOilParams,
    corey_phase_mobilities,
    corey_total_mobility,
    darcy_divergence,
    invert_rock,
    well_cell_rates,
)


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
                    kh = 2.0 * k3[k, j, i] * k3[k, j, i + 1] / (k3[k, j, i] + k3[k, j, i + 1] + 1.0e-30)
                    lamf = 0.5 * (lam3[k, j, i] + lam3[k, j, i + 1])
                    t = kh * lamf * float(dy[j] * dz[k]) / (0.5 * float(dx[i] + dx[i + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if j + 1 < ny:
                    nb = grid.index(i, j + 1, k)
                    kh = 2.0 * k3[k, j, i] * k3[k, j + 1, i] / (k3[k, j, i] + k3[k, j + 1, i] + 1.0e-30)
                    lamf = 0.5 * (lam3[k, j, i] + lam3[k, j + 1, i])
                    t = kh * lamf * float(dx[i] * dz[k]) / (0.5 * float(dy[j] + dy[j + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if k + 1 < nz:
                    nb = grid.index(i, j, k + 1)
                    kh = 2.0 * k3[k, j, i] * k3[k + 1, j, i] / (k3[k, j, i] + k3[k + 1, j, i] + 1.0e-30)
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
    params: BlackOilParams | None = None,
    sw: NDArray[np.float64] | None = None,
    so: NDArray[np.float64] | None = None,
    sg: NDArray[np.float64] | None = None,
) -> dict[str, float]:
    """Forward-simulate observations from true k/phi, invert, and score the fit.

    Saturations are prescribed (constant 0.4/0.4/0.2 unless given); the pressure
    history comes from the forward solve. The inversion is run against the exact
    fields, so the reported errors isolate the inversion's reconstruction power.
    """
    oil = params or BlackOilParams()
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
    params: BlackOilParams,
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
        lam_w, lam_o, lam_g = corey_phase_mobilities(sw, so, sg, params)
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
            while remaining > 1.0e-12:
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
                lam_w, lam_o, lam_g = corey_phase_mobilities(sw, so, sg, params)
                p = _solve_pressure_with_source(grid, permeability, lam_w + lam_o + lam_g, qw_t + qo_t + qg_t, ref_cell, ref_p)
    return p_hist, sw_hist, so_hist, sg_hist
