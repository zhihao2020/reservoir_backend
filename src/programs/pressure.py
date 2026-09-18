"""Program 2: pressure reconstruction from probes and well bottomhole pressures.

The default ``"auto"`` method selects among IDW / RBF / kriging / residual
kriging / a closed-model solve by leave-one-out error. The ``"closed"`` method
solves the steady incompressible closed model
``div(-k lambda grad p) = 0`` with the injector/producer bottomhole pressures
held as Dirichlet boundary conditions and no-flow on every outer face, then
adds a kriged correction so the probe observations are still honoured.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidObservation
from ..core.cartesian import CartesianGrid
from .interpolate import interpolate_field, select_interpolator

_CLOSED = {"closed", "closed_model", "pressure_solve"}


def interpolate_pressure(
    grid: CartesianGrid,
    probe_xyz: NDArray[np.float64],
    probe_pressure: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    well_pressure: NDArray[np.float64],
    *,
    method: str = "auto",
    power: float = 2.0,
    well_cells: list[NDArray[np.int64]] | None = None,
) -> NDArray[np.float64]:
    """Return cell-centered pressure ``(n_cells,)`` for one time slice."""
    probes, p_probe, wells, p_well = _split(probe_xyz, probe_pressure, well_xyz, well_pressure)
    name = str(method).strip().lower()
    if name == "auto":
        name, _ = select_pressure_method(grid, probes, p_probe, wells, p_well, power=power, well_cells=well_cells)
    if name in {"residual_kriging", "rk"}:
        return _residual_kriging(grid, probes, p_probe, wells, p_well, power=power)
    if name in _CLOSED:
        return _closed_kriging(grid, probes, p_probe, wells, p_well, power=power, well_cells=well_cells)
    points, values = _combine(probes, p_probe, wells, p_well)
    return interpolate_field(points, values, grid.cell_centers(), method=name, power=power)


def select_pressure_method(
    grid: CartesianGrid,
    probe_xyz: NDArray[np.float64],
    probe_pressure: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    well_pressure: NDArray[np.float64],
    *,
    power: float = 2.0,
    well_cells: list[NDArray[np.int64]] | None = None,
) -> tuple[str, float]:
    """Pick the pressure interpolator with the lowest leave-one-out RMSE."""
    probes, p_probe, wells, p_well = _split(probe_xyz, probe_pressure, well_xyz, well_pressure)
    if probes.shape[0] < 3:
        return "idw", float("nan")
    candidates = ("idw", "rbf", "kriging", "universal_kriging")
    points, values = _combine(probes, p_probe, wells, p_well)
    best, rmse = select_interpolator(points, values, power=power, candidates=candidates)
    if wells.shape[0] >= 1:
        rk_rmse = _residual_loocv(probes, p_probe, wells, p_well, power=power)
        if np.isfinite(rk_rmse) and rk_rmse < rmse:
            best, rmse = "residual_kriging", rk_rmse
        closed_rmse = _closed_loocv(grid, probes, p_probe, wells, p_well, power=power, well_cells=well_cells)
        if np.isfinite(closed_rmse) and closed_rmse < rmse:
            best, rmse = "closed", closed_rmse
    return best, rmse


def solve_closed_pressure(
    grid: CartesianGrid,
    well_cells: NDArray[np.int64] | list[int] | tuple[int, ...],
    well_bhp: NDArray[np.float64] | list[float] | tuple[float, ...],
    *,
    transmissivity: float = 1.0,
) -> NDArray[np.float64]:
    """Steady closed-model pressure: Dirichlet BHP at wells, no-flow elsewhere.

    Solves the two-point-flux Laplacian ``div(-k lambda grad p) = 0`` with the
    well cells fixed to their bottomhole pressures and Neumann-zero (no-flow)
    on every outer face. ``transmissivity`` scales every face uniformly and so
    cancels out of the harmonic solution; it is kept only for dimensional
    clarity.
    """
    cells = np.asarray(well_cells, dtype=np.int64).ravel()
    bhp = np.asarray(well_bhp, dtype=float).ravel()
    if cells.size != bhp.size:
        raise InvalidObservation("well cells and bottomhole pressures must match")
    if cells.size == 0:
        raise InvalidObservation("closed pressure solve needs at least one well")
    n = int(grid.n_cells)
    if not np.all((cells >= 0) & (cells < n)):
        raise InvalidObservation("well cell index out of range")
    bhp_by_cell: dict[int, float] = {}
    for c, value in zip(cells, bhp):
        bhp_by_cell[int(c)] = float(value)
    fixed = np.array(sorted(bhp_by_cell), dtype=np.int64)
    fixed_bhp = np.array([bhp_by_cell[int(c)] for c in fixed], dtype=float)

    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    diag = np.zeros(n, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = grid.index(i, j, k)
                if i + 1 < nx:
                    nb = grid.index(i + 1, j, k)
                    t = transmissivity * float(dy[j] * dz[k]) / (0.5 * float(dx[i] + dx[i + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if j + 1 < ny:
                    nb = grid.index(i, j + 1, k)
                    t = transmissivity * float(dx[i] * dz[k]) / (0.5 * float(dy[j] + dy[j + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
                if k + 1 < nz:
                    nb = grid.index(i, j, k + 1)
                    t = transmissivity * float(dx[i] * dy[j]) / (0.5 * float(dz[k] + dz[k + 1]))
                    diag[c] += t
                    diag[nb] += t
                    rows += [c, nb]
                    cols += [nb, c]
                    vals += [-t, -t]
    rows += list(range(n))
    cols += list(range(n))
    vals += diag.tolist()
    laplacian = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    fixed_set = set(int(c) for c in fixed)
    free = np.array([i for i in range(n) if i not in fixed_set], dtype=np.int64)
    if free.size == 0:
        p = np.zeros(n, dtype=float)
        p[fixed] = fixed_bhp
        return p
    a_ff = laplacian[free][:, free]
    a_fc = laplacian[free][:, fixed]
    p_free = spsolve(a_ff, -(a_fc @ fixed_bhp))
    p = np.zeros(n, dtype=float)
    p[free] = p_free
    p[fixed] = fixed_bhp
    return p


def _closed_kriging(
    grid: CartesianGrid,
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
    *,
    power: float,
    well_cells: list[NDArray[np.int64]] | None = None,
) -> NDArray[np.float64]:
    centers = grid.cell_centers()
    if wells.shape[0] == 0:
        return interpolate_field(probes, p_probe, centers, method="kriging", power=power)
    bc_cells, bc_bhp = _well_bc(grid, wells, p_well, well_cells)
    base = solve_closed_pressure(grid, bc_cells, bc_bhp)
    if probes.shape[0] == 0:
        return base
    probe_cells = _cell_indices(grid, probes)
    residual = p_probe - base[probe_cells]
    correction = interpolate_field(probes, residual, centers, method="kriging")
    correction[bc_cells] = 0.0
    return base + correction


def _closed_loocv(
    grid: CartesianGrid,
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
    *,
    power: float,
    well_cells: list[NDArray[np.int64]] | None = None,
) -> float:
    n = int(probes.shape[0])
    if n < 3 or wells.shape[0] == 0:
        return float("inf")
    bc_cells, bc_bhp = _well_bc(grid, wells, p_well, well_cells)
    base = solve_closed_pressure(grid, bc_cells, bc_bhp)
    probe_cells = _cell_indices(grid, probes)
    residual = p_probe - base[probe_cells]
    err = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = interpolate_field(probes[mask], residual[mask], probes[i : i + 1], method="kriging")
        err[i] = base[probe_cells[i]] + float(np.asarray(pred).ravel()[0]) - p_probe[i]
    return float(np.sqrt(np.mean(err**2)))


def _cell_indices(grid: CartesianGrid, xyz: NDArray[np.float64]) -> NDArray[np.int64]:
    pts = np.asarray(xyz, dtype=float)
    out = np.empty(pts.shape[0], dtype=np.int64)
    for n, (x, y, z) in enumerate(pts):
        out[n] = grid.locate_cell(float(x), float(y), float(z))
    return out


def _well_bc(
    grid: CartesianGrid,
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
    well_cells: list[NDArray[np.int64]] | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Dirichlet bottomhole boundary condition: cells and their pressures."""
    if well_cells is not None:
        cells: list[int] = []
        bhp: list[float] = []
        for i, cs in enumerate(well_cells):
            for c in cs:
                cells.append(int(c))
                bhp.append(float(p_well[i]))
        return np.array(cells, dtype=np.int64), np.array(bhp, dtype=float)
    return _cell_indices(grid, wells), np.asarray(p_well, dtype=float).ravel()


def _residual_kriging(
    grid: CartesianGrid,
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
    *,
    power: float,
) -> NDArray[np.float64]:
    centers = grid.cell_centers()
    if wells.shape[0] == 0:
        return interpolate_field(probes, p_probe, centers, method="kriging", power=power)
    trend_probes = interpolate_field(wells, p_well, probes, method="idw", power=power)
    trend_grid = interpolate_field(wells, p_well, centers, method="idw", power=power)
    residual = p_probe - trend_probes
    residual_grid = interpolate_field(probes, residual, centers, method="kriging")
    return trend_grid + residual_grid


def _residual_loocv(
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
    *,
    power: float,
) -> float:
    n = int(probes.shape[0])
    if n < 3 or wells.shape[0] == 0:
        return float("inf")
    err = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        trend = interpolate_field(wells, p_well, probes[i : i + 1], method="idw", power=power)
        resid = p_probe[mask] - interpolate_field(wells, p_well, probes[mask], method="idw", power=power)
        pred = interpolate_field(probes[mask], resid, probes[i : i + 1], method="kriging")
        err[i] = float(np.asarray(pred).ravel()[0] + trend[0] - p_probe[i])
    return float(np.sqrt(np.mean(err**2)))


def _split(
    probe_xyz: NDArray[np.float64],
    probe_pressure: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    well_pressure: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    probes = np.asarray(probe_xyz, dtype=float)
    p_probe = np.asarray(probe_pressure, dtype=float).ravel()
    wells = np.asarray(well_xyz, dtype=float)
    p_well = np.asarray(well_pressure, dtype=float).ravel()
    if probes.size:
        if probes.ndim != 2 or probes.shape[0] != p_probe.size:
            raise InvalidObservation("pressure probe coordinates and values mismatch")
    else:
        probes = np.zeros((0, 3), dtype=float)
        p_probe = np.zeros(0, dtype=float)
    if wells.size:
        if wells.ndim != 2 or wells.shape[0] != p_well.size:
            raise InvalidObservation("well coordinates and pressures mismatch")
    else:
        wells = np.zeros((0, 3), dtype=float)
        p_well = np.zeros(0, dtype=float)
    if probes.shape[0] == 0 and wells.shape[0] == 0:
        raise InvalidObservation("pressure interpolation needs probes or wells")
    finite_p = np.isfinite(p_probe)
    probes, p_probe = probes[finite_p], p_probe[finite_p]
    finite_w = np.isfinite(p_well)
    wells, p_well = wells[finite_w], p_well[finite_w]
    return probes, p_probe, wells, p_well


def _combine(
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parts_xyz = []
    parts_val = []
    if probes.shape[0]:
        parts_xyz.append(probes)
        parts_val.append(p_probe)
    if wells.shape[0]:
        parts_xyz.append(wells)
        parts_val.append(p_well)
    return np.vstack(parts_xyz), np.concatenate(parts_val)
