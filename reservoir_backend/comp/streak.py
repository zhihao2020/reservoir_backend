"""EXAMPLE two-region permeability (high-k streak in a low-k matrix).

Not a Jiyang / 济阳 card, not site-calibrated, not industrial-grade.
Values are documented EXAMPLE contrasts only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.grid.cartesian import CartesianGrid

# EXAMPLE permeabilities (m²). Contrast only — not a field measurement.
K_MATRIX_M2 = 1.0e-18
K_STREAK_M2 = 1.0e-12


def example_two_region_k(
    grid: CartesianGrid,
    streak_cells: NDArray[np.int_] | list[int],
    *,
    k_matrix: float = K_MATRIX_M2,
    k_streak: float = K_STREAK_M2,
) -> NDArray[np.float64]:
    """Cell-wise k: matrix everywhere, ``k_streak`` on ``streak_cells``."""
    if k_matrix <= 0.0 or k_streak <= 0.0:
        raise ValueError("EXAMPLE permeabilities must be positive (m²)")
    k = np.full(grid.n_cells, float(k_matrix), dtype=float)
    idx = np.asarray(streak_cells, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("streak_cells must be non-empty")
    if np.any(idx < 0) or np.any(idx >= grid.n_cells):
        raise ValueError("streak cell index out of range")
    k[idx] = float(k_streak)
    return k


def example_drive_pressure(
    grid: CartesianGrid,
    well_cell: int,
    *,
    p0: float,
    drop_pa: float,
) -> NDArray[np.float64]:
    """Prescribed p decreasing with Manhattan distance from the well.

    Not a pressure solve. ``drop_pa > 0`` makes the well a source;
    ``drop_pa < 0`` makes it a sink (produce drawdown).
    """
    # drop_pa > 0: well is a source (p highest). drop_pa < 0: well is a sink.
    iw, jw, kw = grid.ijk(int(well_cell))
    p = np.empty(grid.n_cells, dtype=float)
    for c in range(grid.n_cells):
        i, j, k = grid.ijk(c)
        dist = abs(i - iw) + abs(j - jw) + abs(k - kw)
        p[c] = float(p0) - float(drop_pa) * float(dist)
    return p


def moles_per_pv(
    n: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
    comp_index: int,
    cells: NDArray[np.int_] | list[int],
) -> float:
    """Mean ``n_i / Vp`` of one component over ``cells`` (inventory density)."""
    idx = np.asarray(cells, dtype=int).ravel()
    vp = np.asarray(pore_volume, dtype=float).ravel()[idx]
    ni = np.asarray(n, dtype=float)[idx, int(comp_index)]
    return float(np.mean(ni / vp))


def added_moles_per_pv(
    n: NDArray[np.float64],
    n0: NDArray[np.float64],
    pore_volume: NDArray[np.float64],
    comp_index: int,
    cells: NDArray[np.int_] | list[int],
) -> float:
    """Mean added moles of one component per pore-volume over ``cells``."""
    idx = np.asarray(cells, dtype=int).ravel()
    vp = np.asarray(pore_volume, dtype=float).ravel()[idx]
    dn = np.asarray(n, dtype=float)[idx, int(comp_index)] - np.asarray(n0, dtype=float)[idx, int(comp_index)]
    return float(np.mean(dn / vp))
