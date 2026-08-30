"""Static DPDP data that does not change with ensemble C_f."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.solver.dpdp_jacobian import DPDPJacobianPattern, build_sparsity_pattern
from reservoir_backend.solver.fi_comp import _neighbor_cells


def cartesian_cell_colors(grid: CartesianGrid) -> NDArray[np.int64]:
    """Distance-2 coloring of the 7-point stencil: (i + 2j + 3k) mod 7."""
    n = grid.n_cells
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    c = np.arange(n, dtype=np.int64)
    i = c % nx
    j = (c // nx) % ny
    k = c // (nx * ny)
    return (i + 2 * j + 3 * k) % 7


def verify_coloring_no_row_collision(neighbors: list[list[int]], colors: NDArray[np.int64]) -> None:
    """Same-color cells must not share a residual row (graph distance > 2)."""
    n = len(neighbors)
    colors = np.asarray(colors, dtype=np.int64).ravel()
    n_colors = int(np.max(colors)) + 1 if n else 0
    for color in range(n_colors):
        owner = np.full(n, -1, dtype=np.int64)
        for c in np.flatnonzero(colors == color):
            c = int(c)
            seen = {c, *neighbors[c]}
            for r in seen:
                prev = int(owner[r])
                if prev >= 0 and prev != c:
                    raise ValueError(f"color {color} collides on residual {r}: cells {prev} and {c}")
                owner[r] = c


def _scale_t(
    t_unit: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    k: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    s = float(k)
    return t_unit[0] * s, t_unit[1] * s, t_unit[2] * s


@dataclass
class DPDPModelContext:
    """Topology, coloring, sparsity, unit transmissibility. Built once per grid."""

    grid: CartesianGrid
    n_comp: int
    neighbors: list[list[int]]
    colors: NDArray[np.int64]
    color_cells: list[NDArray[np.int64]]
    pattern: DPDPJacobianPattern
    cell_volumes: NDArray[np.float64]
    t_unit: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    matrix_intercell: bool = True
    sensor_cells: NDArray[np.int64] | None = None
    sensor_names: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        grid: CartesianGrid,
        n_comp: int,
        *,
        matrix_intercell: bool = True,
        sensors: list | None = None,
    ) -> DPDPModelContext:
        n = grid.n_cells
        nu = int(n_comp) + 1
        neighbors = [_neighbor_cells(grid, c) for c in range(n)]
        colors = cartesian_cell_colors(grid)
        verify_coloring_no_row_collision(neighbors, colors)
        n_colors = int(np.max(colors)) + 1 if n else 0
        color_cells = [np.flatnonzero(colors == color) for color in range(n_colors)]
        pattern = build_sparsity_pattern(n, nu, neighbors)
        ones = np.ones(n, dtype=float)
        t_unit = geometric_transmissibility(grid, ones)
        names: tuple[str, ...] = ()
        cells = None
        if sensors:
            names = tuple(str(s.name) for s in sensors)
            cells = np.asarray([int(grid.locate_cell(s.x, s.y, s.z)) for s in sensors], dtype=np.int64)
        return cls(
            grid=grid,
            n_comp=int(n_comp),
            neighbors=neighbors,
            colors=colors,
            color_cells=color_cells,
            pattern=pattern,
            cell_volumes=grid.cell_volumes(),
            t_unit=t_unit,
            matrix_intercell=bool(matrix_intercell),
            sensor_cells=cells,
            sensor_names=names,
        )

    def transmissibilities(self, dual_rock: DualRock):
        """V1: uniform C_f and k_m, so T = k * T(1)."""
        cf = float(np.mean(dual_rock.fracture.permeability))
        km = float(np.mean(dual_rock.matrix.permeability))
        t_f = _scale_t(self.t_unit, cf)
        t_m = _scale_t(self.t_unit, 0.0 if not self.matrix_intercell else km)
        return t_f, t_m
