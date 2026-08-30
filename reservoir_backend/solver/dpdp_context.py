"""Static DPDP data that does not change with ensemble C_f."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.discretization.tpfa import geometric_transmissibility
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.dual_rock import DualRock
from reservoir_backend.solver.dpdp_jacobian import DPDPJacobianPattern, build_sparsity_pattern
from reservoir_backend.solver.fi_comp import _cell_colors, _neighbor_cells


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
        colors = _cell_colors(grid)
        n_colors = int(np.max(colors)) + 1
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
