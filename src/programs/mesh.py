"""Program 1: sample bounds + probe/well coordinates → cell ids and positions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import GridError
from ..core.cartesian import CartesianGrid


@dataclass(frozen=True)
class PointMap:
    ids: tuple[str, ...]
    xyz: NDArray[np.float64]
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    k: NDArray[np.int64]
    cell: NDArray[np.int64]


@dataclass(frozen=True)
class WellMap:
    ids: tuple[str, ...]
    xyz: NDArray[np.float64]  # heel coordinates (n_wells, 3)
    cells: tuple[NDArray[np.int64], ...]  # completion cells per well
    directions: tuple[NDArray[np.float64], ...] = ()  # unit heel->toe vector per well


@dataclass(frozen=True)
class MeshResult:
    grid: CartesianGrid
    probes: PointMap
    wells: WellMap

    def mapping_rows(self, kind: str) -> list[dict[str, object]]:
        if kind == "probe":
            points = self.probes
            rows = []
            for n, name in enumerate(points.ids):
                rows.append(
                    {
                        "kind": kind,
                        "id": name,
                        "x": float(points.xyz[n, 0]),
                        "y": float(points.xyz[n, 1]),
                        "z": float(points.xyz[n, 2]),
                        "i": int(points.i[n]),
                        "j": int(points.j[n]),
                        "k": int(points.k[n]),
                        "cell": int(points.cell[n]),
                    }
                )
            return rows
        rows = []
        for n, name in enumerate(self.wells.ids):
            for c in self.wells.cells[n]:
                i, j, k = self.grid.ijk(int(c))
                rows.append(
                    {
                        "kind": kind,
                        "id": name,
                        "x": float(self.wells.xyz[n, 0]),
                        "y": float(self.wells.xyz[n, 1]),
                        "z": float(self.wells.xyz[n, 2]),
                        "i": i,
                        "j": j,
                        "k": k,
                        "cell": int(c),
                    }
                )
        return rows


def build_grid(
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    nx: int,
    ny: int,
    nz: int,
) -> CartesianGrid:
    lx, ly, lz = (float(extent[0]), float(extent[1]), float(extent[2]))
    if min(lx, ly, lz) <= 0.0:
        raise GridError("extent must be positive")
    return CartesianGrid(
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        dx=np.full(int(nx), lx / int(nx)),
        dy=np.full(int(ny), ly / int(ny)),
        dz=np.full(int(nz), lz / int(nz)),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
    )


def min_probe_chebyshev(xyz: NDArray[np.float64]) -> float:
    """Smallest Chebyshev distance between any two probes. Inf if fewer than two."""
    pts = np.asarray(xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return float("inf")
    dmin = np.inf
    for i in range(pts.shape[0] - 1):
        cheb = np.max(np.abs(pts[i + 1 :] - pts[i]), axis=1)
        dmin = min(dmin, float(np.min(cheb)))
    return float(dmin)


def locate_clamped(grid: CartesianGrid, x: float, y: float, z: float) -> int:
    """Map a point to a cell after checking it lies in the bounding box."""
    return grid.locate_cell(float(x), float(y), float(z))


def map_points(
    grid: CartesianGrid,
    ids: list[str] | tuple[str, ...],
    xyz: NDArray[np.float64],
) -> PointMap:
    coords = np.asarray(xyz, dtype=float)
    names = tuple(str(name) for name in ids)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise GridError("point coordinates must have shape (n, 3)")
    if coords.shape[0] != len(names):
        raise GridError("point ids and coordinates length mismatch")
    cells = np.empty(coords.shape[0], dtype=np.int64)
    ii = np.empty(coords.shape[0], dtype=np.int64)
    jj = np.empty(coords.shape[0], dtype=np.int64)
    kk = np.empty(coords.shape[0], dtype=np.int64)
    for n, (x, y, z) in enumerate(coords):
        cell = locate_clamped(grid, x, y, z)
        i, j, k = grid.ijk(cell)
        cells[n] = cell
        ii[n], jj[n], kk[n] = i, j, k
    return PointMap(ids=names, xyz=coords, i=ii, j=jj, k=kk, cell=cells)


_MAX_AXIS = 256


def _refine_colliding_axes(
    nx: int,
    ny: int,
    nz: int,
    probes: PointMap,
) -> tuple[int, int, int]:
    groups: dict[int, list[int]] = defaultdict(list)
    for n, cell in enumerate(probes.cell):
        groups[int(cell)].append(n)
    bumped = False
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                delta = np.abs(probes.xyz[idxs[a]] - probes.xyz[idxs[b]])
                if float(np.max(delta)) <= 1.0e-12:
                    raise GridError("two probes share the same coordinates")
                axis = int(np.argmax(delta))
                if axis == 0:
                    nx += 1
                elif axis == 1:
                    ny += 1
                else:
                    nz += 1
                bumped = True
    if not bumped:
        nx, ny, nz = nx + 1, ny + 1, nz + 1
    return nx, ny, nz


def build_mesh(
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    nx: int,
    ny: int,
    nz: int,
    probe_ids: list[str] | tuple[str, ...],
    probe_xyz: NDArray[np.float64],
    well_ids: list[str] | tuple[str, ...],
    well_xyz: NDArray[np.float64],
    well_trajectories: list[tuple[tuple[float, float, float], ...]] | None = None,
) -> MeshResult:
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min_probe_chebyshev(probe_xyz) <= 1.0e-12 and np.asarray(probe_xyz).shape[0] >= 2:
        raise GridError("two probes share the same coordinates")
    while True:
        if max(nx, ny, nz) > _MAX_AXIS:
            raise GridError("cannot place neighbouring probes in distinct cells")
        grid = build_grid(origin, extent, nx, ny, nz)
        probes = map_points(grid, probe_ids, probe_xyz)
        if probes.xyz.shape[0] < 2 or int(np.unique(probes.cell).size) == int(probes.cell.size):
            break
        nx, ny, nz = _refine_colliding_axes(nx, ny, nz, probes)
    return MeshResult(
        grid=grid,
        probes=probes,
        wells=_map_wells(grid, well_ids, well_xyz, well_trajectories),
    )


def _map_wells(
    grid: CartesianGrid,
    ids: list[str] | tuple[str, ...],
    well_xyz: NDArray[np.float64],
    trajectories: list[tuple[tuple[float, float, float], ...]] | None = None,
) -> WellMap:
    names = tuple(str(name) for name in ids)
    xyz = np.asarray(well_xyz, dtype=float)
    if trajectories is None:
        trajectories = [((float(x), float(y), float(z)),) for x, y, z in xyz]
    cells: list[NDArray[np.int64]] = [_trajectory_cells(grid, traj) for traj in trajectories]
    directions = []
    for traj in trajectories:
        pts = [np.asarray(p, dtype=float) for p in traj]
        if len(pts) >= 2:
            d = pts[-1] - pts[0]
            norm = float(np.linalg.norm(d))
            directions.append(d / norm if norm > 1.0e-12 else np.zeros(3, dtype=float))
        else:
            directions.append(np.zeros(3, dtype=float))
    return WellMap(ids=names, xyz=xyz, cells=tuple(cells), directions=tuple(directions))


def _trajectory_cells(
    grid: CartesianGrid,
    points: tuple[tuple[float, float, float], ...],
) -> NDArray[np.int64]:
    pts = [tuple(float(c) for c in p) for p in points]
    if len(pts) == 1:
        x, y, z = pts[0]
        return np.array([grid.locate_cell(x, y, z)], dtype=np.int64)
    acc: set[int] = set()
    for (x0, y0, z0), (x1, y1, z1) in zip(pts[:-1], pts[1:]):
        acc.update(int(c) for c in grid.segment_cells((x0, y0, z0), (x1, y1, z1)))
    return np.array(sorted(acc), dtype=np.int64)
