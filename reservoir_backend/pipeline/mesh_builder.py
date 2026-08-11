"""Build structured mesh from bounds, spacing, and well locations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from reservoir_backend.core.grid import Grid3D, SpacingInput
from reservoir_backend.pipeline.state import AxisAlignedBounds, MeshBundle, WellPoint


def build_mesh(
    bounds: AxisAlignedBounds,
    dx: SpacingInput,
    dy: SpacingInput,
    dz: SpacingInput,
    wells: Sequence[WellPoint] | None = None,
) -> MeshBundle:
    """Create a structured orthogonal mesh filling the axis-aligned box.

    Cell counts are chosen so that ``n * spacing ≈ length`` using the mean
    spacing when vectors are provided. Wells are mapped to containing cells.
    """
    wells = list(wells or [])
    lx = bounds.xmax - bounds.xmin
    ly = bounds.ymax - bounds.ymin
    lz = bounds.zmax - bounds.zmin

    dx_arr = _as_vector(dx)
    dy_arr = _as_vector(dy)
    dz_arr = _as_vector(dz)

    if dx_arr.size == 1:
        nx = max(2, int(np.ceil(lx / float(dx_arr[0]))))
        dx_use = float(dx_arr[0])
    else:
        nx = int(dx_arr.size)
        dx_use = dx_arr
        lx_span = float(np.sum(dx_arr))
        if not np.isclose(lx_span, lx, rtol=1e-3, atol=1e-6):
            # rescale spacing to match bounds
            dx_use = dx_arr * (lx / lx_span)

    if dy_arr.size == 1:
        ny = max(2, int(np.ceil(ly / float(dy_arr[0]))))
        dy_use = float(dy_arr[0])
    else:
        ny = int(dy_arr.size)
        dy_use = dy_arr
        ly_span = float(np.sum(dy_arr))
        if not np.isclose(ly_span, ly, rtol=1e-3, atol=1e-6):
            dy_use = dy_arr * (ly / ly_span)

    if dz_arr.size == 1:
        nz = max(2, int(np.ceil(lz / float(dz_arr[0]))))
        dz_use = float(dz_arr[0])
    else:
        nz = int(dz_arr.size)
        dz_use = dz_arr
        lz_span = float(np.sum(dz_arr))
        if not np.isclose(lz_span, lz, rtol=1e-3, atol=1e-6):
            dz_use = dz_arr * (lz / lz_span)

    # If uniform spacing derived from bounds, force exact fill
    if np.isscalar(dx_use) or (isinstance(dx_use, float)):
        dx_use = lx / nx
    if np.isscalar(dy_use) or (isinstance(dy_use, float)):
        dy_use = ly / ny
    if np.isscalar(dz_use) or (isinstance(dz_use, float)):
        dz_use = lz / nz

    grid = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx_use, dy=dy_use, dz=dz_use)
    centers = grid.cell_centers()
    # shift origin to bounds.xmin etc.
    x = centers[..., 0] + bounds.xmin
    y = centers[..., 1] + bounds.ymin
    z = centers[..., 2] + bounds.zmin

    cell_id = np.arange(grid.total_cells, dtype=np.int64)
    i_list = np.empty(grid.total_cells, dtype=np.int64)
    j_list = np.empty(grid.total_cells, dtype=np.int64)
    k_list = np.empty(grid.total_cells, dtype=np.int64)
    x_flat = np.empty(grid.total_cells, dtype=float)
    y_flat = np.empty(grid.total_cells, dtype=float)
    z_flat = np.empty(grid.total_cells, dtype=float)
    for k in range(grid.nz):
        for j in range(grid.ny):
            for i in range(grid.nx):
                idx = grid.index(i, j, k)
                i_list[idx] = i
                j_list[idx] = j
                k_list[idx] = k
                x_flat[idx] = x[k, j, i]
                y_flat[idx] = y[k, j, i]
                z_flat[idx] = z[k, j, i]

    well_map: dict[str, int] = {}
    well_role: dict[str, str] = {}
    for well in wells:
        if well.name in well_map:
            raise ValueError(f"duplicate well name: {well.name}")
        # locate in local coordinates
        xl = well.x - bounds.xmin
        yl = well.y - bounds.ymin
        zl = well.z - bounds.zmin
        try:
            ii, jj, kk = grid.locate_cell(xl, yl, zl)
        except Exception as exc:
            raise ValueError(f"well {well.name} lies outside bounds") from exc
        well_map[well.name] = grid.index(ii, jj, kk)
        well_role[well.name] = well.role

    return MeshBundle(
        grid=grid,
        cell_id=cell_id,
        i=i_list,
        j=j_list,
        k=k_list,
        x=x_flat,
        y=y_flat,
        z=z_flat,
        well_cell_id=well_map,
        well_role=well_role,
        bounds=bounds,
    )


def _as_vector(value: SpacingInput) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("spacing must not be empty")
    return arr
