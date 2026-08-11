"""Orthogonal mesh refinement guided by a shape indicator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.state import AxisAlignedBounds, MeshBundle, WellPoint


def refine_mesh_by_indicator(
    mesh: MeshBundle,
    indicator: NDArray[np.float64],
    *,
    factor: int = 2,
    threshold: float = 0.35,
    mode: str = "global_from_bbox",
) -> tuple[MeshBundle, dict[str, float]]:
    """Build a finer orthogonal mesh focused on high-indicator regions.

    Modes:
    - ``global_from_bbox``: refine the axis-aligned bbox of cells with
      indicator >= threshold (default). Falls back to full-domain refine.
    - ``full_domain``: refine the entire original bounds by ``factor``.
    """
    if factor < 2:
        raise ValueError("factor must be >= 2")
    if mesh.bounds is None:
        raise ValueError("mesh.bounds is required for refinement")

    bounds = mesh.bounds
    wells = [
        WellPoint(
            name=n,
            x=float(mesh.x[c]),
            y=float(mesh.y[c]),
            z=float(mesh.z[c]),
            role=mesh.well_role.get(n, "observer"),
        )
        for n, c in mesh.well_cell_id.items()
    ]
    # recover well true coords from centers is ok for refine continuity

    if mode == "full_domain":
        new_bounds = bounds
    else:
        new_bounds = _bbox_from_indicator(mesh, indicator, threshold=threshold)
        if new_bounds is None:
            new_bounds = bounds

    # mean coarse spacing
    dx0 = float(np.mean(np.asarray(mesh.grid.dx, dtype=float)))
    dy0 = float(np.mean(np.asarray(mesh.grid.dy, dtype=float)))
    dz0 = float(np.mean(np.asarray(mesh.grid.dz, dtype=float)))
    fine = build_mesh(
        new_bounds,
        dx=dx0 / factor,
        dy=dy0 / factor,
        dz=dz0 / max(1, factor // 2) if mesh.grid.nz > 2 else dz0,
        wells=wells if wells else None,
    )

    # map indicator onto fine mesh by nearest coarse cell
    fine_ind = map_field_to_mesh(mesh, indicator, fine)
    stats = {
        "coarse_cells": float(mesh.n_cells),
        "fine_cells": float(fine.n_cells),
        "refine_factor": float(factor),
        "fine_indicator_mean": float(np.mean(fine_ind)),
        "bbox_xmin": float(new_bounds.xmin),
        "bbox_xmax": float(new_bounds.xmax),
        "bbox_ymin": float(new_bounds.ymin),
        "bbox_ymax": float(new_bounds.ymax),
        "bbox_zmin": float(new_bounds.zmin),
        "bbox_zmax": float(new_bounds.zmax),
    }
    return fine, stats


def map_field_to_mesh(
    source: MeshBundle,
    field: NDArray[np.float64],
    target: MeshBundle,
) -> NDArray[np.float64]:
    """Nearest-neighbor map of a cell-centered field from source to target mesh."""
    field = np.asarray(field, dtype=float)
    if field.shape != source.grid.shape:
        raise ValueError("field shape must match source grid")
    out = np.zeros(target.grid.shape, dtype=float)
    # build source centers in physical space already on mesh.x/y/z
    src_xyz = np.column_stack([source.x, source.y, source.z])
    for n in range(target.n_cells):
        q = np.array([target.x[n], target.y[n], target.z[n]], dtype=float)
        d2 = np.sum((src_xyz - q) ** 2, axis=1)
        sidx = int(np.argmin(d2))
        i, j, k = int(source.i[sidx]), int(source.j[sidx]), int(source.k[sidx])
        ti, tj, tk = int(target.i[n]), int(target.j[n]), int(target.k[n])
        out[tk, tj, ti] = field[k, j, i]
    return out


def _bbox_from_indicator(
    mesh: MeshBundle,
    indicator: NDArray[np.float64],
    *,
    threshold: float,
) -> AxisAlignedBounds | None:
    ind = np.asarray(indicator, dtype=float)
    sel = ind >= float(threshold)
    if not np.any(sel):
        return None
    xs, ys, zs = [], [], []
    for n in range(mesh.n_cells):
        i, j, k = int(mesh.i[n]), int(mesh.j[n]), int(mesh.k[n])
        if sel[k, j, i]:
            xs.append(mesh.x[n])
            ys.append(mesh.y[n])
            zs.append(mesh.z[n])
    # pad by one mean cell
    dx = float(np.mean(np.asarray(mesh.grid.dx, dtype=float)))
    dy = float(np.mean(np.asarray(mesh.grid.dy, dtype=float)))
    dz = float(np.mean(np.asarray(mesh.grid.dz, dtype=float)))
    b = mesh.bounds
    assert b is not None
    return AxisAlignedBounds(
        xmin=max(b.xmin, float(np.min(xs)) - dx),
        xmax=min(b.xmax, float(np.max(xs)) + dx),
        ymin=max(b.ymin, float(np.min(ys)) - dy),
        ymax=min(b.ymax, float(np.max(ys)) + dy),
        zmin=max(b.zmin, float(np.min(zs)) - dz),
        zmax=min(b.zmax, float(np.max(zs)) + dz),
    )
