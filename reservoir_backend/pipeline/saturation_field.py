"""Reconstruct oil/gas/water saturations from sparse well sensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle, SensorSample


def reconstruct_saturation(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    pressure: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Step 3: well saturations (+ boundary flux cues) → full-grid sw, so, sg.

    Well saturations are hard-set. Remaining cells use inverse-distance
    weighting. Optional cues:

    - Injecting face flux (positive into domain) adds a high-sw boundary anchor.
    - Producing face flux (negative / outflow) adds a low-sw boundary anchor.
    - If ``pressure`` is provided, along-gradient distance soft-anisotropy
      improves interpolation toward flow paths (still sensor-driven, not full transport).
    """
    notes = [
        "saturation reconstruction uses IDW from well sensors (+ optional flux anchors)",
        "phase saturations are clipped and renormalized so sw+so+sg=1",
    ]
    grid = mesh.grid
    sw = np.full(grid.shape, np.nan, dtype=float)
    so = np.full(grid.shape, np.nan, dtype=float)
    sg = np.full(grid.shape, np.nan, dtype=float)

    anchors: list[tuple[float, float, float, float, float, float]] = []
    for name, phases in sample.well_saturation.items():
        if name not in mesh.well_cell_id:
            raise KeyError(f"sensor well {name} is not on the mesh")
        cell = mesh.well_cell_id[name]
        sw_v, so_v, sg_v = _normalize_phases(float(phases[0]), float(phases[1]), float(phases[2]))
        i, j, k = mesh.grid.ijk(cell)
        sw[k, j, i] = sw_v
        so[k, j, i] = so_v
        sg[k, j, i] = sg_v
        anchors.append((float(mesh.x[cell]), float(mesh.y[cell]), float(mesh.z[cell]), sw_v, so_v, sg_v))

    # Boundary flux anchors (requirement 3 mentions boundary flux)
    bounds = mesh.bounds
    if bounds is not None and sample.boundary.flux:
        for side, q in sample.boundary.flux.items():
            q = float(q)
            if abs(q) < 1.0e-30:
                continue
            # positive q = into domain → injection-like high sw; negative → production
            if q > 0.0:
                sw_b, so_b, sg_b = 0.85, 0.15, 0.0
            else:
                sw_b, so_b, sg_b = 0.25, 0.75, 0.0
            pt = _face_anchor_point(bounds, side)
            if pt is not None:
                anchors.append((*pt, sw_b, so_b, sg_b))
                notes.append(f"flux anchor on {side}: q={q:.3e} m3/s")

    if not anchors:
        sw.fill(0.2)
        so.fill(0.8)
        sg.fill(0.0)
        notes.append("no well saturation sensors; used default sw=0.2, so=0.8, sg=0")
        return sw, so, sg, notes

    # Flow-anisotropic IDW when pressure is available on moderate probe nets
    # (better channel-front Dice than isotropic auto-kriging). Dense nets
    # (n_a > 12) still use auto spatial IDW/kriging/stack.
    pts = np.asarray([[a[0], a[1], a[2]] for a in anchors], dtype=float)
    sw_v = np.asarray([a[3] for a in anchors], dtype=float)
    so_v = np.asarray([a[4] for a in anchors], dtype=float)
    sg_v = np.asarray([a[5] for a in anchors], dtype=float)
    n_a = pts.shape[0]
    use_aniso = (
        pressure is not None
        and pressure.shape == grid.shape
        and n_a <= 14
    )

    if not use_aniso:
        from reservoir_backend.pipeline.spatial_interp import auto_interpolate_to_grid

        sw_res = auto_interpolate_to_grid(mesh, pts, sw_v, clip=(0.0, 1.0))
        so_res = auto_interpolate_to_grid(mesh, pts, so_v, clip=(0.0, 1.0))
        sg_res = auto_interpolate_to_grid(mesh, pts, sg_v, clip=(0.0, 1.0))
        sw = sw_res.values.copy()
        so = so_res.values.copy()
        sg = sg_res.values.copy()
        # re-pin hard sensor cells exactly
        for ax, ay, az, asw, aso, asg in anchors:
            # nearest cell to anchor
            d2 = (mesh.x - ax) ** 2 + (mesh.y - ay) ** 2 + (mesh.z - az) ** 2
            cid = int(np.argmin(d2))
            i, j, k = int(mesh.i[cid]), int(mesh.j[cid]), int(mesh.k[cid])
            sw_n, so_n, sg_n = _normalize_phases(asw, aso, asg)
            sw[k, j, i], so[k, j, i], sg[k, j, i] = sw_n, so_n, sg_n
        # cell-wise renorm
        for idx in range(mesh.n_cells):
            i, j, k = int(mesh.i[idx]), int(mesh.j[idx]), int(mesh.k[idx])
            sw[k, j, i], so[k, j, i], sg[k, j, i] = _normalize_phases(
                float(sw[k, j, i]), float(so[k, j, i]), float(sg[k, j, i])
            )
        notes.append(
            f"saturation auto-spatial n={n_a} "
            f"(sw={sw_res.method}, so={so_res.method}, sg={sg_res.method})"
        )
        notes.extend(sw_res.notes[-2:])
        return sw, so, sg, notes

    notes.append("IDW distance softly anisotropic along pressure gradient (few anchors)")
    power = 2.0
    eps = 1.0e-12
    p = np.asarray(pressure, dtype=float)

    for idx in range(mesh.n_cells):
        i = int(mesh.i[idx])
        j = int(mesh.j[idx])
        k = int(mesh.k[idx])
        if np.isfinite(sw[k, j, i]):
            continue
        px, py, pz = float(mesh.x[idx]), float(mesh.y[idx]), float(mesh.z[idx])
        weights = []
        sws, sos, sgs = [], [], []
        for ax, ay, az, asw, aso, asg in anchors:
            dx = px - ax
            dy = py - ay
            dz = pz - az
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > eps:
                gi, gj, gk = _grad_p_components(p, grid, i, j, k)
                gnorm = np.sqrt(gi * gi + gj * gj + gk * gk) + 1.0e-30
                ux, uy, uz = gi / gnorm, gj / gnorm, gk / gnorm
                par = dx * ux + dy * uy + dz * uz
                per2 = max(0.0, dist * dist - par * par)
                dist = np.sqrt(0.35 * par * par + 1.0 * per2)
            if dist < eps:
                weights = [1.0]
                sws, sos, sgs = [asw], [aso], [asg]
                break
            w = 1.0 / (dist**power)
            weights.append(w)
            sws.append(asw)
            sos.append(aso)
            sgs.append(asg)
        wsum = float(np.sum(weights))
        sw_n, so_n, sg_n = _normalize_phases(
            float(np.dot(weights, sws) / wsum),
            float(np.dot(weights, sos) / wsum),
            float(np.dot(weights, sgs) / wsum),
        )
        sw[k, j, i] = sw_n
        so[k, j, i] = so_n
        sg[k, j, i] = sg_n

    return sw, so, sg, notes


def _face_anchor_point(bounds, side: str) -> tuple[float, float, float] | None:
    xm = 0.5 * (bounds.xmin + bounds.xmax)
    ym = 0.5 * (bounds.ymin + bounds.ymax)
    zm = 0.5 * (bounds.zmin + bounds.zmax)
    side = side.lower()
    if side == "left":
        return bounds.xmin, ym, zm
    if side == "right":
        return bounds.xmax, ym, zm
    if side == "front":
        return xm, bounds.ymin, zm
    if side == "back":
        return xm, bounds.ymax, zm
    if side == "bottom":
        return xm, ym, bounds.zmin
    if side == "top":
        return xm, ym, bounds.zmax
    return None


def _grad_p_components(p: NDArray[np.float64], grid, i: int, j: int, k: int) -> tuple[float, float, float]:
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dxi = float(np.asarray(grid.dx).ravel()[min(i, nx - 1)])
    dyj = float(np.asarray(grid.dy).ravel()[min(j, ny - 1)])
    dzk = float(np.asarray(grid.dz).ravel()[min(k, nz - 1)])
    if 0 < i < nx - 1:
        gi = (p[k, j, i + 1] - p[k, j, i - 1]) / (2.0 * dxi)
    elif i == 0 and nx > 1:
        gi = (p[k, j, 1] - p[k, j, 0]) / dxi
    elif nx > 1:
        gi = (p[k, j, -1] - p[k, j, -2]) / dxi
    else:
        gi = 0.0
    if 0 < j < ny - 1:
        gj = (p[k, j + 1, i] - p[k, j - 1, i]) / (2.0 * dyj)
    elif j == 0 and ny > 1:
        gj = (p[k, 1, i] - p[k, 0, i]) / dyj
    elif ny > 1:
        gj = (p[k, -1, i] - p[k, -2, i]) / dyj
    else:
        gj = 0.0
    if 0 < k < nz - 1:
        gk = (p[k + 1, j, i] - p[k - 1, j, i]) / (2.0 * dzk)
    elif k == 0 and nz > 1:
        gk = (p[1, j, i] - p[0, j, i]) / dzk
    elif nz > 1:
        gk = (p[-1, j, i] - p[-2, j, i]) / dzk
    else:
        gk = 0.0
    return float(gi), float(gj), float(gk)


def _normalize_phases(sw: float, so: float, sg: float) -> tuple[float, float, float]:
    sw = max(0.0, min(1.0, sw))
    so = max(0.0, min(1.0, so))
    sg = max(0.0, min(1.0, sg))
    total = sw + so + sg
    if total <= 0.0:
        return 0.2, 0.8, 0.0
    return sw / total, so / total, sg / total
