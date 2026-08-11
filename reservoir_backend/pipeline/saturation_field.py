"""Reconstruct oil/gas/water saturations from sparse well sensors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle, SensorSample


def reconstruct_saturation(
    mesh: MeshBundle,
    sample: SensorSample,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Return ``sw, so, sg`` arrays with shape ``(nz, ny, nx)``.

    Well saturations are hard-set on their cells. Remaining cells use
    inverse-distance weighting from well sensors, then closure and clipping.
    """
    notes = [
        "saturation reconstruction uses inverse-distance weighting from well sensors",
        "phase saturations are clipped and renormalized so sw+so+sg=1",
    ]
    grid = mesh.grid
    sw = np.full(grid.shape, np.nan, dtype=float)
    so = np.full(grid.shape, np.nan, dtype=float)
    sg = np.full(grid.shape, np.nan, dtype=float)

    anchors: list[tuple[float, float, float, float, float, float]] = []
    # (x,y,z,sw,so,sg)
    for name, phases in sample.well_saturation.items():
        if name not in mesh.well_cell_id:
            raise KeyError(f"sensor well {name} is not on the mesh")
        cell = mesh.well_cell_id[name]
        sw_v, so_v, sg_v = (float(phases[0]), float(phases[1]), float(phases[2]))
        sw_v, so_v, sg_v = _normalize_phases(sw_v, so_v, sg_v)
        i, j, k = mesh.grid.ijk(cell)
        sw[k, j, i] = sw_v
        so[k, j, i] = so_v
        sg[k, j, i] = sg_v
        anchors.append((float(mesh.x[cell]), float(mesh.y[cell]), float(mesh.z[cell]), sw_v, so_v, sg_v))

    if not anchors:
        # default residual-like fill
        sw.fill(0.2)
        so.fill(0.8)
        sg.fill(0.0)
        notes.append("no well saturation sensors; used default sw=0.2, so=0.8, sg=0")
        return sw, so, sg, notes

    power = 2.0
    eps = 1.0e-12
    for idx in range(mesh.n_cells):
        i = int(mesh.i[idx])
        j = int(mesh.j[idx])
        k = int(mesh.k[idx])
        if np.isfinite(sw[k, j, i]):
            continue
        px, py, pz = mesh.x[idx], mesh.y[idx], mesh.z[idx]
        weights = []
        sws, sos, sgs = [], [], []
        for ax, ay, az, asw, aso, asg in anchors:
            dist = np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
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
        sw_v = float(np.dot(weights, sws) / wsum)
        so_v = float(np.dot(weights, sos) / wsum)
        sg_v = float(np.dot(weights, sgs) / wsum)
        sw_v, so_v, sg_v = _normalize_phases(sw_v, so_v, sg_v)
        sw[k, j, i] = sw_v
        so[k, j, i] = so_v
        sg[k, j, i] = sg_v

    return sw, so, sg, notes


def _normalize_phases(sw: float, so: float, sg: float) -> tuple[float, float, float]:
    sw = max(0.0, min(1.0, sw))
    so = max(0.0, min(1.0, so))
    sg = max(0.0, min(1.0, sg))
    total = sw + so + sg
    if total <= 0.0:
        return 0.2, 0.8, 0.0
    return sw / total, so / total, sg / total
