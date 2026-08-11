"""IDW spatial interpolation of scalar point data onto the mesh."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle


def idw_points_to_grid(
    mesh: MeshBundle,
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    power: float = 2.0,
    fill: float | None = None,
) -> NDArray[np.float64]:
    """Inverse-distance weight scalar samples onto cell centers.

    ``points_xyz`` shape ``(n, 3)``, ``values`` shape ``(n,)``.
    """
    pts = np.asarray(points_xyz, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be (n, 3)")
    if vals.size != pts.shape[0]:
        raise ValueError("values length must match number of points")
    if pts.shape[0] == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        return np.full(mesh.grid.shape, float(fill), dtype=float)

    out = np.zeros(mesh.grid.shape, dtype=float)
    eps = 1.0e-12
    for n in range(mesh.n_cells):
        q = np.array([mesh.x[n], mesh.y[n], mesh.z[n]], dtype=float)
        d2 = np.sum((pts - q) ** 2, axis=1)
        if np.any(d2 < eps * eps):
            out.flat[n] = float(vals[int(np.argmin(d2))])
            continue
        w = 1.0 / np.power(np.sqrt(d2), float(power))
        out.flat[n] = float(np.dot(w, vals) / np.sum(w))
    return out
