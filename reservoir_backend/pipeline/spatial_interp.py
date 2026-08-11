"""Spatial interpolation of point data onto the mesh.

- IDW always available
- Ordinary Kriging when point count / geometry allow
- Auto selection by LOO-CV (no user configuration)

Pressure fields stay on TPFA elsewhere; this module is for scalar maps
(especially point k and φ → full grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.state import MeshBundle

# Built-in policy constants (not exposed as YAML/CLI)
N_MIN_KRIGING = 8
CV_MARGIN = 0.05
NUGGET_FRAC = 0.05
EPS = 1.0e-12


@dataclass
class InterpResult:
    """Grid field plus auto-selection diagnostics."""

    values: NDArray[np.float64]
    method: str  # idw | kriging | stack
    notes: list[str]
    loo_rmse_idw: float | None = None
    loo_rmse_kriging: float | None = None
    n_points: int = 0


def idw_points_to_grid(
    mesh: MeshBundle,
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    power: float = 2.0,
    fill: float | None = None,
) -> NDArray[np.float64]:
    """Inverse-distance weight scalar samples onto cell centers."""
    pts, vals = _validate_points(points_xyz, values)
    if pts.shape[0] == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        return np.full(mesh.grid.shape, float(fill), dtype=float)

    out = np.zeros(mesh.grid.shape, dtype=float)
    for n in range(mesh.n_cells):
        out.flat[n] = _idw_at(pts, vals, mesh.x[n], mesh.y[n], mesh.z[n], power=power)
    return out


def ordinary_kriging_to_grid(
    mesh: MeshBundle,
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    fill: float | None = None,
) -> NDArray[np.float64]:
    """Ordinary kriging with a simple exponential covariance (auto range/sill)."""
    pts, vals = _validate_points(points_xyz, values)
    n = pts.shape[0]
    if n == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        return np.full(mesh.grid.shape, float(fill), dtype=float)
    if n == 1:
        return np.full(mesh.grid.shape, float(vals[0]), dtype=float)

    cov_params = _fit_exponential_cov(pts, vals)
    k_mat = _cov_matrix(pts, cov_params)
    # [C 1; 1 0]
    a = np.zeros((n + 1, n + 1), dtype=float)
    a[:n, :n] = k_mat
    a[:n, n] = 1.0
    a[n, :n] = 1.0
    a[n, n] = 0.0

    try:
        from scipy.linalg import lu_factor, lu_solve

        lu = lu_factor(a)
        use_lu = True
        inv_a = None
    except Exception:
        use_lu = False
        try:
            inv_a = np.linalg.inv(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("kriging matrix singular") from exc

    out = np.zeros(mesh.grid.shape, dtype=float)
    rhs = np.ones(n + 1, dtype=float)
    for idx in range(mesh.n_cells):
        q = np.array([mesh.x[idx], mesh.y[idx], mesh.z[idx]], dtype=float)
        # exact hit
        d2 = np.sum((pts - q) ** 2, axis=1)
        if np.any(d2 < EPS * EPS):
            out.flat[idx] = float(vals[int(np.argmin(d2))])
            continue
        c0 = _cov_vector(pts, q, cov_params)
        rhs[:n] = c0
        rhs[n] = 1.0
        if use_lu:
            sol = lu_solve(lu, rhs)
        else:
            sol = inv_a @ rhs  # type: ignore[operator]
        lam = sol[:n]
        out.flat[idx] = float(np.dot(lam, vals))
    return out


def leave_one_out_rmse(
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    method: str,
) -> float:
    """Leave-one-out RMSE for idw or kriging (point→point, not full grid)."""
    pts, vals = _validate_points(points_xyz, values)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        train_p, train_v = pts[mask], vals[mask]
        q = pts[i]
        if method == "idw":
            pred = _idw_at(train_p, train_v, q[0], q[1], q[2])
        elif method == "kriging":
            try:
                pred = _krige_at(train_p, train_v, q)
            except Exception:
                return float("inf")
        else:
            raise ValueError(method)
        errs.append((pred - vals[i]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def points_geometry_ok(points_xyz: NDArray[np.float64]) -> bool:
    """Reject near-collinear / collapsed clouds for kriging."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] < N_MIN_KRIGING:
        return False
    c = pts - pts.mean(axis=0, keepdims=True)
    # rank of covariance
    try:
        s = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if s.size < 2:
        return False
    # if second singular value tiny → nearly collinear
    if s[0] < EPS:
        return False
    if s[1] / s[0] < 1.0e-4:
        return False
    return True


def auto_interpolate_to_grid(
    mesh: MeshBundle,
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    fill: float | None = None,
    log_transform: bool = False,
    clip: tuple[float, float] | None = None,
) -> InterpResult:
    """Auto IDW / ordinary kriging / stack by LOO-CV (no user method config)."""
    pts, vals = _validate_points(points_xyz, values)
    n = pts.shape[0]
    notes: list[str] = []

    if n == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        field = np.full(mesh.grid.shape, float(fill), dtype=float)
        return InterpResult(values=field, method="idw", notes=["auto-interp: empty points, fill"], n_points=0)

    work = vals.copy()
    if log_transform:
        work = np.log(np.clip(work, 1.0e-30, None))
        notes.append("auto-interp: log-transform on")

    # --- force IDW when too few or geometry bad ---
    if n < N_MIN_KRIGING or not points_geometry_ok(pts):
        field = idw_points_to_grid(mesh, pts, work, fill=fill if not log_transform else None)
        if log_transform:
            field = np.exp(field)
        field = _clip(field, clip)
        reason = f"n={n}<{N_MIN_KRIGING}" if n < N_MIN_KRIGING else "geometry_degenerate"
        notes.append(f"auto-interp: idw ({reason})")
        return InterpResult(values=field, method="idw", notes=notes, n_points=n)

    # --- LOO CV both methods ---
    rmse_i = leave_one_out_rmse(pts, work, method="idw")
    try:
        rmse_k = leave_one_out_rmse(pts, work, method="kriging")
        krig_ok = np.isfinite(rmse_k)
    except Exception:
        rmse_k = float("inf")
        krig_ok = False

    method = "idw"
    if not krig_ok:
        method = "idw"
        notes.append("auto-interp: kriging LOO failed → idw")
    elif rmse_k < rmse_i * (1.0 - CV_MARGIN):
        method = "kriging"
    elif rmse_i < rmse_k * (1.0 - CV_MARGIN):
        method = "idw"
    else:
        method = "stack"

    try:
        if method == "idw":
            field = idw_points_to_grid(mesh, pts, work, fill=None)
        elif method == "kriging":
            field = ordinary_kriging_to_grid(mesh, pts, work, fill=None)
        else:
            f_i = idw_points_to_grid(mesh, pts, work, fill=None)
            f_k = ordinary_kriging_to_grid(mesh, pts, work, fill=None)
            w_i = 1.0 / (EPS + rmse_i * rmse_i)
            w_k = 1.0 / (EPS + rmse_k * rmse_k)
            field = (w_i * f_i + w_k * f_k) / (w_i + w_k)
            notes.append(f"auto-interp stack weights idw={w_i:.3g} krig={w_k:.3g}")
    except Exception as exc:
        field = idw_points_to_grid(mesh, pts, work, fill=None)
        method = "idw"
        notes.append(f"auto-interp: kriging/stack failed ({exc}); idw")

    if log_transform:
        field = np.exp(field)
    field = _clip(field, clip)
    notes.append(
        f"auto-interp: {method} (n={n}, loo_idw={rmse_i:.4g}, loo_krig={rmse_k if krig_ok else float('nan'):.4g})"
    )
    return InterpResult(
        values=field,
        method=method,
        notes=notes,
        loo_rmse_idw=rmse_i,
        loo_rmse_kriging=rmse_k if krig_ok else None,
        n_points=n,
    )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _validate_points(
    points_xyz: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pts = np.asarray(points_xyz, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float), vals
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be (n, 3)")
    if vals.size != pts.shape[0]:
        raise ValueError("values length must match number of points")
    return pts, vals


def _idw_at(
    pts: NDArray[np.float64],
    vals: NDArray[np.float64],
    x: float,
    y: float,
    z: float,
    *,
    power: float = 2.0,
) -> float:
    q = np.array([x, y, z], dtype=float)
    d2 = np.sum((pts - q) ** 2, axis=1)
    if np.any(d2 < EPS * EPS):
        return float(vals[int(np.argmin(d2))])
    w = 1.0 / np.power(np.sqrt(d2), float(power))
    return float(np.dot(w, vals) / np.sum(w))


def _fit_exponential_cov(
    pts: NDArray[np.float64], vals: NDArray[np.float64]
) -> dict[str, float]:
    """Heuristic exponential covariance parameters from point cloud."""
    n = pts.shape[0]
    # pairwise distances
    dlist = []
    glist = []
    mean = float(np.mean(vals))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d < EPS:
                continue
            g = 0.5 * (vals[i] - vals[j]) ** 2
            dlist.append(d)
            glist.append(g)
    if not dlist:
        sill = float(np.var(vals)) + EPS
        return {"range": 1.0, "sill": sill, "nugget": NUGGET_FRAC * sill}

    darr = np.asarray(dlist, dtype=float)
    garr = np.asarray(glist, dtype=float)
    sill = float(max(float(np.percentile(garr, 75)), float(np.var(vals)), EPS))
    # range ~ median distance of pairs with gamma < 0.5 * sill, else median d
    mask = garr < 0.5 * sill
    if np.any(mask):
        rang = float(np.median(darr[mask]))
    else:
        rang = float(np.median(darr))
    rang = max(rang, float(np.percentile(darr, 25)), EPS)
    nugget = NUGGET_FRAC * sill
    return {"range": rang, "sill": sill, "nugget": nugget}


def _cov_exp(h: float, p: dict[str, float]) -> float:
    # C(h) = sill * exp(-3 h / range)  (practical range)
    return float(p["sill"]) * float(np.exp(-3.0 * h / max(p["range"], EPS)))


def _cov_matrix(pts: NDArray[np.float64], p: dict[str, float]) -> NDArray[np.float64]:
    n = pts.shape[0]
    k = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            h = float(np.linalg.norm(pts[i] - pts[j]))
            c = _cov_exp(h, p)
            if i == j:
                c += p["nugget"]
            k[i, j] = c
            k[j, i] = c
    return k


def _cov_vector(pts: NDArray[np.float64], q: NDArray[np.float64], p: dict[str, float]) -> NDArray[np.float64]:
    n = pts.shape[0]
    c0 = np.zeros(n, dtype=float)
    for i in range(n):
        h = float(np.linalg.norm(pts[i] - q))
        c0[i] = _cov_exp(h, p)
    return c0


def _krige_at(pts: NDArray[np.float64], vals: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    n = pts.shape[0]
    if n == 1:
        return float(vals[0])
    p = _fit_exponential_cov(pts, vals)
    k_mat = _cov_matrix(pts, p)
    a = np.zeros((n + 1, n + 1), dtype=float)
    a[:n, :n] = k_mat
    a[:n, n] = 1.0
    a[n, :n] = 1.0
    c0 = _cov_vector(pts, q, p)
    rhs = np.ones(n + 1, dtype=float)
    rhs[:n] = c0
    sol = np.linalg.solve(a, rhs)
    return float(np.dot(sol[:n], vals))


def _clip(field: NDArray[np.float64], clip: tuple[float, float] | None) -> NDArray[np.float64]:
    if clip is None:
        return field
    return np.clip(field, float(clip[0]), float(clip[1]))
