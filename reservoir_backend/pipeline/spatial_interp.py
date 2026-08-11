"""Spatial interpolation of point data onto the mesh.

- IDW always available (vectorized)
- Ordinary Kriging when point count / geometry allow (batch RHS solve)
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
    """Inverse-distance weight scalar samples onto cell centers (vectorized)."""
    pts, vals = _validate_points(points_xyz, values)
    if pts.shape[0] == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        return np.full(mesh.grid.shape, float(fill), dtype=float)

    targets = np.column_stack([mesh.x, mesh.y, mesh.z])
    flat = _idw_many(pts, vals, targets, power=float(power))
    return flat.reshape(mesh.grid.shape)


def ordinary_kriging_to_grid(
    mesh: MeshBundle,
    points_xyz: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    fill: float | None = None,
) -> NDArray[np.float64]:
    """Ordinary kriging with exponential covariance; batch multi-RHS solve."""
    pts, vals = _validate_points(points_xyz, values)
    n = pts.shape[0]
    if n == 0:
        if fill is None:
            raise ValueError("no points to interpolate")
        return np.full(mesh.grid.shape, float(fill), dtype=float)
    if n == 1:
        return np.full(mesh.grid.shape, float(vals[0]), dtype=float)

    targets = np.column_stack([mesh.x, mesh.y, mesh.z])
    flat = _krige_many(pts, vals, targets)
    return flat.reshape(mesh.grid.shape)


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
    if method == "idw":
        # vectorized LOO: exclude self via infinite self-distance
        d2 = _pairwise_d2(pts, pts)
        np.fill_diagonal(d2, np.inf)
        w = 1.0 / np.power(np.sqrt(d2), 2.0)
        wsum = np.sum(w, axis=1)
        pred = (w @ vals) / np.maximum(wsum, EPS)
        return float(np.sqrt(np.mean((pred - vals) ** 2)))
    if method == "kriging":
        errs = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            try:
                pred = float(_krige_many(pts[mask], vals[mask], pts[i : i + 1])[0])
            except Exception:
                return float("inf")
            errs.append((pred - vals[i]) ** 2)
        return float(np.sqrt(np.mean(errs)))
    raise ValueError(method)


def points_geometry_ok(points_xyz: NDArray[np.float64]) -> bool:
    """Reject near-collinear / collapsed clouds for kriging."""
    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] < N_MIN_KRIGING:
        return False
    c = pts - pts.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(c, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if s.size < 2:
        return False
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

    if n < N_MIN_KRIGING or not points_geometry_ok(pts):
        field = idw_points_to_grid(mesh, pts, work, fill=fill if not log_transform else None)
        if log_transform:
            field = np.exp(field)
        field = _clip(field, clip)
        reason = f"n={n}<{N_MIN_KRIGING}" if n < N_MIN_KRIGING else "geometry_degenerate"
        notes.append(f"auto-interp: idw ({reason})")
        return InterpResult(values=field, method="idw", notes=notes, n_points=n)

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
        f"auto-interp: {method} (n={n}, loo_idw={rmse_i:.4g}, "
        f"loo_krig={rmse_k if krig_ok else float('nan'):.4g})"
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
# vectorized kernels
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


def _pairwise_d2(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared distances between rows of a (na,3) and b (nb,3) → (na, nb)."""
    # ||a-b||^2 = |a|^2 + |b|^2 - 2 a·b
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0)


def _idw_many(
    pts: NDArray[np.float64],
    vals: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    power: float = 2.0,
) -> NDArray[np.float64]:
    d2 = _pairwise_d2(targets, pts)
    # exact hits
    hit = d2 < (EPS * EPS)
    any_hit = np.any(hit, axis=1)
    out = np.empty(targets.shape[0], dtype=float)
    if np.any(any_hit):
        # first matching sample index
        out[any_hit] = vals[np.argmax(hit[any_hit], axis=1)]
    rest = ~any_hit
    if np.any(rest):
        w = 1.0 / np.power(np.sqrt(d2[rest]), float(power))
        wsum = np.sum(w, axis=1)
        out[rest] = np.sum(w * vals[None, :], axis=1) / np.maximum(wsum, EPS)
    return out


def _idw_at(
    pts: NDArray[np.float64],
    vals: NDArray[np.float64],
    x: float,
    y: float,
    z: float,
    *,
    power: float = 2.0,
) -> float:
    q = np.array([[x, y, z]], dtype=float)
    return float(_idw_many(pts, vals, q, power=power)[0])


def _fit_exponential_cov(
    pts: NDArray[np.float64], vals: NDArray[np.float64]
) -> dict[str, float]:
    """Heuristic exponential covariance from pairwise cloud (vectorized)."""
    n = pts.shape[0]
    if n < 2:
        sill = float(np.var(vals)) + EPS
        return {"range": 1.0, "sill": sill, "nugget": NUGGET_FRAC * sill}

    d2 = _pairwise_d2(pts, pts)
    iu = np.triu_indices(n, k=1)
    darr = np.sqrt(d2[iu])
    ok = darr > EPS
    if not np.any(ok):
        sill = float(np.var(vals)) + EPS
        return {"range": 1.0, "sill": sill, "nugget": NUGGET_FRAC * sill}
    darr = darr[ok]
    vi = vals[iu[0][ok]]
    vj = vals[iu[1][ok]]
    garr = 0.5 * (vi - vj) ** 2
    sill = float(max(float(np.percentile(garr, 75)), float(np.var(vals)), EPS))
    mask = garr < 0.5 * sill
    if np.any(mask):
        rang = float(np.median(darr[mask]))
    else:
        rang = float(np.median(darr))
    rang = max(rang, float(np.percentile(darr, 25)), EPS)
    return {"range": rang, "sill": sill, "nugget": NUGGET_FRAC * sill}


def _cov_matrix(pts: NDArray[np.float64], p: dict[str, float]) -> NDArray[np.float64]:
    h = np.sqrt(_pairwise_d2(pts, pts))
    k = float(p["sill"]) * np.exp(-3.0 * h / max(float(p["range"]), EPS))
    k = k + float(p["nugget"]) * np.eye(pts.shape[0])
    return k


def _cov_to_targets(
    pts: NDArray[np.float64], targets: NDArray[np.float64], p: dict[str, float]
) -> NDArray[np.float64]:
    """Covariance pts→targets, shape (n_pts, n_targets)."""
    h = np.sqrt(_pairwise_d2(pts, targets))
    return float(p["sill"]) * np.exp(-3.0 * h / max(float(p["range"]), EPS))


def _krige_many(
    pts: NDArray[np.float64],
    vals: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    n = pts.shape[0]
    nt = targets.shape[0]
    if n == 0:
        raise ValueError("no points")
    if n == 1:
        return np.full(nt, float(vals[0]), dtype=float)

    cov_params = _fit_exponential_cov(pts, vals)
    k_mat = _cov_matrix(pts, cov_params)
    a = np.zeros((n + 1, n + 1), dtype=float)
    a[:n, :n] = k_mat
    a[:n, n] = 1.0
    a[n, :n] = 1.0

    c0 = _cov_to_targets(pts, targets, cov_params)  # (n, nt)
    rhs = np.ones((n + 1, nt), dtype=float)
    rhs[:n, :] = c0

    try:
        sol = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("kriging matrix singular") from exc

    lam = sol[:n, :]  # (n, nt)
    out = vals @ lam  # (nt,)

    # exact sample hits
    d2 = _pairwise_d2(targets, pts)
    hit = d2 < (EPS * EPS)
    any_hit = np.any(hit, axis=1)
    if np.any(any_hit):
        out = out.copy()
        out[any_hit] = vals[np.argmax(hit[any_hit], axis=1)]
    return out


def _clip(field: NDArray[np.float64], clip: tuple[float, float] | None) -> NDArray[np.float64]:
    if clip is None:
        return field
    return np.clip(field, float(clip[0]), float(clip[1]))
