"""Scatter-to-grid interpolators with fitted variograms and leave-one-out selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidObservation

_KRIGING = {"kriging", "ok", "ordinary_kriging"}
_UNIVERSAL = {"uk", "universal", "universal_kriging"}

# Relative-to-sill nugget floor. A nugget this small keeps the ordinary-kriging
# covariance system's condition number bounded (≲1e6) in float64, while still
# being negligible versus the signal; co-located points otherwise make the LU
# factorization exactly singular (lu_factor only warns, never raises).
_NUGGET_REL_FLOOR = 1.0e-6

# Upper bound on the IDW power. ``1/d**p`` with large ``p`` under/overflows for
# the typical sub-metre cell spacing; ``p`` beyond ~8 is effectively nearest
# neighbour anyway and adds nothing but numerical fragility.
_IDW_MAX_POWER = 8.0


@dataclass(frozen=True)
class VariogramModel:
    kind: str = "exponential"
    nugget: float = 1.0e-8
    sill: float = 1.0
    range_h: float = 0.1
    range_v: float = 0.1

    def lag(self, dh: NDArray[np.float64], dv: NDArray[np.float64]) -> NDArray[np.float64]:
        ah = max(float(self.range_h), 1.0e-12)
        av = max(float(self.range_v), 1.0e-12)
        return np.sqrt((np.asarray(dh, dtype=float) / ah) ** 2 + (np.asarray(dv, dtype=float) / av) ** 2)

    def correlation(self, dh: NDArray[np.float64], dv: NDArray[np.float64]) -> NDArray[np.float64]:
        # Exponential correlation (fixed).
        return np.exp(-np.clip(self.lag(dh, dv), 0.0, None))

    def covariance(self, dh: NDArray[np.float64], dv: NDArray[np.float64]) -> NDArray[np.float64]:
        corr = self.correlation(dh, dv)
        nugget = abs(float(self.nugget))
        sill = max(float(self.sill), 0.0)
        cov = sill * corr
        # Nugget only on the diagonal. Two *distinct* co-located points still
        # share the signal covariance ``sill`` (their block is [[sill+n, sill],
        # [sill, sill+n]] with eigenvalue n); adding ``sill+n`` to the off-diagonal
        # would make that block [[sill+n, sill+n],[sill+n, sill+n]] — exactly
        # singular, which is what lu_factor was warning about.
        if cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
            idx = np.diag_indices(int(cov.shape[0]))
            cov[idx] = sill + nugget
        return cov


@dataclass
class KrigingReport:
    model: VariogramModel
    loocv_rmse: float
    loocv_bias: float
    variance_mean: float
    anisotropic: bool
    n_pairs: int
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.model.kind,
            "nugget": self.model.nugget,
            "sill": self.model.sill,
            "range_h": self.model.range_h,
            "range_v": self.model.range_v,
            "loocv_rmse": self.loocv_rmse,
            "loocv_bias": self.loocv_bias,
            "variance_mean": self.variance_mean,
            "anisotropic": self.anisotropic,
            "n_pairs": self.n_pairs,
            **self.extras,
        }


def interpolate_field(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    method: str = "idw",
    power: float = 2.0,
    model: VariogramModel | None = None,
    return_variance: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate scalar values from scatter ``points`` onto ``targets``."""
    pts, vals, tgt = _clean_scatter(points, values, targets)
    name = str(method).strip().lower()
    if name == "idw":
        out = inverse_distance(pts, vals, tgt, power=power)
        var = np.zeros_like(out)
    elif name in _KRIGING | _UNIVERSAL:
        trend = "linear" if name in _UNIVERSAL else "constant"
        out, var = ordinary_kriging(pts, vals, tgt, model=model, trend=trend)
    else:
        raise InvalidObservation(f"unknown interpolator {method!r}")
    if return_variance:
        return out, var
    return out


def _clean_scatter(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    tgt = np.asarray(targets, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise InvalidObservation("scatter points must have shape (n, 3)")
    if tgt.ndim != 2 or tgt.shape[1] != 3:
        raise InvalidObservation("targets must have shape (m, 3)")
    if pts.shape[0] != vals.size:
        raise InvalidObservation("scatter points and values length mismatch")
    finite = np.isfinite(vals) & np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    vals = vals[finite]
    if pts.shape[0] == 0:
        raise InvalidObservation("no finite scatter values to interpolate")
    return pts, vals, tgt


def inverse_distance(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    power: float = 2.0,
    eps: float = 1.0e-18,
) -> NDArray[np.float64]:
    """Inverse-distance weighting. Coincident targets copy the scatter value."""
    if power <= 0.0 or power > _IDW_MAX_POWER:
        raise InvalidObservation(f"IDW power must be in (0, {_IDW_MAX_POWER}]")
    if points.shape[0] == 1:
        return np.full(targets.shape[0], float(values[0]), dtype=float)
    delta = targets[:, None, :] - points[None, :, :]
    dist2 = np.einsum("mnd,mnd->mn", delta, delta)
    exact = dist2 <= eps
    out = np.empty(targets.shape[0], dtype=float)
    has_exact = exact.any(axis=1)
    if has_exact.any():
        first = np.argmax(exact, axis=1)
        out[has_exact] = values[first[has_exact]]
    rest = ~has_exact
    if rest.any():
        weights = np.exp(-0.5 * power * np.log(dist2[rest] + eps))
        denom = weights.sum(axis=1, keepdims=True)
        out[rest] = (weights * values[None, :]).sum(axis=1) / denom[:, 0]
    return out


def _split_horizontal_vertical(delta: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dh = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2)
    dv = np.abs(delta[..., 2])
    return dh, dv


def default_variogram(points: NDArray[np.float64], values: NDArray[np.float64]) -> VariogramModel:
    n = int(points.shape[0])
    if n < 2:
        return VariogramModel()
    delta = points[:, None, :] - points[None, :, :]
    dh, dv = _split_horizontal_vertical(delta)
    dist = np.sqrt(dh**2 + dv**2)
    nn = np.min(dist + np.eye(n) * 1.0e18, axis=1)
    length = max(float(np.mean(nn)) * 3.0, 1.0e-6)
    sill = float(np.var(values))
    if not np.isfinite(sill) or sill <= 0.0:
        sill = 1.0
    return VariogramModel(kind="exponential", nugget=max(1.0e-8 * sill, _NUGGET_REL_FLOOR * sill), sill=sill, range_h=length, range_v=length)


def ordinary_kriging(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    model: VariogramModel | None = None,
    trend: str = "constant",
    eps: float = 1.0e-18,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Ordinary or universal kriging. Returns estimates and kriging variance."""
    n = int(points.shape[0])
    m = int(targets.shape[0])
    if n == 1 or float(np.var(values)) <= 0.0:
        fill = np.full(m, float(values[0]), dtype=float)
        return fill, np.zeros(m, dtype=float)
    spec = model or default_variogram(points, values)
    linear = str(trend).strip().lower() in {"linear", "universal"} and n >= 5
    n_trend = 4 if linear else 1
    delta_pp = points[:, None, :] - points[None, :, :]
    dh_pp, dv_pp = _split_horizontal_vertical(delta_pp)
    cov = spec.covariance(dh_pp, dv_pp)
    trend_pts = _trend_matrix(points, linear)
    system = np.zeros((n + n_trend, n + n_trend), dtype=float)
    system[:n, :n] = cov
    system[:n, n:] = trend_pts
    system[n:, :n] = trend_pts.T
    try:
        from scipy.linalg import lu_factor, lu_solve

        factor = lu_factor(system)
    except (np.linalg.LinAlgError, ValueError):
        pred = inverse_distance(points, values, targets)
        return pred, np.zeros(m, dtype=float)

    delta_tp = targets[:, None, :] - points[None, :, :]
    dh_tp, dv_tp = _split_horizontal_vertical(delta_tp)
    rhs = np.zeros((n + n_trend, m), dtype=float)
    rhs[:n, :] = spec.covariance(dh_tp, dv_tp).T
    rhs[n:, :] = _trend_matrix(targets, linear).T
    exact = (dh_tp**2 + dv_tp**2) <= eps
    has_exact = exact.any(axis=1)
    try:
        weights = lu_solve(factor, rhs)
    except np.linalg.LinAlgError:
        pred = inverse_distance(points, values, targets)
        return pred, np.zeros(m, dtype=float)
    if not np.isfinite(weights).all():
        # Singular LU factorisation (e.g. co-located points): lu_factor only
        # warns, so the garbage solve is caught here and demoted to IDW.
        pred = inverse_distance(points, values, targets)
        return pred, np.zeros(m, dtype=float)
    out = values @ weights[:n, :]
    c0 = float(spec.sill + abs(spec.nugget))
    variance = np.maximum(c0 - np.sum(weights[:n, :] * rhs[:n, :], axis=0) - np.sum(weights[n:, :] * rhs[n:, :], axis=0), 0.0)
    if has_exact.any():
        first = np.argmax(exact, axis=1)
        out[has_exact] = values[first[has_exact]]
        variance[has_exact] = 0.0
    return np.asarray(out, dtype=float), np.asarray(variance, dtype=float)


def _trend_matrix(xyz: NDArray[np.float64], linear: bool) -> NDArray[np.float64]:
    ones = np.ones((xyz.shape[0], 1), dtype=float)
    if not linear:
        return ones
    return np.column_stack([np.ones(xyz.shape[0]), xyz[:, 0], xyz[:, 1], xyz[:, 2]])


def leave_one_out(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    method: str,
    power: float = 2.0,
    model: VariogramModel | None = None,
) -> tuple[float, float]:
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    n = int(pts.shape[0])
    if n < 3:
        return float("nan"), float("nan")
    err = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = interpolate_field(
            pts[mask], vals[mask], pts[i : i + 1], method=method, power=power, model=model
        )
        err[i] = float(np.asarray(pred).ravel()[0] - vals[i])
    return float(np.sqrt(np.mean(err**2))), float(np.mean(err))


def fit_variogram(points: NDArray[np.float64], values: NDArray[np.float64]) -> KrigingReport:
    """Fit an exponential variogram (fixed kind) and report its LOOCV RMSE."""
    pts = np.asarray(points, dtype=float)
    vals = np.asarray(values, dtype=float).ravel()
    n = int(pts.shape[0])
    base = default_variogram(pts, vals)
    if n < 4 or float(np.var(vals)) <= 0.0:
        rmse, bias = leave_one_out(pts, vals, method="kriging", model=base)
        return KrigingReport(base, rmse, bias, 0.0, False, max(n * (n - 1) // 2, 0))

    delta = pts[:, None, :] - pts[None, :, :]
    dh, dv = _split_horizontal_vertical(delta)
    dist = np.sqrt(dh**2 + dv**2)
    iu, ju = np.triu_indices(n, k=1)
    lags = dist[iu, ju]
    gamma = 0.5 * (vals[iu] - vals[ju]) ** 2
    n_pairs = int(lags.size)
    anisotropic = _enough_anisotropy(dh[iu, ju], dv[iu, ju])
    range_h, range_v = _axis_ranges(dh[iu, ju], dv[iu, ju], lags, base)
    if not anisotropic:
        range_h = range_v = float(base.range_h)

    sill = max(float(np.var(vals)), 1.0e-18)
    nugget = min(0.05 * sill, float(np.median(gamma[: min(5, gamma.size)])) if gamma.size else 0.0)
    model = VariogramModel(
        kind="exponential",
        nugget=max(nugget, _NUGGET_REL_FLOOR * sill),
        sill=sill,
        range_h=range_h,
        range_v=range_v,
    )
    rmse, bias = leave_one_out(pts, vals, method="kriging", model=model)
    _, variance = ordinary_kriging(pts, vals, pts, model=model)
    return KrigingReport(
        model,
        rmse,
        bias,
        float(np.mean(variance)),
        anisotropic,
        n_pairs,
    )


def _enough_anisotropy(dh: NDArray[np.float64], dv: NDArray[np.float64]) -> bool:
    horiz = int(np.sum((dh > 1.0e-9) & (dv <= 0.25 * np.maximum(dh, 1.0e-12))))
    vert = int(np.sum((dv > 1.0e-9) & (dh <= 0.25 * np.maximum(dv, 1.0e-12))))
    return horiz >= 8 and vert >= 8


def _axis_ranges(
    dh: NDArray[np.float64],
    dv: NDArray[np.float64],
    lags: NDArray[np.float64],
    base: VariogramModel,
) -> tuple[float, float]:
    horiz = dh[(dh > 1.0e-9) & (dv <= 0.25 * np.maximum(dh, 1.0e-12))]
    vert = dv[(dv > 1.0e-9) & (dh <= 0.25 * np.maximum(dv, 1.0e-12))]
    range_h = float(np.median(horiz) * 3.0) if horiz.size else float(base.range_h)
    range_v = float(np.median(vert) * 3.0) if vert.size else float(base.range_v)
    fallback = max(float(np.median(lags)) * 2.0, 1.0e-6) if lags.size else float(base.range_h)
    return max(range_h, 1.0e-6), max(range_v if vert.size else fallback, 1.0e-6)


def interpolation_weights(
    points: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    method: str = "idw",
    power: float = 2.0,
    model: VariogramModel | None = None,
    trend: str = "constant",
) -> NDArray[np.float64]:
    """Return the ``(n_points, n_targets)`` weight matrix ``W`` of a linear
    interpolator, so ``values @ W`` reproduces ``interpolate_field``.

    IDW and (universal) kriging weights depend only on point/target geometry
    and the variogram model, never on the values. Precomputing ``W`` once turns
    an optimisation inner loop into a single matrix-vector product.
    """
    pts = np.asarray(points, dtype=float)
    tgt = np.asarray(targets, dtype=float)
    name = str(method).strip().lower()
    if name == "idw":
        return _idw_weights(pts, tgt, power=power)
    if name in _KRIGING | _UNIVERSAL:
        trend_name = "linear" if name in _UNIVERSAL else "constant"
        return _kriging_weights(pts, tgt, model=model, trend=trend_name)
    raise InvalidObservation(f"interpolator {method!r} has no precomputable weights")


def _idw_weights(
    points: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    power: float = 2.0,
    eps: float = 1.0e-18,
) -> NDArray[np.float64]:
    if power <= 0.0 or power > _IDW_MAX_POWER:
        raise InvalidObservation(f"IDW power must be in (0, {_IDW_MAX_POWER}]")
    n, m = points.shape[0], targets.shape[0]
    if n == 1:
        return np.ones((1, m), dtype=float)
    delta = targets[:, None, :] - points[None, :, :]
    dist2 = np.einsum("mnd,mnd->mn", delta, delta)
    weights = np.exp(-0.5 * power * np.log(dist2 + eps))
    w = (weights / weights.sum(axis=1, keepdims=True)).T
    has_exact = (dist2 <= eps).any(axis=1)
    if has_exact.any():
        first = np.argmax(dist2 <= eps, axis=1)
        for t in np.flatnonzero(has_exact):
            w[:, t] = 0.0
            w[first[t], t] = 1.0
    return w


def _kriging_weights(
    points: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    model: VariogramModel | None = None,
    trend: str = "constant",
    eps: float = 1.0e-18,
) -> NDArray[np.float64]:
    n, m = points.shape[0], targets.shape[0]
    if n == 1:
        return np.ones((1, m), dtype=float)
    spec = model or default_variogram(points, np.zeros(n, dtype=float))
    linear = str(trend).strip().lower() in {"linear", "universal"} and n >= 5
    n_trend = 4 if linear else 1
    dh_pp, dv_pp = _split_horizontal_vertical(points[:, None, :] - points[None, :, :])
    cov = spec.covariance(dh_pp, dv_pp)
    trend_pts = _trend_matrix(points, linear)
    system = np.zeros((n + n_trend, n + n_trend), dtype=float)
    system[:n, :n] = cov
    system[:n, n:] = trend_pts
    system[n:, :n] = trend_pts.T
    try:
        from scipy.linalg import lu_factor, lu_solve

        factor = lu_factor(system)
    except (np.linalg.LinAlgError, ValueError):
        return _idw_weights(points, targets, power=2.0)
    dh_tp, dv_tp = _split_horizontal_vertical(targets[:, None, :] - points[None, :, :])
    rhs = np.zeros((n + n_trend, m), dtype=float)
    rhs[:n, :] = spec.covariance(dh_tp, dv_tp).T
    rhs[n:, :] = _trend_matrix(targets, linear).T
    try:
        weights = lu_solve(factor, rhs)
    except np.linalg.LinAlgError:
        return _idw_weights(points, targets, power=2.0)
    if not np.isfinite(weights).all():
        return _idw_weights(points, targets, power=2.0)
    w = weights[:n, :]
    has_exact = ((dh_tp**2 + dv_tp**2) <= eps).any(axis=1)
    if has_exact.any():
        first = np.argmax((dh_tp**2 + dv_tp**2) <= eps, axis=1)
        for t in np.flatnonzero(has_exact):
            w[:, t] = 0.0
            w[first[t], t] = 1.0
    return w

