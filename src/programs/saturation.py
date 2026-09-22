"""Program 3: three-phase saturation interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidObservation
from ..core.cartesian import CartesianGrid
from .interpolate import interpolate_field


def project_saturations(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Clip to non-negative and enforce Sw + So + Sg = 1 with Sg >= 0."""
    sw_p = np.maximum(np.asarray(sw, dtype=float), 0.0)
    so_p = np.maximum(np.asarray(so, dtype=float), 0.0)
    total = sw_p + so_p
    over = total > 1.0
    if np.any(over):
        sw_p = sw_p.copy()
        so_p = so_p.copy()
        sw_p[over] /= total[over]
        so_p[over] /= total[over]
    sg = np.maximum(1.0 - sw_p - so_p, 0.0)
    return sw_p, so_p, sg


def project_saturations3(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Clip to non-negative and enforce Sw + So + Sg = 1 for all three phases."""
    sw_p = np.maximum(np.asarray(sw, dtype=float), 0.0)
    so_p = np.maximum(np.asarray(so, dtype=float), 0.0)
    sg_p = np.maximum(np.asarray(sg, dtype=float), 0.0)
    total = sw_p + so_p + sg_p
    total = np.where(total > 0.0, total, 1.0)
    return sw_p / total, so_p / total, sg_p / total


def interpolate_saturation(
    grid: CartesianGrid,
    probe_xyz: NDArray[np.float64],
    sw_probe: NDArray[np.float64],
    so_probe: NDArray[np.float64],
    sg_probe: NDArray[np.float64] | None = None,
    *,
    method: str = "kriging",
    swc: float = 0.05,
    sgc: float = 0.02,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Interpolate water/oil/gas saturations from scatter probes.

    Each probe may observe only one phase (missing phases are NaN). Observed
    phases are interpolated independently, then the field is projected onto the
    simplex ``Sw + So + Sg = 1``; unobserved phases are recovered by closure or
    residual fill.
    """
    xyz = np.asarray(probe_xyz, dtype=float)
    sw_v = np.asarray(sw_probe, dtype=float).ravel()
    so_v = np.asarray(so_probe, dtype=float).ravel()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise InvalidObservation("saturation probes must have shape (n, 3)")
    if xyz.shape[0] != sw_v.size or xyz.shape[0] != so_v.size:
        raise InvalidObservation("saturation probe coordinates and values mismatch")
    sg_v = np.asarray(sg_probe, dtype=float).ravel() if sg_probe is not None else None
    if sg_v is not None and sg_v.size != xyz.shape[0]:
        raise InvalidObservation("saturation sg probe coordinates and values mismatch")

    targets = grid.cell_centers()
    name = str(method).strip().lower()

    sw_f = _interp_phase(xyz, sw_v, targets, name)
    so_f = _interp_phase(xyz, so_v, targets, name)
    sg_f = _interp_phase(xyz, sg_v, targets, name)
    n_c = int(targets.shape[0])

    known = int(sw_f is not None) + int(so_f is not None) + int(sg_f is not None)
    if known == 0:
        sw_f = np.full(n_c, swc)
        so_f = np.full(n_c, 1.0 - swc - sgc)
        sg_f = np.full(n_c, sgc)
    elif known == 1:
        if sw_f is None:
            sw_f = np.full(n_c, swc)
        if so_f is None:
            so_f = np.full(n_c, 1.0 - swc - sgc)
        if sg_f is None:
            sg_f = np.full(n_c, sgc)
    elif known == 2:
        if sg_f is None:
            sg_f = 1.0 - sw_f - so_f
        elif so_f is None:
            so_f = 1.0 - sw_f - sg_f
        else:
            sw_f = 1.0 - so_f - sg_f
    return project_saturations3(sw_f, so_f, sg_f)


def _interp_phase(
    xyz: NDArray[np.float64],
    values: NDArray[np.float64] | None,
    targets: NDArray[np.float64],
    method: str,
) -> NDArray[np.float64] | None:
    """Interpolate one phase; return ``None`` if it has no finite observations."""
    if values is None:
        return None
    vals = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(vals)
    if not finite.any():
        return None
    return interpolate_field(xyz[finite], vals[finite], targets, method=method)


def smooth_fields(
    fields: NDArray[np.float64],
    *,
    jump_rel: float = 0.2,
    alpha: float = 0.3,
) -> NDArray[np.float64]:
    """Weak temporal smoothing; skip steps that look like a true jump."""
    arr = np.asarray(fields, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return arr
    out = arr.copy()
    for t in range(1, arr.shape[0]):
        scale = float(np.sqrt(np.mean(out[t - 1] ** 2)))
        rel = float(np.sqrt(np.mean((arr[t] - out[t - 1]) ** 2))) / scale if scale > 0.0 else 0.0
        if rel < jump_rel:
            out[t] = (1.0 - alpha) * arr[t] + alpha * out[t - 1]
        else:
            out[t] = arr[t]
    return out
