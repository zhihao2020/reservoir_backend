"""FIM time-step helpers. Shared by single-porosity and DPDP compositional Newton."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TARGET_ITERATION_COUNT = 5
ITERATION_OFFSET = 5
MAX_RELATIVE_DT = 2.0
MIN_RELATIVE_DT = 0.5


def index_nearest_time(times: NDArray[np.float64] | list[float], t: float) -> int:
    """Nearest stored step to a report/probe time."""
    arr = np.asarray(times, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty times")
    return int(np.argmin(np.abs(arr - float(t))))


def clip_dt_to_report_times(
    t: float,
    dt: float,
    report_times: NDArray[np.float64] | None,
    t_end: float,
) -> float:
    """Do not step over the next report time."""
    remaining = float(t_end) - float(t)
    dt = min(float(dt), remaining)
    if report_times is None or remaining <= 0.0:
        return dt
    later = np.unique(np.asarray(report_times, dtype=float).ravel())
    later = later[later > float(t) + 1.0e-15]
    if later.size:
        dt = min(dt, float(later[0]) - float(t))
    return dt


def iteration_count_timestep(
    dt1: float,
    its1: int,
    *,
    dt0: float | None = None,
    its0: int | None = None,
    target: int = TARGET_ITERATION_COUNT,
    offset: int = ITERATION_OFFSET,
    maxits: int = 12,
    dt_min: float = 0.0,
    dt_max: float | None = None,
    max_rel: float = MAX_RELATIVE_DT,
    min_rel: float = MIN_RELATIVE_DT,
) -> float:
    """Next dt from Newton iteration history, with relative clamps."""
    maxits_w = max(int(maxits) + int(offset), 1)
    tol = (float(target) + float(offset)) / maxits_w
    le1 = (float(its1) + float(offset)) / maxits_w
    le1 = max(le1, 1.0e-12)
    if dt0 is None or its0 is None:
        dt_new = (tol / le1) * float(dt1)
    else:
        le0 = (float(its0) + float(offset)) / maxits_w
        dt_new = (float(dt1) / max(float(dt0), 1.0e-30)) * (tol * le0 / (le1 * le1)) * float(dt1)
    change = dt_new / max(float(dt1), 1.0e-30)
    change = min(max(change, float(min_rel)), float(max_rel))
    dt = float(dt1) * change
    if dt_max is not None:
        dt = min(dt, float(dt_max))
    return max(dt, float(dt_min))


def dt_from_newton_iters(
    dt: float,
    newton_iters: int,
    *,
    dt0: float | None = None,
    its0: int | None = None,
    dt_min: float = 1.0e-6,
    dt_max: float = 1.0e30,
    target_its: int = TARGET_ITERATION_COUNT,
) -> float:
    """Grow/shrink Δt from successful Newton iteration count."""
    return float(
        iteration_count_timestep(
            float(dt),
            int(newton_iters),
            dt0=dt0,
            its0=its0,
            target=int(target_its),
            dt_min=float(dt_min),
            dt_max=float(dt_max),
        )
    )
