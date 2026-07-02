"""Material balance utilities."""

from __future__ import annotations

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.wells import Well, WellType


def compute_constant_rate_well_balance(wells: list[Well], dt: float) -> dict[str, float]:
    """Return injected/produced volumes and relative balance error for rate wells."""
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise InvalidPhysicalValueError("dt must be a positive finite value")

    injection_rate = sum(well.rate or 0.0 for well in wells if well.well_type == WellType.INJECTION)
    production_rate = sum(well.rate or 0.0 for well in wells if well.well_type == WellType.PRODUCTION)
    injected_volume = float(injection_rate * dt)
    produced_volume = float(production_rate * dt)
    net_volume = injected_volume - produced_volume
    scale = max(abs(injected_volume), abs(produced_volume), 1.0)
    return {
        "injected_volume": injected_volume,
        "produced_volume": produced_volume,
        "net_volume": net_volume,
        "relative_error": abs(net_volume) / scale,
    }
