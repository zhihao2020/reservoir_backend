"""Sequential simulation helpers.

This package contains lightweight orchestration layers that combine existing
pressure, velocity, saturation, CFL, and production diagnostics without
rewriting the underlying numerical solvers.
"""

from reservoir_backend.simulation.impes import (
    IMPESConfig,
    IMPESRunResult,
    IMPESStepResult,
    compute_mobility_fields,
    create_synthetic_waterflood_case,
    run_impes_simulation,
    run_impes_step,
)
from reservoir_backend.simulation.production import (
    build_production_summary,
    detect_breakthrough_time,
)

__all__ = [
    "IMPESConfig",
    "IMPESRunResult",
    "IMPESStepResult",
    "build_production_summary",
    "compute_mobility_fields",
    "create_synthetic_waterflood_case",
    "detect_breakthrough_time",
    "run_impes_simulation",
    "run_impes_step",
]
