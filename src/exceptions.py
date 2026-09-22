"""Physical and numerical errors. Flat list, no hierarchy beyond the base."""

from __future__ import annotations

__all__ = [
    "AssimilationError",
    "CaseSchemaError",
    "FlashCalculationError",
    "GridError",
    "InvalidControl",
    "InvalidObservation",
    "InvalidPermeability",
    "InvalidSaturation",
    "LinearSolveFailure",
    "MassBalanceError",
    "PhysicsConvergenceError",
    "ReservoirError",
    "TimeStepUnderflow",
    "UnitConversionError",
]


class ReservoirError(Exception):
    """Base error for the laboratory digital-twin backend."""


class InvalidSaturation(ReservoirError, ValueError):
    """Saturation is outside [0, 1] or does not close."""


class InvalidPermeability(ReservoirError, ValueError):
    """Permeability is non-positive, NaN, or otherwise illegal."""


class MassBalanceError(ReservoirError, ValueError):
    """Reported when a step violates conservation beyond tolerance."""


class LinearSolveFailure(ReservoirError, RuntimeError):
    """Pressure linear system failed."""


class TimeStepUnderflow(ReservoirError, RuntimeError):
    """Adaptive dt fell below the configured minimum."""


class InvalidObservation(ReservoirError, ValueError):
    """Observation geometry or data is inconsistent."""


class CaseSchemaError(ReservoirError, ValueError):
    """Laboratory case YAML or CSV cannot be parsed."""

    def __init__(self, issues: list[str] | str) -> None:
        if isinstance(issues, str):
            issues = [issues]
        self.issues = [str(item) for item in issues]
        super().__init__("; ".join(self.issues))


class InvalidControl(ReservoirError, ValueError):
    """A port has conflicting or missing controls."""


class GridError(ReservoirError, ValueError):
    """Grid construction or indexing failed."""


class UnitConversionError(ReservoirError, ValueError):
    """Unsupported unit conversion."""


class PhysicsConvergenceError(ReservoirError, RuntimeError):
    """Forward solve failed to converge, produced NaN, or underflowed dt."""


class FlashCalculationError(ReservoirError, RuntimeError):
    """EOS / flash evaluation failed."""


class AssimilationError(ReservoirError, RuntimeError):
    """Ensemble or parameter update failed."""
