"""Physical and numerical errors. Flat list, no hierarchy beyond the base."""


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
