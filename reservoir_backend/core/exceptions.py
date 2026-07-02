"""Project-specific exception types."""


class ReservoirBackendError(Exception):
    """Base class for reservoir backend errors."""


class GridIndexError(ReservoirBackendError, IndexError):
    """Raised when a grid index or `(i, j, k)` location is out of bounds."""


class FieldShapeError(ReservoirBackendError, ValueError):
    """Raised when field values do not match the owning grid shape."""


class GridMismatchError(ReservoirBackendError, ValueError):
    """Raised when operations are attempted across incompatible grids."""


class InvalidPhysicalValueError(ReservoirBackendError, ValueError):
    """Raised when a physical quantity has an invalid numeric value."""


class CFLViolationError(ReservoirBackendError, ValueError):
    """Raised when an explicit transport time step violates the CFL limit."""


class NonNeighborCellError(ReservoirBackendError, ValueError):
    """Raised when a face-based operation receives cells that are not face neighbors."""


class UnitConversionError(ReservoirBackendError, ValueError):
    """Raised when a requested unit conversion is unsupported."""


class WellControlError(ReservoirBackendError, ValueError):
    """Raised when a well definition or control mode is invalid."""
