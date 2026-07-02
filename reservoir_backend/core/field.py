"""3D field container tied to a :class:`Grid3D`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import FieldShapeError, GridMismatchError
from reservoir_backend.core.grid import Grid3D


@dataclass
class Field3D:
    """A named 3D scalar field stored on a `Grid3D`."""

    grid: Grid3D
    values: ArrayLike
    name: str = ""
    unit: str = ""
    timestamp: datetime | None = None
    confidence: ArrayLike | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != self.grid.shape:
            raise FieldShapeError(
                f"values shape {values.shape} does not match grid shape {self.grid.shape}"
            )
        if np.isinf(values).any():
            raise FieldShapeError("values must not contain Inf")
        self.values = values.copy()

        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=float)
            if confidence.shape != self.grid.shape:
                raise FieldShapeError(
                    "confidence shape "
                    f"{confidence.shape} does not match grid shape {self.grid.shape}"
                )
            if np.isnan(confidence).any() or np.isinf(confidence).any():
                raise FieldShapeError("confidence must be finite")
            if (confidence < 0.0).any() or (confidence > 1.0).any():
                raise FieldShapeError("confidence must be within [0, 1]")
            self.confidence = confidence.copy()

    @classmethod
    def from_constant(
        cls,
        grid: Grid3D,
        value: float,
        *,
        name: str = "",
        unit: str = "",
        timestamp: datetime | None = None,
        confidence: float | ArrayLike | None = None,
    ) -> "Field3D":
        """Create a field filled with one scalar value."""
        values = np.full(grid.shape, float(value), dtype=float)
        if confidence is None:
            confidence_values = None
        elif np.isscalar(confidence):
            confidence_values = np.full(grid.shape, float(confidence), dtype=float)
        else:
            confidence_values = confidence
        return cls(
            grid=grid,
            values=values,
            name=name,
            unit=unit,
            timestamp=timestamp,
            confidence=confidence_values,
        )

    def copy(self, *, name: str | None = None) -> "Field3D":
        """Return a deep copy of this field."""
        return Field3D(
            grid=self.grid,
            values=self.values.copy(),
            name=self.name if name is None else name,
            unit=self.unit,
            timestamp=self.timestamp,
            confidence=None if self.confidence is None else self.confidence.copy(),
        )

    def clip(self, minimum: float, maximum: float) -> "Field3D":
        """Return a copy with values clipped into `[minimum, maximum]`."""
        if minimum > maximum:
            raise ValueError("minimum must be less than or equal to maximum")
        return Field3D(
            grid=self.grid,
            values=np.clip(self.values, minimum, maximum),
            name=self.name,
            unit=self.unit,
            timestamp=self.timestamp,
            confidence=None if self.confidence is None else self.confidence.copy(),
        )

    def fill_nan(self, value: float) -> "Field3D":
        """Return a copy with NaN values replaced by `value`."""
        filled = np.where(np.isnan(self.values), float(value), self.values)
        return Field3D(
            grid=self.grid,
            values=filled,
            name=self.name,
            unit=self.unit,
            timestamp=self.timestamp,
            confidence=None if self.confidence is None else self.confidence.copy(),
        )

    def to_numpy(self, *, copy: bool = True) -> NDArray[np.float64]:
        """Return field values as a NumPy array."""
        return self.values.copy() if copy else self.values

    def assert_same_grid(self, other: "Field3D") -> None:
        """Raise if `other` is not defined on an equivalent grid."""
        if not isinstance(other, Field3D):
            raise GridMismatchError("other must be a Field3D")
        if self.grid != other.grid:
            raise GridMismatchError("fields are defined on different grids")
