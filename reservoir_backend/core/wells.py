"""Well definitions and validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError, WellControlError
from reservoir_backend.core.grid import Grid3D


class WellType(StrEnum):
    """Supported well types."""

    INJECTION = "injection"
    PRODUCTION = "production"


class ControlType(StrEnum):
    """Supported well control modes."""

    RATE = "rate"
    BHP = "bhp"


@dataclass(frozen=True)
class Well:
    """A single-grid-cell injection or production well.

    Rate-controlled wells use positive `rate` values. The signed source term is
    positive for injection and negative for production.
    """

    name: str
    well_type: WellType | str
    grid: Grid3D
    cell_index: int | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    control: ControlType | str = ControlType.RATE
    rate: float | None = None
    bhp: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise WellControlError("well name must not be empty")

        try:
            object.__setattr__(self, "well_type", WellType(self.well_type))
        except ValueError as exc:
            raise WellControlError(f"unsupported well_type: {self.well_type}") from exc

        try:
            object.__setattr__(self, "control", ControlType(self.control))
        except ValueError as exc:
            raise WellControlError(f"unsupported well control: {self.control}") from exc

        has_cell_index = self.cell_index is not None
        has_ijk = self.i is not None or self.j is not None or self.k is not None
        if has_cell_index and has_ijk:
            raise WellControlError("define either cell_index or i/j/k, not both")
        if not has_cell_index and not has_ijk:
            raise WellControlError("well location requires cell_index or i/j/k")

        if has_cell_index:
            # Validation is delegated to Grid3D.
            self.grid.ijk(int(self.cell_index))
            object.__setattr__(self, "cell_index", int(self.cell_index))
        else:
            if self.i is None or self.j is None or self.k is None:
                raise WellControlError("i, j, and k must all be provided")
            object.__setattr__(self, "cell_index", self.grid.index(int(self.i), int(self.j), int(self.k)))

        if self.control == ControlType.RATE:
            if self.rate is None:
                raise WellControlError("rate-controlled well requires rate")
            rate = float(self.rate)
            if not np.isfinite(rate) or rate <= 0.0:
                raise InvalidPhysicalValueError("well rate must be a positive finite value")
            object.__setattr__(self, "rate", rate)
        elif self.control == ControlType.BHP:
            raise NotImplementedError("BHP well control is reserved for a later stage")

    @property
    def location(self) -> tuple[int, int, int]:
        """Return the well location as `(i, j, k)`."""
        assert self.cell_index is not None
        return self.grid.ijk(self.cell_index)

    @property
    def signed_rate(self) -> float:
        """Return signed volumetric rate for source-term assembly."""
        if self.rate is None:
            raise WellControlError("well rate is not available for this control mode")
        if self.well_type == WellType.INJECTION:
            return self.rate
        return -self.rate
