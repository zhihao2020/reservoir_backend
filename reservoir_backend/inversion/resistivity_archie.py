"""Archie resistivity saturation inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D

InvalidPolicy = Literal["raise", "low_confidence"]


@dataclass(frozen=True)
class ArchieInverter:
    """Invert water saturation from resistivity using Archie's law.

    The implemented relation is:

    `Sw = ((a * Rw) / (phi**m * Rt)) ** (1 / n)`

    By default invalid physical inputs raise `InvalidPhysicalValueError`. Passing
    `invalid_policy="low_confidence"` keeps valid cells, marks invalid cells with
    zero confidence, and sets their saturation value to NaN.
    """

    a: float = 1.0
    m: float = 2.0
    n: float = 2.0
    swi: float = 0.0
    sor: float = 0.0
    invalid_policy: InvalidPolicy = "raise"

    def __post_init__(self) -> None:
        for name in ("a", "m", "n"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise InvalidPhysicalValueError(f"Archie parameter {name} must be positive")
            object.__setattr__(self, name, value)

        for name in ("swi", "sor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or value >= 1.0:
                raise InvalidPhysicalValueError(f"{name} must be within [0, 1)")
            object.__setattr__(self, name, value)

        if self.swi + self.sor >= 1.0:
            raise InvalidPhysicalValueError("swi + sor must be less than 1")
        if self.invalid_policy not in ("raise", "low_confidence"):
            raise ValueError("invalid_policy must be 'raise' or 'low_confidence'")

    def invert(
        self,
        rt: float | ArrayLike | Field3D,
        rw: float | ArrayLike | Field3D,
        phi: float | ArrayLike | Field3D,
    ) -> float | NDArray[np.float64] | Field3D:
        """Return water saturation for scalar, ndarray, or `Field3D` inputs."""
        saturation, confidence, template = self._compute(rt, rw, phi)
        if template is None:
            if saturation.shape == ():
                return float(saturation)
            return saturation

        return Field3D(
            grid=template.grid,
            values=saturation,
            name="sw_archie",
            unit="fraction",
            timestamp=template.timestamp,
            confidence=confidence,
        )

    def invert_with_confidence(
        self,
        rt: float | ArrayLike | Field3D,
        rw: float | ArrayLike | Field3D,
        phi: float | ArrayLike | Field3D,
    ) -> tuple[float | NDArray[np.float64], float | NDArray[np.float64]] | tuple[Field3D, Field3D]:
        """Return saturation and confidence as separate outputs."""
        saturation, confidence, template = self._compute(rt, rw, phi)
        if template is None:
            if saturation.shape == ():
                return float(saturation), float(confidence)
            return saturation, confidence

        sw_field = Field3D(
            grid=template.grid,
            values=saturation,
            name="sw_archie",
            unit="fraction",
            timestamp=template.timestamp,
            confidence=confidence,
        )
        confidence_field = Field3D(
            grid=template.grid,
            values=confidence,
            name="sw_archie_confidence",
            unit="fraction",
            timestamp=template.timestamp,
        )
        return sw_field, confidence_field

    def forward_resistivity(
        self,
        sw: float | ArrayLike,
        rw: float | ArrayLike,
        phi: float | ArrayLike,
    ) -> float | NDArray[np.float64]:
        """Compute resistivity from saturation using Archie's law."""
        sw_arr, rw_arr, phi_arr = np.broadcast_arrays(
            np.asarray(sw, dtype=float),
            np.asarray(rw, dtype=float),
            np.asarray(phi, dtype=float),
        )
        invalid = (
            ~np.isfinite(sw_arr)
            | ~np.isfinite(rw_arr)
            | ~np.isfinite(phi_arr)
            | (sw_arr <= 0.0)
            | (rw_arr <= 0.0)
            | (phi_arr <= 0.0)
        )
        if invalid.any():
            raise InvalidPhysicalValueError("sw, rw, and phi must be positive finite values")

        rt = (self.a * rw_arr) / ((phi_arr**self.m) * (sw_arr**self.n))
        if rt.shape == ():
            return float(rt)
        return rt

    def _compute(
        self,
        rt: float | ArrayLike | Field3D,
        rw: float | ArrayLike | Field3D,
        phi: float | ArrayLike | Field3D,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Field3D | None]:
        template = self._template_field(rt, rw, phi)
        if template is not None:
            self._assert_field_grids_match(template, rt, rw, phi)

        rt_values = self._values(rt)
        rw_values = self._values(rw)
        phi_values = self._values(phi)
        rt_arr, rw_arr, phi_arr = np.broadcast_arrays(rt_values, rw_values, phi_values)
        rt_arr = rt_arr.astype(float, copy=False)
        rw_arr = rw_arr.astype(float, copy=False)
        phi_arr = phi_arr.astype(float, copy=False)

        invalid = (
            ~np.isfinite(rt_arr)
            | ~np.isfinite(rw_arr)
            | ~np.isfinite(phi_arr)
            | (rt_arr <= 0.0)
            | (rw_arr <= 0.0)
            | (phi_arr <= 0.0)
        )
        if invalid.any() and self.invalid_policy == "raise":
            raise InvalidPhysicalValueError("Rt, Rw, and phi must be positive finite values")

        saturation = np.full(rt_arr.shape, np.nan, dtype=float)
        confidence = np.where(invalid, 0.0, 1.0).astype(float)
        valid = ~invalid
        if valid.any():
            raw = ((self.a * rw_arr[valid]) / ((phi_arr[valid] ** self.m) * rt_arr[valid])) ** (
                1.0 / self.n
            )
            saturation[valid] = np.clip(raw, self.swi, 1.0 - self.sor)

            clipped = (raw < self.swi) | (raw > 1.0 - self.sor)
            if clipped.any():
                valid_indices = np.flatnonzero(valid)
                confidence.reshape(-1)[valid_indices[clipped]] = 0.5

        if template is not None and saturation.shape != template.grid.shape:
            raise GridMismatchError(
                f"broadcast result shape {saturation.shape} does not match field grid "
                f"shape {template.grid.shape}"
            )

        return saturation, confidence, template

    @staticmethod
    def _values(value: float | ArrayLike | Field3D) -> NDArray[np.float64]:
        if isinstance(value, Field3D):
            return value.values
        return np.asarray(value, dtype=float)

    @staticmethod
    def _template_field(*values: float | ArrayLike | Field3D) -> Field3D | None:
        for value in values:
            if isinstance(value, Field3D):
                return value
        return None

    @staticmethod
    def _assert_field_grids_match(
        template: Field3D, *values: float | ArrayLike | Field3D
    ) -> None:
        for value in values:
            if isinstance(value, Field3D):
                template.assert_same_grid(value)
