"""Archie resistivity saturation inversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reservoir_backend.core.exceptions import GridMismatchError, InvalidPhysicalValueError
from reservoir_backend.core.field import Field3D

InvalidPolicy = Literal["raise", "low_confidence"]


def invert_saturation_archie(
    resistivity: float | ArrayLike,
    water_resistivity: float | ArrayLike,
    porosity: float | ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    clip: bool = True,
    return_report: bool = False,
) -> float | NDArray[np.float64] | tuple[float | NDArray[np.float64], dict]:
    """Invert water saturation from Archie's law with validation and reporting."""
    _validate_positive("water_resistivity", water_resistivity)
    for name, value in {"a": a, "m": m, "n": n}.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise InvalidPhysicalValueError(f"{name} must be positive")

    rt = np.asarray(resistivity, dtype=float)
    rw = np.asarray(water_resistivity, dtype=float)
    phi = np.asarray(porosity, dtype=float)
    rt, rw, phi = np.broadcast_arrays(rt, rw, phi)

    if (~np.isfinite(rt)).any() or (rt <= 0.0).any():
        raise InvalidPhysicalValueError("resistivity must be positive and finite")
    if (~np.isfinite(rw)).any() or (rw <= 0.0).any():
        raise InvalidPhysicalValueError("water_resistivity must be positive and finite")
    if (~np.isfinite(phi)).any() or (phi <= 0.0).any() or (phi > 1.0).any():
        raise InvalidPhysicalValueError("porosity must be finite and within (0, 1]")

    raw = ((float(a) * rw) / ((phi ** float(m)) * rt)) ** (1.0 / float(n))
    if (~np.isfinite(raw)).any():
        raise InvalidPhysicalValueError("Archie inversion produced non-finite saturation")

    if clip:
        saturation = np.clip(raw, 0.0, 1.0)
    else:
        saturation = raw.copy()

    report = _build_inversion_report(
        method="archie",
        saturation=saturation,
        raw_saturation=raw,
        warnings=[],
    )
    result = _to_scalar_if_needed(saturation)
    if return_report:
        report["saturation"] = result
        return result, report
    return result


def archie_sensitivity_report(
    resistivity: float | ArrayLike,
    water_resistivity: float | ArrayLike,
    porosity: float | ArrayLike,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    perturbation: float = 0.01,
) -> dict:
    """Return finite-difference sensitivity of Archie saturation to key inputs."""
    if not np.isfinite(float(perturbation)) or float(perturbation) <= 0.0:
        raise InvalidPhysicalValueError("perturbation must be positive")

    base = np.asarray(
        invert_saturation_archie(resistivity, water_resistivity, porosity, a=a, m=m, n=n),
        dtype=float,
    )
    parameters = {
        "Rt": (resistivity, water_resistivity, porosity, a, m, n),
        "Rw": (resistivity, water_resistivity, porosity, a, m, n),
        "phi": (resistivity, water_resistivity, porosity, a, m, n),
        "m": (resistivity, water_resistivity, porosity, a, m, n),
        "n": (resistivity, water_resistivity, porosity, a, m, n),
    }
    sensitivity: dict[str, float] = {}
    relative: dict[str, float] = {}
    for name, values in parameters.items():
        rt, rw, phi, aa, mm, nn = values
        if name == "Rt":
            step = np.asarray(rt, dtype=float) * perturbation
            shifted = invert_saturation_archie(np.asarray(rt, dtype=float) + step, rw, phi, a=aa, m=mm, n=nn)
        elif name == "Rw":
            step = np.asarray(rw, dtype=float) * perturbation
            shifted = invert_saturation_archie(rt, np.asarray(rw, dtype=float) + step, phi, a=aa, m=mm, n=nn)
        elif name == "phi":
            step = np.asarray(phi, dtype=float) * perturbation
            shifted = invert_saturation_archie(rt, rw, np.asarray(phi, dtype=float) + step, a=aa, m=mm, n=nn)
        elif name == "m":
            step = float(mm) * perturbation
            shifted = invert_saturation_archie(rt, rw, phi, a=aa, m=float(mm) + step, n=nn)
        else:
            step = float(nn) * perturbation
            shifted = invert_saturation_archie(rt, rw, phi, a=aa, m=mm, n=float(nn) + step)

        shifted_arr = np.asarray(shifted, dtype=float)
        step_scale = np.asarray(step, dtype=float)
        deriv = np.mean((shifted_arr - base) / step_scale)
        sensitivity[name] = float(deriv)
        denom = max(float(np.mean(np.abs(base))), 1.0e-12)
        relative[name] = float(deriv * float(np.mean(np.asarray(values[0 if name == "Rt" else 1 if name == "Rw" else 2], dtype=float))) / denom) if name in {"Rt", "Rw", "phi"} else float(deriv * (float(m if name == "m" else n)) / denom)

    values = np.asarray(list(sensitivity.values()) + list(relative.values()), dtype=float)
    return {
        "method": "archie_finite_difference_sensitivity",
        "base_saturation": _to_scalar_if_needed(base),
        "sensitivity": sensitivity,
        "relative_sensitivity": relative,
        "warnings": [],
        "has_nan": bool(np.isnan(values).any()),
        "has_inf": bool(np.isinf(values).any()),
    }


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


def _validate_positive(name: str, value: float | ArrayLike) -> None:
    arr = np.asarray(value, dtype=float)
    if (~np.isfinite(arr)).any() or (arr <= 0.0).any():
        raise InvalidPhysicalValueError(f"{name} must be positive and finite")


def _build_inversion_report(
    method: str,
    saturation: NDArray[np.float64],
    raw_saturation: NDArray[np.float64],
    warnings: list[str],
) -> dict:
    raw = np.asarray(raw_saturation, dtype=float)
    sw = np.asarray(saturation, dtype=float)
    return {
        "method": method,
        "success": True,
        "saturation": _to_scalar_if_needed(sw),
        "raw_saturation_min": float(np.min(raw)),
        "raw_saturation_max": float(np.max(raw)),
        "saturation_min": float(np.min(sw)),
        "saturation_max": float(np.max(sw)),
        "num_clipped_low": int(np.sum(raw < 0.0)),
        "num_clipped_high": int(np.sum(raw > 1.0)),
        "warnings": warnings,
        "has_nan": bool(np.isnan(sw).any() or np.isnan(raw).any()),
        "has_inf": bool(np.isinf(sw).any() or np.isinf(raw).any()),
    }


def _to_scalar_if_needed(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr
