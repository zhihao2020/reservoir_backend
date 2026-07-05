"""Scale descriptor data model for cross-scale similarity criteria."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError


@dataclass(frozen=True)
class ScaleDescriptor:
    """Physical scales required for dimensionless-number comparisons."""

    length_scale_m: float
    time_scale_s: float
    pressure_scale_pa: float
    permeability_scale_m2: float
    porosity: float
    viscosity_pa_s: float
    density_kg_m3: float
    velocity_scale_m_s: float
    flow_rate_m3_s: float
    temperature_scale_k: float | None = None
    interfacial_tension_n_m: float | None = None
    diffusivity_m2_s: float | None = None
    delta_density_kg_m3: float | None = None
    gravity_m_s2: float = 9.80665
    pressure_drop_pa: float | None = None
    elapsed_time_s: float | None = None
    mobility_displacing: float | None = None
    mobility_displaced: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaleDescriptor":
        """Build a descriptor from a mapping, rejecting missing required fields."""
        required = [
            "length_scale_m",
            "time_scale_s",
            "pressure_scale_pa",
            "permeability_scale_m2",
            "porosity",
            "viscosity_pa_s",
            "density_kg_m3",
            "velocity_scale_m_s",
            "flow_rate_m3_s",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise InvalidPhysicalValueError(f"missing required scale descriptor fields: {', '.join(missing)}")
        allowed = set(required) | {
            "temperature_scale_k",
            "interfacial_tension_n_m",
            "diffusivity_m2_s",
            "delta_density_kg_m3",
            "gravity_m_s2",
            "pressure_drop_pa",
            "elapsed_time_s",
            "mobility_displacing",
            "mobility_displaced",
        }
        values = {name: data[name] for name in allowed if name in data}
        return cls(**values)

    def to_dict(self) -> dict[str, float | None]:
        """Return a JSON-serializable descriptor dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate required and optional scale fields."""
        _require_positive("length_scale_m", self.length_scale_m)
        _require_positive("time_scale_s", self.time_scale_s)
        _require_positive("pressure_scale_pa", self.pressure_scale_pa)
        _require_positive("permeability_scale_m2", self.permeability_scale_m2)
        _require_interval("porosity", self.porosity, lower=0.0, upper=1.0, lower_open=True, upper_open=False)
        _require_positive("viscosity_pa_s", self.viscosity_pa_s)
        _require_positive("density_kg_m3", self.density_kg_m3)
        _require_positive("velocity_scale_m_s", self.velocity_scale_m_s)
        _require_nonnegative("flow_rate_m3_s", self.flow_rate_m3_s)
        _require_positive("gravity_m_s2", self.gravity_m_s2)

        _require_optional_positive("temperature_scale_k", self.temperature_scale_k)
        _require_optional_positive("interfacial_tension_n_m", self.interfacial_tension_n_m)
        _require_optional_positive("diffusivity_m2_s", self.diffusivity_m2_s)
        _require_optional_finite("delta_density_kg_m3", self.delta_density_kg_m3)
        _require_optional_nonnegative("pressure_drop_pa", self.pressure_drop_pa)
        _require_optional_nonnegative("elapsed_time_s", self.elapsed_time_s)
        _require_optional_positive("mobility_displacing", self.mobility_displacing)
        _require_optional_positive("mobility_displaced", self.mobility_displaced)


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise InvalidPhysicalValueError(f"{name} must be a positive finite value")


def _require_nonnegative(name: str, value: float) -> None:
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise InvalidPhysicalValueError(f"{name} must be a non-negative finite value")


def _require_interval(name: str, value: float, *, lower: float, upper: float, lower_open: bool, upper_open: bool) -> None:
    number = float(value)
    if not np.isfinite(number):
        raise InvalidPhysicalValueError(f"{name} must be finite")
    lower_ok = number > lower if lower_open else number >= lower
    upper_ok = number < upper if upper_open else number <= upper
    if not lower_ok or not upper_ok:
        lower_bracket = "(" if lower_open else "["
        upper_bracket = ")" if upper_open else "]"
        raise InvalidPhysicalValueError(f"{name} must be in {lower_bracket}{lower}, {upper}{upper_bracket}")


def _require_optional_positive(name: str, value: float | None) -> None:
    if value is not None:
        _require_positive(name, value)


def _require_optional_nonnegative(name: str, value: float | None) -> None:
    if value is not None:
        _require_nonnegative(name, value)


def _require_optional_finite(name: str, value: float | None) -> None:
    if value is not None and not np.isfinite(float(value)):
        raise InvalidPhysicalValueError(f"{name} must be finite if provided")
