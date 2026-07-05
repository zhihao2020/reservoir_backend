"""Scale-effect analysis for lab-field cross-scale comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.similarity import CRITERIA, compute_dimensionless_numbers

DEFAULT_THRESHOLDS: dict[str, float] = {
    "capillary_dominated_ca_threshold": 1.0e-5,
    "viscous_dominated_ca_threshold": 1.0e-3,
    "gravity_dominated_ng_threshold": 1.0,
    "convection_dominated_pe_threshold": 10.0,
    "diffusion_dominated_pe_threshold": 1.0,
    "inertial_re_threshold": 1.0,
    "creeping_re_threshold": 0.01,
}

SCALE_RATIO_KEYS = (
    "scale_ratio_length",
    "scale_ratio_time",
    "scale_ratio_pressure",
    "scale_ratio_permeability",
    "scale_ratio_velocity",
    "scale_ratio_flow_rate",
    "scale_ratio_porosity",
    "scale_ratio_temperature",
)


def compute_scale_ratios(lab_descriptor: ScaleDescriptor, field_descriptor: ScaleDescriptor) -> dict[str, Any]:
    """Compute field-to-lab scale ratios with warnings for missing optional fields."""
    warnings: list[str] = []
    ratios = {
        "scale_ratio_length": _ratio("length_scale_m", lab_descriptor.length_scale_m, field_descriptor.length_scale_m),
        "scale_ratio_time": _ratio("time_scale_s", lab_descriptor.time_scale_s, field_descriptor.time_scale_s),
        "scale_ratio_pressure": _ratio(
            "pressure_scale_pa", lab_descriptor.pressure_scale_pa, field_descriptor.pressure_scale_pa
        ),
        "scale_ratio_permeability": _ratio(
            "permeability_scale_m2", lab_descriptor.permeability_scale_m2, field_descriptor.permeability_scale_m2
        ),
        "scale_ratio_velocity": _ratio(
            "velocity_scale_m_s", lab_descriptor.velocity_scale_m_s, field_descriptor.velocity_scale_m_s
        ),
        "scale_ratio_flow_rate": _ratio(
            "flow_rate_m3_s", lab_descriptor.flow_rate_m3_s, field_descriptor.flow_rate_m3_s
        ),
        "scale_ratio_porosity": _ratio("porosity", lab_descriptor.porosity, field_descriptor.porosity),
        "scale_ratio_temperature": _optional_ratio(
            "temperature_scale_k", lab_descriptor.temperature_scale_k, field_descriptor.temperature_scale_k, warnings
        ),
    }
    finite_values = [value for value in ratios.values() if value is not None]
    has_nan = any(np.isnan(float(value)) for value in finite_values)
    has_inf = any(np.isinf(float(value)) for value in finite_values)
    if has_nan or has_inf:
        raise InvalidPhysicalValueError("scale ratios must be finite")
    return {
        **ratios,
        "warnings": warnings,
        "missing_ratios": [key for key, value in ratios.items() if value is None],
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def classify_flow_regime(dimensionless_numbers: dict[str, Any], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Classify flow regime from engineering threshold heuristics."""
    t = _thresholds(thresholds)
    warnings: list[str] = []

    re = _optional_number(dimensionless_numbers, "reynolds", warnings)
    ca = _optional_number(dimensionless_numbers, "capillary", warnings)
    pe = _optional_number(dimensionless_numbers, "peclet", warnings)
    ng = _optional_number(dimensionless_numbers, "gravity_number", warnings)

    inertia_role = "inertia_uncertain"
    if re is not None:
        if re < t["creeping_re_threshold"]:
            inertia_role = "creeping_flow"
        elif re > t["inertial_re_threshold"]:
            inertia_role = "inertial_effect_possible"
        else:
            inertia_role = "laminar_viscous_flow"

    capillary_role = "capillary_uncertain"
    if ca is not None:
        if ca < t["capillary_dominated_ca_threshold"]:
            capillary_role = "capillary_dominated"
        elif ca > t["viscous_dominated_ca_threshold"]:
            capillary_role = "viscous_dominated"
        else:
            capillary_role = "capillary_viscous_transition"

    gravity_role = "gravity_uncertain"
    if ng is not None:
        if abs(ng) > t["gravity_dominated_ng_threshold"]:
            gravity_role = "gravity_dominated"
        else:
            gravity_role = "gravity_minor"

    transport_role = "transport_uncertain"
    if pe is not None:
        if pe > t["convection_dominated_pe_threshold"]:
            transport_role = "convection_dominated"
        elif pe < t["diffusion_dominated_pe_threshold"]:
            transport_role = "diffusion_dominated"
        else:
            transport_role = "convection_diffusion_transition"

    if gravity_role == "gravity_dominated":
        dominant_force = "gravity"
    elif capillary_role == "capillary_dominated":
        dominant_force = "capillary"
    elif capillary_role == "viscous_dominated":
        dominant_force = "viscous"
    else:
        dominant_force = "mixed_or_uncertain"

    flow_regime = "_".join([dominant_force, transport_role, inertia_role])
    return {
        "flow_regime": flow_regime,
        "dominant_force": dominant_force,
        "capillary_role": capillary_role,
        "gravity_role": gravity_role,
        "transport_role": transport_role,
        "inertia_role": inertia_role,
        "warnings": warnings,
    }


def detect_regime_shift(lab_regime: dict[str, Any], field_regime: dict[str, Any]) -> dict[str, Any]:
    """Detect whether key regime labels changed from lab to field scale."""
    comparisons = {
        "dominant_force_changed": ("dominant_force", "dominant force"),
        "transport_role_changed": ("transport_role", "transport role"),
        "capillary_role_changed": ("capillary_role", "capillary role"),
        "gravity_role_changed": ("gravity_role", "gravity role"),
        "inertia_role_changed": ("inertia_role", "inertia role"),
    }
    result: dict[str, Any] = {}
    summary: list[str] = []
    warnings: list[str] = []
    for output_key, (regime_key, label) in comparisons.items():
        lab_value = lab_regime.get(regime_key)
        field_value = field_regime.get(regime_key)
        if lab_value is None or field_value is None:
            changed = False
            warnings.append(f"{label} is missing from one regime")
        else:
            changed = lab_value != field_value
            if changed:
                summary.append(f"{label} changed from {lab_value} to {field_value}")
        result[output_key] = changed

    result["regime_shift_detected"] = any(bool(result[key]) for key in comparisons)
    result["shift_summary"] = summary
    result["warnings"] = warnings
    return result


def build_scale_effect_report(
    lab_descriptor: ScaleDescriptor,
    field_descriptor: ScaleDescriptor,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a pure in-memory scale-effect analysis report."""
    lab_numbers = compute_dimensionless_numbers(lab_descriptor)
    field_numbers = compute_dimensionless_numbers(field_descriptor)
    scale_ratios = compute_scale_ratios(lab_descriptor, field_descriptor)
    lab_regime = classify_flow_regime(lab_numbers, thresholds)
    field_regime = classify_flow_regime(field_numbers, thresholds)
    regime_shift = detect_regime_shift(lab_regime, field_regime)

    warnings = [
        *(f"lab: {warning}" for warning in lab_numbers["warnings"]),
        *(f"field: {warning}" for warning in field_numbers["warnings"]),
        *(f"scale ratio: {warning}" for warning in scale_ratios["warnings"]),
        *(f"lab regime: {warning}" for warning in lab_regime["warnings"]),
        *(f"field regime: {warning}" for warning in field_regime["warnings"]),
        *(f"regime shift: {warning}" for warning in regime_shift["warnings"]),
    ]
    has_nan = bool(lab_numbers["has_nan"] or field_numbers["has_nan"] or scale_ratios["has_nan"])
    has_inf = bool(lab_numbers["has_inf"] or field_numbers["has_inf"] or scale_ratios["has_inf"])
    return {
        "success": not has_nan and not has_inf,
        "scale_ratios": _scale_ratios_only(scale_ratios),
        "dimensionless_numbers_lab": _numbers_only(lab_numbers),
        "dimensionless_numbers_field": _numbers_only(field_numbers),
        "regime_lab": lab_regime,
        "regime_field": field_regime,
        "regime_shift": regime_shift,
        "dominant_force_lab": lab_regime["dominant_force"],
        "dominant_force_field": field_regime["dominant_force"],
        "regime_shift_detected": regime_shift["regime_shift_detected"],
        "warnings": warnings,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _ratio(name: str, lab_value: float, field_value: float) -> float:
    denominator = float(lab_value)
    numerator = float(field_value)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise InvalidPhysicalValueError(f"{name} lab value must be positive for scale ratio")
    if not np.isfinite(numerator):
        raise InvalidPhysicalValueError(f"{name} field value must be finite for scale ratio")
    value = numerator / denominator
    if not np.isfinite(value):
        raise InvalidPhysicalValueError(f"{name} scale ratio must be finite")
    return float(value)


def _optional_ratio(name: str, lab_value: float | None, field_value: float | None, warnings: list[str]) -> float | None:
    if lab_value is None or field_value is None:
        warnings.append(f"{name} is missing; scale ratio is not computed")
        return None
    return _ratio(name, lab_value, field_value)


def _thresholds(thresholds: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        merged.update(thresholds)
    for name, value in merged.items():
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise InvalidPhysicalValueError(f"threshold {name} must be a positive finite value")
        merged[name] = number
    if merged["capillary_dominated_ca_threshold"] >= merged["viscous_dominated_ca_threshold"]:
        raise InvalidPhysicalValueError("capillary dominated threshold must be less than viscous dominated threshold")
    if merged["diffusion_dominated_pe_threshold"] >= merged["convection_dominated_pe_threshold"]:
        raise InvalidPhysicalValueError("diffusion dominated threshold must be less than convection dominated threshold")
    if merged["creeping_re_threshold"] >= merged["inertial_re_threshold"]:
        raise InvalidPhysicalValueError("creeping Re threshold must be less than inertial Re threshold")
    return merged


def _optional_number(numbers: dict[str, Any], key: str, warnings: list[str]) -> float | None:
    value = numbers.get(key)
    if value is None:
        warnings.append(f"{key} is missing; regime classification is uncertain")
        return None
    number = float(value)
    if not np.isfinite(number):
        raise InvalidPhysicalValueError(f"{key} must be finite for regime classification")
    return number


def _scale_ratios_only(scale_ratios: dict[str, Any]) -> dict[str, float | None]:
    return {name: scale_ratios[name] for name in SCALE_RATIO_KEYS}


def _numbers_only(numbers: dict[str, Any]) -> dict[str, float | None]:
    return {name: numbers.get(name) for name in CRITERIA}
