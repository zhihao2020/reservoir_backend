"""Dimensionless similarity criteria for lab-field scale comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor

CRITERIA = (
    "reynolds",
    "capillary",
    "peclet",
    "mobility_ratio",
    "gravity_number",
    "dimensionless_pressure",
    "dimensionless_time",
)


@dataclass(frozen=True)
class CriterionResult:
    """Result for one dimensionless criterion."""

    name: str
    value: float | None
    success: bool
    warning: str | None = None

    def __post_init__(self) -> None:
        if self.value is not None and not np.isfinite(float(self.value)):
            raise InvalidPhysicalValueError(f"{self.name} result must be finite")


def compute_reynolds_number(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `Re = rho * v * L / mu`."""
    value = descriptor.density_kg_m3 * descriptor.velocity_scale_m_s * descriptor.length_scale_m / descriptor.viscosity_pa_s
    return _ok("reynolds", value)


def compute_capillary_number(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `Ca = mu * v / sigma`."""
    if descriptor.interfacial_tension_n_m is None:
        return _missing("capillary", "interfacial_tension_n_m is required for capillary number")
    value = descriptor.viscosity_pa_s * descriptor.velocity_scale_m_s / descriptor.interfacial_tension_n_m
    return _ok("capillary", value)


def compute_peclet_number(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `Pe = v * L / D`."""
    if descriptor.diffusivity_m2_s is None:
        return _missing("peclet", "diffusivity_m2_s is required for Peclet number")
    value = descriptor.velocity_scale_m_s * descriptor.length_scale_m / descriptor.diffusivity_m2_s
    return _ok("peclet", value)


def compute_mobility_ratio(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `M = mobility_displacing / mobility_displaced`."""
    if descriptor.mobility_displacing is None or descriptor.mobility_displaced is None:
        return _missing("mobility_ratio", "mobility_displacing and mobility_displaced are required for mobility ratio")
    value = descriptor.mobility_displacing / descriptor.mobility_displaced
    return _ok("mobility_ratio", value)


def compute_gravity_number(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `Ng = delta_rho * g * k / (mu * v)`."""
    if descriptor.delta_density_kg_m3 is None:
        return _missing("gravity_number", "delta_density_kg_m3 is required for gravity number")
    value = (
        descriptor.delta_density_kg_m3
        * descriptor.gravity_m_s2
        * descriptor.permeability_scale_m2
        / (descriptor.viscosity_pa_s * descriptor.velocity_scale_m_s)
    )
    return _ok("gravity_number", value)


def compute_dimensionless_pressure(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `Pi_p = pressure_drop_pa / pressure_scale_pa`."""
    if descriptor.pressure_drop_pa is None:
        return _missing("dimensionless_pressure", "pressure_drop_pa is required for dimensionless pressure")
    value = descriptor.pressure_drop_pa / descriptor.pressure_scale_pa
    return _ok("dimensionless_pressure", value)


def compute_dimensionless_time(descriptor: ScaleDescriptor) -> CriterionResult:
    """Compute `tD = elapsed_time_s * velocity_scale_m_s / length_scale_m`."""
    if descriptor.elapsed_time_s is None:
        return _missing("dimensionless_time", "elapsed_time_s is required for dimensionless time")
    value = descriptor.elapsed_time_s * descriptor.velocity_scale_m_s / descriptor.length_scale_m
    return _ok("dimensionless_time", value)


def compute_dimensionless_numbers(descriptor: ScaleDescriptor) -> dict[str, Any]:
    """Compute all supported dimensionless numbers and missing-parameter warnings."""
    results = _criterion_results(descriptor)
    warnings = [result.warning for result in results.values() if result.warning]
    missing = [name for name, result in results.items() if not result.success]
    values = {name: result.value for name, result in results.items()}
    finite_values = [value for value in values.values() if value is not None]
    has_nan = any(np.isnan(float(value)) for value in finite_values)
    has_inf = any(np.isinf(float(value)) for value in finite_values)
    return {
        **values,
        "warnings": warnings,
        "missing_criteria": missing,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def compute_criterion_similarity_score(lab_value: float | None, field_value: float | None) -> CriterionResult:
    """Compute one criterion similarity score in `[0, 1]` using log-ratio distance."""
    if lab_value is None or field_value is None:
        return CriterionResult("criterion_similarity", None, False, "lab or field criterion value is missing")
    lab = float(lab_value)
    field = float(field_value)
    if not np.isfinite(lab) or not np.isfinite(field):
        raise InvalidPhysicalValueError("criterion values must be finite")
    if lab <= 0.0 or field <= 0.0:
        return CriterionResult("criterion_similarity", None, False, "criterion values must be positive for log-ratio score")
    score = exp(-abs(log(field / lab)))
    return _ok("criterion_similarity", float(np.clip(score, 0.0, 1.0)))


def compute_overall_similarity_score(
    lab_numbers: dict[str, Any],
    field_numbers: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute weighted overall similarity for all shared valid criteria."""
    criterion_scores: dict[str, float | None] = {}
    missing: list[str] = []
    warnings: list[str] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for name in CRITERIA:
        score = compute_criterion_similarity_score(lab_numbers.get(name), field_numbers.get(name))
        criterion_scores[name] = score.value
        if score.value is None:
            missing.append(name)
            if score.warning:
                warnings.append(f"{name}: {score.warning}")
            continue
        weight = _weight_for(name, weights)
        weighted_sum += weight * score.value
        total_weight += weight

    if total_weight <= 0.0:
        warnings.append("no valid criteria available for overall similarity score")
        overall_score: float | None = None
    else:
        overall_score = float(np.clip(weighted_sum / total_weight, 0.0, 1.0))

    return {
        "overall_score": overall_score,
        "criterion_scores": criterion_scores,
        "missing_criteria": missing,
        "warnings": warnings,
    }


def build_similarity_report(
    lab_descriptor: ScaleDescriptor,
    field_descriptor: ScaleDescriptor,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a pure in-memory lab-field similarity report."""
    lab_numbers = compute_dimensionless_numbers(lab_descriptor)
    field_numbers = compute_dimensionless_numbers(field_descriptor)
    score_report = compute_overall_similarity_score(lab_numbers, field_numbers, weights)
    warnings = [
        *(f"lab: {warning}" for warning in lab_numbers["warnings"]),
        *(f"field: {warning}" for warning in field_numbers["warnings"]),
        *score_report["warnings"],
    ]
    missing = sorted(set(lab_numbers["missing_criteria"]) | set(field_numbers["missing_criteria"]) | set(score_report["missing_criteria"]))
    has_nan = bool(lab_numbers["has_nan"] or field_numbers["has_nan"])
    has_inf = bool(lab_numbers["has_inf"] or field_numbers["has_inf"])
    return {
        "success": not has_nan and not has_inf,
        "dimensionless_numbers_lab": _numbers_only(lab_numbers),
        "dimensionless_numbers_field": _numbers_only(field_numbers),
        "criterion_scores": score_report["criterion_scores"],
        "overall_similarity_score": score_report["overall_score"],
        "missing_criteria": missing,
        "warnings": warnings,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def _criterion_results(descriptor: ScaleDescriptor) -> dict[str, CriterionResult]:
    return {
        "reynolds": compute_reynolds_number(descriptor),
        "capillary": compute_capillary_number(descriptor),
        "peclet": compute_peclet_number(descriptor),
        "mobility_ratio": compute_mobility_ratio(descriptor),
        "gravity_number": compute_gravity_number(descriptor),
        "dimensionless_pressure": compute_dimensionless_pressure(descriptor),
        "dimensionless_time": compute_dimensionless_time(descriptor),
    }


def _ok(name: str, value: float) -> CriterionResult:
    if not np.isfinite(float(value)):
        raise InvalidPhysicalValueError(f"{name} result must be finite")
    return CriterionResult(name=name, value=float(value), success=True)


def _missing(name: str, warning: str) -> CriterionResult:
    return CriterionResult(name=name, value=None, success=False, warning=warning)


def _weight_for(name: str, weights: dict[str, float] | None) -> float:
    if weights is None:
        return 1.0
    value = float(weights.get(name, 1.0))
    if not np.isfinite(value) or value < 0.0:
        raise InvalidPhysicalValueError(f"weight for {name} must be finite and non-negative")
    return value


def _numbers_only(numbers: dict[str, Any]) -> dict[str, float | None]:
    return {name: numbers.get(name) for name in CRITERIA}
