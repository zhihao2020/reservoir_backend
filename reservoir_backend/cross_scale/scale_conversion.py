from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor


SCALE_FIELDS = (
    ("length_scale", "length_scale_m"),
    ("time_scale", "time_scale_s"),
    ("pressure_scale", "pressure_scale_pa"),
    ("permeability_scale", "permeability_scale_m2"),
    ("velocity_scale", "velocity_scale_m_s"),
    ("flow_rate_scale", "flow_rate_m3_s"),
    ("porosity", "porosity"),
)


def build_scale_conversion_report(lab_descriptor: ScaleDescriptor, field_descriptor: ScaleDescriptor) -> dict[str, Any]:
    """Build a lab-to-field scale conversion report.

    Ratios are field value divided by lab value. The report is diagnostic only;
    it does not imply deterministic equivalence between lab and field systems.
    """
    report: dict[str, Any] = {
        "success": True,
        "warnings": [],
        "has_nan": False,
        "has_inf": False,
        "limitations": [
            "Scale ratios are diagnostic, not deterministic equivalence rules.",
            "No complex upscaling solver is implemented.",
        ],
    }
    for prefix, attr in SCALE_FIELDS:
        lab_value = float(getattr(lab_descriptor, attr))
        field_value = float(getattr(field_descriptor, attr))
        ratio = _ratio(prefix, lab_value, field_value)
        report[f"{prefix}_lab"] = lab_value
        report[f"{prefix}_field"] = field_value
        report[f"{prefix}_ratio"] = ratio

    numbers = [value for key, value in report.items() if isinstance(value, (int, float))]
    report["has_nan"] = any(np.isnan(float(value)) for value in numbers)
    report["has_inf"] = any(np.isinf(float(value)) for value in numbers)
    report["success"] = not report["has_nan"] and not report["has_inf"]
    return report


def descriptors_from_config(config: dict[str, Any]) -> tuple[ScaleDescriptor, ScaleDescriptor]:
    lab = _descriptor_section(config, "lab_case")
    field = _descriptor_section(config, "field_case")
    return lab, field


def descriptor_to_json(descriptor: ScaleDescriptor) -> dict[str, Any]:
    return asdict(descriptor)


def _descriptor_section(config: dict[str, Any], key: str) -> ScaleDescriptor:
    section = config.get(key)
    if not isinstance(section, dict):
        raise InvalidPhysicalValueError(f"{key} must be a mapping")
    data = section.get("descriptor", section)
    if not isinstance(data, dict):
        raise InvalidPhysicalValueError(f"{key}.descriptor must be a mapping")
    return ScaleDescriptor.from_dict(data)


def _ratio(name: str, lab_value: float, field_value: float) -> float:
    if not np.isfinite(lab_value) or lab_value <= 0.0:
        raise InvalidPhysicalValueError(f"{name} lab value must be positive and finite")
    if not np.isfinite(field_value):
        raise InvalidPhysicalValueError(f"{name} field value must be finite")
    value = field_value / lab_value
    if not np.isfinite(value):
        raise InvalidPhysicalValueError(f"{name} ratio must be finite")
    return float(value)
