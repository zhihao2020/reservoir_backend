"""Unit conversion helpers for reservoir backend inputs."""

from __future__ import annotations

from reservoir_backend.core.exceptions import UnitConversionError

MD_TO_M2 = 9.869233e-16


def pressure_to_pa(value: float, unit: str) -> float:
    """Convert a pressure value to pascal."""
    normalized = _normalize(unit)
    if normalized == "pa":
        return float(value)
    if normalized == "kpa":
        return float(value) * 1.0e3
    if normalized == "mpa":
        return float(value) * 1.0e6
    if normalized == "bar":
        return float(value) * 1.0e5
    raise UnitConversionError(f"unsupported pressure unit: {unit}")


def permeability_to_m2(value: float, unit: str) -> float:
    """Convert a permeability value to square meters."""
    normalized = _normalize(unit)
    if normalized in {"m2", "m^2"}:
        return float(value)
    if normalized == "md":
        return float(value) * MD_TO_M2
    if normalized == "d":
        return float(value) * 1000.0 * MD_TO_M2
    raise UnitConversionError(f"unsupported permeability unit: {unit}")


def viscosity_to_pa_s(value: float, unit: str) -> float:
    """Convert a dynamic viscosity value to pascal-second."""
    normalized = _normalize(unit)
    if normalized in {"pa.s", "pa*s", "pas"}:
        return float(value)
    if normalized == "cp":
        return float(value) * 1.0e-3
    raise UnitConversionError(f"unsupported viscosity unit: {unit}")


def fraction_to_decimal(value: float, unit: str = "fraction") -> float:
    """Convert a fraction or percent value to decimal fraction."""
    normalized = _normalize(unit)
    if normalized in {"fraction", "decimal", "frac"}:
        return float(value)
    if normalized in {"%", "percent"}:
        return float(value) / 100.0
    raise UnitConversionError(f"unsupported fraction unit: {unit}")


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert common project units into SI base units.

    `to_unit` must be one of `Pa`, `m2`, `Pa.s`, or `fraction`.
    """
    target = _normalize(to_unit)
    if target == "pa":
        return pressure_to_pa(value, from_unit)
    if target in {"m2", "m^2"}:
        return permeability_to_m2(value, from_unit)
    if target in {"pa.s", "pa*s", "pas"}:
        return viscosity_to_pa_s(value, from_unit)
    if target in {"fraction", "decimal", "frac"}:
        return fraction_to_decimal(value, from_unit)
    raise UnitConversionError(f"unsupported target unit: {to_unit}")


def _normalize(unit: str) -> str:
    return unit.strip().lower().replace(" ", "")
