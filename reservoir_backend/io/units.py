"""Convert engineering units to SI at the IO boundary."""

from __future__ import annotations

from reservoir_backend.exceptions import UnitConversionError

MD_TO_M2 = 9.869233e-16
PSI_TO_PA = 6894.757293168
ML_PER_MIN_TO_M3_S = 1.0e-6 / 60.0


def _norm(unit: str) -> str:
    return unit.strip().lower().replace(" ", "").replace("³", "3").replace("·", ".")


def to_metres(value: float, unit: str) -> float:
    u = _norm(unit)
    if u in {"m", "meter", "metre"}:
        return float(value)
    if u == "mm":
        return float(value) * 1.0e-3
    if u == "cm":
        return float(value) * 1.0e-2
    if u in {"ft", "feet"}:
        return float(value) * 0.3048
    raise UnitConversionError(f"unsupported length unit: {unit}")


def to_seconds(value: float, unit: str) -> float:
    u = _norm(unit)
    if u in {"s", "sec", "second", "seconds"}:
        return float(value)
    if u in {"min", "minute", "minutes"}:
        return float(value) * 60.0
    if u in {"h", "hr", "hour", "hours"}:
        return float(value) * 3600.0
    if u in {"d", "day", "days"}:
        return float(value) * 86400.0
    raise UnitConversionError(f"unsupported time unit: {unit}")


def to_pa(value: float, unit: str) -> float:
    u = _norm(unit)
    if u == "pa":
        return float(value)
    if u == "kpa":
        return float(value) * 1.0e3
    if u == "mpa":
        return float(value) * 1.0e6
    if u == "bar":
        return float(value) * 1.0e5
    if u == "psi":
        return float(value) * PSI_TO_PA
    raise UnitConversionError(f"unsupported pressure unit: {unit}")


def to_m2(value: float, unit: str) -> float:
    u = _norm(unit)
    if u in {"m2", "m^2"}:
        return float(value)
    if u == "md":
        return float(value) * MD_TO_M2
    if u in {"d", "darcy"}:
        return float(value) * 1000.0 * MD_TO_M2
    raise UnitConversionError(f"unsupported permeability unit: {unit}")


def to_pa_s(value: float, unit: str) -> float:
    u = _norm(unit)
    if u in {"pa.s", "pa*s", "pas"}:
        return float(value)
    if u == "cp":
        return float(value) * 1.0e-3
    raise UnitConversionError(f"unsupported viscosity unit: {unit}")


def to_m3_s(value: float, unit: str) -> float:
    u = _norm(unit)
    if u in {"m3/s", "m^3/s"}:
        return float(value)
    if u in {"ml/min", "mlmin", "cm3/min"}:
        return float(value) * ML_PER_MIN_TO_M3_S
    if u in {"m3/d", "m^3/d", "m3/day"}:
        return float(value) / 86400.0
    raise UnitConversionError(f"unsupported rate unit: {unit}")
