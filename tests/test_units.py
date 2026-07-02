from __future__ import annotations

import pytest

from reservoir_backend.core.exceptions import UnitConversionError
from reservoir_backend.core.units import (
    convert,
    fraction_to_decimal,
    permeability_to_m2,
    pressure_to_pa,
    viscosity_to_pa_s,
)


def test_mpa_to_pa() -> None:
    assert pressure_to_pa(1.0, "MPa") == pytest.approx(1.0e6)
    assert convert(1.0, "MPa", "Pa") == pytest.approx(1.0e6)


def test_md_to_m2() -> None:
    assert permeability_to_m2(1.0, "mD") == pytest.approx(9.869233e-16)
    assert convert(1.0, "mD", "m2") == pytest.approx(9.869233e-16)


def test_cp_to_pa_s() -> None:
    assert viscosity_to_pa_s(1.0, "cP") == pytest.approx(1.0e-3)
    assert convert(1.0, "cP", "Pa.s") == pytest.approx(1.0e-3)


def test_fraction_percent_conversion() -> None:
    assert fraction_to_decimal(20.0, "%") == pytest.approx(0.2)
    assert fraction_to_decimal(0.2, "fraction") == pytest.approx(0.2)
    assert convert(20.0, "percent", "fraction") == pytest.approx(0.2)


def test_unknown_unit_raises() -> None:
    with pytest.raises(UnitConversionError):
        pressure_to_pa(1.0, "psi")
    with pytest.raises(UnitConversionError):
        convert(1.0, "mD", "unknown")
