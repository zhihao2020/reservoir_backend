"""Standard experimental data schema for reservoir backend inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """Schema rule for one standard experimental-data field."""

    name: str
    aliases: tuple[str, ...]
    canonical_unit: str
    required: bool = False
    lower: float | None = None
    upper: float | None = None
    strict_lower: bool = False
    description: str = ""


@dataclass
class ExperimentalField:
    """One numeric field with unit and metadata."""

    name: str
    values: Any
    unit: str
    source_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = normalize_field_name(self.name)
        self.values = np.asarray(self.values, dtype=float).copy()
        self.unit = str(self.unit)
        self.source_name = str(self.source_name)
        self.metadata = dict(self.metadata)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)


@dataclass
class ExperimentalDataset:
    """Unified internal representation returned by all data readers."""

    fields: dict[str, ExperimentalField]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_name: str = ""
    input_file: str = ""
    input_format: str = ""

    def __post_init__(self) -> None:
        normalized: dict[str, ExperimentalField] = {}
        for name, item in self.fields.items():
            field_name = normalize_field_name(name)
            if isinstance(item, ExperimentalField):
                data_field = item
            else:
                raise TypeError("fields must contain ExperimentalField values")
            data_field.name = field_name
            if not data_field.source_name:
                data_field.source_name = self.source_name
            normalized[field_name] = data_field
        self.fields = normalized
        self.metadata = dict(self.metadata)
        self.source_name = str(self.source_name)
        self.input_file = str(self.input_file)
        self.input_format = str(self.input_format)

    @property
    def field_names(self) -> list[str]:
        return sorted(self.fields)

    @property
    def shape(self) -> tuple[int, ...] | None:
        if not self.fields:
            return None
        return next(iter(self.fields.values())).shape

    def get(self, name: str) -> ExperimentalField:
        return self.fields[normalize_field_name(name)]

    def with_fields(self, fields: dict[str, ExperimentalField]) -> "ExperimentalDataset":
        return ExperimentalDataset(
            fields=fields,
            metadata=self.metadata.copy(),
            source_name=self.source_name,
            input_file=self.input_file,
            input_format=self.input_format,
        )


STANDARD_FIELD_SPECS: dict[str, FieldSpec] = {
    "resistivity": FieldSpec("resistivity", ("rt", "res", "resistivity_ohm_m"), "ohm_m", lower=0.0, strict_lower=True),
    "electromagnetic_response": FieldSpec("electromagnetic_response", ("em", "em_response", "electromagnetic"), "dimensionless"),
    "acoustic_response": FieldSpec("acoustic_response", ("acoustic", "acoustic_velocity", "vp"), "m_s"),
    "pressure": FieldSpec("pressure", ("p", "pressure_pa", "pressure_mpa"), "Pa"),
    "saturation": FieldSpec("saturation", ("sw", "water_saturation", "saturation_fraction"), "fraction", lower=0.0, upper=1.0),
    "porosity": FieldSpec("porosity", ("phi", "poro", "porosity_fraction"), "fraction", lower=0.0, upper=1.0),
    "permeability": FieldSpec("permeability", ("perm", "k", "kx", "permeability_md"), "m2", lower=0.0, strict_lower=True),
    "temperature": FieldSpec("temperature", ("temp", "temperature_c", "temperature_k"), "K"),
    "time": FieldSpec("time", ("t", "time_s", "time_day"), "s"),
    "x": FieldSpec("x", ("x_m", "i", "coord_x"), "m"),
    "y": FieldSpec("y", ("y_m", "j", "coord_y"), "m"),
    "z": FieldSpec("z", ("z_m", "k", "coord_z"), "m"),
    "confidence": FieldSpec("confidence", ("conf", "quality"), "fraction", lower=0.0, upper=1.0),
    "variance": FieldSpec("variance", ("var", "uncertainty_variance"), "variance", lower=0.0),
}


UNIT_ALIASES = {
    "pa": "Pa",
    "kpa": "kPa",
    "mpa": "MPa",
    "bar": "bar",
    "m2": "m2",
    "m^2": "m2",
    "md": "mD",
    "d": "D",
    "fraction": "fraction",
    "frac": "fraction",
    "decimal": "fraction",
    "%": "percent",
    "percent": "percent",
    "ohm_m": "ohm_m",
    "ohm-m": "ohm_m",
    "ohmm": "ohm_m",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "h": "h",
    "hr": "h",
    "day": "day",
    "days": "day",
    "m": "m",
    "cm": "cm",
    "mm": "mm",
    "k": "K",
    "c": "C",
    "degc": "C",
    "m_s": "m_s",
    "m/s": "m_s",
    "dimensionless": "dimensionless",
    "variance": "variance",
}


def normalize_field_name(name: str) -> str:
    """Normalize a source field name into a known schema field if possible."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    for standard, spec in STANDARD_FIELD_SPECS.items():
        aliases = {standard, *spec.aliases}
        if key in aliases:
            return standard
    # Drop a unit suffix and try again, e.g. pressure_mpa -> pressure.
    parts = key.split("_")
    if len(parts) > 1:
        prefix = "_".join(parts[:-1])
        for standard, spec in STANDARD_FIELD_SPECS.items():
            if prefix in {standard, *spec.aliases}:
                return standard
    return key


def infer_unit_from_name(name: str, default: str | None = None) -> str:
    """Infer unit from a column/array name suffix."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    suffix = key.split("_")[-1]
    if suffix in UNIT_ALIASES:
        return UNIT_ALIASES[suffix]
    normalized = normalize_field_name(key)
    if normalized in STANDARD_FIELD_SPECS:
        if key != normalized:
            return "unknown" if default is None else default
        return STANDARD_FIELD_SPECS[normalized].canonical_unit if default is None else default
    return "unknown" if default is None else default


def canonical_unit(field_name: str) -> str:
    """Return canonical unit for a standard field."""
    normalized = normalize_field_name(field_name)
    if normalized not in STANDARD_FIELD_SPECS:
        return "unknown"
    return STANDARD_FIELD_SPECS[normalized].canonical_unit


def field_spec(field_name: str) -> FieldSpec | None:
    """Return schema rule for a standard field, if known."""
    return STANDARD_FIELD_SPECS.get(normalize_field_name(field_name))


def dataset_from_arrays(
    arrays: dict[str, Any],
    *,
    units: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
    source_name: str = "",
    input_file: str | Path = "",
    input_format: str = "",
) -> ExperimentalDataset:
    """Build a dataset from arrays and unit mapping."""
    units = {} if units is None else dict(units)
    fields = {}
    for raw_name, values in arrays.items():
        name = normalize_field_name(raw_name)
        unit = units.get(raw_name, units.get(name, infer_unit_from_name(raw_name)))
        fields[name] = ExperimentalField(name=name, values=values, unit=unit, source_name=source_name)
    return ExperimentalDataset(
        fields=fields,
        metadata={} if metadata is None else dict(metadata),
        source_name=source_name,
        input_file=str(input_file),
        input_format=input_format,
    )
