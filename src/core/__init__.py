"""Shared core: Cartesian grid, SI unit conversion, and the lab-case contract."""

from .cartesian import CartesianGrid
from .lab_case import LabCase, load_lab_case, summarize_lab_case
from .units import MD_TO_M2, to_m2, to_m3_s, to_metres, to_pa, to_pa_s, to_seconds

__all__ = [
    "CartesianGrid",
    "LabCase",
    "MD_TO_M2",
    "load_lab_case",
    "summarize_lab_case",
    "to_m2",
    "to_m3_s",
    "to_metres",
    "to_pa",
    "to_pa_s",
    "to_seconds",
]
