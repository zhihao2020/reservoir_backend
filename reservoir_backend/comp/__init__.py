"""Standalone compositional mass conservation + TPFA.

Per-cell ``flash_tp``, accumulation ``n_i = V_pore (ξ_L S_L x_i + ξ_V S_V y_i)``,
and an explicit closed-domain mole update with two-point phase-potential
upwind. Sits on ``reservoir_backend.eos``. Not the FIM residual: do not
import this package from ``solver/fi.py``, IMPES, DigitalTwin, CLI, or apply.

EXAMPLE fluids only. Not field-validated.
"""

from reservoir_backend.comp.accumulation import CellFlash, component_moles, flash_cell
from reservoir_backend.comp.flux import InteriorFace, interior_faces, phase_molar_flux
from reservoir_backend.comp.step import CompFields, accumulate_system, explicit_step
from reservoir_backend.comp.well import RateInjector, example_rate_injector, peaceman_wi

__all__ = [
    "CellFlash",
    "CompFields",
    "InteriorFace",
    "RateInjector",
    "accumulate_system",
    "component_moles",
    "example_rate_injector",
    "explicit_step",
    "flash_cell",
    "interior_faces",
    "peaceman_wi",
    "phase_molar_flux",
]
