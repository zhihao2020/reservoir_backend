"""Standalone compositional mass conservation + TPFA.

Per-cell ``flash_tp``, accumulation ``n_i = V_pore (ξ_L S_L x_i + ξ_V S_V y_i)``,
and an explicit closed-domain mole update with two-point phase-potential
upwind. Sits on ``reservoir_backend.eos``. Not the FIM residual: do not
import this package from ``solver/fi.py``, IMPES, DigitalTwin, CLI, or apply.

EXAMPLE fluids only. Not field-validated.
"""

from reservoir_backend.comp.accumulation import CellFlash, component_moles, flash_cell
from reservoir_backend.comp.flux import InteriorFace, interior_faces, phase_molar_flux
from reservoir_backend.comp.cycle import (
    CycleLedger,
    CycleRecord,
    MultiCycleLedger,
    injector_well_cell_z_co2,
    perforated_z_co2,
    produced_stream_z_co2,
    run_horizontal_huff_and_puff,
    run_horizontal_huff_and_puff_implicit,
    run_huff_and_puff,
    run_huff_and_puff_cycles,
    run_huff_and_puff_implicit,
    run_inject_soak_produce,
)
from reservoir_backend.comp.streak import (
    K_MATRIX_M2,
    K_STREAK_M2,
    added_moles_per_pv,
    example_drive_pressure,
    example_two_region_k,
)
from reservoir_backend.comp.implicit import ImplicitPeriodLedger, ImplicitStepReport, implicit_newton_step, run_implicit_period
from reservoir_backend.comp.step import CompFields, StepReport, WellLedger, accumulate_system, explicit_step, run_steps
from reservoir_backend.comp.well import (
    RateInjector,
    RateProducer,
    example_co2_rich_stream,
    example_horizontal_well,
    example_huff_n_puff_well,
    example_producer,
    example_rate_injector,
    peaceman_wi,
    well_cell_molar_z,
)

__all__ = [
    "CellFlash",
    "CompFields",
    "CycleLedger",
    "CycleRecord",
    "ImplicitPeriodLedger",
    "ImplicitStepReport",
    "InteriorFace",
    "K_MATRIX_M2",
    "K_STREAK_M2",
    "MultiCycleLedger",
    "RateInjector",
    "RateProducer",
    "StepReport",
    "WellLedger",
    "accumulate_system",
    "added_moles_per_pv",
    "component_moles",
    "example_co2_rich_stream",
    "example_drive_pressure",
    "example_horizontal_well",
    "example_huff_n_puff_well",
    "example_two_region_k",
    "example_producer",
    "example_rate_injector",
    "explicit_step",
    "flash_cell",
    "implicit_newton_step",
    "injector_well_cell_z_co2",
    "interior_faces",
    "peaceman_wi",
    "perforated_z_co2",
    "phase_molar_flux",
    "produced_stream_z_co2",
    "run_horizontal_huff_and_puff",
    "run_horizontal_huff_and_puff_implicit",
    "run_huff_and_puff",
    "run_huff_and_puff_cycles",
    "run_huff_and_puff_implicit",
    "run_implicit_period",
    "run_inject_soak_produce",
    "run_steps",
    "well_cell_molar_z",
]
