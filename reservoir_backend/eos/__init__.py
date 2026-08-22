"""Standalone Peng–Robinson EOS and isothermal two-phase flash.

Compositional kernel (P3). Not wired into FIM, IMPES, DigitalTwin, CLI,
or apply. Example PR parameters only — not field-validated.
"""

from reservoir_backend.eos.example_library import (
    EXAMPLE_LIBRARY_MARKER,
    example_eight_component_mixture,
    example_feed_z,
)
from reservoir_backend.eos.load import (
    DEFAULT_EXAMPLE_FLUID_YAML,
    load_eos_mixture_yaml,
    load_feed_z_yaml,
    resolve_fluid_yaml,
)
from reservoir_backend.eos.flash import FlashResult, flash_tp, solve_rachford_rice, wilson_k
from reservoir_backend.eos.inventory import PhaseInventory, component_inventory
from reservoir_backend.eos.peng_robinson import (
    GAS_CONSTANT,
    EosMixture,
    compressibility_factor,
    compressibility_roots,
    fugacity_coefficients,
    mass_density,
    molar_volume,
    peng_robinson_ab,
    select_z,
)
from reservoir_backend.eos.stability import StabilityResult, michelsen_stability, tangent_plane_distance

__all__ = [
    "DEFAULT_EXAMPLE_FLUID_YAML",
    "EXAMPLE_LIBRARY_MARKER",
    "GAS_CONSTANT",
    "EosMixture",
    "FlashResult",
    "PhaseInventory",
    "StabilityResult",
    "component_inventory",
    "compressibility_factor",
    "compressibility_roots",
    "example_eight_component_mixture",
    "example_feed_z",
    "load_eos_mixture_yaml",
    "load_feed_z_yaml",
    "resolve_fluid_yaml",
    "flash_tp",
    "fugacity_coefficients",
    "mass_density",
    "michelsen_stability",
    "molar_volume",
    "peng_robinson_ab",
    "select_z",
    "solve_rachford_rice",
    "tangent_plane_distance",
    "wilson_k",
]
