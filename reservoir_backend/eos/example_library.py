"""EXAMPLE eight-component PR library (public literature, not a field card).

Numbers live in ``eos/fluids/example_c1_c7plus_co2.yaml``. This module
is a thin adapter so existing flash/comp tests keep calling
``example_eight_component_mixture()``.

STATUS: standalone example flash, not wired, example PR params.
Not site-calibrated. Not field-validated.
"""

from __future__ import annotations

import numpy as np

from reservoir_backend.eos.load import DEFAULT_EXAMPLE_FLUID_YAML, load_eos_mixture_yaml, load_feed_z_yaml
from reservoir_backend.eos.peng_robinson import EosMixture

# Tests assert this token so the table cannot be mistaken for a GEM card.
# Must match ``marker`` in eos/fluids/example_c1_c7plus_co2.yaml.
EXAMPLE_LIBRARY_MARKER = (
    "EXAMPLE / public literature — standalone example flash, not wired, "
    "example PR params; NOT a Jiyang GEM card; NOT site-calibrated"
)


def example_eight_component_mixture() -> EosMixture:
    """Return the EXAMPLE 8-component library loaded from YAML. Not GEM."""
    return load_eos_mixture_yaml(DEFAULT_EXAMPLE_FLUID_YAML)


def example_feed_z() -> np.ndarray:
    """EXAMPLE 8-component feed mole fractions (not a Jiyang / GEM composition)."""
    return load_feed_z_yaml(DEFAULT_EXAMPLE_FLUID_YAML)
