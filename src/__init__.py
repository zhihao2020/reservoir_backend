"""3D shale-oil inversion: mesh, pressure, saturation, porosity/permeability.

This package is the compiled payload. GitHub Actions builds ``src.so`` on Linux
and the thin repo-root loader imports it as ``src`` / ``reservoir_backend.src``.
"""

from .core.cartesian import CartesianGrid
from .core.lab_case import LabCase, load_lab_case, summarize_lab_case
from .version import __version__
from .programs.forward import black_oil_forward, validate_inversion
from .programs.mesh import MeshResult, WellMap, build_mesh
from .programs.pipeline import ProgramFields, field_scale_view, run_pipeline, write_output
from .programs.session import InversionSession
from .programs.pressure import interpolate_pressure
from .programs.protocol import PROTOCOL_VERSION, pack_step, unpack_step
from .programs.results import summarize_results
from .programs.rock import (
    FluidParams,
    RockDiagnostics,
    WellModelParams,
    invert_rock,
    invert_rock_coarse_to_fine,
    invert_rock_three_phase,
    peaceman_wi,
    transient_weights,
    well_cell_rates_weighted,
)
from .programs.saturation import interpolate_saturation
from .programs.similarity import (
    MODEL_HEIGHT_M,
    MODEL_LENGTH_M,
    MODEL_VOLUME_M3,
    MODEL_VOLUME_ML,
    MODEL_WIDTH_M,
    VOLUME_BASES,
    complete_injection,
    field_to_lab,
    field_to_lab_coords,
    field_to_lab_point,
    field_to_lab_rate,
    field_to_lab_time,
    field_to_lab_volume,
    lab_to_field,
    lab_to_field_coords,
    lab_to_field_point,
    lab_to_field_rate,
    lab_to_field_time,
    lab_to_field_volume,
    similarity_ratios,
)
from .programs.udp import IncompleteStreamError, UdpPublisher, parse_host_port, publish_fields

__all__ = [
    "FluidParams",
    "CartesianGrid",
    "IncompleteStreamError",
    "LabCase",
    "MODEL_HEIGHT_M",
    "MODEL_LENGTH_M",
    "MODEL_VOLUME_M3",
    "MODEL_VOLUME_ML",
    "MODEL_WIDTH_M",
    "MeshResult",
    "PROTOCOL_VERSION",
    "ProgramFields",
    "RockDiagnostics",
    "UdpPublisher",
    "VOLUME_BASES",
    "WellMap",
    "WellModelParams",
    "black_oil_forward",
    "build_mesh",
    "complete_injection",
    "field_scale_view",
    "field_to_lab",
    "field_to_lab_coords",
    "field_to_lab_point",
    "field_to_lab_rate",
    "field_to_lab_time",
    "field_to_lab_volume",
    "InversionSession",
    "interpolate_pressure",
    "interpolate_saturation",
    "invert_rock",
    "invert_rock_coarse_to_fine",
    "invert_rock_three_phase",
    "lab_to_field",
    "lab_to_field_coords",
    "lab_to_field_point",
    "lab_to_field_rate",
    "lab_to_field_time",
    "lab_to_field_volume",
    "load_lab_case",
    "pack_step",
    "parse_host_port",
    "peaceman_wi",
    "publish_fields",
    "run_pipeline",
    "similarity_ratios",
    "summarize_lab_case",
    "summarize_results",
    "transient_weights",
    "unpack_step",
    "validate_inversion",
    "well_cell_rates_weighted",
    "write_output",
    "__version__",
]
