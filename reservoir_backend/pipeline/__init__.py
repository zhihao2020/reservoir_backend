"""Sensor-driven four-field pipeline: mesh, pressure, saturation, rock properties."""

from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.mesh_refine import map_field_to_mesh, refine_mesh_by_indicator
from reservoir_backend.pipeline.pressure_field import reconstruct_pressure
from reservoir_backend.pipeline.property_field import invert_rock_properties
from reservoir_backend.pipeline.run import run_time_slice, save_fields
from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
from reservoir_backend.pipeline.shape_indicator import (
    indicator_to_active_mask,
    infer_shape_indicator,
)
from reservoir_backend.pipeline.state import (
    AxisAlignedBounds,
    BoundaryConditions,
    FieldBundle,
    MeshBundle,
    SensorSample,
    WellPoint,
)
from reservoir_backend.pipeline.synthetic_twin import (
    SyntheticTwin,
    build_channel_twin,
    mask_overlap,
)
from reservoir_backend.pipeline.time_series import (
    DiscoveryResult,
    run_shape_discovery,
    run_time_series,
    save_discovery,
)

__all__ = [
    "AxisAlignedBounds",
    "BoundaryConditions",
    "DiscoveryResult",
    "FieldBundle",
    "MeshBundle",
    "SensorSample",
    "SyntheticTwin",
    "WellPoint",
    "build_channel_twin",
    "build_mesh",
    "indicator_to_active_mask",
    "infer_shape_indicator",
    "invert_rock_properties",
    "map_field_to_mesh",
    "mask_overlap",
    "reconstruct_pressure",
    "reconstruct_saturation",
    "refine_mesh_by_indicator",
    "run_shape_discovery",
    "run_time_series",
    "run_time_slice",
    "save_discovery",
    "save_fields",
]
