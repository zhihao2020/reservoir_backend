"""Sensor-driven four-field pipeline: mesh, pressure, saturation, rock properties."""

from reservoir_backend.pipeline.ensemble_math import (
    gaspari_cohn,
    normalize_alpha_weights,
)
from reservoir_backend.pipeline.fractional_flow import water_fractional_flow
from reservoir_backend.pipeline.point_workflow import run_point_first_slice
from reservoir_backend.pipeline.spatial_interp import InterpResult, auto_interpolate_to_grid
from reservoir_backend.pipeline.esmda import ESMdaResult, generate_logk_ensemble, run_esmda_permeability
from reservoir_backend.pipeline.mesh_builder import build_mesh
from reservoir_backend.pipeline.mesh_refine import map_field_to_mesh, refine_mesh_by_indicator
from reservoir_backend.pipeline.pressure_field import reconstruct_pressure
from reservoir_backend.pipeline.property_field import invert_rock_properties
from reservoir_backend.pipeline.run import run_time_slice, save_fields
from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
from reservoir_backend.pipeline.sensor_io import (
    load_sensor_series,
    load_well_series_csv,
    write_boundary_series_csv,
    write_well_series_csv,
)
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
    build_faulted_channel_twin,
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
    "ESMdaResult",
    "FieldBundle",
    "InterpResult",
    "MeshBundle",
    "SensorSample",
    "SyntheticTwin",
    "WellPoint",
    "auto_interpolate_to_grid",
    "build_channel_twin",
    "build_faulted_channel_twin",
    "build_mesh",
    "gaspari_cohn",
    "generate_logk_ensemble",
    "water_fractional_flow",
    "indicator_to_active_mask",
    "normalize_alpha_weights",
    "infer_shape_indicator",
    "invert_rock_properties",
    "load_sensor_series",
    "load_well_series_csv",
    "map_field_to_mesh",
    "mask_overlap",
    "reconstruct_pressure",
    "reconstruct_saturation",
    "refine_mesh_by_indicator",
    "run_esmda_permeability",
    "run_point_first_slice",
    "run_shape_discovery",
    "run_time_series",
    "run_time_slice",
    "save_discovery",
    "save_fields",
    "write_boundary_series_csv",
    "write_well_series_csv",
]
