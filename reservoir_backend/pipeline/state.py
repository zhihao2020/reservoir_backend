"""Data contracts for the sensor four-field pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.grid import Grid3D


@dataclass(frozen=True)
class AxisAlignedBounds:
    """Axis-aligned domain box in metres."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def __post_init__(self) -> None:
        if not (self.xmax > self.xmin and self.ymax > self.ymin and self.zmax > self.zmin):
            raise ValueError("bounds must satisfy max > min on each axis")


@dataclass(frozen=True)
class WellPoint:
    """A named well location in physical coordinates."""

    name: str
    x: float
    y: float
    z: float


@dataclass
class MeshBundle:
    """Mesh geometry plus well-to-cell mapping."""

    grid: Grid3D
    cell_id: NDArray[np.int64]  # flat index, shape (n,)
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    k: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    well_cell_id: dict[str, int] = field(default_factory=dict)
    bounds: AxisAlignedBounds | None = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_id.size)


@dataclass
class BoundaryConditions:
    """Face boundary pressures (Pa) and optional net face fluxes (m^3/s)."""

    pressure: dict[str, float] = field(default_factory=dict)
    # keys: left,right,front,back,bottom,top
    flux: dict[str, float] = field(default_factory=dict)


@dataclass
class SensorSample:
    """Sparse sensor readings at one time stamp."""

    time: float
    well_pressure: Mapping[str, float]  # well name -> Pa
    well_saturation: Mapping[str, tuple[float, float, float]]  # name -> (sw, so, sg)
    boundary: BoundaryConditions = field(default_factory=BoundaryConditions)


@dataclass
class FieldBundle:
    """Four-field state on the mesh at one time.

    Maps to 软件要求 steps 2–4: pressure, saturations (sw/so/sg), rock (k, φ).
    Optional face fluxes support step-4 Darcy inversion diagnostics.
    """

    time: float
    pressure: NDArray[np.float64]  # (nz,ny,nx)
    sw: NDArray[np.float64]
    so: NDArray[np.float64]
    sg: NDArray[np.float64]
    permeability: NDArray[np.float64]  # m^2
    porosity: NDArray[np.float64]
    notes: list[str] = field(default_factory=list)
    # optional face volumetric fluxes (m^3/s), shapes (nz,ny,nx+1) etc.
    flux_x: NDArray[np.float64] | None = None
    flux_y: NDArray[np.float64] | None = None
    flux_z: NDArray[np.float64] | None = None
