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
    """A named location on the mesh (injector / producer / single-quantity probe).

    ``role``:
    - ``injector`` / ``producer``: may measure p and/or S and may have rates
    - ``observer_p``: **pressure-only** probe (no S, no rate)
    - ``observer_s``: **saturation-only** probe (no p, no rate)
    - ``observer``: legacy alias; must still provide only p **or** only S
    """

    name: str
    x: float
    y: float
    z: float
    role: str = "observer_p"

    def __post_init__(self) -> None:
        role = str(self.role).lower().strip()
        if role in ("inj", "injection", "injector"):
            role = "injector"
        elif role in ("prod", "production", "producer"):
            role = "producer"
        elif role in (
            "obs_p",
            "observer_p",
            "pressure_probe",
            "p_probe",
            "pressure_only",
        ):
            role = "observer_p"
        elif role in (
            "obs_s",
            "observer_s",
            "saturation_probe",
            "s_probe",
            "sw_probe",
            "saturation_only",
        ):
            role = "observer_s"
        elif role in ("obs", "probe", "sensor", "monitor", "observation", "observer"):
            role = "observer"
        else:
            raise ValueError(f"unsupported well role: {self.role}")
        object.__setattr__(self, "role", role)


@dataclass
class MeshBundle:
    """Mesh geometry plus well/probe-to-cell mapping."""

    grid: Grid3D
    cell_id: NDArray[np.int64]  # flat index, shape (n,)
    i: NDArray[np.int64]
    j: NDArray[np.int64]
    k: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    z: NDArray[np.float64]
    well_cell_id: dict[str, int] = field(default_factory=dict)
    well_role: dict[str, str] = field(default_factory=dict)
    bounds: AxisAlignedBounds | None = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_id.size)

    def observer_names(self) -> list[str]:
        return [
            n
            for n, r in self.well_role.items()
            if r in ("observer", "observer_p", "observer_s")
        ]

    def pressure_probe_names(self) -> list[str]:
        return [n for n, r in self.well_role.items() if r == "observer_p"]

    def saturation_probe_names(self) -> list[str]:
        return [n for n, r in self.well_role.items() if r == "observer_s"]

    def active_well_names(self) -> list[str]:
        return [n for n, r in self.well_role.items() if r in ("injector", "producer")]


@dataclass
class BoundaryConditions:
    """Face boundary pressures (Pa) and optional net face fluxes (m^3/s)."""

    pressure: dict[str, float] = field(default_factory=dict)
    # keys: left,right,front,back,bottom,top
    flux: dict[str, float] = field(default_factory=dict)


@dataclass
class SensorSample:
    """Sparse sensor readings at one time stamp.

    Names in ``well_pressure`` / ``well_saturation`` may be:

    - **injectors / producers**: may appear in either/both maps and in ``well_rate``
    - **observer_p**: **only** in ``well_pressure`` (never saturation)
    - **observer_s**: **only** in ``well_saturation`` (never pressure)

    After field interpolation, every hard location is *assigned* the missing
    quantity from the field so that point k,φ can be estimated, then IDW to grid.
    """

    time: float
    well_pressure: Mapping[str, float]  # name -> Pa
    well_saturation: Mapping[str, tuple[float, float, float]]  # name -> (sw, so, sg)
    boundary: BoundaryConditions = field(default_factory=BoundaryConditions)
    # signed volumetric rates (m^3/s): +injection, -production; probes omit this
    well_rate: Mapping[str, float] = field(default_factory=dict)

    def observation_names(self, mesh: MeshBundle | None = None) -> list[str]:
        """Probe names (no fluid rate)."""
        rate_names = set(self.well_rate or {})
        names = set(self.well_pressure) | set(self.well_saturation)
        if mesh is not None and mesh.well_role:
            obs_roles = {"observer", "observer_p", "observer_s"}
            return sorted(
                n
                for n in names
                if mesh.well_role.get(n) in obs_roles or n not in rate_names
            )
        return sorted(n for n in names if n not in rate_names)

    def flowing_names(self) -> list[str]:
        return sorted(self.well_rate or {})


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
