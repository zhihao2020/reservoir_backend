"""Simplified rate-controlled well source/sink utilities.

This module provides engineering helpers for pressure-system source terms. It
does not implement industrial well controls or a Peaceman well model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.core.grid import Grid3D


_INJECTOR_TYPES = {"injector", "injection"}
_PRODUCER_TYPES = {"producer", "production"}


@dataclass(frozen=True)
class RateControlledWell:
    """Single-cell rate-controlled well.

    Positive source terms denote injection into the reservoir. Producer rates
    are assembled as negative source terms even when `rate` is supplied as a
    positive production magnitude.
    """

    well_id: str
    well_type: str
    rate: float
    cell_index: int | None = None
    i: int | None = None
    j: int | None = None
    k: int | None = None
    control_type: str = "rate"
    unit: str = "m3/s"
    phase: str = "total"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.well_id:
            raise ValueError("well_id must not be empty")
        normalized_type = self.well_type.lower()
        if normalized_type not in _INJECTOR_TYPES | _PRODUCER_TYPES:
            raise ValueError("well_type must be injector or producer")
        object.__setattr__(self, "well_type", "injector" if normalized_type in _INJECTOR_TYPES else "producer")
        if self.control_type != "rate":
            raise NotImplementedError("only rate-controlled wells are supported")
        rate = float(self.rate)
        if not np.isfinite(rate) or rate < 0.0:
            raise ValueError("rate must be a nonnegative finite magnitude")
        object.__setattr__(self, "rate", rate)
        has_cell = self.cell_index is not None
        has_ijk = self.i is not None or self.j is not None or self.k is not None
        if has_cell and has_ijk:
            raise ValueError("define either cell_index or i/j/k, not both")
        if not has_cell and not has_ijk:
            raise ValueError("well location requires cell_index or i/j/k")
        if has_ijk and (self.i is None or self.j is None or self.k is None):
            raise ValueError("i, j, and k must all be provided")

    @property
    def signed_rate(self) -> float:
        """Return signed reservoir source term in the configured rate unit."""
        if self.well_type == "injector":
            return float(self.rate)
        return -float(self.rate)

    def resolved_cell_index(self, grid: Grid3D | tuple[int, int, int]) -> int:
        """Return x-fastest flattened cell index for the configured location."""
        shape = _grid_shape(grid)
        nz, ny, nx = shape
        total = nx * ny * nz
        if self.cell_index is not None:
            idx = int(self.cell_index)
            if idx < 0 or idx >= total:
                raise ValueError(f"cell_index {idx} outside valid range [0, {total})")
            return idx
        assert self.i is not None and self.j is not None and self.k is not None
        i, j, k = int(self.i), int(self.j), int(self.k)
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise ValueError(f"well coordinates {(i, j, k)} outside grid shape {shape}")
        return k * ny * nx + j * nx + i


def validate_well(well: RateControlledWell, grid: Grid3D | tuple[int, int, int]) -> None:
    """Validate a well against a Cartesian grid shape."""
    well.resolved_cell_index(grid)


def build_well_contribution_vector(
    wells: list[RateControlledWell],
    grid: Grid3D | tuple[int, int, int],
) -> NDArray[np.float64]:
    """Assemble a flattened source/sink vector from rate-controlled wells."""
    shape = _grid_shape(grid)
    nz, ny, nx = shape
    contribution = np.zeros(nx * ny * nz, dtype=float)
    for well in wells:
        idx = well.resolved_cell_index(grid)
        contribution[idx] += well.signed_rate
    return contribution


def summarize_wells(wells: list[RateControlledWell], grid: Grid3D | tuple[int, int, int]) -> dict:
    """Build JSON-serializable well source/sink diagnostics."""
    contribution = build_well_contribution_vector(wells, grid)
    diagnostics = []
    for well in wells:
        diagnostics.append(
            {
                "well_id": well.well_id,
                "well_type": well.well_type,
                "control_type": well.control_type,
                "cell_index": int(well.resolved_cell_index(grid)),
                "signed_rate": float(well.signed_rate),
                "unit": well.unit,
                "phase": well.phase,
                "metadata": dict(well.metadata),
            }
        )
    total_injection = float(sum(max(well.signed_rate, 0.0) for well in wells))
    total_production = float(sum(-min(well.signed_rate, 0.0) for well in wells))
    return {
        "success": True,
        "num_wells": len(wells),
        "total_injection_rate": total_injection,
        "total_production_rate": total_production,
        "net_source_rate": float(np.sum(contribution)),
        "well_contribution_shape": list(contribution.shape),
        "well_diagnostics": diagnostics,
        "warnings": [],
    }


def well_to_dict(well: RateControlledWell, grid: Grid3D | tuple[int, int, int] | None = None) -> dict:
    """Serialize a rate-controlled well."""
    data = {
        "well_id": well.well_id,
        "well_type": well.well_type,
        "control_type": well.control_type,
        "rate": float(well.rate),
        "signed_rate": float(well.signed_rate),
        "unit": well.unit,
        "phase": well.phase,
        "metadata": dict(well.metadata),
    }
    if grid is not None:
        data["cell_index"] = int(well.resolved_cell_index(grid))
    elif well.cell_index is not None:
        data["cell_index"] = int(well.cell_index)
    else:
        data.update({"i": well.i, "j": well.j, "k": well.k})
    return data


def _grid_shape(grid: Grid3D | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(grid, Grid3D):
        return grid.shape
    if len(grid) != 3:
        raise ValueError("grid shape must be (nz, ny, nx)")
    nz, ny, nx = (int(v) for v in grid)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("grid shape entries must be positive")
    return nz, ny, nx
