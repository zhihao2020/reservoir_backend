"""Program 2: pressure reconstruction from probes and well bottomhole pressures.

Pressure is reconstructed by interpolating the probe and well bottomhole
observations onto the full grid with a single interpolator (kriging by default).
The multi-method "auto" selection (IDW / RBF / residual / closed-model with
leave-one-out) was removed: one method is used throughout.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..exceptions import InvalidObservation
from ..core.cartesian import CartesianGrid
from .interpolate import interpolate_field


def interpolate_pressure(
    grid: CartesianGrid,
    probe_xyz: NDArray[np.float64],
    probe_pressure: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    well_pressure: NDArray[np.float64],
    *,
    method: str = "kriging",
) -> NDArray[np.float64]:
    """Return cell-centered pressure ``(n_cells,)`` for one time slice."""
    probes, p_probe, wells, p_well = _split(probe_xyz, probe_pressure, well_xyz, well_pressure)
    points, values = _combine(probes, p_probe, wells, p_well)
    return interpolate_field(points, values, grid.cell_centers(), method=method)


def _split(
    probe_xyz: NDArray[np.float64],
    probe_pressure: NDArray[np.float64],
    well_xyz: NDArray[np.float64],
    well_pressure: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    probes = np.asarray(probe_xyz, dtype=float)
    p_probe = np.asarray(probe_pressure, dtype=float).ravel()
    wells = np.asarray(well_xyz, dtype=float)
    p_well = np.asarray(well_pressure, dtype=float).ravel()
    if probes.size:
        if probes.ndim != 2 or probes.shape[0] != p_probe.size:
            raise InvalidObservation("pressure probe coordinates and values mismatch")
    else:
        probes = np.zeros((0, 3), dtype=float)
        p_probe = np.zeros(0, dtype=float)
    if wells.size:
        if wells.ndim != 2 or wells.shape[0] != p_well.size:
            raise InvalidObservation("well coordinates and pressures mismatch")
    else:
        wells = np.zeros((0, 3), dtype=float)
        p_well = np.zeros(0, dtype=float)
    if probes.shape[0] == 0 and wells.shape[0] == 0:
        raise InvalidObservation("pressure interpolation needs probes or wells")
    finite_p = np.isfinite(p_probe)
    probes, p_probe = probes[finite_p], p_probe[finite_p]
    finite_w = np.isfinite(p_well)
    wells, p_well = wells[finite_w], p_well[finite_w]
    return probes, p_probe, wells, p_well


def _combine(
    probes: NDArray[np.float64],
    p_probe: NDArray[np.float64],
    wells: NDArray[np.float64],
    p_well: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parts_xyz = []
    parts_val = []
    if probes.shape[0]:
        parts_xyz.append(probes)
        parts_val.append(p_probe)
    if wells.shape[0]:
        parts_xyz.append(wells)
        parts_val.append(p_well)
    return np.vstack(parts_xyz), np.concatenate(parts_val)
