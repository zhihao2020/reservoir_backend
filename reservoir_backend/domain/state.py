"""Continuum and dual-continuum state. Fields are SI; pressure in Pa."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _copy(arr: NDArray[np.float64] | None) -> NDArray[np.float64] | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=float).copy()


@dataclass
class ContinuumState:
    """One continuum (matrix or fracture). Arrays are ``(n_cells,)`` or ``(n_cells, n_comp)``."""

    pressure: NDArray[np.float64]
    saturation: NDArray[np.float64]
    composition: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        self.pressure = np.asarray(self.pressure, dtype=float).ravel()
        self.saturation = np.asarray(self.saturation, dtype=float).ravel()
        if self.pressure.size != self.saturation.size:
            raise ValueError("pressure and saturation must share n_cells")
        if self.composition is not None:
            z = np.asarray(self.composition, dtype=float)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            if z.shape[0] != self.pressure.size:
                raise ValueError("composition rows must equal n_cells")
            self.composition = z

    def copy(self) -> ContinuumState:
        return ContinuumState(
            pressure=self.pressure.copy(),
            saturation=self.saturation.copy(),
            composition=_copy(self.composition),
        )


@dataclass
class DualContinuumState:
    """Matrix and fracture continua at one time. Time in seconds."""

    matrix: ContinuumState
    fracture: ContinuumState
    time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.matrix.pressure.size != self.fracture.pressure.size:
            raise ValueError("matrix and fracture n_cells must match")
        self.time_s = float(self.time_s)

    def copy(self) -> DualContinuumState:
        return DualContinuumState(
            matrix=self.matrix.copy(),
            fracture=self.fracture.copy(),
            time_s=float(self.time_s),
        )
