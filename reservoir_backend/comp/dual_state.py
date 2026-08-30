"""Compositional dual-continuum state. Primary variables are moles and pressure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class CompositionalContinuumState:
    """One continuum. ``pressure`` is (n_cells,), ``moles`` is (n_cells, n_comp)."""

    pressure: NDArray[np.float64]
    moles: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.pressure = np.asarray(self.pressure, dtype=float).ravel()
        self.moles = np.asarray(self.moles, dtype=float)
        if self.moles.ndim != 2 or self.moles.shape[0] != self.pressure.size:
            raise ValueError("moles must be (n_cells, n_comp)")

    @property
    def n_cells(self) -> int:
        return int(self.pressure.size)

    @property
    def n_comp(self) -> int:
        return int(self.moles.shape[1])

    def copy(self) -> CompositionalContinuumState:
        return CompositionalContinuumState(self.pressure.copy(), self.moles.copy())


@dataclass
class DualCompositionalState:
    """Fracture and matrix compositional states at one time (seconds)."""

    fracture: CompositionalContinuumState
    matrix: CompositionalContinuumState
    time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.fracture.n_cells != self.matrix.n_cells:
            raise ValueError("matrix and fracture n_cells must match")
        if self.fracture.n_comp != self.matrix.n_comp:
            raise ValueError("matrix and fracture n_comp must match")
        self.time_s = float(self.time_s)

    def copy(self) -> DualCompositionalState:
        return DualCompositionalState(
            fracture=self.fracture.copy(),
            matrix=self.matrix.copy(),
            time_s=float(self.time_s),
        )

    def total_moles(self) -> NDArray[np.float64]:
        """Component moles summed over both continua and all cells."""
        return self.fracture.moles.sum(axis=0) + self.matrix.moles.sum(axis=0)
