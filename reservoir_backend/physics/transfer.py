"""Matrix–fracture transfer. V1: Warren–Root, shape factor and k_m fixed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WarrenRootTransfer:
    """``q_mf = σ k_m V (p_m - p_f)``.

    ``shape_factor`` is in 1/m². ``k_matrix_m2`` is in m².
    Transfer volume rate is m³/s per cell when pressure is Pa
    (and viscosity is absorbed into the caller if needed).
    V1 does not invert these coefficients.
    """

    shape_factor: float
    k_matrix_m2: float

    def __post_init__(self) -> None:
        if float(self.shape_factor) < 0.0:
            raise ValueError("shape_factor must be >= 0")
        if float(self.k_matrix_m2) <= 0.0:
            raise ValueError("k_matrix_m2 must be positive")

    def compute_transfer(
        self,
        p_matrix: NDArray[np.float64] | float,
        p_fracture: NDArray[np.float64] | float,
        cell_volume: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        """Volumetric matrix→fracture transfer (m³/s), positive when p_m > p_f."""
        pm = np.asarray(p_matrix, dtype=float)
        pf = np.asarray(p_fracture, dtype=float)
        vol = np.asarray(cell_volume, dtype=float)
        return float(self.shape_factor) * float(self.k_matrix_m2) * vol * (pm - pf)


MatrixFractureTransferModel = WarrenRootTransfer
