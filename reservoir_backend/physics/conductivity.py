"""Effective fracture conductivity → cell permeability. Units: C_f in m²."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidPermeability


@dataclass
class FractureConductivityModel:
    """Map scalar (or zonal) ``C_f`` onto fracture cells. Matrix k is fixed.

    ``C_f`` is effective fracture permeability ``k_f^eff`` in m², not a
    geometric aperture of a discrete fracture.
    """

    n_cells: int
    fracture_mask: NDArray[np.bool_]
    k_matrix_m2: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.fracture_mask, dtype=bool).ravel()
        if mask.size != int(self.n_cells):
            raise ValueError(f"fracture_mask size {mask.size} != n_cells {self.n_cells}")
        if float(self.k_matrix_m2) <= 0.0:
            raise InvalidPermeability("k_matrix_m2 must be positive")
        self.fracture_mask = mask

    def permeability(self, cf_m2: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Legacy single-continuum paint. Prefer ``dual_rock`` for DPDP."""
        kf = self._scalar_cf(cf_m2)
        k = np.full(self.n_cells, float(self.k_matrix_m2), dtype=float)
        k[self.fracture_mask] = kf
        return k

    def dual_rock(self, cf_m2: float | NDArray[np.float64], *, phi_matrix: float, phi_fracture: float):
        """C_f → DualRock. Only the fracture continuum permeability changes."""
        from reservoir_backend.physics.dual_rock import DualRock

        return DualRock.from_cf(
            self.n_cells,
            k_matrix_m2=float(self.k_matrix_m2),
            phi_matrix=float(phi_matrix),
            cf_m2=self._scalar_cf(cf_m2),
            phi_fracture=float(phi_fracture),
        )

    def _scalar_cf(self, cf_m2: float | NDArray[np.float64]) -> float:
        cf = np.asarray(cf_m2, dtype=float).ravel()
        if cf.size != 1:
            raise ValueError("V1 FractureConductivityModel accepts a scalar C_f")
        kf = float(cf[0])
        if kf <= 0.0 or not np.isfinite(kf):
            raise InvalidPermeability("C_f must be positive and finite")
        return kf
