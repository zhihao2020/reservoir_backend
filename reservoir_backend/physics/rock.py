"""Static rock parameters and transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidPermeability


LOGK_MIN = float(np.log(1.0e-22))
LOGK_MAX = float(np.log(1.0e-9))


def as_cell_field(value: float | NDArray[np.float64], n_cells: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_cells, float(arr), dtype=float)
    flat = arr.ravel()
    if flat.size != n_cells:
        raise ValueError(f"{name} size {flat.size} != n_cells {n_cells}")
    return flat.astype(float, copy=True)


def log_permeability(k: NDArray[np.float64] | float) -> NDArray[np.float64]:
    arr = np.asarray(k, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise InvalidPermeability("permeability must be positive and finite")
    return np.log(arr)


def exp_permeability(theta: NDArray[np.float64] | float) -> NDArray[np.float64]:
    arr = np.clip(np.asarray(theta, dtype=float), LOGK_MIN, LOGK_MAX)
    return np.exp(arr)


@dataclass
class Rock:
    """Static cell properties. k is kx=ky; optional kz."""

    permeability: NDArray[np.float64]
    porosity: NDArray[np.float64]
    kz: NDArray[np.float64] | None = None

    @classmethod
    def uniform(cls, n_cells: int, k: float = 1.0e-12, phi: float = 0.20, kz: float | None = None) -> Rock:
        kz_arr = None if kz is None else np.full(n_cells, float(kz), dtype=float)
        return cls(
            permeability=np.full(n_cells, float(k), dtype=float),
            porosity=np.full(n_cells, float(phi), dtype=float),
            kz=kz_arr,
        )

    def __post_init__(self) -> None:
        self.permeability = np.asarray(self.permeability, dtype=float).ravel()
        self.porosity = np.asarray(self.porosity, dtype=float).ravel()
        if self.kz is not None:
            self.kz = as_cell_field(self.kz, self.permeability.size, "kz")
            if np.any(self.kz <= 0.0) or not np.all(np.isfinite(self.kz)):
                raise InvalidPermeability("kz must be positive and finite")
        if self.permeability.size != self.porosity.size:
            raise ValueError("k and phi must have the same length")
        if np.any(self.permeability <= 0.0) or not np.all(np.isfinite(self.permeability)):
            raise InvalidPermeability("permeability must be positive and finite")
        if np.any(self.porosity <= 0.0) or np.any(self.porosity >= 1.0):
            raise ValueError("porosity must lie in (0, 1)")

    def vertical_permeability(self) -> NDArray[np.float64]:
        return self.permeability if self.kz is None else self.kz
