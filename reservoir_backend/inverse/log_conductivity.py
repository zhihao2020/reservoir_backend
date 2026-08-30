"""Log-space parameterization for effective fracture conductivity.

V1 inverts a scalar ``C_f`` (effective fracture permeability, m²).
The assimilator updates ``m = log(C_f)`` so updates cannot produce C_f <= 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN

LOG_CF_MIN = LOGK_MIN
LOG_CF_MAX = LOGK_MAX


@dataclass
class LogConductivityParameterization:
    """Scalar ``m = log(C_f)``. ``n_params`` is 1 in V1 (Level 1).

    ``C_f`` is effective fracture permeability (m²), not a discrete-fracture k.
    """

    n_zones: int = 1
    log_min: float = LOG_CF_MIN
    log_max: float = LOG_CF_MAX
    phi: float = 0.20
    conductivity: FractureConductivityModel | None = None
    prior_mean: float | NDArray[np.float64] = float(np.log(1.0e-13))
    prior_std: float | NDArray[np.float64] = 1.0

    def __post_init__(self) -> None:
        if int(self.n_zones) < 1:
            raise ValueError("n_zones must be >= 1")
        self.n_zones = int(self.n_zones)

    @property
    def n_params(self) -> int:
        return int(self.n_zones)

    def encode(self, physical_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """C_f (m²) → m = log(C_f)."""
        cf = np.asarray(physical_parameter, dtype=float).ravel()
        if cf.size != self.n_params:
            raise ValueError(f"C_f size {cf.size} != {self.n_params}")
        if np.any(cf <= 0.0) or not np.all(np.isfinite(cf)):
            raise InvalidPermeability("C_f must be positive and finite")
        return np.clip(np.log(cf), self.log_min, self.log_max)

    def decode(self, latent_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """m → C_f = exp(m) (m²)."""
        m = np.asarray(latent_parameter, dtype=float).ravel()
        if m.size != self.n_params:
            raise ValueError(f"latent size {m.size} != {self.n_params}")
        if not np.all(np.isfinite(m)):
            raise InvalidPermeability("log C_f must be finite")
        return np.exp(np.clip(m, self.log_min, self.log_max))

    def project(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.asarray(theta, dtype=float).ravel()
        if th.size != self.n_params:
            raise ValueError(f"theta size {th.size} != {self.n_params}")
        return np.clip(th, self.log_min, self.log_max)

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map log C_f onto the fracture cells. Matrix k stays fixed."""
        if self.conductivity is None:
            raise ValueError("FractureConductivityModel is required to expand C_f")
        cf = self.decode(theta)
        return self.conductivity.permeability(cf)
