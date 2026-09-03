"""Log-space parameterization for effective fracture conductivity.

V1 inverts a scalar ``C_f`` (effective fracture permeability, m²).
The assimilator updates ``m = log(C_f / C_ref)`` so the prior is centred at 0
when ``C_f = C_ref`` and updates cannot produce C_f <= 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.physics.rock import LOGK_MAX, LOGK_MIN

CF_REF_M2 = 1.0e-13
LOG_CF_MIN = float(LOGK_MIN - np.log(CF_REF_M2))
LOG_CF_MAX = float(LOGK_MAX - np.log(CF_REF_M2))


@dataclass
class LogConductivityParameterization:
    """Scalar ``m = log(C_f / C_ref)``. ``n_params`` is 1 in V1 (Level 1).

    ``C_f`` is effective fracture permeability (m²), not a discrete-fracture k.
    """

    n_zones: int = 1
    log_min: float = LOG_CF_MIN
    log_max: float = LOG_CF_MAX
    phi: float = 0.08
    phi_fracture: float = 0.02
    c_ref_m2: float = CF_REF_M2
    conductivity: FractureConductivityModel | None = None
    prior_mean: float | NDArray[np.float64] = 0.0
    prior_std: float | NDArray[np.float64] = 1.0

    def __post_init__(self) -> None:
        if int(self.n_zones) < 1:
            raise ValueError("n_zones must be >= 1")
        self.n_zones = int(self.n_zones)
        if float(self.c_ref_m2) <= 0.0:
            raise InvalidPermeability("C_ref must be positive")

    @property
    def n_params(self) -> int:
        return int(self.n_zones)

    def encode(self, physical_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """C_f (m²) → m = log(C_f / C_ref)."""
        cf = np.asarray(physical_parameter, dtype=float).ravel()
        if cf.size != self.n_params:
            raise ValueError(f"C_f size {cf.size} != {self.n_params}")
        if np.any(cf <= 0.0) or not np.all(np.isfinite(cf)):
            raise InvalidPermeability("C_f must be positive and finite")
        return np.clip(np.log(cf / float(self.c_ref_m2)), self.log_min, self.log_max)

    def decode(self, latent_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """m → C_f = C_ref * exp(m) (m²)."""
        m = np.asarray(latent_parameter, dtype=float).ravel()
        if m.size != self.n_params:
            raise ValueError(f"latent size {m.size} != {self.n_params}")
        if not np.all(np.isfinite(m)):
            raise InvalidPermeability("log C_f must be finite")
        return float(self.c_ref_m2) * np.exp(np.clip(m, self.log_min, self.log_max))

    def project(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.asarray(theta, dtype=float).ravel()
        if th.size != self.n_params:
            raise ValueError(f"theta size {th.size} != {self.n_params}")
        return np.clip(th, self.log_min, self.log_max)

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fracture-continuum permeability field. Uniform C_f in V1 DPDP."""
        cf = float(self.decode(theta)[0])
        n = int(self.conductivity.n_cells) if self.conductivity is not None else 1
        return np.full(n, cf, dtype=float)

    def dual_rock(self, theta: NDArray[np.float64]):
        """C_f → DualRock. Matrix rock is unchanged."""
        if self.conductivity is None:
            raise ValueError("FractureConductivityModel is required to build DualRock")
        cf = self.decode(theta)
        return self.conductivity.dual_rock(
            cf, phi_matrix=float(self.phi), phi_fracture=float(self.phi_fracture)
        )
