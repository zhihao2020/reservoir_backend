"""Joint log-space parameterization: fracture conductivity and T_mf multiplier.

V1 theta is
    theta[0] = log(C_f / C_ref)
    theta[1] = log(beta_mf)
with T_mf = beta_mf * T_mf^ref. Shape factor and k_m stay in T_mf^ref.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.inverse.log_conductivity import CF_REF_M2, LOG_CF_MAX, LOG_CF_MIN
from reservoir_backend.physics.conductivity import FractureConductivityModel

LOG_BETA_MIN = -4.0
LOG_BETA_MAX = 4.0


@dataclass
class LogCfTmfParameterization:
    """Two-parameter V1: effective C_f and matrix-fracture exchange multiplier."""

    phi: float = 0.08
    phi_fracture: float = 0.02
    c_ref_m2: float = CF_REF_M2
    tmf_multiplier_ref: float = 1.0
    conductivity: FractureConductivityModel | None = None
    prior_mean: NDArray[np.float64] | list[float] | float = (0.0, 0.0)
    prior_std: NDArray[np.float64] | list[float] | float = (0.8, 0.5)
    log_min: float = min(LOG_CF_MIN, LOG_BETA_MIN)
    log_max: float = max(LOG_CF_MAX, LOG_BETA_MAX)

    def __post_init__(self) -> None:
        if float(self.c_ref_m2) <= 0.0:
            raise InvalidPermeability("C_ref must be positive")
        if float(self.tmf_multiplier_ref) <= 0.0:
            raise InvalidPermeability("tmf_multiplier_ref must be positive")
        self.prior_mean = np.asarray(self.prior_mean, dtype=float).ravel()
        self.prior_std = np.asarray(self.prior_std, dtype=float).ravel()
        if self.prior_mean.size == 1:
            self.prior_mean = np.array([float(self.prior_mean[0]), 0.0], dtype=float)
        if self.prior_std.size == 1:
            self.prior_std = np.array([float(self.prior_std[0]), 0.5], dtype=float)
        if self.prior_mean.size != 2 or self.prior_std.size != 2:
            raise ValueError("prior_mean and prior_std must have length 2")

    @property
    def n_params(self) -> int:
        return 2

    def encode(self, physical_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """[C_f, beta_mf] → theta."""
        phys = np.asarray(physical_parameter, dtype=float).ravel()
        if phys.size != 2:
            raise ValueError(f"physical size {phys.size} != 2")
        if np.any(phys <= 0.0) or not np.all(np.isfinite(phys)):
            raise InvalidPermeability("C_f and beta_mf must be positive and finite")
        th = np.array(
            [
                np.log(phys[0] / float(self.c_ref_m2)),
                np.log(phys[1] / float(self.tmf_multiplier_ref)),
            ],
            dtype=float,
        )
        return np.clip(th, self.log_min, self.log_max)

    def decode(self, latent_parameter: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """theta → [C_f, beta_mf]."""
        m = np.asarray(latent_parameter, dtype=float).ravel()
        if m.size != 2:
            raise ValueError(f"latent size {m.size} != 2")
        if not np.all(np.isfinite(m)):
            raise InvalidPermeability("log parameters must be finite")
        m = np.clip(m, self.log_min, self.log_max)
        return np.array(
            [
                float(self.c_ref_m2) * float(np.exp(m[0])),
                float(self.tmf_multiplier_ref) * float(np.exp(m[1])),
            ],
            dtype=float,
        )

    def decode_physical(self, latent_parameter: float | NDArray[np.float64]) -> dict[str, float]:
        phys = self.decode(latent_parameter)
        return {"cf_m2": float(phys[0]), "tmf_multiplier": float(phys[1])}

    def project(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        th = np.asarray(theta, dtype=float).ravel()
        if th.size != 2:
            raise ValueError(f"theta size {th.size} != 2")
        return np.clip(th, self.log_min, self.log_max)

    def expand(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fracture-continuum permeability field from C_f only."""
        cf = float(self.decode(theta)[0])
        n = int(self.conductivity.n_cells) if self.conductivity is not None else 1
        return np.full(n, cf, dtype=float)

    def dual_rock(self, theta: NDArray[np.float64]):
        if self.conductivity is None:
            raise ValueError("FractureConductivityModel is required to build DualRock")
        phys = self.decode_physical(theta)
        return self.conductivity.dual_rock(
            phys["cf_m2"], phi_matrix=float(self.phi), phi_fracture=float(self.phi_fracture)
        )
