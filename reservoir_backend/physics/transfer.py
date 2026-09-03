"""Matrix–fracture transfer. Transfer > 0 is always matrix → fracture.

V1 driving force is potential Φ = p (no capillary). Mobility is upstream.
Component molar rate:
    N_i = ξ_L x_i q_L + ξ_V y_i q_V  [+ water]
with
    q_α = V σ k_m λ_{α,up} (p_m − p_f)
Units: q_α in m³/s, N_i in mol/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.properties import PhaseProps


@dataclass
class TransferRates:
    """Per-cell transfer. ``molar_rate`` is (n_cells, n_comp), matrix → fracture."""

    molar_rate: NDArray[np.float64]
    phase_liquid_rate: NDArray[np.float64]
    phase_vapor_rate: NDArray[np.float64]
    water_rate: NDArray[np.float64]


@dataclass(frozen=True)
class WarrenRootTransfer:
    """Pressure-only volumetric rate ``q = σ k_m V (p_m − p_f)`` (λ = 1)."""

    shape_factor: float
    k_matrix_m2: float
    transfer_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if float(self.shape_factor) < 0.0:
            raise ValueError("shape_factor must be >= 0")
        if float(self.k_matrix_m2) <= 0.0:
            raise ValueError("k_matrix_m2 must be positive")
        if float(self.transfer_multiplier) <= 0.0:
            raise ValueError("transfer_multiplier must be positive")

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
        return (
            float(self.shape_factor)
            * float(self.k_matrix_m2)
            * float(self.transfer_multiplier)
            * vol
            * (pm - pf)
        )


@dataclass(frozen=True)
class ComponentTransfer:
    r"""Multiphase component transfer.

    ``shape_factor`` and ``k_matrix_m2`` build \(T_{mf}^{ref}\).
    ``transfer_multiplier`` is \(\beta_{mf}\) applied once: \(T_{mf}=\beta T^{ref}\).
    """

    shape_factor: float
    k_matrix_m2: float
    transfer_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if float(self.shape_factor) < 0.0:
            raise ValueError("shape_factor must be >= 0")
        if float(self.k_matrix_m2) <= 0.0:
            raise ValueError("k_matrix_m2 must be positive")
        if float(self.transfer_multiplier) <= 0.0:
            raise ValueError("transfer_multiplier must be positive")

    def compute(
        self,
        p_matrix: NDArray[np.float64],
        p_fracture: NDArray[np.float64],
        cell_volume: NDArray[np.float64],
        props_matrix: PhaseProps,
        props_fracture: PhaseProps,
        k_matrix: NDArray[np.float64] | float | None = None,
    ) -> TransferRates:
        pm = np.asarray(p_matrix, dtype=float).ravel()
        pf = np.asarray(p_fracture, dtype=float).ravel()
        vol = np.asarray(cell_volume, dtype=float).ravel()
        dphi = pm - pf
        if k_matrix is None:
            km = float(self.k_matrix_m2)
        else:
            km_arr = np.asarray(k_matrix, dtype=float)
            km = km_arr if km_arr.shape == vol.shape else float(np.ravel(km_arr)[0])
        cond = float(self.shape_factor) * km * vol * float(self.transfer_multiplier)
        from_m = dphi >= 0.0
        lam_l = np.where(from_m, props_matrix.lam_l, props_fracture.lam_l)
        lam_v = np.where(from_m, props_matrix.lam_v, props_fracture.lam_v)
        q_l = cond * lam_l * dphi
        q_v = cond * lam_v * dphi
        x_up = np.where(q_l[:, None] >= 0.0, props_matrix.x, props_fracture.x)
        y_up = np.where(q_v[:, None] >= 0.0, props_matrix.y, props_fracture.y)
        xi_l = np.where(q_l >= 0.0, props_matrix.xi_l, props_fracture.xi_l)
        xi_v = np.where(q_v >= 0.0, props_matrix.xi_v, props_fracture.xi_v)
        molar_hc = xi_l[:, None] * x_up * q_l[:, None] + xi_v[:, None] * y_up * q_v[:, None]
        q_w = np.zeros_like(q_l)
        if props_matrix.has_water:
            lam_w = np.where(from_m, props_matrix.lam_w, props_fracture.lam_w)
            q_w = cond * lam_w * dphi
            xi_w = np.where(q_w >= 0.0, props_matrix.xi_w, props_fracture.xi_w)
            n_w = (xi_w * q_w)[:, None]
            molar = np.concatenate([molar_hc, n_w], axis=1)
        else:
            molar = molar_hc
        return TransferRates(
            molar_rate=molar,
            phase_liquid_rate=q_l,
            phase_vapor_rate=q_v,
            water_rate=q_w,
        )


MatrixFractureTransferModel = ComponentTransfer
