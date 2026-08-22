"""Per-cell flash and component molar accumulation.

Units (SI):
    T          kelvin
    p          pascal
    V_pore     m³
    v          m³/mol   (PR ``Z R T / p``)
    ξ          mol/m³   (``1 / v``)
    S          —        (phase volume fraction of pore space)
    x, y, z    —        (mole fractions)
    n_i        mol

    n_i = V_pore * (ξ_L S_L x_i + ξ_V S_V y_i)

Saturations follow from the flash vapor mole fraction ``ν`` and the PR
molar volumes: ``S_V = ν v_V / (ν v_V + (1−ν) v_L)``, ``S_L = 1 − S_V``.
Equivalent closed form: ``n = V_pore * z / v_mix`` with
``v_mix = ν v_V + (1−ν) v_L``.

Standalone kernel helper. Not a FIM accumulation term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.eos.peng_robinson import EosMixture, molar_volume


@dataclass(frozen=True)
class CellFlash:
    """Flashed cell properties used by accumulation and TPFA flux."""

    z: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    nu: float  # vapor mole fraction from flash (``V``)
    S_liquid: float
    S_vapor: float
    xi_liquid: float  # mol/m³
    xi_vapor: float
    v_liquid: float  # m³/mol
    v_vapor: float
    rho_liquid: float  # kg/m³
    rho_vapor: float
    phase_state: str
    marker: str = ""


def flash_cell(
    z: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
) -> CellFlash:
    """Flash overall composition ``z`` at ``(T [K], p [Pa])`` and pack saturations."""
    result = flash_tp(z, T, p, mixture)
    if result.phase_state == "two-phase":
        v_l = float(result.v_liquid) if result.v_liquid is not None else molar_volume(
            result.x, T, p, mixture, phase="liquid"
        )
        v_v = float(result.v_vapor) if result.v_vapor is not None else molar_volume(
            result.y, T, p, mixture, phase="vapor"
        )
        nu = float(result.V)
        x = result.x.copy()
        y = result.y.copy()
    elif result.phase_state == "liquid":
        v_l = molar_volume(result.z, T, p, mixture, phase="liquid")
        v_v = v_l
        nu = 0.0
        x = result.z.copy()
        y = result.z.copy()
    else:
        v_v = molar_volume(result.z, T, p, mixture, phase="vapor")
        v_l = v_v
        nu = 1.0
        x = result.z.copy()
        y = result.z.copy()

    v_l = max(v_l, 1.0e-16)
    v_v = max(v_v, 1.0e-16)
    v_mix = nu * v_v + (1.0 - nu) * v_l
    s_v = float(nu * v_v / v_mix) if v_mix > 0.0 else 0.0
    s_l = 1.0 - s_v
    rho_l = float(result.rho_liquid) if result.rho_liquid is not None else 0.0
    rho_v = float(result.rho_vapor) if result.rho_vapor is not None else 0.0
    if result.phase_state == "liquid" and mixture.Mw is not None:
        rho_l = float((x @ mixture.Mw) / v_l)
        rho_v = rho_l
    elif result.phase_state == "vapor" and mixture.Mw is not None:
        rho_v = float((y @ mixture.Mw) / v_v)
        rho_l = rho_v
    return CellFlash(
        z=result.z.copy(),
        x=x,
        y=y,
        nu=nu,
        S_liquid=s_l,
        S_vapor=s_v,
        xi_liquid=1.0 / v_l,
        xi_vapor=1.0 / v_v,
        v_liquid=v_l,
        v_vapor=v_v,
        rho_liquid=rho_l,
        rho_vapor=rho_v,
        phase_state=result.phase_state,
        marker=mixture.marker,
    )


def component_moles(cell: CellFlash, pore_volume: float) -> NDArray[np.float64]:
    """``n_i = V_pore (ξ_L S_L x_i + ξ_V S_V y_i)`` in mol."""
    if pore_volume < 0.0:
        raise ValueError("pore volume must be non-negative (m³)")
    return float(pore_volume) * (
        cell.xi_liquid * cell.S_liquid * cell.x + cell.xi_vapor * cell.S_vapor * cell.y
    )
