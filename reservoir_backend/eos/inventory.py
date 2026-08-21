"""Standalone component-mass inventory after a TP flash.

Given feed ``z``, ``T`` [K], ``p`` [Pa] and a basis of 1 mol or 1 kg of feed,
returns phase mole fractions, vapor fraction ``V``, phase masses, and
per-component moles. Material balance: ``n_liquid + n_vapor = n_feed``.

This is a kernel helper, not a FIM accumulation term. Do not import it from
``solver/fi.py`` or ``physics/pvt.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.flash import FlashResult, flash_tp
from reservoir_backend.eos.peng_robinson import EosMixture, _normalize_composition, molar_mass


@dataclass(frozen=True)
class PhaseInventory:
    """Component moles and phase masses for one flash at a stated basis."""

    T: float
    p: float
    basis: str  # "mol" (1 mol feed) | "kg" (1 kg feed)
    z: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    V: float
    feed_moles: float
    feed_mass: float
    n_feed: NDArray[np.float64]
    n_liquid: NDArray[np.float64]
    n_vapor: NDArray[np.float64]
    mass_liquid: float
    mass_vapor: float
    phase_state: str
    marker: str = ""
    flash: FlashResult | None = None

    def mole_balance_residual(self) -> NDArray[np.float64]:
        return self.n_feed - (self.n_liquid + self.n_vapor)


def component_inventory(
    z: NDArray[np.float64] | float,
    T: float,
    p: float,
    mixture: EosMixture,
    *,
    basis: str = "mol",
) -> PhaseInventory:
    """Flash ``z`` and report phase / component inventory.

    ``basis='mol'`` uses 1 mol of feed; ``basis='kg'`` uses 1 kg of feed.
    ``mixture.Mw`` must be set (kg/mol). EXAMPLE library values are public
    literature, not a Jiyang / GEM card.
    """
    if basis not in ("mol", "kg"):
        raise ValueError("basis must be 'mol' (1 mol feed) or 'kg' (1 kg feed)")
    if mixture.Mw is None:
        raise ValueError("mixture.Mw (kg/mol) is required for component inventory")
    z_arr = _normalize_composition(z, mixture.n_components)
    flashed = flash_tp(z_arr, T, p, mixture)
    m_feed_per_mol = molar_mass(flashed.z, mixture)
    n_total = 1.0 if basis == "mol" else 1.0 / m_feed_per_mol
    n_feed = n_total * flashed.z
    n_liquid = n_total * (1.0 - flashed.V) * flashed.x
    n_vapor = n_total * flashed.V * flashed.y
    mass_liquid = float(n_liquid @ mixture.Mw)
    mass_vapor = float(n_vapor @ mixture.Mw)
    return PhaseInventory(
        T=float(T),
        p=float(p),
        basis=basis,
        z=flashed.z.copy(),
        x=flashed.x.copy(),
        y=flashed.y.copy(),
        V=float(flashed.V),
        feed_moles=float(n_total),
        feed_mass=float(n_total * m_feed_per_mol),
        n_feed=n_feed,
        n_liquid=n_liquid,
        n_vapor=n_vapor,
        mass_liquid=mass_liquid,
        mass_vapor=mass_vapor,
        phase_state=flashed.phase_state,
        marker=mixture.marker,
        flash=flashed,
    )
