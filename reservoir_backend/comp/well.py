"""Peaceman-style rate-controlled injector for the standalone comp kernel.

Well index (vertical well, isotropic Cartesian cell, skin ``s``):

    r_e = 0.14 * sqrt(dx² + dy²)     Peaceman, SPEJ 1983
    WI  = 2 π k h / (ln(r_e / r_w) + s)

``k`` in m², ``h = dz`` in m, so ``WI`` is m³ (same convention as the lab
ports: ``WI λ Δp`` would be m³/s). First cut is **rate-controlled
injection** only: specified molar rate of an EXAMPLE stream is added to
the well cell. Not BHP-controlled, not a producer, not industrial-grade,
not a GEM well. Do not import from ``solver/fi.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.peng_robinson import EosMixture
from reservoir_backend.grid.cartesian import CartesianGrid

# Peaceman (1983) isotropic equivalent-radius factor.
_PEACEMAN_RE = 0.14


@dataclass(frozen=True)
class RateInjector:
    """Specified molar injection into one cell. ``rate`` is mol/s (>0)."""

    cell: int
    rate: float
    z_inj: NDArray[np.float64]
    well_index: float
    r_e: float
    r_w: float
    marker: str = ""


def peaceman_equivalent_radius(dx: float, dy: float) -> float:
    """``r_e = 0.14 sqrt(dx² + dy²)`` for an isotropic Cartesian cell."""
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive (m)")
    return _PEACEMAN_RE * float(np.sqrt(dx * dx + dy * dy))


def peaceman_wi(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    *,
    r_w: float | None = None,
    skin: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(WI [m³], r_e [m], r_w [m])`` for a vertical well in ``cell``.

    Default ``r_w`` is 10% of min(dx, dy), clipped below ``r_e``.
    """
    i, j, k = grid.ijk(int(cell))
    dx, dy, dz = float(grid.dx[i]), float(grid.dy[j]), float(grid.dz[k])
    r_e = peaceman_equivalent_radius(dx, dy)
    rw = float(r_w) if r_w is not None else 0.10 * min(dx, dy)
    if rw <= 0.0:
        raise ValueError("r_w must be positive (m)")
    if rw >= r_e:
        raise ValueError(f"r_w={rw} must be < Peaceman r_e={r_e}")
    denom = float(np.log(r_e / rw) + skin)
    if denom <= 0.0:
        raise ValueError("ln(r_e/r_w) + skin must be positive")
    if permeability < 0.0:
        raise ValueError("permeability must be non-negative (m²)")
    wi = 2.0 * np.pi * float(permeability) * dz / denom
    return float(wi), r_e, rw


def example_rate_injector(
    grid: CartesianGrid,
    cell: int,
    permeability: float,
    mixture: EosMixture,
    *,
    rate: float,
    stream: str = "CO2",
    r_w: float | None = None,
    skin: float = 0.0,
) -> RateInjector:
    """Rate-controlled injector of an EXAMPLE library stream (pure CO2 or named)."""
    if rate < 0.0:
        raise ValueError("injection rate must be non-negative (mol/s)")
    if stream not in mixture.names:
        raise ValueError(f"{stream!r} is not in the EXAMPLE mixture")
    z_inj = np.zeros(mixture.n_components, dtype=float)
    z_inj[mixture.names.index(stream)] = 1.0
    wi, r_e, rw = peaceman_wi(grid, cell, permeability, r_w=r_w, skin=skin)
    return RateInjector(
        cell=int(cell),
        rate=float(rate),
        z_inj=z_inj,
        well_index=wi,
        r_e=r_e,
        r_w=rw,
        marker=mixture.marker,
    )


def injection_moles(injector: RateInjector, dt: float) -> NDArray[np.float64]:
    """Component moles added in ``dt`` seconds."""
    if dt < 0.0:
        raise ValueError("dt must be non-negative (s)")
    z = np.asarray(injector.z_inj, dtype=float)
    total = float(z.sum())
    if total <= 0.0:
        raise ValueError("injector stream sums to zero")
    return float(injector.rate) * float(dt) * (z / total)
