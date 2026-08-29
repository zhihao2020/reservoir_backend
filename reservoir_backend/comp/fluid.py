"""Compositional fluid spec. EXAMPLE only unless a real gem_deck is given."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.pr import PengRobinson


@dataclass
class CompSpec:
    """Isothermal hydrocarbon VL flash. Optional immiscible water (not in the EOS)."""

    eos: PengRobinson
    temperature_k: float = 350.0
    mu_liquid: float = 3.0e-4
    mu_vapor: float = 2.0e-5
    sorg: float = 0.15
    sgr: float = 0.02
    kro0: float = 1.0
    krg0: float = 1.0
    no: float = 2.0
    ng: float = 2.0
    z_init: NDArray[np.float64] = field(default_factory=lambda: np.array([0.55, 0.45], dtype=float))
    z_inj: NDArray[np.float64] = field(default_factory=lambda: np.array([0.95, 0.05], dtype=float))
    has_water: bool = False
    sw_init: float = 0.0
    mu_water: float = 5.0e-4
    vw0: float = 1.8068e-5
    cw: float = 4.5e-10
    p_ref: float = 1.0e5
    swc: float = 0.20
    krw0: float = 0.40
    nw: float = 2.0

    def __post_init__(self) -> None:
        self.z_init = _frac(self.z_init, self.eos.nc)
        self.z_inj = _frac(self.z_inj, self.eos.nc)
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if min(self.mu_liquid, self.mu_vapor, self.kro0, self.krg0) <= 0.0:
            raise ValueError("viscosities and kr endpoints must be positive")
        if self.has_water:
            if min(self.mu_water, self.vw0, self.krw0, self.nw) <= 0.0:
                raise ValueError("water viscosity, vw0, and krw0 must be positive")
            self.sw_init = float(np.clip(self.sw_init, 0.0, 0.95))

    @property
    def n_hc(self) -> int:
        return int(self.eos.nc)

    @property
    def nc(self) -> int:
        """Moles per cell: hydrocarbon components, plus water if enabled."""
        return self.n_hc + (1 if self.has_water else 0)

    def water_vw(self, pressure: float | NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.asarray(pressure, dtype=float)
        b = 1.0 + float(self.cw) * (p - float(self.p_ref))
        return float(self.vw0) / np.maximum(b, 1.0e-8)


def _frac(z: NDArray[np.float64], nc: int) -> NDArray[np.float64]:
    z = np.asarray(z, dtype=float).ravel()
    if z.size != nc:
        raise ValueError(f"z size {z.size} != nc {nc}")
    z = np.maximum(z, 0.0)
    s = float(np.sum(z))
    if s <= 0.0:
        raise ValueError("composition sums to 0")
    return z / s


def fluid_from_name(name: str, **kwargs) -> CompSpec:
    key = str(name).strip().lower()
    if key in {"example", "c1_nc10", "example_c1_nc10"}:
        return CompSpec(eos=example_c1_nc10(), **kwargs)
    raise ValueError(
        f"unknown compositional fluid {name!r}; use 'example' or supply fluid.gem_deck "
        "(refuse invented Jiyang Tc/Pc)"
    )
