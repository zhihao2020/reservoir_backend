"""Fluid property interface. FlowSolver must not embed a specific EOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import FlashCalculationError
from reservoir_backend.physics.pvt import BlackOilPVT


@dataclass
class FluidProperties:
    """Phase properties at one (p, T, z). SI: density kg/m³, viscosity Pa·s."""

    density: NDArray[np.float64]
    viscosity: NDArray[np.float64]
    phase_fraction: NDArray[np.float64]
    phase_composition: NDArray[np.float64] | None = None
    two_phase: bool = False


class FluidModel(Protocol):
    def evaluate(
        self,
        pressure: float | NDArray[np.float64],
        temperature: float,
        composition: NDArray[np.float64] | None = None,
    ) -> FluidProperties: ...


@dataclass
class SimpleFluidModel:
    """Black-oil / incompressible adapter. Composition is ignored."""

    pvt: BlackOilPVT
    temperature_k: float = 293.15

    def evaluate(
        self,
        pressure: float | NDArray[np.float64],
        temperature: float,
        composition: NDArray[np.float64] | None = None,
    ) -> FluidProperties:
        p = np.asarray(pressure, dtype=float)
        bw = np.asarray(self.pvt.b_w(p), dtype=float)
        bo = np.asarray(self.pvt.b_o(p), dtype=float)
        rho_w = float(self.pvt.rho_w_sc) * bw
        rho_o = float(self.pvt.rho_o_sc) * bo
        mu_w = np.asarray(self.pvt.mu_w_of(p) if hasattr(self.pvt, "mu_w_of") else self.pvt.mu_w, dtype=float)
        mu_o = np.asarray(self.pvt.mu_o_of(p) if hasattr(self.pvt, "mu_o_of") else self.pvt.mu_o, dtype=float)
        dens = np.stack([np.broadcast_to(rho_w, p.shape), np.broadcast_to(rho_o, p.shape)], axis=-1)
        visc = np.stack([np.broadcast_to(mu_w, p.shape), np.broadcast_to(mu_o, p.shape)], axis=-1)
        frac = np.stack([np.full(p.shape, 0.5), np.full(p.shape, 0.5)], axis=-1)
        _ = temperature
        _ = composition
        return FluidProperties(density=dens, viscosity=visc, phase_fraction=frac, two_phase=False)


@dataclass
class PengRobinsonFluidModel:
    """Compositional adapter around the existing PT flash."""

    eos: object
    mu_liquid: float = 3.0e-4
    mu_vapor: float = 2.0e-5

    def evaluate(
        self,
        pressure: float | NDArray[np.float64],
        temperature: float,
        composition: NDArray[np.float64] | None = None,
    ) -> FluidProperties:
        from reservoir_backend.eos.flash import flash_tp
        from reservoir_backend.eos.pr import R_GAS

        if composition is None:
            raise FlashCalculationError("PengRobinsonFluidModel needs composition z")
        p = float(np.asarray(pressure, dtype=float).ravel()[0])
        z = np.asarray(composition, dtype=float).ravel()
        try:
            flash = flash_tp(self.eos, p, float(temperature), z)
        except Exception as exc:
            raise FlashCalculationError(str(exc)) from exc
        rt = float(R_GAS) * float(temperature)
        rho_l = p / max(flash.z_liq * rt, 1.0e-30)
        rho_v = p / max(flash.z_vap * rt, 1.0e-30)
        dens = np.array([rho_l, rho_v], dtype=float)
        visc = np.array([self.mu_liquid, self.mu_vapor], dtype=float)
        frac = np.array([1.0 - flash.vapor_frac, flash.vapor_frac], dtype=float)
        xy = np.stack([flash.x, flash.y], axis=0)
        return FluidProperties(
            density=dens,
            viscosity=visc,
            phase_fraction=frac,
            phase_composition=xy,
            two_phase=bool(flash.two_phase),
        )
