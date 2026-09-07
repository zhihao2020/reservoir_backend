"""Forward trajectory types shared by compositional FIM (single and DPDP)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import State


@dataclass
class MassBalance:
    initial_mass: float
    final_mass: float
    injected_mass: float
    produced_mass: float
    boundary_flux: float
    balance_error: float
    relative_balance_error: float
    gas_initial_mass: float = 0.0
    gas_final_mass: float = 0.0
    gas_injected_mass: float = 0.0
    gas_produced_mass: float = 0.0
    gas_balance_error: float = 0.0
    gas_relative_balance_error: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "initial_mass": self.initial_mass,
            "final_mass": self.final_mass,
            "injected_mass": self.injected_mass,
            "produced_mass": self.produced_mass,
            "boundary_flux": self.boundary_flux,
            "balance_error": self.balance_error,
            "relative_balance_error": self.relative_balance_error,
            "gas_initial_mass": self.gas_initial_mass,
            "gas_final_mass": self.gas_final_mass,
            "gas_injected_mass": self.gas_injected_mass,
            "gas_produced_mass": self.gas_produced_mass,
            "gas_balance_error": self.gas_balance_error,
            "gas_relative_balance_error": self.gas_relative_balance_error,
        }


@dataclass
class StepReport:
    time_s: float
    dt: float
    max_cfl: float
    max_ds: float
    mass: MassBalance
    port_rates: dict[str, float]
    notes: list[str] = field(default_factory=list)
    newton_its: int | None = None


@dataclass
class Trajectory:
    times_s: NDArray[np.float64]
    states: list[State]
    reports: list[StepReport]
    port_rates: list[dict[str, float]]
    port_bhp: list[dict[str, float]] = field(default_factory=list)

    def state_at(self, t: float) -> State:
        times = np.asarray(self.times_s, dtype=float)
        if times.size == 0:
            raise ValueError("empty trajectory")
        idx = int(np.argmin(np.abs(times - float(t))))
        return self.states[idx]

    def rates_and_bhp_at(self, t: float) -> tuple[dict[str, float], dict[str, float]]:
        """Last stored well report at or before ``t`` (step-post / GEM DATE)."""
        times = np.asarray(self.times_s, dtype=float)
        if times.size == 0:
            return {}, {}
        idx = int(np.searchsorted(times, float(t), side="right") - 1)
        idx = int(np.clip(idx, 0, times.size - 1))
        rates = self.port_rates[idx] if idx < len(self.port_rates) else {}
        bhp = self.port_bhp[idx] if idx < len(self.port_bhp) else {}
        return rates, bhp
