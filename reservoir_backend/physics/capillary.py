"""Capillary pressure models. Lab cases must pick one explicitly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidSaturation


@dataclass(frozen=True)
class NoCapillary:
    name: str = "none"

    def pc(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return np.zeros_like(np.asarray(sw, dtype=float), dtype=float)

    def dpc_dsw(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return np.zeros_like(np.asarray(sw, dtype=float), dtype=float)


@dataclass(frozen=True)
class BrooksCorey:
    """Pc = Pe * Se^(-1/lambda)."""

    entry_pressure: float = 2.0e3
    lambda_pc: float = 2.0
    swi: float = 0.20
    sor: float = 0.20
    name: str = "brooks_corey"

    def __post_init__(self) -> None:
        if self.entry_pressure <= 0.0 or self.lambda_pc <= 0.0:
            raise ValueError("Brooks-Corey Pe and lambda must be positive")
        if self.swi < 0.0 or self.sor < 0.0 or self.swi + self.sor >= 1.0:
            raise InvalidSaturation("invalid residual saturations")

    def se(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        sw_arr = np.asarray(sw, dtype=float)
        denom = 1.0 - self.swi - self.sor
        return np.clip((sw_arr - self.swi) / denom, 1.0e-8, 1.0)

    def pc(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        se = self.se(sw)
        return self.entry_pressure * np.power(se, -1.0 / self.lambda_pc)

    def dpc_dsw(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        se = self.se(sw)
        denom = 1.0 - self.swi - self.sor
        dpc_dse = self.entry_pressure * (-1.0 / self.lambda_pc) * np.power(se, -1.0 / self.lambda_pc - 1.0)
        return dpc_dse / denom


@dataclass(frozen=True)
class VanGenuchten:
    p0: float = 2.0e3
    m: float = 0.5
    n: float = 2.0
    swi: float = 0.20
    sor: float = 0.20
    name: str = "van_genuchten"

    def se(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        sw_arr = np.asarray(sw, dtype=float)
        denom = 1.0 - self.swi - self.sor
        return np.clip((sw_arr - self.swi) / denom, 1.0e-8, 1.0)

    def pc(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        se = self.se(sw)
        term = np.maximum(np.power(se, -1.0 / self.m) - 1.0, 0.0)
        return self.p0 * np.power(term, 1.0 / self.n)

    def dpc_dsw(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        # numeric derivative is enough for diagnostics
        sw_arr = np.asarray(sw, dtype=float)
        eps = 1.0e-5
        return (self.pc(sw_arr + eps) - self.pc(sw_arr - eps)) / (2.0 * eps)


@dataclass(frozen=True)
class TableCapillary:
    """Pc(Sw) table. IMEX *SWT last column stored in Pascals."""

    sw: NDArray[np.float64]
    pc_pa: NDArray[np.float64]
    name: str = "table"

    def __post_init__(self) -> None:
        sw = np.asarray(self.sw, dtype=float).ravel()
        pc = np.asarray(self.pc_pa, dtype=float).ravel()
        if sw.size < 2 or sw.size != pc.size:
            raise ValueError("Pc table must align and have >= 2 rows")
        if np.any(np.diff(sw) <= 0.0):
            raise ValueError("Pc Sw table must be strictly increasing")
        object.__setattr__(self, "sw", sw)
        object.__setattr__(self, "pc_pa", pc)

    def pc(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        s = np.clip(np.asarray(sw, dtype=float), float(self.sw[0]), float(self.sw[-1]))
        return np.interp(s, self.sw, self.pc_pa)

    def dpc_dsw(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        sw_arr = np.asarray(sw, dtype=float)
        eps = 1.0e-5
        return (self.pc(sw_arr + eps) - self.pc(sw_arr - eps)) / (2.0 * eps)

    @classmethod
    def cmg_swt(cls) -> TableCapillary:
        psi = 6894.757293168
        return cls(
            sw=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.00]),
            pc_pa=np.array([7.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5, 0.0]) * psi,
            name="cmg_swt",
        )


def capillary_from_name(name: str, **kwargs) -> NoCapillary | BrooksCorey | VanGenuchten | TableCapillary:
    key = str(name).strip().lower()
    if key in {"none", "off", "no", "nocapillary"}:
        return NoCapillary()
    if key in {"brooks_corey", "brookscorey", "bc"}:
        return BrooksCorey(**{k: v for k, v in kwargs.items() if k in BrooksCorey.__dataclass_fields__})
    if key in {"van_genuchten", "vangenuchten", "vg"}:
        return VanGenuchten(**{k: v for k, v in kwargs.items() if k in VanGenuchten.__dataclass_fields__})
    if key in {"table", "cmg_swt", "swt"}:
        return TableCapillary.cmg_swt()
    raise ValueError(f"unknown capillary model: {name}")
