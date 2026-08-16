"""Two-phase relative permeability. Corey is the P0 model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.exceptions import InvalidSaturation


@dataclass(frozen=True)
class CoreyTwoPhase:
    swi: float = 0.20
    sor: float = 0.20
    krw0: float = 1.0
    kro0: float = 1.0
    nw: float = 2.0
    no: float = 2.0
    mu_w: float = 1.0e-3
    mu_o: float = 5.0e-3

    def __post_init__(self) -> None:
        if self.swi < 0.0 or self.sor < 0.0 or self.swi + self.sor >= 1.0:
            raise InvalidSaturation("swi, sor must be >= 0 and swi+sor < 1")
        if min(self.krw0, self.kro0, self.nw, self.no, self.mu_w, self.mu_o) <= 0.0:
            raise ValueError("Corey endpoints, exponents, and viscosities must be positive")

    def se(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        sw_arr = np.asarray(sw, dtype=float)
        denom = 1.0 - self.swi - self.sor
        return np.clip((sw_arr - self.swi) / denom, 0.0, 1.0)

    def kr(self, sw: NDArray[np.float64] | float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        se = self.se(sw)
        krw = self.krw0 * np.power(se, self.nw)
        kro = self.kro0 * np.power(1.0 - se, self.no)
        return krw, kro

    def mobility(self, sw: NDArray[np.float64] | float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        krw, kro = self.kr(sw)
        lw = krw / self.mu_w
        lo = kro / self.mu_o
        return lw, lo, lw + lo

    def fractional_flow(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        lw, _lo, lt = self.mobility(sw)
        return np.divide(lw, lt, out=np.zeros_like(lt, dtype=float), where=lt > 0.0)


@dataclass(frozen=True)
class TableTwoPhase:
    """Tabulated two-phase kr (MRST ``tabulatedSatFunc`` / IMEX *SWT)."""

    sw: NDArray[np.float64]
    krw: NDArray[np.float64]
    kro: NDArray[np.float64]
    mu_w: float = 1.0e-3
    mu_o: float = 5.0e-3

    def __post_init__(self) -> None:
        sw = np.asarray(self.sw, dtype=float).ravel()
        krw = np.asarray(self.krw, dtype=float).ravel()
        kro = np.asarray(self.kro, dtype=float).ravel()
        if sw.size < 2 or sw.size != krw.size or sw.size != kro.size:
            raise ValueError("relperm table columns must align and have >= 2 rows")
        if np.any(np.diff(sw) <= 0.0):
            raise ValueError("relperm Sw table must be strictly increasing")
        object.__setattr__(self, "sw", sw)
        object.__setattr__(self, "krw", np.clip(krw, 0.0, None))
        object.__setattr__(self, "kro", np.clip(kro, 0.0, None))
        if min(self.mu_w, self.mu_o) <= 0.0:
            raise ValueError("table viscosities must be positive")

    @property
    def swi(self) -> float:
        return float(self.sw[0])

    @property
    def sor(self) -> float:
        wet = np.where(self.kro <= 1.0e-14)[0]
        sw_or = float(self.sw[wet[0]]) if wet.size else float(self.sw[-1])
        return max(0.0, 1.0 - sw_or)

    def kr(self, sw: NDArray[np.float64] | float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.clip(np.asarray(sw, dtype=float), float(self.sw[0]), float(self.sw[-1]))
        return np.interp(s, self.sw, self.krw), np.interp(s, self.sw, self.kro)

    def mobility(self, sw: NDArray[np.float64] | float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        krw, kro = self.kr(sw)
        lw = krw / self.mu_w
        lo = kro / self.mu_o
        return lw, lo, lw + lo

    def fractional_flow(self, sw: NDArray[np.float64] | float) -> NDArray[np.float64]:
        lw, _lo, lt = self.mobility(sw)
        return np.divide(lw, lt, out=np.zeros_like(lt, dtype=float), where=lt > 0.0)

    @classmethod
    def cmg_seawater(cls, *, mu_w: float = 1.1e-3, mu_o: float = 0.64e-3) -> TableTwoPhase:
        """IMEX *SWT from the virtual-experiment decks (not inverted)."""
        return cls(
            sw=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.00]),
            krw=np.array([0.00, 0.07, 0.15, 0.24, 0.33, 0.65, 0.83, 1.00]),
            kro=np.array([1.00, 0.40, 0.125, 0.0649, 0.0048, 0.00, 0.00, 0.00]),
            mu_w=float(mu_w),
            mu_o=float(mu_o),
        )


@dataclass(frozen=True)
class CoreyThreePhase:
    """Independent Corey three-phase model (P1 baseline, not Stone).

    ``So = 1 - Sw - Sg``. Residuals must satisfy ``swi + sor + sgr < 1``.
    """

    swi: float = 0.20
    sor: float = 0.15
    sgr: float = 0.05
    krw0: float = 1.0
    kro0: float = 1.0
    krg0: float = 1.0
    nw: float = 2.0
    no: float = 2.0
    ng: float = 2.0
    mu_w: float = 1.0e-3
    mu_o: float = 5.0e-3
    mu_g: float = 2.0e-5

    def __post_init__(self) -> None:
        if min(self.swi, self.sor, self.sgr) < 0.0 or self.swi + self.sor + self.sgr >= 1.0:
            raise InvalidSaturation("three-phase residuals must be >= 0 and sum to < 1")
        if min(self.krw0, self.kro0, self.krg0, self.nw, self.no, self.ng, self.mu_w, self.mu_o, self.mu_g) <= 0.0:
            raise ValueError("Corey three-phase parameters must be positive")

    def _se(self, sw, sg) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        sw_a = np.asarray(sw, dtype=float)
        sg_a = np.asarray(sg, dtype=float)
        denom = 1.0 - self.swi - self.sor - self.sgr
        sew = np.clip((sw_a - self.swi) / denom, 0.0, 1.0)
        seg = np.clip((sg_a - self.sgr) / denom, 0.0, 1.0)
        seo = np.clip(1.0 - sew - seg, 0.0, 1.0)
        return sew, seo, seg

    def kr(
        self, sw: NDArray[np.float64] | float, sg: NDArray[np.float64] | float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        sew, seo, seg = self._se(sw, sg)
        return (
            self.krw0 * np.power(sew, self.nw),
            self.kro0 * np.power(seo, self.no),
            self.krg0 * np.power(seg, self.ng),
        )

    def mobility(
        self, sw: NDArray[np.float64] | float, sg: NDArray[np.float64] | float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        krw, kro, krg = self.kr(sw, sg)
        lw = krw / self.mu_w
        lo = kro / self.mu_o
        lg = krg / self.mu_g
        return lw, lo, lg, lw + lo + lg

    def fractional_flow(
        self, sw: NDArray[np.float64] | float, sg: NDArray[np.float64] | float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        lw, lo, lg, lt = self.mobility(sw, sg)
        fw = np.divide(lw, lt, out=np.zeros_like(lt), where=lt > 0.0)
        fo = np.divide(lo, lt, out=np.zeros_like(lt), where=lt > 0.0)
        fg = np.divide(lg, lt, out=np.zeros_like(lt), where=lt > 0.0)
        return fw, fo, fg
