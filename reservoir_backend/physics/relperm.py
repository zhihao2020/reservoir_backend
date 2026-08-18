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
    """Tabulated two-phase kr (IMEX *SWT)."""

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


@dataclass(frozen=True)
class TableThreePhase:
    """IMEX *SWT + *SLT. Default oil rule is Stone II.

    Sl = 1 − Sg. Not inverted; tables are the experiment fluid.
    """

    sw: NDArray[np.float64]
    krw: NDArray[np.float64]
    krow: NDArray[np.float64]
    sl: NDArray[np.float64]
    krg: NDArray[np.float64]
    krog: NDArray[np.float64]
    mu_w: float = 1.1e-3
    mu_o: float = 0.64e-3
    mu_g: float = 2.08e-5
    oil_rule: str = "stone2"

    def __post_init__(self) -> None:
        sw = np.asarray(self.sw, dtype=float).ravel()
        krw = np.asarray(self.krw, dtype=float).ravel()
        krow = np.asarray(self.krow, dtype=float).ravel()
        sl = np.asarray(self.sl, dtype=float).ravel()
        krg = np.asarray(self.krg, dtype=float).ravel()
        krog = np.asarray(self.krog, dtype=float).ravel()
        if sw.size < 2 or sw.size != krw.size or sw.size != krow.size:
            raise ValueError("SWT columns must align and have >= 2 rows")
        if sl.size < 2 or sl.size != krg.size or sl.size != krog.size:
            raise ValueError("SLT columns must align and have >= 2 rows")
        if np.any(np.diff(sw) <= 0.0) or np.any(np.diff(sl) <= 0.0):
            raise ValueError("SWT/SLT saturations must be strictly increasing")
        object.__setattr__(self, "sw", sw)
        object.__setattr__(self, "krw", np.clip(krw, 0.0, None))
        object.__setattr__(self, "krow", np.clip(krow, 0.0, None))
        object.__setattr__(self, "sl", sl)
        object.__setattr__(self, "krg", np.clip(krg, 0.0, None))
        object.__setattr__(self, "krog", np.clip(krog, 0.0, None))
        if min(self.mu_w, self.mu_o, self.mu_g) <= 0.0:
            raise ValueError("table viscosities must be positive")
        rule = str(self.oil_rule).strip().lower()
        if rule not in {"product", "stone1", "stone2", "baker"}:
            raise ValueError(f"unknown oil_rule {self.oil_rule}")
        object.__setattr__(self, "oil_rule", rule)

    @property
    def swi(self) -> float:
        return float(self.sw[0])

    @property
    def sgr(self) -> float:
        wet = np.where(self.krg[::-1] > 1.0e-14)[0]
        sl_g = float(self.sl[::-1][wet[0]]) if wet.size else float(self.sl[0])
        return max(0.0, 1.0 - sl_g)

    def kr(
        self, sw: NDArray[np.float64] | float, sg: NDArray[np.float64] | float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        sw_a = np.clip(np.asarray(sw, dtype=float), float(self.sw[0]), float(self.sw[-1]))
        sg_a = np.clip(np.asarray(sg, dtype=float), 0.0, 1.0)
        sl_a = np.clip(1.0 - sg_a, float(self.sl[0]), float(self.sl[-1]))
        krw = np.interp(sw_a, self.sw, self.krw)
        krg = np.interp(sl_a, self.sl, self.krg)
        krow = np.interp(sw_a, self.sw, self.krow)
        krog = np.interp(sl_a, self.sl, self.krog)
        kro_cw = max(float(self.krow[0]), 1.0e-30)
        if self.oil_rule == "stone2":
            kro = kro_cw * ((krow / kro_cw + krw) * (krog / kro_cw + krg) - krw - krg)
            kro = np.clip(kro, 0.0, kro_cw)
        elif self.oil_rule == "stone1":
            swi = float(self.swi)
            sgr = float(self.sgr)
            sor = max(0.0, 1.0 - float(self.sw[np.where(self.krow <= 1.0e-14)[0][0]])) if np.any(self.krow <= 1.0e-14) else 0.20
            so = np.clip(1.0 - sw_a - sg_a, 0.0, 1.0)
            denom = max(1.0 - swi - sor - sgr, 1.0e-8)
            so_s = np.clip((so - sor) / denom, 0.0, 1.0)
            sw_s = np.clip((sw_a - swi) / max(1.0 - swi - sor, 1.0e-8), 0.0, 1.0)
            sg_s = np.clip(sg_a / max(1.0 - swi - sgr, 1.0e-8), 0.0, 1.0)
            kro = so_s * np.divide(krow, np.maximum(1.0 - sw_s, 1.0e-8)) * np.divide(krog, np.maximum(1.0 - sg_s, 1.0e-8))
            kro = np.clip(kro, 0.0, kro_cw)
        elif self.oil_rule == "baker":
            so = np.clip(1.0 - sw_a - sg_a, 0.0, 1.0)
            swi = float(self.swi)
            a = np.divide(so, np.maximum(so + sw_a - swi, 1.0e-8))
            b = np.divide(so, np.maximum(so + sg_a, 1.0e-8))
            kro = np.clip(a * krow + b * krog, 0.0, kro_cw)
        else:
            kro = krow * krog
        return krw, kro, krg

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

    @classmethod
    def cmg_seawater(
        cls, *, mu_w: float = 1.1e-3, mu_o: float = 0.64e-3, mu_g: float = 2.08e-5
    ) -> TableThreePhase:
        """IMEX *SWT + *SLT from the virtual-experiment decks."""
        return cls(
            sw=np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 1.00]),
            krw=np.array([0.00, 0.07, 0.15, 0.24, 0.33, 0.65, 0.83, 1.00]),
            krow=np.array([1.00, 0.40, 0.125, 0.0649, 0.0048, 0.00, 0.00, 0.00]),
            sl=np.array([0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.88, 0.95, 0.98, 0.999, 1.00]),
            krg=np.array([0.98, 0.94, 0.87, 0.72, 0.60, 0.41, 0.19, 0.125, 0.075, 0.025, 0.005, 0.00, 0.00, 0.00]),
            krog=np.array([0.00, 0.00, 1.0e-4, 0.001, 0.010, 0.021, 0.09, 0.20, 0.35, 0.70, 0.98, 0.997, 1.00, 1.00]),
            mu_w=float(mu_w),
            mu_o=float(mu_o),
            mu_g=float(mu_g),
        )
