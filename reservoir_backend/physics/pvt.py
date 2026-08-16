"""Black-oil PVT: reciprocal FVF b=1/B, rock/fluid compressibility.

Follows MRST ``TwoPhaseOilWater`` / ``getFluxAndProps*_BO``:

    water:  (1/dt) (pv bW Sw − pv0 bW0 Sw0) + Div(bW vW) = qW^s
    oil:    (1/dt) (pv bO So − pv0 bO0 So0) + Div(bO vO) = qO^s

``bα(p) = (1 + cα (p − pref)) / Bα_ref`` is the IMEX linear form
(``*BWI``, ``*CW``, undersaturated ``*CO``). Dead oil (Rs unused in the
two-phase conservation). Live-oil Rs is stored only so a later gas
equation can pick it up; it does not change oil/water mass.

Not inverted. Fluid of the experiment is known. θ stays log K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PSI = 6894.757293168
SCF_PER_STB = 0.17810760667903525  # sm3/sm3 per scf/stb
LB_FT3 = 16.01846337396014  # kg/m3


@dataclass(frozen=True)
class BlackOilPVT:
    """Slightly compressible dead-oil / undersaturated water–oil PVT."""

    bw_ref: float = 1.0
    bo_ref: float = 1.0
    bg_ref: float = 1.0
    cw: float = 0.0
    co: float = 0.0
    cg: float = 0.0
    cr: float = 0.0
    pref_w: float = 1.0e5
    pref_o: float = 1.0e5
    pref_r: float = 1.0e5
    mu_w: float = 1.0e-3
    mu_o: float = 5.0e-3
    mu_g: float = 2.0e-5
    rs_sc: float = 0.0
    rho_w_sc: float = 1000.0
    rho_o_sc: float = 800.0
    rho_g_sc: float = 1.0

    def _b(self, p: NDArray[np.float64] | float, c: float, pref: float, bref: float) -> NDArray[np.float64]:
        return (1.0 + float(c) * (np.asarray(p, dtype=float) - float(pref))) / max(float(bref), 1.0e-30)

    def b_w(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return self._b(p, self.cw, self.pref_w, self.bw_ref)

    def b_o(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return self._b(p, self.co, self.pref_o, self.bo_ref)

    def b_g(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return self._b(p, self.cg, self.pref_o, self.bg_ref)

    def pv_mult(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return 1.0 + float(self.cr) * (np.asarray(p, dtype=float) - float(self.pref_r))

    def ct(
        self,
        sw: NDArray[np.float64] | float,
        so: NDArray[np.float64] | float | None = None,
        sg: NDArray[np.float64] | float | None = None,
    ) -> NDArray[np.float64]:
        sw_a = np.asarray(sw, dtype=float)
        sg_a = 0.0 if sg is None else np.asarray(sg, dtype=float)
        so_a = (1.0 - sw_a - sg_a) if so is None else np.asarray(so, dtype=float)
        return self.cr + sw_a * self.cw + so_a * self.co + sg_a * self.cg

    def has_storage(self) -> bool:
        return max(abs(self.cw), abs(self.co), abs(self.cg), abs(self.cr)) > 0.0

    @classmethod
    def incompressible(cls, *, mu_w: float = 1.0e-3, mu_o: float = 5.0e-3, mu_g: float = 2.0e-5) -> BlackOilPVT:
        return cls(mu_w=float(mu_w), mu_o=float(mu_o), mu_g=float(mu_g))

    @classmethod
    def slightly_compressible(
        cls,
        ct: float,
        *,
        pref: float = 1.0e5,
        mu_w: float = 1.0e-3,
        mu_o: float = 5.0e-3,
    ) -> BlackOilPVT:
        """Uniform total compressibility parked on the rock (old scalar ct)."""
        return cls(cr=float(ct), pref_r=float(pref), pref_w=float(pref), pref_o=float(pref), mu_w=mu_w, mu_o=mu_o)

    @classmethod
    def cmg_seawater(cls, *, p_init: float = 3000.0 * PSI, pb: float = 2500.0 * PSI) -> BlackOilPVT:
        """IMEX SPE-style *PVT used by the virtual-experiment decks.

        Undersaturated oil: Bob = 1.5 at *PB, *CO = 1.3687e-5 /psi.
        Water: *BWI = 1.04 at *REFPW = 14.7 psi, *CW = 3.04e-6 /psi.
        Rock: *CPOR = 3.0e-6 /psi at *PRPOR = 14.7 psi.
        Viscosity: SVISC 1.1 cP water, table viso ≈ 0.64 cP near pb.
        """
        return cls(
            bw_ref=1.04,
            bo_ref=1.50,
            cw=3.04e-6 / PSI,
            co=1.3687e-5 / PSI,
            cr=3.0e-6 / PSI,
            pref_w=14.7 * PSI,
            pref_o=float(pb),
            pref_r=14.7 * PSI,
            mu_w=1.1e-3,
            mu_o=0.64e-3,
            rs_sc=775.0 * SCF_PER_STB,
            rho_w_sc=62.238 * LB_FT3,
            rho_o_sc=46.244 * LB_FT3,
            rho_g_sc=0.0647 * LB_FT3,
        )
