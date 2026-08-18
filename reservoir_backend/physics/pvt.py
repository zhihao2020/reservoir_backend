"""Black-oil PVT: reciprocal FVF b=1/B, rock/fluid compressibility.

Black-oil surface-volume form:

    water:  (1/dt) (pv bW Sw − pv0 bW0 Sw0) + Div(bW vW) = qW^s
    oil:    (1/dt) (pv bO So − pv0 bO0 So0) + Div(bO vO) = qO^s

``bα(p) = (1 + cα (p − pref)) / Bα_ref`` is the IMEX linear form
(``*BWI``, ``*CW``, undersaturated ``*CO``). Two-phase oil-water is dead
oil (Rs unused). Three-phase live oil tracks dissolved gas:

    G^s = φ (b_g S_g + R_s b_o S_o)

and the IMPES pressure storage picks up ``S_o (b_o/b_g) dR_s/dp``
below the bubble point.

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
    """Slightly compressible water-oil PVT, optional live-oil tables.

    Linear undersaturated form when tables are absent. With IMEX *PVT
    tables, saturated Rs, Bo, Eg=1/Bg are interpolated; above pb oil is
    undersaturated (Rs capped, Bo from *CO).
    """

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
    pb: float | None = None
    p_tab: NDArray[np.float64] | None = None
    rs_tab: NDArray[np.float64] | None = None
    bo_tab: NDArray[np.float64] | None = None
    eg_tab: NDArray[np.float64] | None = None
    muo_tab: NDArray[np.float64] | None = None
    mug_tab: NDArray[np.float64] | None = None

    def _b(self, p: NDArray[np.float64] | float, c: float, pref: float, bref: float) -> NDArray[np.float64]:
        return (1.0 + float(c) * (np.asarray(p, dtype=float) - float(pref))) / max(float(bref), 1.0e-30)

    def _interp(self, p: NDArray[np.float64], xp: NDArray[np.float64], fp: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.interp(np.asarray(p, dtype=float), xp, fp)

    def rs(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.rs_tab is None:
            return np.full(p_a.shape, float(self.rs_sc), dtype=float)
        rs = self._interp(p_a, self.p_tab, self.rs_tab)
        if self.pb is not None:
            rs_b = float(np.interp(float(self.pb), self.p_tab, self.rs_tab))
            rs = np.minimum(rs, rs_b)
        return rs

    def b_w(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return self._b(p, self.cw, self.pref_w, self.bw_ref)

    def pbub_of_rs(self, rs: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Invert the saturated Rs(p) table to a bubble-point pressure."""
        rs_a = np.asarray(rs, dtype=float)
        if self.p_tab is None or self.rs_tab is None:
            pb = float(self.pb) if self.pb is not None else float(self.pref_o)
            return np.full(rs_a.shape, pb, dtype=float)
        p = np.asarray(self.p_tab, dtype=float)
        r = np.asarray(self.rs_tab, dtype=float)
        pb = float(self.pb) if self.pb is not None else float(np.max(p))
        keep = p <= pb + 50.0 * PSI
        p_s = p[keep]
        r_s = r[keep]
        order = np.argsort(r_s)
        r_s = r_s[order]
        p_s = p_s[order]
        r_s, idx = np.unique(r_s, return_index=True)
        p_s = p_s[idx]
        return np.interp(rs_a, r_s, p_s)

    def b_o(
        self,
        p: NDArray[np.float64] | float,
        rs: NDArray[np.float64] | float | None = None,
        saturated: NDArray[np.bool_] | bool | None = None,
    ) -> NDArray[np.float64]:
        """Oil shrinkage b_o=1/Bo.

        Saturated: table Bo(p). Undersaturated: Bo(Rs)/(1+co*(p-pb(Rs))),
        the *CO branch used when only a saturated curve is stored (same as
        interpolating PVTO in p, Rs). rs is None keeps the old
        pressure-only path (global pb).
        """
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.bo_tab is None:
            return self._b(p_a, self.co, self.pref_o, self.bo_ref)
        pb = float(self.pb) if self.pb is not None else float(self.pref_o)
        bo_sat = self._interp(p_a, self.p_tab, self.bo_tab)
        if rs is None:
            bo_b = float(np.interp(pb, self.p_tab, self.bo_tab))
            bo_unsat = bo_b / np.maximum(1.0 + float(self.co) * (p_a - pb), 1.0e-8)
            bo = np.where(p_a >= pb, bo_unsat, bo_sat)
            return 1.0 / np.maximum(bo, 1.0e-30)
        rs_a = np.broadcast_to(np.asarray(rs, dtype=float), p_a.shape)
        rs_sat = self.rs(p_a)
        if saturated is None:
            sat = (rs_a >= rs_sat - 1.0e-10) & (p_a <= pb)
        else:
            sat = np.broadcast_to(np.asarray(saturated, dtype=bool), p_a.shape)
        pb_rs = self.pbub_of_rs(rs_a)
        bo_bub = self._interp(pb_rs, self.p_tab, self.bo_tab)
        bo_unsat = bo_bub / np.maximum(1.0 + float(self.co) * (p_a - pb_rs), 1.0e-8)
        bo = np.where(sat, bo_sat, bo_unsat)
        return 1.0 / np.maximum(bo, 1.0e-30)

    def b_g(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is not None and self.eg_tab is not None:
            return self._interp(p_a, self.p_tab, self.eg_tab)
        return self._b(p_a, self.cg, self.pref_o, self.bg_ref)

    def viscosity_o(
        self,
        p: NDArray[np.float64] | float,
        rs: NDArray[np.float64] | float | None = None,
        saturated: NDArray[np.bool_] | bool | None = None,
    ) -> NDArray[np.float64]:
        """Oil viscosity. Saturated: table vs p. Undersaturated: table at pb(Rs)."""
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.muo_tab is None:
            return np.full(p_a.shape, float(self.mu_o), dtype=float)
        mu_sat = self._interp(p_a, self.p_tab, self.muo_tab)
        if rs is None:
            return mu_sat
        pb = float(self.pb) if self.pb is not None else float(self.pref_o)
        rs_a = np.broadcast_to(np.asarray(rs, dtype=float), p_a.shape)
        rs_sat = self.rs(p_a)
        if saturated is None:
            sat = (rs_a >= rs_sat - 1.0e-10) & (p_a <= pb)
        else:
            sat = np.broadcast_to(np.asarray(saturated, dtype=bool), p_a.shape)
        pb_rs = self.pbub_of_rs(rs_a)
        mu_unsat = self._interp(pb_rs, self.p_tab, self.muo_tab)
        return np.where(sat, mu_sat, mu_unsat)

    def viscosity_g(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.mug_tab is None:
            return np.full(p_a.shape, float(self.mu_g), dtype=float)
        return self._interp(p_a, self.p_tab, self.mug_tab)

    def has_live_oil(self) -> bool:
        return self.p_tab is not None and self.rs_tab is not None

    def density_o(
        self,
        p: NDArray[np.float64] | float,
        rs: NDArray[np.float64] | float | None = None,
        bo: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Mass density of oil including dissolved gas: (rho_o^s + Rs rho_g^s) b_o."""
        rs_a = self.rs(p) if rs is None else np.asarray(rs, dtype=float)
        b = self.b_o(p, rs=rs_a) if bo is None else np.asarray(bo, dtype=float)
        return (float(self.rho_o_sc) + rs_a * float(self.rho_g_sc)) * b

    def density_w(self, p: NDArray[np.float64] | float, bw: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        b = self.b_w(p) if bw is None else np.asarray(bw, dtype=float)
        return float(self.rho_w_sc) * b

    def density_g(self, p: NDArray[np.float64] | float, bg: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        b = self.b_g(p) if bg is None else np.asarray(bg, dtype=float)
        return float(self.rho_g_sc) * b

    def drs_dp(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """dRs/dp. Zero above pb (Rs capped)."""
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.rs_tab is None:
            return np.zeros(p_a.shape, dtype=float)
        slope = np.gradient(self.rs_tab, self.p_tab)
        out = self._interp(p_a, self.p_tab, slope)
        if self.pb is not None:
            out = np.where(p_a >= float(self.pb), 0.0, out)
        return out

    def cg_of(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Gas compressibility from *PVT Eg: (1/Eg) dEg/dp."""
        p_a = np.asarray(p, dtype=float)
        if self.p_tab is None or self.eg_tab is None:
            return np.full(p_a.shape, float(self.cg), dtype=float)
        slope = np.gradient(self.eg_tab, self.p_tab)
        dbg = self._interp(p_a, self.p_tab, slope)
        return dbg / np.maximum(self.b_g(p_a), 1.0e-30)

    def surface_gas_holdup(
        self,
        sw: NDArray[np.float64] | float,
        sg: NDArray[np.float64] | float,
        p: NDArray[np.float64] | float,
        rs: NDArray[np.float64] | float | None = None,
    ) -> NDArray[np.float64]:
        """Surface gas per reservoir volume: free + dissolved."""
        sw_a = np.asarray(sw, dtype=float)
        sg_a = np.asarray(sg, dtype=float)
        so = np.clip(1.0 - sw_a - sg_a, 0.0, 1.0)
        rs_a = self.rs(p) if rs is None else np.asarray(rs, dtype=float)
        return self.b_g(p) * sg_a + rs_a * self.b_o(p, rs=rs_a) * so

    def vo_unsat(self, sg: NDArray[np.float64] | float, eps: float = 1.0e-8) -> NDArray[np.bool_]:
        """No free gas (volatile-oil status 1 when disgas, no vapoil)."""
        return np.asarray(sg, dtype=float).ravel() <= float(eps)

    def vo_encode(self, sg: NDArray[np.float64], rs: NDArray[np.float64], unsat: NDArray[np.bool_]) -> NDArray[np.float64]:
        """x = Rs in undersaturated cells, x = Sg when both oil and gas are present."""
        return np.where(unsat, np.asarray(rs, dtype=float).ravel(), np.asarray(sg, dtype=float).ravel())

    def vo_decode(
        self,
        x: NDArray[np.float64],
        sw: NDArray[np.float64],
        rs_sat: NDArray[np.float64],
        bo: NDArray[np.float64],
        bg: NDArray[np.float64],
        unsat: NDArray[np.bool_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Decode x → (Sg, Rs) and switch status if a cell crosses the bubble point."""
        x_a = np.asarray(x, dtype=float).ravel()
        sw_a = np.asarray(sw, dtype=float).ravel()
        rs_sat = np.asarray(rs_sat, dtype=float).ravel()
        bo = np.asarray(bo, dtype=float).ravel()
        bg = np.maximum(np.asarray(bg, dtype=float).ravel(), 1.0e-30)
        unsat = np.asarray(unsat, dtype=bool).ravel()
        sl = np.maximum(1.0 - sw_a, 0.0)
        sg = np.zeros_like(x_a)
        rs = rs_sat.copy()
        sat = ~unsat
        sg[sat] = np.clip(x_a[sat], 0.0, sl[sat])
        rs[unsat] = np.clip(x_a[unsat], 0.0, None)
        grow = unsat & (rs > rs_sat)
        extra = (rs - rs_sat) * bo * sl
        denom = np.maximum(bg - rs_sat * bo, 1.0e-12)
        sg[grow] = np.clip(extra[grow] / denom[grow], 0.0, sl[grow])
        rs[grow] = rs_sat[grow]
        dry = sat & (sg <= 1.0e-8)
        sg[dry] = 0.0
        unsat = (unsat | dry) & ~grow
        rs = np.where(unsat, np.minimum(rs, rs_sat), rs_sat)
        return sg, rs, unsat

    def flash_from_total(
        self,
        sw: NDArray[np.float64] | float,
        g_per_pv: NDArray[np.float64] | float,
        p: NDArray[np.float64] | float,
    ) -> NDArray[np.float64]:
        """Split total surface gas into Sg at ``p``. Water is unchanged."""
        sw_a = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
        g = np.asarray(g_per_pv, dtype=float)
        sl = np.clip(1.0 - sw_a, 0.0, 1.0)
        bg = np.maximum(self.b_g(p), 1.0e-30)
        rs = self.rs(p)
        bo = self.b_o(p, rs=rs)
        denom = bg - rs * bo
        safe = np.where(np.abs(denom) > 1.0e-12, denom, 1.0e-12)
        sg = (g - rs * bo * sl) / safe
        return np.clip(sg, 0.0, sl)

    def flash_sg(
        self,
        sw: NDArray[np.float64],
        sg: NDArray[np.float64],
        p: NDArray[np.float64],
        p_old: NDArray[np.float64],
        rs: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Liberate or dissolve gas so total surface gas is kept. Water unchanged."""
        if not self.has_live_oil():
            return np.asarray(sg, dtype=float)
        sw_a = np.clip(np.asarray(sw, dtype=float), 0.0, 1.0)
        sg_a = np.clip(np.asarray(sg, dtype=float), 0.0, 1.0 - sw_a)
        g = self.surface_gas_holdup(sw_a, sg_a, p_old, rs=rs)
        return self.flash_from_total(sw_a, g, p)

    def pv_mult(self, p: NDArray[np.float64] | float) -> NDArray[np.float64]:
        return 1.0 + float(self.cr) * (np.asarray(p, dtype=float) - float(self.pref_r))

    def ct(
        self,
        sw: NDArray[np.float64] | float,
        so: NDArray[np.float64] | float | None = None,
        sg: NDArray[np.float64] | float | None = None,
        p: NDArray[np.float64] | float | None = None,
    ) -> NDArray[np.float64]:
        sw_a = np.asarray(sw, dtype=float)
        sg_a = 0.0 if sg is None else np.asarray(sg, dtype=float)
        so_a = (1.0 - sw_a - sg_a) if so is None else np.asarray(so, dtype=float)
        cg = self.cg_of(p) if p is not None else float(self.cg)
        out = self.cr + sw_a * self.cw + so_a * self.co + sg_a * cg
        if p is not None and self.has_live_oil():
            bg = np.maximum(self.b_g(p), 1.0e-30)
            out = out + so_a * (self.b_o(p) / bg) * self.drs_dp(p)
        return out

    def has_storage(self) -> bool:
        return self.has_live_oil() or max(abs(self.cw), abs(self.co), abs(self.cg), abs(self.cr)) > 0.0

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
        p_psi = np.array([14.7, 264.7, 514.7, 1014.7, 2014.7, 2514.7, 3014.7, 4014.7, 5014.7, 9014.7])
        rs_scf = np.array([1.0, 90.5, 180.0, 371.0, 636.0, 775.0, 930.0, 1270.0, 1600.0, 2984.0])
        bo = np.array([1.062, 1.15, 1.207, 1.295, 1.435, 1.5, 1.565, 1.695, 1.827, 2.36])
        eg = np.array([6.0, 82.692, 159.388, 312.793, 619.579, 772.798, 925.926, 1233.046, 1540.832, 2590.674])
        viso = np.array([1.04, 0.975, 0.91, 0.83, 0.69, 0.64, 0.594, 0.51, 0.449, 0.203]) * 1.0e-3
        visg = np.array([0.0080, 0.0096, 0.0112, 0.0140, 0.0189, 0.0208, 0.0228, 0.0268, 0.0309, 0.0470]) * 1.0e-3
        return cls(
            bw_ref=1.04,
            bo_ref=1.50,
            bg_ref=1.0 / (772.798 * SCF_PER_STB),
            cw=3.04e-6 / PSI,
            co=1.3687e-5 / PSI,
            cr=3.0e-6 / PSI,
            pref_w=14.7 * PSI,
            pref_o=float(pb),
            pref_r=14.7 * PSI,
            mu_w=1.1e-3,
            mu_o=0.64e-3,
            mu_g=0.0208e-3,
            rs_sc=775.0 * SCF_PER_STB,
            rho_w_sc=62.238 * LB_FT3,
            rho_o_sc=46.244 * LB_FT3,
            rho_g_sc=0.0647 * LB_FT3,
            pb=float(pb),
            p_tab=p_psi * PSI,
            rs_tab=rs_scf * SCF_PER_STB,
            bo_tab=bo,
            eg_tab=eg * SCF_PER_STB,
            muo_tab=viso,
            mug_tab=visg,
        )
