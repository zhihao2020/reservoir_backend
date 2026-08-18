"""Sequential black-oil helpers: well mixture, CNV, Newton relaxation, dt.

Used by implicit transport and the IMPES/SFI driver. Not a MATLAB runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Sequential black-oil Newton / CNV defaults (dsMax 0.20 matches *NORM SATUR).
DS_MAX_ABS = 0.20
TOLERANCE_CNV = 1.0e-3
TOLERANCE_MB = 1.0e-7
VOLUME_DISCREPANCY_TOL = 1.0e-3
INC_TOL_SATURATION = 1.0e-3
TARGET_ITERATION_COUNT = 5
ITERATION_OFFSET = 5
MAX_RELATIVE_DT = 2.0
MIN_RELATIVE_DT = 0.5


def cross_flow_mixture(
    flux: NDArray[np.float64],
    compi: NDArray[np.float64],
    perf2well: NDArray[np.int64],
    n_wells: int,
    *,
    conserve_mass: bool = False,
) -> NDArray[np.float64]:
    """Wellbore mixture for injection perfs with crossflow.

    ``flux`` is (nperf, nph) mobility-weighted wellbore flux, positive out
    of the well into the reservoir. Inflow from producing perfs plus any
    net topside injection sets the composition seen by injecting perfs.
    """
    flux = np.asarray(flux, dtype=float)
    compi = np.asarray(compi, dtype=float).copy()
    if compi.ndim == 1:
        compi = np.broadcast_to(compi, (n_wells, compi.size)).copy()
    p2w = np.asarray(perf2well, dtype=np.int64).ravel()
    nperf, nph = flux.shape
    if nperf == 0 or n_wells <= 0:
        return compi
    flux_in = -np.minimum(flux, 0.0)
    if not np.any(flux_in):
        return compi

    def _sum_perf(v: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.zeros((n_wells, v.shape[1]), dtype=float)
        np.add.at(out, p2w, v)
        return out

    net_flux = _sum_perf(np.sum(flux, axis=1, keepdims=True)).ravel()
    net_injection = np.maximum(net_flux, 0.0)
    sum_in = _sum_perf(flux_in)
    top_in = net_injection[:, None] * compi
    comp = sum_in + top_in
    comp_t = np.sum(comp, axis=1)
    active = comp_t > 0.0
    mixed = np.zeros_like(comp)
    mixed[active] = comp[active] / comp_t[active, None]
    compi = compi.copy()
    compi[active] = mixed[active]
    if conserve_mass:
        act = np.sum(top_in, axis=1) > 0.0
        denom = np.maximum(np.maximum(net_flux, 0.0), 1.0e-30)
        compi[act] = (top_in[act] + sum_in[act]) / denom[act, None]
    return compi


def cross_flow_mixture_density(
    mass_flux: NDArray[np.float64],
    volume_total_flux: NDArray[np.float64],
    mass_flux_from_surface: NDArray[np.float64],
    perf2well: NDArray[np.int64],
    n_wells: int,
) -> NDArray[np.float64]:
    """Wellbore mixture density from inflow plus topside mass."""
    mass = np.asarray(mass_flux, dtype=float)
    if mass.ndim == 1:
        mass = mass[:, None]
    vol = np.asarray(volume_total_flux, dtype=float).ravel()
    top = np.asarray(mass_flux_from_surface, dtype=float)
    if top.ndim == 1:
        top = top.reshape(n_wells, -1)
    p2w = np.asarray(perf2well, dtype=np.int64).ravel()

    def _sum_perf(v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.ndim == 1:
            v = v[:, None]
        out = np.zeros((n_wells, v.shape[1]), dtype=float)
        np.add.at(out, p2w, v)
        return out

    mass_in = _sum_perf(np.maximum(-mass, 0.0))
    vol_in = _sum_perf(np.maximum(-vol, 0.0)).ravel()
    vol_out = _sum_perf(np.maximum(vol, 0.0)).ravel()
    vol_ex = vol_in - vol_out
    total_mass = mass_in + np.maximum(top, 0.0)
    denom = (vol_out + np.maximum(vol_ex, 0.0))[:, None]
    rho = np.divide(total_mass, np.maximum(denom, 1.0e-30))
    bad = (np.sum(total_mass, axis=1) == 0.0) | (vol_out == 0.0)
    rho[bad] = 1.0
    return rho


def cnv_mb(
    residuals: list[NDArray[np.float64]],
    pv: NDArray[np.float64],
    b: list[NDArray[np.float64]],
    dt: float,
    *,
    tol_cnv: float = TOLERANCE_CNV,
    tol_mb: float = TOLERANCE_MB,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """CNV / material-balance residuals used by commercial black-oil codes.

    ``residuals`` are conservation residuals of the form
    ``(acc − acc0) + dt·Div − dt·q`` (surface volume). Then

        CNV_α = B̄_α · max(|R_α| / φV)
        MB_α  = |B̄_α · Σ R_α| / Σ(φV)
    """
    pv_a = np.maximum(np.asarray(pv, dtype=float).ravel(), 1.0e-30)
    pv_tot = float(np.sum(pv_a))
    dt = float(dt)
    nph = len(residuals)
    cnv = np.zeros(nph, dtype=float)
    mb = np.zeros(nph, dtype=float)
    for i, (r, bi) in enumerate(zip(residuals, b)):
        r_a = np.asarray(r, dtype=float).ravel()
        b_a = np.asarray(bi, dtype=float).ravel()
        b_avg = float(np.mean(1.0 / np.maximum(b_a, 1.0e-30)))
        cnv[i] = b_avg * float(np.max(np.abs(r_a) / pv_a))
        mb[i] = abs(b_avg * float(np.sum(r_a))) / max(pv_tot, 1.0e-30)
    ok = bool(np.all(cnv <= tol_cnv) and np.all(mb <= tol_mb))
    return cnv, mb, ok


def limit_update_abs(dx: NDArray[np.float64], ds_max: float = DS_MAX_ABS) -> NDArray[np.float64]:
    """Scale a Newton increment so max |Δx| ≤ ``ds_max``.

    Matches IMEX ``*NORM *SATUR 0.20``.
    """
    x = np.asarray(dx, dtype=float)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= float(ds_max) or peak <= 0.0:
        return x
    return x * (float(ds_max) / peak)


@dataclass
class NewtonRelaxation:
    """Dampen / SOR Newton increments when residuals oscillate or stagnate.

    Sequential implicit transport turns this on by default.
    """

    relaxation: float = 1.0
    relaxation_type: str = "dampen"
    increment: float = 0.1
    min_relaxation: float = 0.5
    max_relaxation: float = 1.0
    stagnate_tol: float = 1.0e-2
    oscillation_threshold: float = 1.0
    previous: NDArray[np.float64] | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.relaxation = self.max_relaxation
        self.previous = None

    def oscillating(self, history: NDArray[np.float64]) -> NDArray[np.bool_]:
        hist = np.asarray(history, dtype=float)
        if hist.shape[0] < 3:
            return np.zeros(hist.shape[1], dtype=bool)
        old, mid, nxt = hist[-3], hist[-2], hist[-1]
        denom = mid - old
        denom = np.where(np.abs(denom) < 1.0e-30, 1.0e-30, denom)
        return ((nxt - mid) / denom) < 0.0

    def stagnated(self, history: NDArray[np.float64]) -> NDArray[np.bool_]:
        hist = np.asarray(history, dtype=float)
        if hist.shape[0] < 2:
            return np.zeros(hist.shape[1], dtype=bool)
        prev, nxt = hist[-2], hist[-1]
        return np.abs(nxt - prev) / np.maximum(np.abs(prev), 1.0e-30) < self.stagnate_tol

    def update(self, history: NDArray[np.float64], converged: NDArray[np.bool_] | None = None) -> float:
        hist = np.asarray(history, dtype=float)
        if hist.ndim == 1:
            hist = hist[:, None]
        nph = hist.shape[1]
        ok = np.zeros(nph, dtype=bool) if converged is None else np.asarray(converged, dtype=bool)
        bad = self.oscillating(hist) | self.stagnated(hist)
        n_open = int(np.sum(~ok))
        relax = n_open > 0 and int(np.sum(bad & ~ok)) >= self.oscillation_threshold * n_open and not bool(np.all(ok))
        if relax:
            self.relaxation = max(self.relaxation - self.increment, self.min_relaxation)
        else:
            self.relaxation = min(self.relaxation + self.increment, self.max_relaxation)
        return self.relaxation

    def apply(self, dx: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.asarray(dx, dtype=float)
        w = float(self.relaxation)
        if w >= 1.0 or self.relaxation_type == "none":
            self.previous = x.copy()
            return x
        if self.relaxation_type == "sor" and self.previous is not None and self.previous.shape == x.shape:
            out = x * w + (1.0 - w) * self.previous
        else:
            out = x * w
        self.previous = x.copy()
        return out


def iteration_count_timestep(
    dt1: float,
    its1: int,
    *,
    dt0: float | None = None,
    its0: int | None = None,
    target: int = TARGET_ITERATION_COUNT,
    offset: int = ITERATION_OFFSET,
    maxits: int = 12,
    dt_min: float = 0.0,
    dt_max: float | None = None,
    max_rel: float = MAX_RELATIVE_DT,
    min_rel: float = MIN_RELATIVE_DT,
) -> float:
    """Next dt from Newton iteration history, with relative clamps."""
    maxits_w = max(int(maxits) + int(offset), 1)
    tol = (float(target) + float(offset)) / maxits_w
    le1 = (float(its1) + float(offset)) / maxits_w
    le1 = max(le1, 1.0e-12)
    if dt0 is None or its0 is None:
        dt_new = (tol / le1) * float(dt1)
    else:
        le0 = (float(its0) + float(offset)) / maxits_w
        dt_new = (float(dt1) / max(float(dt0), 1.0e-30)) * (tol * le0 / (le1 * le1)) * float(dt1)
    change = dt_new / max(float(dt1), 1.0e-30)
    change = min(max(change, float(min_rel)), float(max_rel))
    dt = float(dt1) * change
    if dt_max is not None:
        dt = min(dt, float(dt_max))
    return max(dt, float(dt_min))


def state_change_timestep(
    dt1: float,
    ds: float,
    dp_rel: float = 0.0,
    *,
    target_ds: float = 0.15,
    target_dp_rel: float = 0.20,
    dt_min: float = 0.0,
    dt_max: float | None = None,
    max_rel: float = MAX_RELATIVE_DT,
    min_rel: float = MIN_RELATIVE_DT,
) -> float:
    """Next dt from saturation / pressure change, with relative clamps.

    ``dt_new / dt1 = min(target_ds / ΔS, target_dp_rel / (|Δp|/p))``.
    """
    factors: list[float] = []
    if float(ds) > 1.0e-15 and float(target_ds) > 0.0:
        factors.append(float(target_ds) / float(ds))
    if float(dp_rel) > 1.0e-15 and float(target_dp_rel) > 0.0:
        factors.append(float(target_dp_rel) / float(dp_rel))
    if not factors:
        dt_new = float(dt1)
    else:
        dt_new = float(dt1) * min(factors)
    change = dt_new / max(float(dt1), 1.0e-30)
    change = min(max(change, float(min_rel)), float(max_rel))
    dt = float(dt1) * change
    if dt_max is not None:
        dt = min(dt, float(dt_max))
    return max(dt, float(dt_min))


def cell_status_vo(
    so: NDArray[np.float64],
    sw: NDArray[np.float64],
    sg: NDArray[np.float64],
    *,
    disgas: bool = False,
    vapoil: bool = False,
    status: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Volatile-oil cell status flags.

    * st1: oil present, no free gas  → primary is Rs, Sg = 0
    * st2: gas present, no oil       → primary is Rv (unused here)
    * st3: oil and gas               → primary is Sg, Rs = RsSat
    """
    sw_a = np.asarray(sw, dtype=float).ravel()
    so_a = np.asarray(so, dtype=float).ravel()
    sg_a = np.asarray(sg, dtype=float).ravel()
    if status is None:
        wat_only = sw_a > 1.0 - np.sqrt(np.finfo(float).eps)
        oil_present = np.ones(sw_a.size, dtype=bool) if not vapoil else (so_a > 0.0) | wat_only
        gas_present = np.ones(sw_a.size, dtype=bool) if not disgas else (sg_a > 0.0) | wat_only
        status_a = oil_present.astype(np.int64) + 2 * gas_present.astype(np.int64)
    else:
        status_a = np.asarray(status, dtype=np.int64).ravel()
    st1 = status_a == 1 if disgas else np.zeros(status_a.size, dtype=bool)
    st2 = status_a == 2 if vapoil else np.zeros(status_a.size, dtype=bool)
    st3 = status_a == 3
    return st1, st2, st3


def compute_flash_blackoil(
    sw: NDArray[np.float64],
    so: NDArray[np.float64],
    sg: NDArray[np.float64],
    rs: NDArray[np.float64],
    rs_sat: NDArray[np.float64],
    sw0: NDArray[np.float64],
    so0: NDArray[np.float64],
    sg0: NDArray[np.float64],
    rs0: NDArray[np.float64],
    rs_sat0: NDArray[np.float64],
    status: tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]] | None = None,
    *,
    disgas: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Chop saturations / Rs near black-oil phase transitions.

    Disgas only (no vaporized oil). ``state0`` is the previous Newton
    iterate, not the previous timestep.
    """
    etol = float(np.sqrt(np.finfo(float).eps))
    sw = np.asarray(sw, dtype=float).ravel().copy()
    so = np.asarray(so, dtype=float).ravel().copy()
    sg = np.asarray(sg, dtype=float).ravel().copy()
    rs = np.asarray(rs, dtype=float).ravel().copy()
    rs_sat = np.asarray(rs_sat, dtype=float).ravel()
    sg0 = np.asarray(sg0, dtype=float).ravel()
    rs0 = np.asarray(rs0, dtype=float).ravel()
    rs_sat0 = np.asarray(rs_sat0, dtype=float).ravel()
    if status is None:
        status = cell_status_vo(so, sw, sg, disgas=disgas, vapoil=False)
    st1, _st2, _st3 = status
    wat_only = sw > 1.0 - etol
    if not disgas:
        gas_present = np.ones(sw.size, dtype=bool)
    else:
        gas_present = ((sg > 0.0) | (rs == 0.0)) & ~st1 | wat_only
        gas_present = gas_present | ((sg < 0.0) & (sg0 > etol))
        ix2 = (rs > rs_sat * (1.0 + etol)) & st1 & (rs0 > rs_sat0 * (1.0 - etol))
        sg = np.where(ix2, 0.0, sg)
        gas_present = gas_present | ix2
    ix = sg < 0.0
    den = np.maximum(1.0 - sg, 1.0e-30)
    sw = np.where(ix, sw / den, sw)
    so = np.where(ix, so / den, so)
    sg = np.where(ix, 0.0, sg)
    ix = so < 0.0
    den = np.maximum(1.0 - so, 1.0e-30)
    sw = np.where(ix, sw / den, sw)
    sg = np.where(ix, sg / den, sg)
    so = np.where(ix, 0.0, so)
    ix = sw < 0.0
    den = np.maximum(1.0 - sw, 1.0e-30)
    so = np.where(ix, so / den, so)
    sg = np.where(ix, sg / den, sg)
    sw = np.where(ix, 0.0, sw)
    rs = np.where(gas_present, rs_sat, np.minimum(rs_sat, rs))
    rs = np.maximum(rs, 0.0)
    oil_present = so > 0.0
    new_status = oil_present.astype(np.int64) + 2 * gas_present.astype(np.int64)
    return sw, so, sg, rs, new_status


def hybrid_upwind_flags(v_t: NDArray[np.float64], nph: int = 3) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Viscous vs gravity upwind flags.

    Viscous flags follow total velocity; gravity flags are filled later
    by ``multiphase_upwind_indices`` (Brenier–Jaffré) when potentials exist.
    """
    vt = np.asarray(v_t, dtype=float).ravel()
    flag_v = np.repeat((vt > 0.0)[:, None], int(nph), axis=1)
    flag_g = np.ones_like(flag_v)
    return flag_v, flag_g


def multiphase_upwind_indices(
    pot: NDArray[np.float64],
    v_t: NDArray[np.float64],
    trans: NDArray[np.float64],
    mob_left: NDArray[np.float64],
    mob_right: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Brenier–Jaffré multiphase potential upwind.

    ``pot``, ``mob_*`` are (nface, nph). Returns ``up`` True if the left
    cell is upstream for that phase.
    """
    g = np.asarray(pot, dtype=float)
    nface, nph = g.shape
    t = np.asarray(trans, dtype=float).ravel()
    vt = np.asarray(v_t, dtype=float).ravel()
    kl = np.asarray(mob_left, dtype=float)
    kr = np.asarray(mob_right, dtype=float)
    if nface == 0:
        return np.ones((0, nph), dtype=bool)
    order = np.argsort(g, axis=1)
    g_sorted = np.take_along_axis(g, order, axis=1)
    kl_s = np.take_along_axis(kl, order, axis=1)
    kr_s = np.take_along_axis(kr, order, axis=1)
    theta = np.broadcast_to(vt[:, None], (nface, nph)).copy()
    for l in range(nph):
        for j in range(nph):
            if j == l:
                continue
            kj = kl_s[:, j] if j > l else kr_s[:, j]
            theta[:, l] = theta[:, l] + t * (g_sorted[:, l] - g_sorted[:, j]) * kj
    r = np.zeros(nface, dtype=int)
    for l in range(nph):
        r = np.where(theta[:, l] <= 0.0, l + 1, r)
    up_sorted = np.arange(1, nph + 1)[None, :] > r[:, None]
    up = np.empty_like(up_sorted)
    np.put_along_axis(up, order, up_sorted, axis=1)
    return up


def critical_point_chop(
    x0: NDArray[np.float64],
    x1: NDArray[np.float64],
    xc: NDArray[np.float64] | float,
    eps: float = 1.0e-8,
) -> NDArray[np.float64]:
    """Stop a Newton increment just inside / outside a critical saturation.

    Crossing ``xc`` is chopped: from outside the band first land at
    ``xc ± ε/2``; from inside the band, leave at ``xc ± 3ε/2``.
    """
    x0_a = np.asarray(x0, dtype=float)
    x1_a = np.asarray(x1, dtype=float).copy()
    xc_a = np.broadcast_to(np.asarray(xc, dtype=float), x0_a.shape)
    e = float(eps)
    inc = (x0_a < xc_a) & (x1_a > xc_a)
    dec = (x0_a > xc_a) & (x1_a < xc_a)
    middle = (x0_a < xc_a + e) & (x0_a > xc_a - e)
    lo = inc & ~middle
    hi = inc & middle
    x1_a = np.where(hi, np.minimum(x1_a, xc_a + 1.5 * e), x1_a)
    x1_a = np.where(lo, xc_a - 0.5 * e, x1_a)
    lo_d = dec & middle
    hi_d = dec & ~middle
    x1_a = np.where(lo_d, np.maximum(x1_a, xc_a - 1.5 * e), x1_a)
    x1_a = np.where(hi_d, xc_a + 0.5 * e, x1_a)
    return x1_a


def sequential_phase_fluxes(
    v_t: NDArray[np.float64],
    trans: NDArray[np.float64],
    pot: NDArray[np.float64],
    mob_left: NDArray[np.float64],
    mob_right: NDArray[np.float64],
    *,
    upwind: str = "potential",
) -> NDArray[np.float64]:
    """Sequential phase volume fluxes on one axis.

    ``potential``: one Brenier–Jaffré flag using (G, vT);
    ``q_i = f_i (vT + T Σ_{j≠i} λ_j (G_i−G_j))``.
    ``hybrid``: viscous ``f_i(vT) vT`` plus gravity at vT=0.
    """
    t = np.asarray(trans, dtype=float).ravel()
    vt = np.asarray(v_t, dtype=float).ravel()
    g = np.asarray(pot, dtype=float)
    kl = np.asarray(mob_left, dtype=float)
    kr = np.asarray(mob_right, dtype=float)
    nface, nph = g.shape
    if nface == 0:
        return np.zeros((0, nph), dtype=float)
    mode = str(upwind).lower()
    if mode == "hybrid":
        flags_v = np.repeat((vt >= 0.0)[:, None], nph, axis=1)
        mob_v = np.where(flags_v, kl, kr)
        lt_v = np.maximum(np.sum(mob_v, axis=1, keepdims=True), 1.0e-30)
        q = (mob_v / lt_v) * vt[:, None]
        q = q + sequential_gravity_face(t, g, kl, kr)
        return q
    flags = multiphase_upwind_indices(g, vt, t, kl, kr)
    mob_f = np.where(flags, kl, kr)
    lt = np.maximum(np.sum(mob_f, axis=1, keepdims=True), 1.0e-30)
    frac = mob_f / lt
    q = np.zeros((nface, nph), dtype=float)
    for i in range(nph):
        couple = np.zeros(nface, dtype=float)
        for j in range(nph):
            if i == j:
                continue
            couple = couple + mob_f[:, j] * (g[:, i] - g[:, j])
        q[:, i] = frac[:, i] * (vt + t * couple)
    return q


def sequential_transport_extras(
    v_t: NDArray[np.float64],
    trans: NDArray[np.float64],
    pot: NDArray[np.float64],
    mob_left: NDArray[np.float64],
    mob_right: NDArray[np.float64],
    *,
    upwind: str = "potential",
) -> NDArray[np.float64]:
    """``q_α − f_α(vT) vT`` so implicit transport can keep viscous vT-upwind."""
    vt = np.asarray(v_t, dtype=float).ravel()
    kl = np.asarray(mob_left, dtype=float)
    kr = np.asarray(mob_right, dtype=float)
    nph = kl.shape[1] if kl.ndim == 2 else 1
    flags_v = np.repeat((vt >= 0.0)[:, None], nph, axis=1)
    mob_v = np.where(flags_v, kl, kr)
    lt_v = np.maximum(np.sum(mob_v, axis=1, keepdims=True), 1.0e-30)
    q_visc = (mob_v / lt_v) * vt[:, None]
    q = sequential_phase_fluxes(vt, trans, pot, kl, kr, upwind=upwind)
    return q - q_visc


def sequential_gravity_face(
    trans: NDArray[np.float64],
    pot: NDArray[np.float64],
    mob_left: NDArray[np.float64],
    mob_right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Hybrid gravity/capillary phase fluxes at vT = 0.

    ``q_i = f_i T Σ_{j≠i} λ_j (G_i − G_j)`` with Brenier–Jaffré upwind
    of λ on the potential. ``pot`` and ``mob_*`` are (nface, nph).
    """
    t = np.asarray(trans, dtype=float).ravel()
    g = np.asarray(pot, dtype=float)
    kl = np.asarray(mob_left, dtype=float)
    kr = np.asarray(mob_right, dtype=float)
    nface, nph = g.shape
    if nface == 0:
        return np.zeros((0, nph), dtype=float)
    flags = multiphase_upwind_indices(g, np.zeros(nface), t, kl, kr)
    mob_f = np.where(flags, kl, kr)
    lt = np.maximum(np.sum(mob_f, axis=1, keepdims=True), 1.0e-30)
    frac = mob_f / lt
    q = np.zeros((nface, nph), dtype=float)
    for i in range(nph):
        couple = np.zeros(nface, dtype=float)
        for j in range(nph):
            if i == j:
                continue
            couple = couple + mob_f[:, j] * (g[:, i] - g[:, j])
        q[:, i] = frac[:, i] * t * couple
    return q


def volume_discrepancy(sw: NDArray[np.float64], sg: NDArray[np.float64] | None = None) -> float:
    """‖Σ S − 1‖_∞ for the sequential outer volume check."""
    tot = np.asarray(sw, dtype=float)
    if sg is not None:
        tot = tot + np.asarray(sg, dtype=float)
    so = 1.0 - tot
    return float(np.max(np.abs(sw + so + (0.0 if sg is None else sg) - 1.0))) if tot.size else 0.0


def saturation_increment(
    sw: NDArray[np.float64],
    sw0: NDArray[np.float64],
    sg: NDArray[np.float64] | None = None,
    sg0: NDArray[np.float64] | None = None,
) -> float:
    """max |ΔS| for the sequential outer increment check."""
    ds = float(np.max(np.abs(np.asarray(sw, dtype=float) - np.asarray(sw0, dtype=float)))) if sw.size else 0.0
    if sg is not None and sg0 is not None:
        ds = max(ds, float(np.max(np.abs(np.asarray(sg, dtype=float) - np.asarray(sg0, dtype=float)))))
    return ds


def outer_converged(
    sw: NDArray[np.float64],
    sw_before: NDArray[np.float64],
    sg: NDArray[np.float64] | None = None,
    sg_before: NDArray[np.float64] | None = None,
    *,
    vol_tol: float = VOLUME_DISCREPANCY_TOL,
    inc_tol: float = INC_TOL_SATURATION,
) -> bool:
    """Sequential outer-loop check: volume discrepancy and saturation increment."""
    vol = volume_discrepancy(sw, sg)
    inc = saturation_increment(sw, sw_before, sg, sg_before)
    return vol <= vol_tol and inc <= inc_tol
