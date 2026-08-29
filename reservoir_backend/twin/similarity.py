"""Waterflood similarity groups from PhysicsSpec / PVT / grid.

Reports the conventional-waterflood criteria in 相似准则2.pptx slides 3, 7, 8, 9:
geometric, reservoir (k, phi), fluid (mu and density ratios), dynamic/motion
(capillary vs viscous if Pc is on; compressibility), saturation (Swc, Sor,
movable), and displacement-feature similarity (Sw and p field nRMSE).

Field-prototype ratios are omitted unless the caller passes them. The concept
lab sources do not state a field thickness or well spacing. Thermal / polymer /
shale groups are out of scope.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


REQUIRED_KEYS = (
    "geometric",
    "reservoir",
    "fluid",
    "dynamic",
    "motion",
    "saturation",
    "displacement",
    "skipped",
)


def field_nrmse(pred: ArrayLike, truth: ArrayLike) -> float:
    """nRMSE of pred vs truth, scaled by RMS(truth)."""
    a = np.asarray(pred, dtype=float).ravel()
    b = np.asarray(truth, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"field size {a.size} != {b.size}")
    denom = float(np.sqrt(np.mean(b * b)))
    if denom <= 0.0:
        denom = 1.0
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def _port_xy(twin: Any) -> dict[str, tuple[float, float]]:
    centers = twin.grid.cell_centers()
    out: dict[str, tuple[float, float]] = {}
    for port in twin.ports:
        cells = np.asarray(port.cell_ids, dtype=np.int64).ravel()
        if cells.size == 0:
            continue
        xyz = centers[cells]
        out[str(port.name)] = (float(np.mean(xyz[:, 0])), float(np.mean(xyz[:, 1])))
    return out


def _well_spacing_m(twin: Any) -> float | None:
    xy = _port_xy(twin)
    inj = [xy[p.name] for p in twin.ports if str(p.role).lower() in {"injector", "inj"} and p.name in xy]
    prod = [xy[p.name] for p in twin.ports if str(p.role).lower() in {"producer", "prod"} and p.name in xy]
    if not inj or not prod:
        if len(xy) >= 2:
            pts = list(xy.values())
            return float(np.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]))
        return None
    d = [np.hypot(ix - px, iy - py) for (ix, iy) in inj for (px, py) in prod]
    return float(min(d)) if d else None


def _well_radius_m(twin: Any) -> float | None:
    radii = [float(getattr(p, "rw_m", 0.0) or 0.0) for p in twin.ports]
    positive = [r for r in radii if r > 0.0]
    if not positive:
        return None
    return float(np.mean(positive))


def _injection_rate_m3_s(twin: Any) -> float | None:
    inj = {p.name for p in twin.ports if str(p.role).lower() in {"injector", "inj"} and str(p.control).lower() == "rate"}
    rates: list[float] = []
    for c in getattr(twin.experiment, "controls", []) or []:
        if c.port_name in inj and str(c.kind).lower() == "rate":
            vals = np.asarray(c.values, dtype=float).ravel()
            if vals.size:
                rates.append(float(np.max(np.abs(vals))))
    if not rates:
        return None
    return float(max(rates))


def _k_characteristic_m2(twin: Any, k: ArrayLike | None) -> float:
    if k is not None:
        arr = np.asarray(k, dtype=float).ravel()
        arr = arr[np.isfinite(arr) & (arr > 0.0)]
        if arr.size:
            return float(np.exp(np.mean(np.log(arr))))
    prior = getattr(getattr(twin, "inverse", None), "prior_mean", None)
    if prior is not None:
        return float(np.exp(float(prior)))
    return 1.0e-12


def waterflood_groups(
    twin: Any,
    *,
    k: ArrayLike | None = None,
    field_size_m: tuple[float, float, float] | None = None,
    field_thickness_m: float | None = None,
    field_well_spacing_m: float | None = None,
    field_well_radius_m: float | None = None,
    field_k_m2: float | None = None,
    field_phi: float | None = None,
) -> dict[str, Any]:
    """Lab-side waterflood pi groups. Field ratios stay None unless passed in."""
    phys = twin.physics
    pvt = phys.pvt
    rel = phys.relperm
    grid = twin.grid
    size = tuple(float(x) for x in grid.size_m())
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    k_m2 = _k_characteristic_m2(twin, k)
    mu_w = float(pvt.mu_w)
    mu_o = float(pvt.mu_o)
    rho_w = float(pvt.rho_w_sc)
    rho_o = float(pvt.rho_o_sc)
    swc = float(rel.swi)
    sor = float(rel.sor)
    movable = float(1.0 - swc - sor)

    spacing = _well_spacing_m(twin)
    rw = _well_radius_m(twin)
    q = _injection_rate_m3_s(twin)
    area = size[1] * size[2]
    u = None if q is None or area <= 0.0 else float(q / area)

    cap = phys.capillary
    cap_name = str(getattr(cap, "name", type(cap).__name__))
    pe = getattr(cap, "entry_pressure", None)
    if pe is None:
        pe = getattr(cap, "p0", None)
    pe_f = None if pe is None else float(pe)
    viscous_dp = None
    cap_over_visc = None
    if u is not None and k_m2 > 0.0 and mu_w > 0.0:
        viscous_dp = float(mu_w * u * size[0] / k_m2)
        if pe_f is not None and viscous_dp > 0.0:
            cap_over_visc = float(pe_f / viscous_dp)

    sw0 = float(phys.sw_init)
    so0 = max(0.0, 1.0 - sw0 - float(getattr(phys, "sg_init", 0.0)))
    ct = float(pvt.cr + sw0 * pvt.cw + so0 * pvt.co)
    p_char = float(phys.p_init)
    g = float(getattr(phys, "gravity", 0.0) or 0.0)
    g_over_visc = None
    if g > 0.0 and u is not None and u > 0.0 and mu_w > 0.0:
        g_over_visc = float(k_m2 * abs(rho_w - rho_o) * g / (mu_w * u))

    def _ratio(lab: float | None, field: float | None) -> float | None:
        if lab is None or field is None or field == 0.0:
            return None
        return float(lab / field)

    field_L = None
    if field_thickness_m is not None:
        field_L = float(field_thickness_m)
    elif field_size_m is not None:
        field_L = float(field_size_m[2])

    skipped: dict[str, str] = {
        "thermal": "waterflood only (slides 7-9); thermal groups not reported",
        "polymer": "waterflood only; polymer residual-resistance / rheology skipped",
        "shale": "waterflood only; shale adsorption / dual-porosity skipped",
        "acoustic_em_xyz": "测点位置.pptx states 12 acoustic + 8 EM at 75 mm, no xyz table",
    }
    if field_L is None and field_well_spacing_m is None and field_well_radius_m is None:
        skipped["field_geometric_ratios"] = (
            "concecpt states the 30 cm lab cube only; no field prototype size, "
            "thickness, well spacing, or well radius"
        )
    if g <= 0.0:
        skipped["gravity_over_viscous"] = "physics.gravity is 0 (lab default off)"
    if pe_f is None or cap_name in {"none", "NoCapillary"}:
        skipped["capillary_over_viscous"] = "no Pc model / entry pressure"
    if rw is None:
        skipped["well_radius"] = "concecpt and YAML ports give no rw_m"

    return {
        "scheme": "waterflood",
        "source": "相似准则2.pptx slides 3,7,8,9; lab-side groups from PhysicsSpec/PVT/grid",
        "geometric": {
            "size_m": list(size),
            "well_spacing_m": spacing,
            "well_radius_m": rw,
            "field_size_m": None if field_size_m is None else list(field_size_m),
            "field_thickness_m": None if field_thickness_m is None else float(field_thickness_m),
            "field_well_spacing_m": field_well_spacing_m,
            "field_well_radius_m": field_well_radius_m,
            "size_ratio_lab_over_field": _ratio(size[2], field_L),
            "well_spacing_ratio_lab_over_field": _ratio(spacing, field_well_spacing_m),
            "well_radius_ratio_lab_over_field": _ratio(rw, field_well_radius_m),
            "field_ratios": "unknown" if field_L is None else "set_by_caller",
        },
        "reservoir": {
            "k_m2": k_m2,
            "phi": phi,
            "field_k_m2": field_k_m2,
            "field_phi": field_phi,
        },
        "fluid": {
            "mu_o_over_mu_w": float(mu_o / mu_w) if mu_w else None,
            "rho_o_over_rho_w": float(rho_o / rho_w) if rho_w else None,
            "mu_o_pa_s": mu_o,
            "mu_w_pa_s": mu_w,
            "rho_o_kg_m3": rho_o,
            "rho_w_kg_m3": rho_w,
        },
        "dynamic": {
            "capillary_model": cap_name,
            "pc_entry_pa": pe_f,
            "viscous_delta_p_pa": viscous_dp,
            "capillary_over_viscous": cap_over_visc,
            "compressibility_ct_1_pa": ct,
            "cw_1_pa": float(pvt.cw),
            "co_1_pa": float(pvt.co),
            "cr_1_pa": float(pvt.cr),
            "pi_compressibility": float(ct * p_char),
            "gravity_on": bool(g > 0.0),
            "gravity_over_viscous": g_over_visc,
        },
        "motion": {
            "injection_rate_m3_s": q,
            "darcy_velocity_m_s": u,
            "mobility_oil_m4_n_s": float(k_m2 / mu_o) if mu_o else None,
            "mobility_water_m4_n_s": float(k_m2 / mu_w) if mu_w else None,
        },
        "saturation": {
            "swc": swc,
            "sor": sor,
            "movable": movable,
            "swc_over_movable": float(swc / movable) if movable else None,
            "sor_over_movable": float(sor / movable) if movable else None,
        },
        "displacement": {
            "comparison": "F(m_post) vs F(m_true)",
            "not": "CMG",
            "sw_field_nrmse": None,
            "p_field_nrmse": None,
        },
        "skipped": skipped,
    }


def attach_displacement(
    report: Mapping[str, Any],
    *,
    sw_post: ArrayLike,
    sw_true: ArrayLike,
    p_post: ArrayLike,
    p_true: ArrayLike,
) -> dict[str, Any]:
    """Fill displacement-feature nRMSE: F(m_post) vs F(m_true) Sw and p fields."""
    out = dict(report)
    disp = dict(out.get("displacement") or {})
    disp["comparison"] = "F(m_post) vs F(m_true)"
    disp["not"] = "CMG"
    disp["sw_field_nrmse"] = field_nrmse(sw_post, sw_true)
    disp["p_field_nrmse"] = field_nrmse(p_post, p_true)
    out["displacement"] = disp
    return out


def report_from_trajectories(twin: Any, post_hist: Any, true_hist: Any, **kwargs: Any) -> dict[str, Any]:
    """Groups plus last-time Sw/p field nRMSE of two forwards."""
    k = None
    if getattr(post_hist, "states", None):
        # k is on the rock, not the state; caller may pass k=
        k = kwargs.pop("k", None)
    rep = waterflood_groups(twin, k=k, **kwargs)
    if not getattr(post_hist, "states", None) or not getattr(true_hist, "states", None):
        return rep
    sp = post_hist.states[-1]
    st = true_hist.states[-1]
    return attach_displacement(
        rep,
        sw_post=sp.sw,
        sw_true=st.sw,
        p_post=sp.pressure,
        p_true=st.pressure,
    )
