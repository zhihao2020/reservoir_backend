"""Greenfield high-accuracy sensor inversion (wells + probes only).

Single path:

1. Low-dim geological k parameters (bg / channel / width / z-bias)
2. Multi-time **joint** ES-MDA on θ (not full-grid k)
3. Expand θ → k; one path enhance; hard-pin point-first series

No outer loops, no dual ES-MDA, no CMG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.ensemble_math import (
    esmda_update_step,
    normalize_alpha_weights,
)
from reservoir_backend.pipeline.esmda import (
    _forward_joint_obs,
    _observation_vector,
    build_observation_spec,
)
from reservoir_backend.pipeline.k_param import (
    N_K_PARAMS,
    boost_theta_from_indicator,
    default_k_param_prior,
    enforce_k_channel_contrast,
    enforce_theta_contrast,
    expand_k_from_params,
    fit_corridor_to_indicator,
    project_k_to_params,
    sample_k_param_ensemble,
)
from reservoir_backend.pipeline.pressure_field import _rate_wells_from_sample
from reservoir_backend.pipeline.run import run_time_slice
from reservoir_backend.pipeline.shape_indicator import (
    enhance_permeability_from_indicator,
    infer_shape_indicator,
)
from reservoir_backend.pipeline.state import FieldBundle, MeshBundle, SensorSample


@dataclass
class InversionResult:
    """High-accuracy inversion output."""

    history: list[FieldBundle]
    k_mean: NDArray[np.float64]
    k_std: NDArray[np.float64]
    theta_mean: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(N_K_PARAMS)
    )
    observation_nrmse: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_sensor_inversion(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    permeability_prior_m2: float | NDArray[np.float64] = 1.0e-13,
    porosity_prior: float | NDArray[np.float64] = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    oil_viscosity_pa_s: float = 5.0e-3,
    ne: int = 32,
    n_assimilations: int = 5,
    max_times: int = 8,
    n_k_iterations: int = 2,
    seed: int = 11,
    path_enhance: bool = True,
    # unused legacy kwargs (kept for call stability)
    logk_std: float = 1.0,
    corr_len_cells: float = 3.0,
) -> InversionResult:
    """High-accuracy inversion from injectors/producers/probes only."""
    _ = (logk_std, corr_len_cells)
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: float(s.time))
    samples = _subsample_times(samples, max_times)

    phi0 = float(np.mean(np.asarray(porosity_prior, dtype=float)))
    k0 = np.asarray(permeability_prior_m2, dtype=float)
    if k0.ndim == 3:
        k_mean_scalar = float(np.exp(np.mean(np.log(np.clip(k0, 1e-30, None)))))
        theta0 = project_k_to_params(mesh, k0)
    else:
        k_mean_scalar = float(k0.reshape(-1)[0]) if k0.size else 1.0e-13
        theta0 = None

    notes: list[str] = [
        "greenfield: warm corridor → param ES-MDA → corridor refit → hard series",
    ]
    samples, closed_note = _closed_reservoir_samples(mesh, samples)
    if closed_note:
        notes.append(closed_note)

    # --- 0) warm-start corridor geometry from a cheap draft (prior k) ---
    if theta0 is None:
        theta0 = default_k_param_prior(k_mean_scalar).mean.copy()
    if path_enhance and len(samples) >= 2:
        k_warm = expand_k_from_params(mesh, theta0)
        draft0 = _hard_series(
            mesh,
            samples,
            k_work=k_warm,
            phi_work=phi0,
            viscosity_pa_s=viscosity_pa_s,
            n_k_iterations=1,
        )
        if len(draft0) >= 2:
            ind0, st0 = infer_shape_indicator(
                mesh, draft0, sw_weight=2.0, k_weight=0.02, pressure_weight=0.7
            )
            theta0, align0 = fit_corridor_to_indicator(
                mesh, theta0, ind0, n_amp=9, n_phase=12, n_width=5
            )
            theta0 = boost_theta_from_indicator(mesh, theta0, ind0, strength=0.45)
            notes.append(
                f"warm corridor-fit align={align0:.3f} "
                f"meander=({theta0[4]:.2f},{theta0[5]:.2f}) "
                f"ind_mean={st0.get('indicator_mean', float('nan')):.3f}"
            )

    # --- 1) joint multi-time ES-MDA in parameter space ---
    k_mean, k_std, theta_mean, nrmse, es_notes = run_param_joint_esmda(
        mesh,
        samples,
        ne=ne,
        n_assimilations=n_assimilations,
        k_mean_scalar=k_mean_scalar,
        theta_seed=theta0,
        porosity_prior=phi0,
        viscosity_pa_s=viscosity_pa_s,
        oil_viscosity_pa_s=oil_viscosity_pa_s,
        seed=seed,
    )
    notes.extend(es_notes)

    # --- 2) corridor refit on post-ES-MDA draft ---
    align_post = float("nan")
    if path_enhance and len(samples) >= 2:
        k_mean, theta_mean, align_post, n2 = _geometry_from_indicator(
            mesh,
            samples,
            k_mean,
            theta_mean,
            phi0=phi0,
            viscosity_pa_s=viscosity_pa_s,
            tag="post-ESMDA",
        )
        notes.extend(n2)

    # parametric space: keep a strong channel/matrix gap (linear ≥15×)
    theta_mean = enforce_theta_contrast(theta_mean, min_log_ratio=float(np.log(15.0)))
    k_mean = expand_k_from_params(mesh, theta_mean)
    k_mean, theta_mean, ratio0 = enforce_k_channel_contrast(
        mesh, k_mean, theta_mean, min_ratio=8.0
    )
    notes.append(f"k contrast guard ratio≈{ratio0:.2f}")

    # --- 3) hard series on locked parametric k ---
    history = _hard_series(
        mesh,
        samples,
        k_work=k_mean,
        phi_work=phi0,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=n_k_iterations,
        lock_permeability=True,
    )

    # --- 4) one post-series corridor polish if multi-time history improved geometry ---
    if path_enhance and len(history) >= 2:
        ind_f, st_f = infer_shape_indicator(
            mesh, history, sw_weight=2.0, k_weight=0.02, pressure_weight=0.7
        )
        th2, align2 = fit_corridor_to_indicator(
            mesh, theta_mean, ind_f, n_amp=11, n_phase=14, n_width=7
        )
        # only re-run if alignment clearly improves
        if (np.isfinite(align_post) and align2 > float(align_post) + 0.02) or (
            not np.isfinite(align_post) and align2 > 0.05
        ):
            th2 = boost_theta_from_indicator(mesh, th2, ind_f, strength=0.85)
            k2 = expand_k_from_params(mesh, th2)
            k2 = enhance_permeability_from_indicator(
                k2, ind_f, strength=0.30, asymmetric=True
            )
            k2, th2, ratio2 = enforce_k_channel_contrast(
                mesh, k2, th2, min_ratio=8.0
            )
            history = _hard_series(
                mesh,
                samples,
                k_work=k2,
                phi_work=phi0,
                viscosity_pa_s=viscosity_pa_s,
                n_k_iterations=n_k_iterations,
                lock_permeability=True,
            )
            k_mean, theta_mean = k2, th2
            notes.append(
                f"post-series corridor polish align {align_post:.3f}→{align2:.3f} "
                f"k_ch/k_mat≈{ratio2:.2f} ind_mean={st_f.get('indicator_mean', float('nan')):.3f}"
            )
        else:
            notes.append(
                f"post-series polish skipped (align {align_post:.3f} vs {align2:.3f})"
            )

    for h in history:
        h.notes = list(h.notes) + notes[:12]
        h.permeability = np.clip(
            0.97 * k_mean + 0.03 * np.asarray(h.permeability, dtype=float),
            1.0e-18,
            1.0e-10,
        )
        h.permeability, _, _ = enforce_k_channel_contrast(
            mesh, h.permeability, theta_mean, min_ratio=8.0
        )

    # --- 5) Sw polish on the same parametric k (no contrast-killing soft-wide) ---
    if len(history) >= 2:
        history = _sw_polish_series(
            mesh,
            samples,
            history,
            k_fixed=k_mean,
            phi=phi0,
            viscosity_pa_s=viscosity_pa_s,
            lock_k=True,
        )
        for h in history:
            h.permeability = np.asarray(k_mean, dtype=float).copy()
        notes.append("fixed-k Sw polish (parametric k, lock_k blend)")

    return InversionResult(
        history=history,
        k_mean=k_mean,
        k_std=k_std,
        theta_mean=theta_mean,
        observation_nrmse=nrmse,
        notes=notes,
    )


def _closed_reservoir_samples(
    mesh: MeshBundle,
    samples: list[SensorSample],
) -> tuple[list[SensorSample], str]:
    """Drop face Dirichlet that only copies INJ/PROD BHP (closed-box default).

    Does not invent aquifer data. Real user face pressures that differ from
    well BHP are kept.
    """
    inj_p: float | None = None
    prod_p: float | None = None
    for name, role in mesh.well_role.items():
        if not samples:
            break
        p0 = samples[0].well_pressure or {}
        if role == "injector" and name in p0:
            inj_p = float(p0[name])
        if role == "producer" and name in p0:
            prod_p = float(p0[name])
    if inj_p is None or prod_p is None:
        return list(samples), ""

    def _near(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= max(50.0, 1.0e-4 * max(abs(b), 1.0))

    stripped = 0
    out: list[SensorSample] = []
    for s in samples:
        faces = dict(s.boundary.pressure or {})
        pairs = (("left", "right"), ("front", "back"))
        for a, b in pairs:
            if a not in faces or b not in faces:
                continue
            va, vb = float(faces[a]), float(faces[b])
            mimic = (_near(va, inj_p) and _near(vb, prod_p)) or (
                _near(va, prod_p) and _near(vb, inj_p)
            )
            if mimic:
                faces.pop(a, None)
                faces.pop(b, None)
                stripped += 1
        if faces == (s.boundary.pressure or {}):
            out.append(s)
            continue
        from reservoir_backend.pipeline.state import BoundaryConditions

        s2 = SensorSample(
            time=s.time,
            well_pressure=dict(s.well_pressure or {}),
            well_saturation=dict(s.well_saturation or {}),
            well_rate=dict(s.well_rate or {}),
            boundary=BoundaryConditions(pressure=faces, flux=dict(s.boundary.flux or {})),
        )
        out.append(s2)
    note = (
        f"closed-reservoir: stripped well-mimic face Dirichlet on {stripped} samples"
        if stripped
        else ""
    )
    return out, note


def _sw_polish_series(
    mesh: MeshBundle,
    samples: list[SensorSample],
    history: list[FieldBundle],
    *,
    k_fixed: NDArray[np.float64],
    phi: float | NDArray[np.float64],
    viscosity_pa_s: float,
    lock_k: bool = True,
) -> list[FieldBundle]:
    """Recompute p/S with fixed parametric k (no further rock IDW damage)."""
    from reservoir_backend.pipeline.point_workflow import (
        blend_recon_transport_sw,
        filter_sample_for_pressure,
        filter_sample_for_saturation,
    )
    from reservoir_backend.pipeline.pressure_field import reconstruct_pressure
    from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
    from reservoir_backend.pipeline.transport_saturation import (
        phases_from_sw,
        transport_water_saturation,
    )

    out: list[FieldBundle] = []
    prev: FieldBundle | None = None
    for i, sample in enumerate(samples):
        dt = None if prev is None else float(sample.time - prev.time)
        if dt is not None and dt <= 0:
            dt = None
        sample_p = filter_sample_for_pressure(sample, mesh)
        sample_s = filter_sample_for_saturation(sample, mesh)
        p, _ = reconstruct_pressure(
            mesh,
            sample_p,
            permeability_m2=k_fixed,
            viscosity_pa_s=viscosity_pa_s,
            saturation=None if prev is None else prev.sw,
        )
        sw, so, sg, _ = reconstruct_saturation(
            mesh, sample_s, pressure=p, permeability_m2=k_fixed
        )
        if prev is not None and dt is not None and float(dt) > 0.0:
            n_s = len(sample_s.well_saturation)
            n_sub = int(np.clip(round(float(dt) / 4.0) + n_s, 8, 28))
            sw_t, _ = transport_water_saturation(
                mesh,
                0.45 * prev.sw + 0.55 * sw,
                p,
                k_fixed,
                sample,
                porosity=phi,
                viscosity_pa_s=viscosity_pa_s,
                dt=float(dt),
                n_substeps=n_sub,
            )
            sw_b, _rw = blend_recon_transport_sw(
                sw, sw_t, n_s_hard=n_s, lock_k=lock_k
            )
            sw, so, sg = phases_from_sw(sw_b, sample=sample_s, mesh=mesh)
        base = history[i] if i < len(history) else prev
        notes = list(base.notes if base is not None else []) + ["fixed-k Sw polish"]
        fb = FieldBundle(
            time=sample.time,
            pressure=p,
            sw=sw,
            so=so,
            sg=sg,
            permeability=np.asarray(k_fixed, dtype=float).copy(),
            porosity=(
                np.asarray(base.porosity, dtype=float).copy()
                if base is not None
                else np.full(mesh.grid.shape, float(np.mean(np.asarray(phi))))
            ),
            notes=notes,
            flux_x=base.flux_x if base is not None else None,
            flux_y=base.flux_y if base is not None else None,
            flux_z=base.flux_z if base is not None else None,
        )
        out.append(fb)
        prev = fb
    return out


def _k_for_sw_transport(
    mesh: MeshBundle,
    theta: NDArray[np.float64],
    k_rock: NDArray[np.float64],
    *,
    max_ratio: float = 7.5,
    width_boost: float = 0.40,
) -> NDArray[np.float64]:
    """Softer/wider parametric k for Sw transport only (rock k_mean unchanged).

    High rock contrast (~10×) can make the water front too thin vs true channel
    width and hurt ΔSw Dice. Cap transport contrast and widen the corridor a bit.
    """
    th = np.asarray(theta, dtype=float).ravel().copy()
    if th.size >= N_K_PARAMS:
        th[2] = float(th[2] + width_boost)
        gap = float(th[1] - th[0])
        max_gap = float(np.log(max(max_ratio, 1.01)))
        if gap > max_gap:
            mid = 0.5 * (float(th[0]) + float(th[1]))
            th[0] = mid - 0.5 * max_gap
            th[1] = mid + 0.5 * max_gap
        k_sw = expand_k_from_params(mesh, th)
    else:
        k_sw = np.asarray(k_rock, dtype=float).copy()
    # light blend toward rock k so absolute level stays consistent
    return np.clip(
        0.85 * k_sw + 0.15 * np.asarray(k_rock, dtype=float),
        1.0e-18,
        1.0e-10,
    )


def _geometry_from_indicator(
    mesh: MeshBundle,
    samples: list[SensorSample],
    k_mean: NDArray[np.float64],
    theta_mean: NDArray[np.float64],
    *,
    phi0: float,
    viscosity_pa_s: float,
    tag: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, list[str]]:
    """Fit corridor + contrast from a draft series under current k."""
    notes: list[str] = []
    draft = _hard_series(
        mesh,
        samples,
        k_work=k_mean,
        phi_work=phi0,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=1,
    )
    if len(draft) < 2:
        return k_mean, theta_mean, float("nan"), notes
    ind, stats = infer_shape_indicator(
        mesh, draft, sw_weight=2.0, k_weight=0.02, pressure_weight=0.7
    )
    theta_mean, align = fit_corridor_to_indicator(
        mesh, theta_mean, ind, n_amp=11, n_phase=14, n_width=5
    )
    theta_mean = boost_theta_from_indicator(mesh, theta_mean, ind, strength=0.75)
    k_mean = expand_k_from_params(mesh, theta_mean)
    k_mean = enhance_permeability_from_indicator(
        k_mean, ind, strength=0.28, asymmetric=True
    )
    k_mean, theta_mean, ratio = enforce_k_channel_contrast(
        mesh, k_mean, theta_mean, min_ratio=8.0
    )
    notes.append(
        f"{tag} corridor-fit align={align:.3f} "
        f"meander=({theta_mean[4]:.2f},{theta_mean[5]:.2f}) "
        f"ind_mean={stats.get('indicator_mean', float('nan')):.3f} "
        f"k_ch/k_mat≈{ratio:.2f}"
    )
    return k_mean, theta_mean, float(align), notes


def run_param_joint_esmda(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    ne: int,
    n_assimilations: int,
    k_mean_scalar: float,
    theta_seed: NDArray[np.float64] | None,
    porosity_prior: float,
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    seed: int,
    obs_std_frac: float = 0.02,
    obs_std_floor_pa: float = 5.0e4,
    sw_obs_std: float = 0.06,
    qw_obs_std_frac: float = 0.15,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[float],
    list[str],
]:
    """ES-MDA on 4-D k parameters; expand to grid at the end."""
    samples = sorted(samples, key=lambda s: float(s.time))
    ne = int(ne)
    alphas = normalize_alpha_weights(int(n_assimilations))
    notes: list[str] = []

    prior = default_k_param_prior(k_mean_scalar)
    if theta_seed is not None:
        prior = type(prior)(
            mean=0.6 * np.asarray(theta_seed, dtype=float) + 0.4 * prior.mean,
            std=prior.std * 0.85,
        )
    theta_ens = sample_k_param_ensemble(prior, ne=ne, seed=seed)

    specs = [
        build_observation_spec(mesh, s, assimilate_sw=True, assimilate_qw=True)
        for s in samples
    ]
    if not any(specs):
        raise ValueError("no soft observations for param ES-MDA")

    obs_parts, r_parts = [], []
    for s, sp in zip(samples, specs):
        if not sp:
            continue
        o, r, _xyz = _observation_vector(
            mesh,
            s,
            sp,
            obs_std_frac=obs_std_frac,
            obs_std_floor_pa=obs_std_floor_pa,
            sw_obs_std=sw_obs_std,
            qw_obs_std_frac=qw_obs_std_frac,
            oil_viscosity_pa_s=oil_viscosity_pa_s,
            viscosity_pa_s=viscosity_pa_s,
        )
        obs_parts.append(o)
        r_parts.append(r)
    obs = np.concatenate(obs_parts)
    r_diag = np.concatenate(r_parts)
    n_obs = int(obs.size)

    rate_wells = [_rate_wells_from_sample(mesh, s) for s in samples]
    rng = np.random.default_rng(seed + 5)
    nrmse_hist: list[float] = []
    notes.append(
        f"param joint ES-MDA n_θ={N_K_PARAMS} ne={ne} Na={int(alphas.size)} "
        f"n_times={len(samples)} n_obs={n_obs}"
    )

    for alpha in alphas:
        d_sim = np.zeros((ne, n_obs), dtype=float)
        for e in range(ne):
            k_e = expand_k_from_params(mesh, theta_ens[e])
            # sequential multi-time forward (real Δt transport) for physical Sw
            d_sim[e, :] = _forward_member_all_times(
                mesh,
                samples,
                specs,
                k_e,
                rate_wells=rate_wells,
                viscosity_pa_s=viscosity_pa_s,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
                porosity=porosity_prior,
            )

        # update θ directly (n_m = N_K_PARAMS) — no localization needed
        m = theta_ens.copy()
        m = esmda_update_step(
            m,
            d_sim,
            obs,
            r_diag,
            float(alpha),
            rng,
            md_localization=None,
            inflation=1.02,
        )
        # re-clip channel/bg ordering lightly
        for e in range(ne):
            if m[e, 1] < m[e, 0]:
                m[e, 0], m[e, 1] = m[e, 1], m[e, 0]
        from reservoir_backend.pipeline.k_param import _clip_theta

        theta_ens = _clip_theta(m)

        sig = np.sqrt(np.maximum(r_diag, 1.0e-30))
        nrmse = float(np.sqrt(np.mean(((np.mean(d_sim, axis=0) - obs) / sig) ** 2)))
        nrmse_hist.append(nrmse)
        notes.append(f"param MDA nRMSE={nrmse:.4g}")

    theta_mean = np.mean(theta_ens, axis=0)
    theta_std = np.std(theta_ens, axis=0)
    k_mean = expand_k_from_params(mesh, theta_mean)
    # approximate k_std via param std → expand extremes
    k_hi = expand_k_from_params(mesh, theta_mean + theta_std)
    k_lo = expand_k_from_params(mesh, theta_mean - theta_std)
    k_std = 0.5 * np.abs(k_hi - k_lo)
    notes.append(
        f"θ_mean log_bg={theta_mean[0]:.3f} log_ch={theta_mean[1]:.3f} "
        f"log_w={theta_mean[2]:.3f} z_bias={theta_mean[3]:.3f}"
    )
    return k_mean, k_std, theta_mean, nrmse_hist, notes


def _forward_member_all_times(
    mesh: MeshBundle,
    samples: list[SensorSample],
    specs: list,
    k_field: NDArray[np.float64],
    *,
    rate_wells: list,
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    porosity: float,
) -> NDArray[np.float64]:
    """One ensemble member: walk times with carried Sw state."""
    from reservoir_backend.pipeline.esmda import (
        _forward_pressure_cached,
        _sample_obs_from_fields,
    )
    from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
    from reservoir_backend.pipeline.state import SensorSample as SS
    from reservoir_backend.pipeline.transport_saturation import (
        phases_from_sw,
        transport_water_saturation,
    )

    chunks: list[NDArray[np.float64]] = []
    sw_prev: NDArray[np.float64] | None = None
    t_prev: float | None = None

    for si, (sample, sp) in enumerate(zip(samples, specs)):
        if not sp:
            continue
        dt = None if t_prev is None else float(sample.time - t_prev)
        p = _forward_pressure_cached(
            mesh,
            sample,
            permeability_m2=k_field,
            viscosity_pa_s=viscosity_pa_s,
            rate_wells=rate_wells[si],
            saturation=sw_prev,
            oil_viscosity_pa_s=oil_viscosity_pa_s,
        )
        sample_s = SS(
            time=sample.time,
            well_pressure={},
            well_saturation=dict(sample.well_saturation or {}),
            boundary=sample.boundary,
            well_rate={},
        )
        sw_rec, _, _, _ = reconstruct_saturation(
            mesh, sample_s, pressure=p, permeability_m2=k_field
        )
        if sw_prev is None or dt is None or dt <= 0.0 or not sample.well_rate:
            sw_state = sw_rec
        else:
            n_sub = int(np.clip(round(dt / 5.0), 4, 24))
            sw_t, _ = transport_water_saturation(
                mesh,
                0.35 * sw_prev + 0.65 * sw_rec,
                p,
                k_field,
                sample,
                porosity=porosity,
                viscosity_pa_s=viscosity_pa_s,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
                dt=dt,
                n_substeps=n_sub,
            )
            sw_state, _, _ = phases_from_sw(sw_t, sample=sample_s, mesh=mesh)

        chunks.append(
            _sample_obs_from_fields(
                mesh,
                sample,
                sp,
                pressure=p,
                sw=sw_state,
                viscosity_pa_s=viscosity_pa_s,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
            )
        )
        sw_prev = sw_state
        t_prev = float(sample.time)

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)


def _hard_series(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    k_work: float | NDArray[np.float64],
    phi_work: float | NDArray[np.float64],
    viscosity_pa_s: float,
    n_k_iterations: int,
    lock_permeability: bool = True,
    k_anchor: float = 0.90,
) -> list[FieldBundle]:
    """Hard-pin time series.

    ``lock_permeability=True`` keeps pure parametric k for p/transport.
    ``lock_permeability=False`` allows mild point-rock detail while
    ``k_anchor`` pulls each step's prior back toward ``k_work`` (Dice-friendly
    slightly wider fronts without abandoning the parametric channel).
    """
    history: list[FieldBundle] = []
    prev: FieldBundle | None = None
    phi = phi_work
    k_prior = k_work
    a = float(np.clip(k_anchor, 0.5, 1.0))
    for sample in samples:
        dt = None if prev is None else float(sample.time - prev.time)
        if dt is not None and dt <= 0:
            dt = None
        if prev is not None and not lock_permeability:
            phi = prev.porosity
            k_prior = a * np.asarray(k_work, dtype=float) + (1.0 - a) * np.asarray(
                prev.permeability, dtype=float
            )
            k_prior = np.clip(k_prior, 1.0e-18, 1.0e-10)
        else:
            k_prior = k_work
        bundle = run_time_slice(
            mesh,
            sample,
            permeability_prior_m2=k_prior,
            porosity_prior=phi,
            viscosity_pa_s=viscosity_pa_s,
            previous=prev,
            dt=dt,
            n_k_iterations=n_k_iterations,
            mode="point_first",
            lock_permeability=lock_permeability,
        )
        history.append(bundle)
        prev = bundle
    return history


def _subsample_times(samples: list[SensorSample], max_times: int) -> list[SensorSample]:
    n = len(samples)
    max_times = max(1, int(max_times))
    if n <= max_times:
        return list(samples)
    idx = np.linspace(0, n - 1, max_times)
    picked = sorted({int(round(i)) for i in idx})
    return [samples[i] for i in picked]
