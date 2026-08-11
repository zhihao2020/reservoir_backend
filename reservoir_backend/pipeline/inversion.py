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
    default_k_param_prior,
    expand_k_from_params,
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
        "greenfield: low-dim k (bg/channel/width/z) + joint multi-time ES-MDA + hard series",
    ]

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

    # --- 2) single path enhance ---
    if path_enhance and len(samples) >= 2:
        draft = _hard_series(
            mesh,
            samples,
            k_work=k_mean,
            phi_work=phi0,
            viscosity_pa_s=viscosity_pa_s,
            n_k_iterations=1,
        )
        if len(draft) >= 2:
            ind, stats = infer_shape_indicator(
                mesh, draft, sw_weight=1.8, k_weight=0.05, pressure_weight=0.9
            )
            k_mean = enhance_permeability_from_indicator(
                k_mean, ind, strength=0.70, asymmetric=True
            )
            notes.append(
                f"path-k enhance indicator_mean={stats.get('indicator_mean', float('nan')):.3f}"
            )

    # --- 3) hard series ---
    history = _hard_series(
        mesh,
        samples,
        k_work=k_mean,
        phi_work=phi0,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=n_k_iterations,
    )
    for h in history:
        h.notes = list(h.notes) + notes[:8]
        h.permeability = np.clip(
            0.85 * k_mean + 0.15 * np.asarray(h.permeability, dtype=float),
            1.0e-18,
            1.0e-10,
        )

    return InversionResult(
        history=history,
        k_mean=k_mean,
        k_std=k_std,
        theta_mean=theta_mean,
        observation_nrmse=nrmse,
        notes=notes,
    )


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
            chunks = []
            for si, (s, sp) in enumerate(zip(samples, specs)):
                if not sp:
                    continue
                chunks.append(
                    _forward_joint_obs(
                        mesh,
                        s,
                        k_e,
                        obs_spec=sp,
                        viscosity_pa_s=viscosity_pa_s,
                        oil_viscosity_pa_s=oil_viscosity_pa_s,
                        porosity=porosity_prior,
                        rate_wells=rate_wells[si],
                    )
                )
            d_sim[e, :] = np.concatenate(chunks)

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


def _hard_series(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    k_work: float | NDArray[np.float64],
    phi_work: float | NDArray[np.float64],
    viscosity_pa_s: float,
    n_k_iterations: int,
) -> list[FieldBundle]:
    history: list[FieldBundle] = []
    prev: FieldBundle | None = None
    k = k_work
    phi = phi_work
    for sample in samples:
        dt = None if prev is None else float(sample.time - prev.time)
        if dt is not None and dt <= 0:
            dt = None
        if prev is not None:
            k = 0.9 * k_work + 0.1 * prev.permeability
            phi = prev.porosity
        bundle = run_time_slice(
            mesh,
            sample,
            permeability_prior_m2=k,
            porosity_prior=phi,
            viscosity_pa_s=viscosity_pa_s,
            previous=prev,
            dt=dt,
            n_k_iterations=n_k_iterations,
            mode="point_first",
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
