"""Greenfield high-accuracy sensor inversion (wells + probes only).

Single path, no stacked redundant loops:

1. Multi-time **joint** ES-MDA on log(k) using soft p / Sw / qw
2. One path-aware k enhance from multi-time ΔSw indicator
3. One point-first hard-pin series for final p/S/k/φ fields

No CMG, no second/outer ES-MDA forks, no dual refine stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.ensemble_math import (
    esmda_update_step,
    normalize_alpha_weights,
    well_parameter_localization,
)
from reservoir_backend.pipeline.esmda import (
    _forward_joint_obs,
    _observation_vector,
    build_observation_spec,
    generate_logk_ensemble,
    generate_logk_ensemble_around,
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
    ne: int = 24,
    n_assimilations: int = 4,
    logk_std: float = 1.0,
    corr_len_cells: float = 3.0,
    max_times: int = 8,
    n_k_iterations: int = 2,
    seed: int = 11,
    path_enhance: bool = True,
) -> InversionResult:
    """High-accuracy inversion from injectors/producers/probes only.

    Parameters are intentionally few (greenfield): ensemble size, MDA steps,
    and optional path enhance. No outer-loop / second-pass knobs.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: float(s.time))
    samples = _subsample_times(samples, max_times)

    phi0 = float(np.mean(np.asarray(porosity_prior, dtype=float)))
    k0 = np.asarray(permeability_prior_m2, dtype=float)
    if k0.ndim == 3:
        k_prior_field: NDArray[np.float64] | None = k0
        k_mean_scalar = float(np.exp(np.mean(np.log(np.clip(k0, 1e-30, None)))))
    else:
        k_prior_field = None
        k_mean_scalar = float(k0.reshape(-1)[0]) if k0.size else 1.0e-13
    notes: list[str] = [
        "greenfield sensor inversion: joint multi-time ES-MDA → path-k → hard series",
    ]

    # --- 1) joint multi-time ES-MDA ---
    k_mean, k_std, nrmse, es_notes = run_joint_multitime_esmda(
        mesh,
        samples,
        ne=ne,
        n_assimilations=n_assimilations,
        k_prior=k_prior_field,
        k_mean_scalar=k_mean_scalar,
        logk_std=logk_std,
        corr_len_cells=corr_len_cells,
        porosity_prior=phi0,
        viscosity_pa_s=viscosity_pa_s,
        oil_viscosity_pa_s=oil_viscosity_pa_s,
        seed=seed,
    )
    notes.extend(es_notes)

    # --- 2) one path enhance (optional, single shot) ---
    if path_enhance and len(samples) >= 2:
        # cheap probe series to build ΔSw indicator under current k
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
                k_mean, ind, strength=0.85, asymmetric=True
            )
            notes.append(
                f"single path-k enhance indicator_mean={stats.get('indicator_mean', float('nan')):.3f}"
            )

    # --- 3) final hard-pin point-first series ---
    history = _hard_series(
        mesh,
        samples,
        k_work=k_mean,
        phi_work=phi0,
        viscosity_pa_s=viscosity_pa_s,
        n_k_iterations=n_k_iterations,
    )
    for h in history:
        h.notes = list(h.notes) + notes[:6]
        # keep ensemble k structure dominant, allow mild local rock update
        h.permeability = np.clip(
            0.80 * k_mean + 0.20 * np.asarray(h.permeability, dtype=float),
            1.0e-18,
            1.0e-10,
        )

    return InversionResult(
        history=history,
        k_mean=k_mean,
        k_std=k_std,
        observation_nrmse=nrmse,
        notes=notes,
    )


def run_joint_multitime_esmda(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    ne: int,
    n_assimilations: int,
    k_prior: NDArray[np.float64] | None,
    k_mean_scalar: float,
    logk_std: float,
    corr_len_cells: float,
    porosity_prior: float,
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    seed: int,
    obs_std_frac: float = 0.02,
    obs_std_floor_pa: float = 5.0e4,
    sw_obs_std: float = 0.06,
    qw_obs_std_frac: float = 0.15,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float], list[str]]:
    """One ES-MDA: each step matches **all times** jointly (stacked soft obs)."""
    samples = sorted(samples, key=lambda s: float(s.time))
    ne = int(ne)
    alphas = normalize_alpha_weights(int(n_assimilations))
    notes: list[str] = []

    if k_prior is not None:
        k_ens = generate_logk_ensemble_around(
            k_prior, ne=ne, logk_std=logk_std, corr_len_cells=corr_len_cells, seed=seed
        )
    else:
        k_ens = generate_logk_ensemble(
            mesh.grid.shape,
            ne=ne,
            k_mean=k_mean_scalar,
            logk_std=logk_std,
            corr_len_cells=corr_len_cells,
            seed=seed,
        )

    specs = [
        build_observation_spec(mesh, s, assimilate_sw=True, assimilate_qw=True)
        for s in samples
    ]
    if not any(specs):
        raise ValueError("no soft observations available for joint ES-MDA")

    # stacked observation vector (all times)
    obs_parts = []
    r_parts = []
    xyz_parts = []
    for s, sp in zip(samples, specs):
        if not sp:
            continue
        o, r, xyz = _observation_vector(
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
        xyz_parts.extend(xyz)
    obs = np.concatenate(obs_parts)
    r_diag = np.concatenate(r_parts)
    n_obs = int(obs.size)

    dx = float(np.ptp(mesh.x)) if mesh.n_cells else 1.0
    dy = float(np.ptp(mesh.y)) if mesh.n_cells else 1.0
    dz = float(np.ptp(mesh.z)) if mesh.n_cells else 1.0
    diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    loc_r = max(diag * float(np.clip(0.50 / np.sqrt(max(n_obs, 1)), 0.20, 0.45)), 1.0)
    mesh_xyz = np.column_stack([mesh.x, mesh.y, mesh.z])
    md_loc = well_parameter_localization(
        mesh_xyz, np.asarray(xyz_parts, dtype=float), loc_r
    )

    rate_wells = [_rate_wells_from_sample(mesh, s) for s in samples]
    shape = mesh.grid.shape
    n_m = int(np.prod(shape))
    rng = np.random.default_rng(seed + 3)
    nrmse_hist: list[float] = []

    notes.append(
        f"joint multi-time ES-MDA ne={ne} Na={int(alphas.size)} "
        f"n_times={len(samples)} n_obs={n_obs} loc_r={loc_r:.3g}"
    )

    for alpha in alphas:
        d_sim = np.zeros((ne, n_obs), dtype=float)
        for e in range(ne):
            chunks = []
            for si, (s, sp) in enumerate(zip(samples, specs)):
                if not sp:
                    continue
                chunks.append(
                    _forward_joint_obs(
                        mesh,
                        s,
                        k_ens[e],
                        obs_spec=sp,
                        viscosity_pa_s=viscosity_pa_s,
                        oil_viscosity_pa_s=oil_viscosity_pa_s,
                        porosity=porosity_prior,
                        rate_wells=rate_wells[si],
                    )
                )
            d_sim[e, :] = np.concatenate(chunks)

        m = np.log(np.clip(k_ens, 1.0e-20, None)).reshape(ne, n_m)
        m = esmda_update_step(
            m,
            d_sim,
            obs,
            r_diag,
            float(alpha),
            rng,
            md_localization=md_loc,
            inflation=1.02,
        )
        k_ens = np.clip(np.exp(m.reshape(ne, *shape)), 1.0e-18, 1.0e-10)
        sig = np.sqrt(np.maximum(r_diag, 1.0e-30))
        nrmse = float(np.sqrt(np.mean(((np.mean(d_sim, axis=0) - obs) / sig) ** 2)))
        nrmse_hist.append(nrmse)
        notes.append(f"joint MDA step nRMSE={nrmse:.4g}")

    k_mean = np.mean(k_ens, axis=0)
    k_std = np.std(k_ens, axis=0)
    return k_mean, k_std, nrmse_hist, notes


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
            k = prev.permeability
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
