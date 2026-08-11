"""ES-MDA for permeability under well-pressure observations.

Self-contained implementation inspired by Emerick & Reynolds (2013) and
common open-source practice (normalized alpha, R-preconditioning, optional
Gaspari–Cohn localization, ensemble inflation). Does **not** import
``references/`` upstream packages.

Performance (accuracy-preserving):
- vectorized TPFA assembly (solver)
- one process pool for the whole assimilation (amortize Windows spawn)
- workers map only permeability arrays after initializer loads mesh/samples
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.ensemble_math import (
    esmda_update_step,
    normalize_alpha_weights,
    well_parameter_localization,
)
from reservoir_backend.pipeline.pressure_field import _rate_wells_from_sample
from reservoir_backend.pipeline.run import run_time_slice
from reservoir_backend.pipeline.state import FieldBundle, MeshBundle, SensorSample
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d


def _default_workers(*, n_forwards: int = 0) -> int:
    """Workers for ensemble forwards (max accuracy path, max speed).

    Always parallelize when the total forward count is large enough to
    amortize process spawn. Nested pools (probe-study workers) stay serial.
    Override: ``RESERVOIR_BACKEND_WORKERS``.
    """
    if mp.current_process().name != "MainProcess":
        return 1
    env = os.environ.get("RESERVOIR_BACKEND_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    # many forwards → use cores; tiny jobs stay serial
    if int(n_forwards) < 24:
        return 1
    return max(1, min(cpu, 8))


# ---- process worker state (one init per pool lifetime) ----
_WORKER_MESH: MeshBundle | None = None
_WORKER_SAMPLES: list[SensorSample] | None = None
_WORKER_NAMES: list[str] | None = None
_WORKER_MU: float = 1.0e-3
_WORKER_RATE_WELLS: list | None = None  # per-sample cached wells


def _init_forward_worker(
    mesh: MeshBundle,
    samples: list[SensorSample],
    well_names: list[str],
    viscosity_pa_s: float,
) -> None:
    global _WORKER_MESH, _WORKER_SAMPLES, _WORKER_NAMES, _WORKER_MU, _WORKER_RATE_WELLS
    _WORKER_MESH = mesh
    _WORKER_SAMPLES = list(samples)
    _WORKER_NAMES = list(well_names)
    _WORKER_MU = float(viscosity_pa_s)
    _WORKER_RATE_WELLS = [_rate_wells_from_sample(mesh, s) for s in _WORKER_SAMPLES]


def _forward_sample_k(payload: tuple[int, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Child: (sample_index, k) → observed pressures."""
    sample_idx, k = payload
    assert (
        _WORKER_MESH is not None
        and _WORKER_SAMPLES is not None
        and _WORKER_NAMES is not None
        and _WORKER_RATE_WELLS is not None
    )
    sample = _WORKER_SAMPLES[sample_idx]
    rate_wells = _WORKER_RATE_WELLS[sample_idx]
    p = _forward_pressure_cached(
        _WORKER_MESH,
        sample,
        permeability_m2=k,
        viscosity_pa_s=_WORKER_MU,
        rate_wells=rate_wells,
    )
    return _sample_well_pressures(_WORKER_MESH, p, _WORKER_NAMES)


@dataclass
class ESMdaResult:
    """Ensemble smoother – multiple data assimilation output."""

    mesh: MeshBundle
    k_mean: NDArray[np.float64]
    k_std: NDArray[np.float64]
    k_ensemble: NDArray[np.float64]  # (ne, nz, ny, nx)
    phi_mean: NDArray[np.float64]
    history_mean: list[FieldBundle] = field(default_factory=list)
    observation_rmse: list[float] = field(default_factory=list)
    alpha_schedule: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def generate_logk_ensemble(
    shape: tuple[int, int, int],
    *,
    ne: int,
    k_mean: float,
    logk_std: float = 1.0,
    corr_len_cells: float = 3.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Gaussian-smoothed log-k ensemble around ``log(k_mean)``.

    Returns array shape ``(ne, nz, ny, nx)`` in linear k [m^2].
    """
    base = np.full(shape, float(max(k_mean, 1.0e-20)), dtype=float)
    return generate_logk_ensemble_around(
        base, ne=ne, logk_std=logk_std, corr_len_cells=corr_len_cells, seed=seed
    )


def generate_logk_ensemble_around(
    k_base: NDArray[np.float64],
    *,
    ne: int,
    logk_std: float = 1.0,
    corr_len_cells: float = 3.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Ensemble of log-k perturbations around a **spatial** permeability map.

    Used for second-pass / outer-loop ES-MDA so updates refine structure instead
    of re-drawing from a scalar prior (accuracy-critical).
    """
    base = np.clip(np.asarray(k_base, dtype=float), 1.0e-30, None)
    log_base = np.log(base)
    rng = np.random.default_rng(seed)
    members = []
    sigma = max(0.5, float(corr_len_cells) / 2.5)
    for _ in range(int(ne)):
        noise = rng.normal(0.0, 1.0, size=base.shape)
        smooth = _smooth3(noise, sigma=sigma)
        s = float(np.std(smooth)) + 1.0e-30
        smooth = smooth / s * float(logk_std)
        members.append(np.exp(log_base + smooth))
    ens = np.stack(members, axis=0)
    return np.clip(ens, 1.0e-18, 1.0e-10)


def run_esmda_permeability(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    ne: int = 24,
    n_assimilations: int = 4,
    k_mean: float = 1.0e-13,
    k_prior_field: NDArray[np.float64] | None = None,
    logk_std: float = 1.0,
    corr_len_cells: float = 3.0,
    obs_std_frac: float = 0.02,
    obs_std_floor_pa: float = 5.0e4,
    porosity_prior: float = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    seed: int = 42,
    n_k_iterations: int = 1,
    localization_radius_m: float | None = None,
    ensemble_inflation: float = 1.02,
    auto_localize: bool = True,
    n_workers: int | None = None,
) -> ESMdaResult:
    """Multi-time ES-MDA on ``log(k)`` using pressure hard data as soft obs.

    Forward map: TPFA with face Dirichlet + **rate wells** (no cell Dirichlet on
    sensors), so permeability influences predicted BHP/probe pressures.
    Alpha weights satisfy ``sum 1/alpha_i = 1`` (Emerick & Reynolds).
    """
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: s.time)
    ne = int(ne)
    alphas = normalize_alpha_weights(int(n_assimilations))
    na = int(alphas.size)

    if k_prior_field is not None:
        k_ens = generate_logk_ensemble_around(
            np.asarray(k_prior_field, dtype=float),
            ne=ne,
            logk_std=logk_std,
            corr_len_cells=corr_len_cells,
            seed=seed,
        )
    else:
        k_ens = generate_logk_ensemble(
            mesh.grid.shape,
            ne=ne,
            k_mean=float(k_mean),
            logk_std=logk_std,
            corr_len_cells=corr_len_cells,
            seed=seed,
        )
    n_forwards = len(samples) * na * ne
    workers = (
        int(n_workers)
        if n_workers is not None
        else _default_workers(n_forwards=n_forwards)
    )
    workers = max(1, min(workers, ne))
    use_processes = workers > 1 and mp.current_process().name == "MainProcess"
    notes = [
        f"ES-MDA ne={ne} Na={na} alpha={alphas.tolist()}",
        f"prior k_mean={k_mean:.3e} logk_std={logk_std}",
        "forward: boundary Dirichlet + rate wells; soft pressure at sensors",
        f"ensemble forward workers={workers} "
        f"({'process-pool-reused' if use_processes else 'serial'}) "
        f"n_forwards≈{n_forwards}",
        "practice sources: Emerick&Reynolds2013; alpha norm / R-precond / inflation",
    ]

    rmse_hist: list[float] = []
    rng = np.random.default_rng(seed + 7)
    well_names = _ordered_well_names(mesh, samples[0])
    if not well_names:
        raise ValueError("no wells on mesh for ES-MDA observations")

    loc_r = localization_radius_m
    if loc_r is None and auto_localize:
        dx = float(np.ptp(mesh.x)) if mesh.n_cells else 0.0
        dy = float(np.ptp(mesh.y)) if mesh.n_cells else 0.0
        dz = float(np.ptp(mesh.z)) if mesh.n_cells else 0.0
        diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        n_obs = max(1, len(well_names))
        frac = float(np.clip(0.55 / np.sqrt(float(n_obs)), 0.22, 0.50))
        loc_r = max(diag * frac, 1.0)
    md_loc = None
    if loc_r is not None and float(loc_r) > 0.0:
        mesh_xyz = np.column_stack([mesh.x, mesh.y, mesh.z])
        well_xyz = []
        for name in well_names:
            c = mesh.well_cell_id[name]
            well_xyz.append([mesh.x[c], mesh.y[c], mesh.z[c]])
        md_loc = well_parameter_localization(
            mesh_xyz, np.asarray(well_xyz, dtype=float), float(loc_r)
        )
        notes.append(f"Gaspari-Cohn localization radius_m={float(loc_r):.3g}")

    shape = mesh.grid.shape
    n_m = int(np.prod(shape))

    # serial cache for rate wells
    rate_wells_by_t = [_rate_wells_from_sample(mesh, s) for s in samples]

    proc_pool: ProcessPoolExecutor | None = None
    if use_processes:
        try:
            proc_pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_forward_worker,
                initargs=(mesh, samples, well_names, viscosity_pa_s),
            )
        except Exception:
            proc_pool = None
            notes.append("process pool init failed; serial ensemble forwards")

    try:
        for si, sample in enumerate(samples):
            obs = np.array(
                [float(sample.well_pressure[n]) for n in well_names], dtype=float
            )
            sigma = np.maximum(
                np.abs(obs) * float(obs_std_frac), float(obs_std_floor_pa)
            )
            r_diag = sigma**2

            for alpha in alphas:
                if proc_pool is not None:
                    payloads = [(si, k_ens[e]) for e in range(ne)]
                    try:
                        rows = list(proc_pool.map(_forward_sample_k, payloads))
                        d_sim = np.stack(rows, axis=0)
                    except Exception:
                        d_sim = _ensemble_serial(
                            mesh,
                            sample,
                            k_ens,
                            well_names,
                            viscosity_pa_s,
                            rate_wells_by_t[si],
                        )
                else:
                    d_sim = _ensemble_serial(
                        mesh,
                        sample,
                        k_ens,
                        well_names,
                        viscosity_pa_s,
                        rate_wells_by_t[si],
                    )

                m = np.log(np.clip(k_ens, 1.0e-20, None)).reshape(ne, n_m)
                m = esmda_update_step(
                    m,
                    d_sim,
                    obs,
                    r_diag,
                    float(alpha),
                    rng,
                    md_localization=md_loc,
                    inflation=float(ensemble_inflation),
                )
                k_ens = np.exp(m.reshape(ne, *shape))
                k_ens = np.clip(k_ens, 1.0e-18, 1.0e-10)

                rmse = float(np.sqrt(np.mean((np.mean(d_sim, axis=0) - obs) ** 2)))
                rmse_hist.append(rmse)

            notes.append(f"t={sample.time}: final forecast RMSE(Pa)≈{rmse_hist[-1]:.3e}")
    finally:
        if proc_pool is not None:
            proc_pool.shutdown(wait=True)

    k_mean_f = np.mean(k_ens, axis=0)
    k_std_f = np.std(k_ens, axis=0)

    history: list[FieldBundle] = []
    prev: FieldBundle | None = None
    for sample in samples:
        dt = None if prev is None else float(sample.time - prev.time)
        if dt is not None and dt <= 0:
            dt = None
        bundle = run_time_slice(
            mesh,
            sample,
            permeability_prior_m2=k_mean_f if prev is None else prev.permeability,
            porosity_prior=porosity_prior if prev is None else prev.porosity,
            viscosity_pa_s=viscosity_pa_s,
            previous=prev,
            dt=dt,
            n_k_iterations=n_k_iterations,
        )
        bundle.permeability = 0.7 * k_mean_f + 0.3 * bundle.permeability
        bundle.permeability = np.clip(bundle.permeability, 1.0e-18, 1.0e-10)
        bundle.notes = list(bundle.notes) + [
            "permeability blended with ES-MDA ensemble mean"
        ]
        history.append(bundle)
        prev = bundle

    phi_mean = (
        history[-1].porosity
        if history
        else np.full(mesh.grid.shape, porosity_prior)
    )

    return ESMdaResult(
        mesh=mesh,
        k_mean=k_mean_f,
        k_std=k_std_f,
        k_ensemble=k_ens,
        phi_mean=phi_mean,
        history_mean=history,
        observation_rmse=rmse_hist,
        alpha_schedule=[float(a) for a in alphas],
        notes=notes,
    )


def _ensemble_serial(
    mesh: MeshBundle,
    sample: SensorSample,
    k_ens: NDArray[np.float64],
    well_names: list[str],
    viscosity_pa_s: float,
    rate_wells: list,
) -> NDArray[np.float64]:
    ne = int(k_ens.shape[0])
    d_sim = np.zeros((ne, len(well_names)), dtype=float)
    for e in range(ne):
        p = _forward_pressure_cached(
            mesh,
            sample,
            permeability_m2=k_ens[e],
            viscosity_pa_s=viscosity_pa_s,
            rate_wells=rate_wells,
        )
        d_sim[e, :] = _sample_well_pressures(mesh, p, well_names)
    return d_sim


def _forward_pressure_cached(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64],
    viscosity_pa_s: float,
    rate_wells: list | None = None,
) -> NDArray[np.float64]:
    """TPFA pressure: face BC + rate sources; sensor cells free (soft obs)."""
    grid = mesh.grid
    boundaries = {
        key: float(value)
        for key, value in sample.boundary.pressure.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    ref = float(next(iter(sample.well_pressure.values()), 0.0))
    wells = rate_wells if rate_wells is not None else _rate_wells_from_sample(mesh, sample)
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=permeability_m2,
        ky=permeability_m2,
        kz=permeability_m2,
        mu=viscosity_pa_s,
        dirichlet_boundaries=boundaries or None,
        wells=wells or None,
        reference_pressure=ref,
        cell_dirichlet=None,
    )
    return result.pressure.values


def _forward_pressure_no_well_dirichlet(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64],
    viscosity_pa_s: float,
) -> NDArray[np.float64]:
    return _forward_pressure_cached(
        mesh,
        sample,
        permeability_m2=permeability_m2,
        viscosity_pa_s=viscosity_pa_s,
        rate_wells=None,
    )


def _ordered_well_names(mesh: MeshBundle, sample: SensorSample) -> list[str]:
    """Pressure observation names: any hard p on mesh (wells + observer_p)."""
    names = [n for n in sample.well_pressure if n in mesh.well_cell_id]

    def _key(n: str) -> tuple[int, str]:
        r = mesh.well_role.get(n, "")
        if r == "injector":
            return (0, n)
        if r == "producer":
            return (1, n)
        if r == "observer_p":
            return (2, n)
        return (3, n)

    return sorted(names, key=_key)


def _sample_well_pressures(
    mesh: MeshBundle,
    pressure: NDArray[np.float64],
    well_names: list[str],
) -> NDArray[np.float64]:
    out = np.zeros(len(well_names), dtype=float)
    for i, name in enumerate(well_names):
        cell = mesh.well_cell_id[name]
        ii, jj, kk = mesh.grid.ijk(cell)
        out[i] = float(pressure[kk, jj, ii])
    return out


def _smooth3(arr: NDArray[np.float64], *, sigma: float) -> NDArray[np.float64]:
    out = arr.astype(float, copy=True)
    passes = max(1, int(round(sigma)))
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        acc = padded[1:-1, 1:-1, 1:-1] * 6.0
        acc += padded[1:-1, 1:-1, 0:-2]
        acc += padded[1:-1, 1:-1, 2:]
        acc += padded[1:-1, 0:-2, 1:-1]
        acc += padded[1:-1, 2:, 1:-1]
        acc += padded[0:-2, 1:-1, 1:-1]
        acc += padded[2:, 1:-1, 1:-1]
        out = acc / 12.0
    return out
