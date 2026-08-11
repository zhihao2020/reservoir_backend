"""ES-MDA for permeability under well-pressure observations.

Self-contained implementation inspired by Emerick & Reynolds (2013) and
common open-source practice (normalized alpha, R-preconditioning, optional
Gaspari–Cohn localization, ensemble inflation). Does **not** import
``references/`` upstream packages.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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


def _default_workers(*, n_cells: int = 0) -> int:
    """Parallel workers for ensemble forwards.

    Process pools help on **large** grids; on small CMG-like meshes spawn
    overhead dominates. Nested process pools auto-disable.
    Override with env ``RESERVOIR_BACKEND_WORKERS``.
    """
    if mp.current_process().name != "MainProcess":
        return 1
    env = os.environ.get("RESERVOIR_BACKEND_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    # small grids: serial is faster on Windows (spawn cost)
    if int(n_cells) > 0 and int(n_cells) < 2500:
        return 1
    cpu = os.cpu_count() or 4
    return max(1, min(6, cpu))


# Process-pool worker state (initialized once per child)
_WORKER_MESH: MeshBundle | None = None
_WORKER_SAMPLE: SensorSample | None = None
_WORKER_NAMES: list[str] | None = None
_WORKER_MU: float = 1.0e-3


def _init_forward_worker(
    mesh: MeshBundle,
    sample: SensorSample,
    well_names: list[str],
    viscosity_pa_s: float,
) -> None:
    global _WORKER_MESH, _WORKER_SAMPLE, _WORKER_NAMES, _WORKER_MU
    _WORKER_MESH = mesh
    _WORKER_SAMPLE = sample
    _WORKER_NAMES = list(well_names)
    _WORKER_MU = float(viscosity_pa_s)


def _forward_k_only(k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Child worker: forward one permeability member (mesh fixed in process)."""
    assert _WORKER_MESH is not None and _WORKER_SAMPLE is not None and _WORKER_NAMES is not None
    p = _forward_pressure_no_well_dirichlet(
        _WORKER_MESH,
        _WORKER_SAMPLE,
        permeability_m2=k,
        viscosity_pa_s=_WORKER_MU,
    )
    return _sample_well_pressures(_WORKER_MESH, p, _WORKER_NAMES)


def _process_forward_one(
    payload: tuple,
) -> NDArray[np.float64]:
    """Picklable single-member forward (fallback without initializer)."""
    mesh, sample, k, well_names, viscosity_pa_s = payload
    p = _forward_pressure_no_well_dirichlet(
        mesh,
        sample,
        permeability_m2=k,
        viscosity_pa_s=float(viscosity_pa_s),
    )
    return _sample_well_pressures(mesh, p, list(well_names))


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
    rng = np.random.default_rng(seed)
    log_mean = float(np.log(max(k_mean, 1.0e-20)))
    members = []
    for _ in range(ne):
        noise = rng.normal(0.0, 1.0, size=shape)
        smooth = _smooth3(noise, sigma=max(0.5, float(corr_len_cells) / 2.5))
        s = float(np.std(smooth)) + 1.0e-30
        smooth = smooth / s * float(logk_std)
        members.append(np.exp(log_mean + smooth))
    ens = np.stack(members, axis=0)
    return np.clip(ens, 1.0e-18, 1.0e-10)


def run_esmda_permeability(
    mesh: MeshBundle,
    samples: list[SensorSample],
    *,
    ne: int = 24,
    n_assimilations: int = 4,
    k_mean: float = 1.0e-13,
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

    Observations include injectors, producers, and ``observer_p`` entries present
    in ``sample.well_pressure``.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: s.time)
    ne = int(ne)
    alphas = normalize_alpha_weights(int(n_assimilations))
    na = int(alphas.size)

    k_ens = generate_logk_ensemble(
        mesh.grid.shape,
        ne=ne,
        k_mean=k_mean,
        logk_std=logk_std,
        corr_len_cells=corr_len_cells,
        seed=seed,
    )
    workers = (
        int(n_workers)
        if n_workers is not None
        else _default_workers(n_cells=int(np.prod(mesh.grid.shape)))
    )
    workers = max(1, min(workers, ne))
    # process pool only from main process; threads as fallback inside workers
    use_processes = workers > 1 and mp.current_process().name == "MainProcess"
    notes = [
        f"ES-MDA ne={ne} Na={na} alpha={alphas.tolist()}",
        f"prior k_mean={k_mean:.3e} logk_std={logk_std}",
        "forward: boundary Dirichlet + rate wells; soft pressure at sensors",
        f"ensemble forward workers={workers} "
        f"({'process' if use_processes else 'thread/serial'})",
        "practice sources: Emerick&Reynolds2013; alpha norm / R-precond / inflation",
    ]

    rmse_hist: list[float] = []
    rng = np.random.default_rng(seed + 7)
    well_names = _ordered_well_names(mesh, samples[0])
    if not well_names:
        raise ValueError("no wells on mesh for ES-MDA observations")

    # default localization: shrinks as observation count grows (more local updates)
    loc_r = localization_radius_m
    if loc_r is None and auto_localize:
        dx = float(np.ptp(mesh.x)) if mesh.n_cells else 0.0
        dy = float(np.ptp(mesh.y)) if mesh.n_cells else 0.0
        dz = float(np.ptp(mesh.z)) if mesh.n_cells else 0.0
        diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        n_obs = max(1, len(well_names))
        # 2 wells → ~0.45 diag; many probes → ~0.22 diag
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

    for sample in samples:
        obs = np.array([float(sample.well_pressure[n]) for n in well_names], dtype=float)
        sigma = np.maximum(np.abs(obs) * float(obs_std_frac), float(obs_std_floor_pa))
        r_diag = sigma**2

        # one pool per sample time (mesh+sample fixed); map only k arrays
        proc_pool: ProcessPoolExecutor | None = None
        if use_processes and workers > 1:
            try:
                proc_pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_forward_worker,
                    initargs=(mesh, sample, well_names, viscosity_pa_s),
                )
            except Exception:
                proc_pool = None

        try:
            for alpha in alphas:
                d_sim = _ensemble_pressure_obs(
                    mesh,
                    sample,
                    k_ens,
                    well_names=well_names,
                    viscosity_pa_s=viscosity_pa_s,
                    n_workers=workers,
                    use_processes=use_processes and proc_pool is not None,
                    process_pool=proc_pool,
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
        finally:
            if proc_pool is not None:
                proc_pool.shutdown(wait=True)

        notes.append(f"t={sample.time}: final forecast RMSE(Pa)≈{rmse_hist[-1]:.3e}")

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
        bundle.notes = list(bundle.notes) + ["permeability blended with ES-MDA ensemble mean"]
        history.append(bundle)
        prev = bundle

    phi_mean = history[-1].porosity if history else np.full(mesh.grid.shape, porosity_prior)

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


def _ensemble_pressure_obs(
    mesh: MeshBundle,
    sample: SensorSample,
    k_ens: NDArray[np.float64],
    *,
    well_names: list[str],
    viscosity_pa_s: float,
    n_workers: int,
    use_processes: bool = False,
    process_pool: ProcessPoolExecutor | None = None,
) -> NDArray[np.float64]:
    """Parallel forward map: each ensemble member → observed pressures."""
    ne = int(k_ens.shape[0])
    n_obs = len(well_names)
    d_sim = np.zeros((ne, n_obs), dtype=float)

    if n_workers <= 1 or ne <= 1:
        for e in range(ne):
            d_sim[e, :] = _process_forward_one(
                (mesh, sample, k_ens[e], well_names, viscosity_pa_s)
            )
        return d_sim

    if use_processes and process_pool is not None:
        try:
            # only ship k members; mesh/sample live in worker via initializer
            ks = [k_ens[e] for e in range(ne)]
            rows = list(process_pool.map(_forward_k_only, ks))
            for e, row in enumerate(rows):
                d_sim[e, :] = row
            return d_sim
        except Exception:
            pass

    def _one(e: int) -> tuple[int, NDArray[np.float64]]:
        row = _process_forward_one(
            (mesh, sample, k_ens[e], well_names, viscosity_pa_s)
        )
        return e, row

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for e, row in pool.map(_one, range(ne)):
            d_sim[e, :] = row
    return d_sim


def _forward_pressure_no_well_dirichlet(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64],
    viscosity_pa_s: float,
) -> NDArray[np.float64]:
    """TPFA pressure: face BC + rate sources; sensor cells free (soft obs)."""
    grid = mesh.grid
    boundaries = {
        key: float(value)
        for key, value in sample.boundary.pressure.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    ref = float(next(iter(sample.well_pressure.values()), 0.0))
    rate_wells = _rate_wells_from_sample(mesh, sample)
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=permeability_m2,
        ky=permeability_m2,
        kz=permeability_m2,
        mu=viscosity_pa_s,
        dirichlet_boundaries=boundaries or None,
        wells=rate_wells or None,
        reference_pressure=ref,
        cell_dirichlet=None,
    )
    return result.pressure.values


def _ordered_well_names(mesh: MeshBundle, sample: SensorSample) -> list[str]:
    """Pressure observation names: any hard p on mesh (wells + observer_p)."""
    names = [n for n in sample.well_pressure if n in mesh.well_cell_id]
    # stable order: injectors/producers first, then probes
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
