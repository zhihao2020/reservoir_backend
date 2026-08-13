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
_WORKER_OBS_SPEC: list | None = None  # list[list[ObsSpec]] per sample
_WORKER_MU: float = 1.0e-3
_WORKER_MU_O: float = 5.0e-3
_WORKER_PHI: float = 0.2
_WORKER_RATE_WELLS: list | None = None


def _init_forward_worker(
    mesh: MeshBundle,
    samples: list[SensorSample],
    obs_spec_by_t: list,
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    porosity_prior: float,
) -> None:
    global _WORKER_MESH, _WORKER_SAMPLES, _WORKER_OBS_SPEC
    global _WORKER_MU, _WORKER_MU_O, _WORKER_PHI, _WORKER_RATE_WELLS
    _WORKER_MESH = mesh
    _WORKER_SAMPLES = list(samples)
    _WORKER_OBS_SPEC = list(obs_spec_by_t)
    _WORKER_MU = float(viscosity_pa_s)
    _WORKER_MU_O = float(oil_viscosity_pa_s)
    _WORKER_PHI = float(porosity_prior)
    _WORKER_RATE_WELLS = [_rate_wells_from_sample(mesh, s) for s in _WORKER_SAMPLES]


def _forward_sample_k(payload: tuple[int, NDArray[np.float64]]) -> NDArray[np.float64]:
    """Child: (sample_index, k) → joint observation vector (p, Sw, qw)."""
    sample_idx, k = payload
    assert (
        _WORKER_MESH is not None
        and _WORKER_SAMPLES is not None
        and _WORKER_OBS_SPEC is not None
        and _WORKER_RATE_WELLS is not None
    )
    return _forward_joint_obs(
        _WORKER_MESH,
        _WORKER_SAMPLES[sample_idx],
        k,
        obs_spec=_WORKER_OBS_SPEC[sample_idx],
        viscosity_pa_s=_WORKER_MU,
        oil_viscosity_pa_s=_WORKER_MU_O,
        porosity=_WORKER_PHI,
        rate_wells=_WORKER_RATE_WELLS[sample_idx],
    )


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
    sw_obs_std: float = 0.06,
    qw_obs_std_frac: float = 0.15,
    porosity_prior: float = 0.2,
    viscosity_pa_s: float = 1.0e-3,
    oil_viscosity_pa_s: float = 5.0e-3,
    seed: int = 42,
    n_k_iterations: int = 1,
    localization_radius_m: float | None = None,
    ensemble_inflation: float = 1.02,
    auto_localize: bool = True,
    n_workers: int | None = None,
    assimilate_sw: bool = True,
    assimilate_qw: bool = True,
) -> ESMdaResult:
    """Multi-time ES-MDA on ``log(k)`` from well/probe soft observations.

    Soft data (product path — no CMG):
    - **pressure** at injectors/producers/``observer_p``
    - **Sw** at injectors/producers/``observer_s`` (if ``assimilate_sw``)
    - **water-rate proxy** ``|q|·f_w(S)`` at producers with rate+S (if ``assimilate_qw``)

    Forward: soft TPFA pressure + rate wells; Sw via sensor reconstruction
    (+ light transport when rates present). Alpha: ``sum 1/alpha_i = 1``.
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

    obs_spec_by_t = [
        build_observation_spec(
            mesh,
            s,
            assimilate_sw=assimilate_sw,
            assimilate_qw=assimilate_qw,
        )
        for s in samples
    ]
    if not any(obs_spec_by_t):
        raise ValueError("no pressure/Sw/rate observations on mesh for ES-MDA")

    n_forwards = len(samples) * na * ne
    workers = (
        int(n_workers)
        if n_workers is not None
        else _default_workers(n_forwards=n_forwards)
    )
    workers = max(1, min(workers, ne))
    use_processes = workers > 1 and mp.current_process().name == "MainProcess"
    n_p = sum(1 for sp in obs_spec_by_t[0] if sp.kind == "p")
    n_s = sum(1 for sp in obs_spec_by_t[0] if sp.kind == "sw")
    n_q = sum(1 for sp in obs_spec_by_t[0] if sp.kind == "qw")
    notes = [
        f"ES-MDA ne={ne} Na={na} alpha={alphas.tolist()}",
        f"prior k_mean={k_mean:.3e} logk_std={logk_std}",
        f"soft obs: n_p={n_p} n_sw={n_s} n_qw={n_q} (product wells+probes)",
        "forward: soft BHP + rate wells; Sw recon (+light transport)",
        f"ensemble forward workers={workers} "
        f"({'process-pool-reused' if use_processes else 'serial'}) "
        f"n_forwards≈{n_forwards}",
        "practice sources: Emerick&Reynolds2013; alpha norm / R-precond / inflation",
    ]

    rmse_hist: list[float] = []
    rng = np.random.default_rng(seed + 7)

    shape = mesh.grid.shape
    n_m = int(np.prod(shape))
    rate_wells_by_t = [_rate_wells_from_sample(mesh, s) for s in samples]

    proc_pool: ProcessPoolExecutor | None = None
    if use_processes:
        try:
            proc_pool = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_forward_worker,
                initargs=(
                    mesh,
                    samples,
                    obs_spec_by_t,
                    viscosity_pa_s,
                    oil_viscosity_pa_s,
                    porosity_prior,
                ),
            )
        except Exception:
            proc_pool = None
            notes.append("process pool init failed; serial ensemble forwards")

    try:
        for si, sample in enumerate(samples):
            spec = obs_spec_by_t[si]
            if not spec:
                continue
            obs, r_diag, xyz_obs = _observation_vector(
                mesh,
                sample,
                spec,
                obs_std_frac=obs_std_frac,
                obs_std_floor_pa=obs_std_floor_pa,
                sw_obs_std=sw_obs_std,
                qw_obs_std_frac=qw_obs_std_frac,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
                viscosity_pa_s=viscosity_pa_s,
            )
            md_loc = None
            if auto_localize or localization_radius_m is not None:
                loc_r = localization_radius_m
                if loc_r is None:
                    dx = float(np.ptp(mesh.x)) if mesh.n_cells else 0.0
                    dy = float(np.ptp(mesh.y)) if mesh.n_cells else 0.0
                    dz = float(np.ptp(mesh.z)) if mesh.n_cells else 0.0
                    diag = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                    frac = float(
                        np.clip(0.55 / np.sqrt(float(max(len(spec), 1))), 0.22, 0.50)
                    )
                    loc_r = max(diag * frac, 1.0)
                mesh_xyz = np.column_stack([mesh.x, mesh.y, mesh.z])
                md_loc = well_parameter_localization(
                    mesh_xyz, np.asarray(xyz_obs, dtype=float), float(loc_r)
                )

            for alpha in alphas:
                if proc_pool is not None:
                    payloads = [(si, k_ens[e]) for e in range(ne)]
                    try:
                        rows = list(proc_pool.map(_forward_sample_k, payloads))
                        d_sim = np.stack(rows, axis=0)
                    except Exception:
                        d_sim = _ensemble_serial_joint(
                            mesh,
                            sample,
                            k_ens,
                            spec,
                            viscosity_pa_s,
                            oil_viscosity_pa_s,
                            porosity_prior,
                            rate_wells_by_t[si],
                        )
                else:
                    d_sim = _ensemble_serial_joint(
                        mesh,
                        sample,
                        k_ens,
                        spec,
                        viscosity_pa_s,
                        oil_viscosity_pa_s,
                        porosity_prior,
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

                # normalized RMSE (dimensionless) for mixed p/S/q units
                sig = np.sqrt(np.maximum(r_diag, 1.0e-30))
                nrmse = float(
                    np.sqrt(np.mean(((np.mean(d_sim, axis=0) - obs) / sig) ** 2))
                )
                rmse_hist.append(nrmse)

            notes.append(
                f"t={sample.time}: joint soft-obs nRMSE≈{rmse_hist[-1]:.3g} "
                f"(n_obs={len(spec)})"
            )
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


@dataclass(frozen=True)
class ObsSpec:
    """One soft observation channel at a named sensor."""

    kind: str  # p | sw | qw
    name: str
    cell_id: int


def build_observation_spec(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    assimilate_sw: bool = True,
    assimilate_qw: bool = True,
) -> list[ObsSpec]:
    """Build product-path soft obs list from wells + exclusive probes."""
    specs: list[ObsSpec] = []
    for name, _p in (sample.well_pressure or {}).items():
        if name not in mesh.well_cell_id:
            continue
        role = mesh.well_role.get(name, "")
        if role in ("observer_s",):
            continue
        specs.append(ObsSpec("p", name, int(mesh.well_cell_id[name])))

    if assimilate_sw:
        for name, phases in (sample.well_saturation or {}).items():
            if name not in mesh.well_cell_id:
                continue
            role = mesh.well_role.get(name, "")
            if role in ("observer_p",):
                continue
            specs.append(ObsSpec("sw", name, int(mesh.well_cell_id[name])))

    if assimilate_qw:
        for name, q in (sample.well_rate or {}).items():
            if name not in mesh.well_cell_id:
                continue
            if mesh.well_role.get(name) != "producer":
                continue
            if name not in (sample.well_saturation or {}):
                continue
            if float(q) >= 0.0:
                continue
            specs.append(ObsSpec("qw", name, int(mesh.well_cell_id[name])))

    # stable order: p, sw, qw then name
    order = {"p": 0, "sw": 1, "qw": 2}
    specs.sort(key=lambda s: (order.get(s.kind, 9), s.name))
    return specs


def _observation_vector(
    mesh: MeshBundle,
    sample: SensorSample,
    spec: list[ObsSpec],
    *,
    obs_std_frac: float,
    obs_std_floor_pa: float,
    sw_obs_std: float,
    qw_obs_std_frac: float,
    oil_viscosity_pa_s: float,
    viscosity_pa_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[list[float]]]:
    from reservoir_backend.pipeline.fractional_flow import water_fractional_flow

    vals: list[float] = []
    vars_: list[float] = []
    xyz: list[list[float]] = []
    for ch in spec:
        c = ch.cell_id
        xyz.append([float(mesh.x[c]), float(mesh.y[c]), float(mesh.z[c])])
        if ch.kind == "p":
            v = float(sample.well_pressure[ch.name])
            vals.append(v)
            vars_.append(max(abs(v) * float(obs_std_frac), float(obs_std_floor_pa)) ** 2)
        elif ch.kind == "sw":
            v = float(sample.well_saturation[ch.name][0])
            vals.append(v)
            vars_.append(float(sw_obs_std) ** 2)
        else:  # qw water-rate proxy
            q = float(sample.well_rate[ch.name])
            sw = float(sample.well_saturation[ch.name][0])
            fw = float(
                water_fractional_flow(
                    sw, mu_w=viscosity_pa_s, mu_o=oil_viscosity_pa_s
                )
            )
            v = abs(q) * fw
            vals.append(v)
            vars_.append(max(abs(v) * float(qw_obs_std_frac), 1.0e-12) ** 2)
    return np.asarray(vals, dtype=float), np.asarray(vars_, dtype=float), xyz


def _ensemble_serial_joint(
    mesh: MeshBundle,
    sample: SensorSample,
    k_ens: NDArray[np.float64],
    obs_spec: list[ObsSpec],
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    porosity: float,
    rate_wells: list,
) -> NDArray[np.float64]:
    ne = int(k_ens.shape[0])
    rows = []
    for e in range(ne):
        rows.append(
            _forward_joint_obs(
                mesh,
                sample,
                k_ens[e],
                obs_spec=obs_spec,
                viscosity_pa_s=viscosity_pa_s,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
                porosity=porosity,
                rate_wells=rate_wells,
            )
        )
    return np.stack(rows, axis=0)


def _forward_joint_obs(
    mesh: MeshBundle,
    sample: SensorSample,
    permeability_m2: float | NDArray[np.float64],
    *,
    obs_spec: list[ObsSpec],
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
    porosity: float,
    rate_wells: list | None,
    dt: float | None = None,
    sw_prev: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Soft pressure + Sw reconstruction (+ transport) → obs vector."""
    from reservoir_backend.pipeline.saturation_field import reconstruct_saturation
    from reservoir_backend.pipeline.transport_saturation import (
        phases_from_sw,
        transport_water_saturation,
    )

    p = _forward_pressure_cached(
        mesh,
        sample,
        permeability_m2=permeability_m2,
        viscosity_pa_s=viscosity_pa_s,
        rate_wells=rate_wells,
    )
    need_sw = any(ch.kind in ("sw", "qw") for ch in obs_spec)
    sw = None
    if need_sw:
        sample_s = SensorSample(
            time=sample.time,
            well_pressure={},
            well_saturation=dict(sample.well_saturation or {}),
            boundary=sample.boundary,
            well_rate={},
        )
        sw_rec, _so, _sg, _ = reconstruct_saturation(mesh, sample_s, pressure=p)
        if sw_prev is not None and dt is not None and float(dt) > 0.0 and sample.well_rate:
            n_sub = int(np.clip(round(float(dt) / 5.0), 4, 24))
            sw_t, _ = transport_water_saturation(
                mesh,
                0.4 * sw_prev + 0.6 * sw_rec,
                p,
                permeability_m2,
                sample,
                porosity=porosity,
                viscosity_pa_s=viscosity_pa_s,
                oil_viscosity_pa_s=oil_viscosity_pa_s,
                dt=float(dt),
                n_substeps=n_sub,
            )
            sw, _, _ = phases_from_sw(sw_t, sample=sample_s, mesh=mesh)
        else:
            sw = sw_rec

    return _sample_obs_from_fields(
        mesh,
        sample,
        obs_spec,
        pressure=p,
        sw=sw,
        viscosity_pa_s=viscosity_pa_s,
        oil_viscosity_pa_s=oil_viscosity_pa_s,
    )


def _sample_obs_from_fields(
    mesh: MeshBundle,
    sample: SensorSample,
    obs_spec: list[ObsSpec],
    *,
    pressure: NDArray[np.float64],
    sw: NDArray[np.float64] | None,
    viscosity_pa_s: float,
    oil_viscosity_pa_s: float,
) -> NDArray[np.float64]:
    """Extract soft-obs vector from pressure/Sw fields."""
    from reservoir_backend.pipeline.fractional_flow import water_fractional_flow

    out = np.zeros(len(obs_spec), dtype=float)
    for i, ch in enumerate(obs_spec):
        ii, jj, kk = mesh.grid.ijk(ch.cell_id)
        if ch.kind == "p":
            out[i] = float(pressure[kk, jj, ii])
        elif ch.kind == "sw":
            if sw is None:
                out[i] = 0.0
            else:
                out[i] = float(sw[kk, jj, ii])
        else:
            if sw is None:
                out[i] = 0.0
            else:
                q = float((sample.well_rate or {}).get(ch.name, 0.0))
                fw = float(
                    water_fractional_flow(
                        float(sw[kk, jj, ii]),
                        mu_w=viscosity_pa_s,
                        mu_o=oil_viscosity_pa_s,
                    )
                )
                out[i] = abs(q) * fw
    return out


def _forward_pressure_cached(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64],
    viscosity_pa_s: float,
    rate_wells: list | None = None,
    saturation: NDArray[np.float64] | None = None,
    oil_viscosity_pa_s: float = 5.0e-3,
) -> NDArray[np.float64]:
    """TPFA pressure: face BC + rate sources; sensor cells free (soft obs)."""
    grid = mesh.grid
    boundaries = {
        key: float(value)
        for key, value in sample.boundary.pressure.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    if sample.well_pressure:
        ref = float(next(iter(sample.well_pressure.values())))
    else:
        ref = float(next(iter(boundaries.values()), 0.0))
    wells = rate_wells if rate_wells is not None else _rate_wells_from_sample(mesh, sample)
    k_use: float | NDArray[np.float64] = permeability_m2
    if saturation is not None:
        from reservoir_backend.pipeline.fractional_flow import total_mobility

        sw = np.asarray(saturation, dtype=float)
        if sw.shape == grid.shape:
            lam = total_mobility(
                sw, mu_w=viscosity_pa_s, mu_o=oil_viscosity_pa_s
            )
            k_arr = np.asarray(permeability_m2, dtype=float)
            if k_arr.ndim == 0:
                k_arr = np.full(grid.shape, float(k_arr))
            k_use = np.clip(k_arr * lam * float(viscosity_pa_s), 1.0e-22, 1.0e-8)
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=k_use,
        ky=k_use,
        kz=k_use,
        mu=viscosity_pa_s,
        dirichlet_boundaries=boundaries or None,
        wells=wells or None,
        reference_pressure=ref,
        cell_dirichlet=None,
    )
    return result.pressure.values


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
