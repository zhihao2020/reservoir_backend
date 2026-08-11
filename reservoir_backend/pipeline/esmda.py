"""Lightweight ES-MDA for permeability under well-pressure observations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.pipeline.run import run_time_slice
from reservoir_backend.pipeline.state import FieldBundle, MeshBundle, SensorSample
from reservoir_backend.solver.pressure_solver import solve_steady_state_pressure_3d


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
    nz, ny, nx = shape
    log_mean = float(np.log(max(k_mean, 1.0e-20)))
    members = []
    for _ in range(ne):
        noise = rng.normal(0.0, 1.0, size=shape)
        smooth = _smooth3(noise, sigma=max(0.5, float(corr_len_cells) / 2.5))
        # re-std
        s = float(np.std(smooth)) + 1.0e-30
        smooth = smooth / s * float(logk_std)
        logk = log_mean + smooth
        members.append(np.exp(logk))
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
) -> ESMdaResult:
    """Sequential multi-time ES-MDA on ``log(k)`` using well pressure data.

    Forward map per member: TPFA pressure reconstruction with that k field;
    observations are well-cell pressures at each sample time. Between times the
    ensemble is carried forward (no model error). Alpha schedule is the standard
    equal-weight ES-MDA ``alpha = Na``.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    samples = sorted(samples, key=lambda s: s.time)
    ne = int(ne)
    na = max(1, int(n_assimilations))
    alpha = float(na)

    k_ens = generate_logk_ensemble(
        mesh.grid.shape,
        ne=ne,
        k_mean=k_mean,
        logk_std=logk_std,
        corr_len_cells=corr_len_cells,
        seed=seed,
    )
    notes = [
        f"ES-MDA ne={ne} Na={na} alpha={alpha}",
        f"prior k_mean={k_mean:.3e} logk_std={logk_std}",
    ]
    rmse_hist: list[float] = []
    rng = np.random.default_rng(seed + 7)

    well_names = _ordered_well_names(mesh, samples[0])
    if not well_names:
        raise ValueError("no wells on mesh for ES-MDA observations")

    for sample in samples:
        obs = np.array([float(sample.well_pressure[n]) for n in well_names], dtype=float)
        n_obs = obs.size
        # diagonal observation noise
        sigma = np.maximum(np.abs(obs) * float(obs_std_frac), float(obs_std_floor_pa))
        r_diag = sigma**2

        for _ia in range(na):
            # forecast observations: pressure at well cells WITHOUT hard Dirichlet
            # so k actually influences predicted well BHP (boundaries only).
            d_sim = np.zeros((ne, n_obs), dtype=float)
            for e in range(ne):
                p = _forward_pressure_no_well_dirichlet(
                    mesh,
                    sample,
                    permeability_m2=k_ens[e],
                    viscosity_pa_s=viscosity_pa_s,
                )
                d_sim[e, :] = _sample_well_pressures(mesh, p, well_names)

            # ensemble in log space
            m = np.log(np.clip(k_ens, 1.0e-20, None)).reshape(ne, -1)  # (ne, n_state)
            m_mean = np.mean(m, axis=0)
            d_mean = np.mean(d_sim, axis=0)
            am = m - m_mean
            ad = d_sim - d_mean

            # Cov_md (n_state, n_obs), Cov_dd (n_obs, n_obs)
            cov_md = (am.T @ ad) / max(ne - 1, 1)
            cov_dd = (ad.T @ ad) / max(ne - 1, 1)
            cov_dd = cov_dd + alpha * np.diag(r_diag)

            # solve cov_dd X^T = cov_md^T  →  K = cov_md @ inv(cov_dd)
            try:
                k_gain = np.linalg.solve(cov_dd, cov_md.T).T  # (n_state, n_obs)
            except np.linalg.LinAlgError:
                k_gain = cov_md @ np.linalg.pinv(cov_dd)

            for e in range(ne):
                innov = obs + np.sqrt(alpha) * rng.normal(0.0, sigma) - d_sim[e]
                m[e] = m[e] + k_gain @ innov

            k_ens = np.exp(m.reshape(ne, *mesh.grid.shape))
            k_ens = np.clip(k_ens, 1.0e-18, 1.0e-10)

            rmse = float(np.sqrt(np.mean((np.mean(d_sim, axis=0) - obs) ** 2)))
            rmse_hist.append(rmse)

        notes.append(f"t={sample.time}: final forecast RMSE(Pa)≈{rmse_hist[-1]:.3e}")

    k_mean_f = np.mean(k_ens, axis=0)
    k_std_f = np.std(k_ens, axis=0)

    # reconstruct mean history with ensemble-mean k carried across times
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
        # blend ES-MDA mean k with local inversion for mild consistency
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
        notes=notes,
    )


def _forward_pressure_no_well_dirichlet(
    mesh: MeshBundle,
    sample: SensorSample,
    *,
    permeability_m2: float | NDArray[np.float64],
    viscosity_pa_s: float,
) -> NDArray[np.float64]:
    """TPFA pressure using face BC only — well cells remain free unknowns."""
    grid = mesh.grid
    boundaries = {
        key: float(value)
        for key, value in sample.boundary.pressure.items()
        if key in {"left", "right", "front", "back", "bottom", "top"}
    }
    ref = float(next(iter(sample.well_pressure.values()), 0.0))
    result = solve_steady_state_pressure_3d(
        grid=grid,
        kx=permeability_m2,
        ky=permeability_m2,
        kz=permeability_m2,
        mu=viscosity_pa_s,
        dirichlet_boundaries=boundaries or None,
        wells=None,
        reference_pressure=ref,
        cell_dirichlet=None,
    )
    return result.pressure.values


def _ordered_well_names(mesh: MeshBundle, sample: SensorSample) -> list[str]:
    names = []
    for n in sample.well_pressure:
        if n in mesh.well_cell_id:
            names.append(n)
    return names


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
    """Separable box/gaussian-ish smooth via repeated averaging."""
    out = arr.astype(float, copy=True)
    passes = max(1, int(round(sigma)))
    for _ in range(passes):
        # 6-neighbor average + self
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
