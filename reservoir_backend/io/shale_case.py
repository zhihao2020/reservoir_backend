"""Build shale-oil depletion DigitalTwin from truth JSON + IMEX .out."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.frac import (
    FractureStripParameterization,
    MD_TO_M2,
    WellTrack,
    decode_frac_theta,
    default_shale_prior,
    paint_fracture_strips,
    wells_from_truth,
)
from reservoir_backend.io.cmg_out import ft_to_m, parse_grid_series, psi_to_pa
from reservoir_backend.physics.pvt import PSI, BlackOilPVT
from reservoir_backend.physics.relperm import TableThreePhase, TableTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import (
    DigitalTwin,
    InverseSpec,
    PhysicsSpec,
    Posterior,
    predict_from_trajectory,
    stack_observations,
)
from reservoir_backend.solver.impes import Trajectory

DAY_S = 86400.0
STB_TO_M3 = 0.158987294928
TOTAL_Q_M3S = -800.0 * STB_TO_M3 / 86400.0  # IMEX *MAX *STO 800 STB/day (surface)
S5_SHUTIN_DAY = 273.0
S5_REOPEN_DAY = 365.0


def perf_step(n_frac_planes: int) -> int:
    return 2 if int(n_frac_planes) >= 8 else 3


def grid_from_truth(truth: dict) -> CartesianGrid:
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    return CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, ft_to_m(g["di_ft"])),
        dy=np.full(ny, ft_to_m(g["dj_ft"])),
        dz=np.array([ft_to_m(v) for v in g["dk_ft"]], dtype=float),
    )


def frac_mask_from_truth(truth: dict, grid: CartesianGrid) -> NDArray[np.bool_]:
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    mask = np.zeros((nz, ny, nx), dtype=bool)
    blocks = truth.get("high_k_blocks_ijk") or truth.get("channel_blocks_ijk") or []
    for i, j, k in blocks:
        if 1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz:
            mask[k - 1, j - 1, i - 1] = True
    return mask.reshape(-1)


def truth_half_length_m(truth: dict) -> float:
    g = truth["grid"]
    dj = ft_to_m(g["dj_ft"])
    mask = frac_mask_from_truth(truth, grid_from_truth(truth)).reshape(
        int(g["nz"]), int(g["ny"]), int(g["nx"])
    )
    js = np.where(mask.any(axis=(0, 2)))[0]
    if js.size == 0:
        return float("nan")
    return 0.5 * (float(js.max() - js.min()) + 1.0) * dj


def _port_open(day: float, well: WellTrack, scenario: str) -> bool:
    if day < float(well.open_from_day):
        return False
    if scenario.upper() == "S5" and well.name == "HW1":
        if S5_SHUTIN_DAY <= day < S5_REOPEN_DAY:
            return False
    return True


def _build_ports_and_sensors(
    grid: CartesianGrid,
    wells: tuple[WellTrack, ...],
    truth: dict,
) -> tuple[list[FlowPort], list[Sensor]]:
    step = perf_step(len(truth.get("frac_i_planes") or []))
    ports: list[FlowPort] = []
    sensors: list[Sensor] = []
    centers = grid.cell_centers()
    for well in wells:
        for i in range(int(well.i0), int(well.i1) + 1, step):
            cell = int(grid.index(i, well.j, well.k))
            name = f"{well.name}_{i + 1:02d}"
            ports.append(
                FlowPort(
                    name=name,
                    role="producer",
                    control="rate",
                    cell_ids=np.array([cell], dtype=np.int64),
                    use_productivity=True,
                    rw_m=0.25 * 0.3048,
                    geofac=0.34,
                    axis="j",
                    min_bhp_Pa=1500.0 * PSI,
                )
            )
            x, y, z = centers[cell]
            sensors.append(
                Sensor(
                    name=name,
                    kind="pressure",
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    sigma=5.0e4,
                )
            )
    return ports, sensors


def _rate_schedule(
    port_names: list[str],
    wells: tuple[WellTrack, ...],
    truth: dict,
    times_days: NDArray[np.float64],
) -> list[ControlSeries]:
    scenario = str(truth.get("scenario", "S1"))
    well_by_prefix: dict[str, WellTrack] = {w.name: w for w in wells}
    controls: list[ControlSeries] = []
    n_open = np.zeros(times_days.size, dtype=int)
    for t_idx, day in enumerate(times_days):
        for name in port_names:
            prefix = name.rsplit("_", 1)[0]
            well = well_by_prefix[prefix]
            if _port_open(float(day), well, scenario):
                n_open[t_idx] += 1
    for name in port_names:
        prefix = name.rsplit("_", 1)[0]
        well = well_by_prefix[prefix]
        rates = np.zeros(times_days.size, dtype=float)
        for t_idx, day in enumerate(times_days):
            if _port_open(float(day), well, scenario) and n_open[t_idx] > 0:
                rates[t_idx] = TOTAL_Q_M3S / float(n_open[t_idx])
        controls.append(
            ControlSeries(name, "rate", times_days * DAY_S, rates)
        )
    return controls


def observations_from_out(
    truth: dict,
    wells: tuple[WellTrack, ...],
    out_path: Path,
    *,
    n_times: int = 5,
) -> tuple[list[ObservationSeries], NDArray[np.float64]]:
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    p_series = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    if len(p_series) < 2:
        raise ValueError("too few PRES times in IMEX .out")
    idx = np.unique(np.linspace(0, len(p_series) - 1, int(n_times)).astype(int))
    picks = [p_series[int(i)] for i in idx]
    times_days = np.array([t for t, _ in picks], dtype=float)
    scenario = str(truth.get("scenario", "S1"))
    step = perf_step(len(truth.get("frac_i_planes") or []))
    observations: list[ObservationSeries] = []
    for well in wells:
        for i in range(int(well.i0), int(well.i1) + 1, step):
            name = f"{well.name}_{i + 1:02d}"
            vals = []
            sigmas = []
            obs_times = []
            for day, p_psi in picks:
                if not _port_open(float(day), well, scenario):
                    continue
                p_pa = float(psi_to_pa(float(p_psi[well.k, well.j, i])))
                if not np.isfinite(p_pa):
                    continue
                obs_times.append(float(day) * DAY_S)
                vals.append(p_pa)
                sigmas.append(5.0e4)
            if not vals:
                continue
            observations.append(
                ObservationSeries(
                    sensor_name=name,
                    kind="pressure",
                    times_s=np.asarray(obs_times, dtype=float),
                    values=np.asarray(vals, dtype=float),
                    sigma=np.asarray(sigmas, dtype=float),
                    holdout=False,
                )
            )
    if not observations:
        raise ValueError("no finite BHP observations parsed from .out")
    return observations, times_days


def twin_from_shale_truth(
    truth_path: Path | str,
    *,
    out_path: Path | str | None = None,
    n_times: int = 5,
    max_iter: int = 12,
    fully_implicit: bool = False,
    free_geometry: bool = False,
) -> DigitalTwin:
    """Assemble a shale depletion twin for LM inversion.

    Default uses sequential black-oil + 4-D θ (n_frac/phase fixed from truth).
    Pass ``fully_implicit=True`` for FIM; ``free_geometry=True`` for 6-D θ.
    """
    tp = Path(truth_path)
    truth = json.loads(tp.read_text(encoding="utf-8"))
    grid = grid_from_truth(truth)
    wells = wells_from_truth(truth)
    ports, sensors = _build_ports_and_sensors(grid, wells, truth)
    if out_path is None:
        case = str(truth.get("scenario", "S1")).lower()
        out_path = tp.parent / f"mxshale_{case}.out"
    op = Path(out_path)
    if not op.is_file():
        raise FileNotFoundError(f"missing IMEX .out: {op}")
    observations, times_days = observations_from_out(truth, wells, op, n_times=n_times)
    controls = _rate_schedule([p.name for p in ports], wells, truth, times_days)
    mean, std = default_shale_prior(truth, free_geometry=bool(free_geometry))
    di = float(np.mean(grid.dx))
    planes = truth.get("frac_i_planes") or []
    n_frac_fixed = float(max(len(planes), 1))
    param = FractureStripParameterization(
        grid,
        wells,
        phi=0.08,
        prior_mean=mean,
        prior_std=std,
        frac_aperture_m=di,
        free_geometry=bool(free_geometry),
        fixed_n_frac=n_frac_fixed,
        fixed_phase=0.0,
    )
    p_init = 3000.0 * PSI
    # Two-phase sequential is enough for BHP-led depletion twin; three-phase/FIM
    # optional for expert IMEX field matching (slow on 21×31×5).
    physics = PhysicsSpec(
        relperm=TableTwoPhase.cmg_seawater(),
        three_phase=TableThreePhase.cmg_seawater() if fully_implicit else None,
        pvt=BlackOilPVT.cmg_seawater(p_init=p_init),
        sw_init=0.20,
        sg_init=0.0,
        p_init=p_init,
        dt_init=DAY_S,
        dt_min=3600.0,
        dt_max=60.0 * DAY_S,
        max_cfl=1.0,
        max_ds=0.25,
        fully_implicit=bool(fully_implicit),
        implicit_transport=True,
        kz_over_kx=0.1,
    )
    t_end = float(times_days.max()) * DAY_S
    experiment = Experiment(
        size_m=(float(grid.dx.sum()), float(grid.dy.sum()), float(grid.dz.sum())),
        sensors=sensors,
        controls=controls,
        observations=observations,
        history_end_s=t_end,
    )
    twin = DigitalTwin(
        grid,
        experiment,
        ports,
        physics,
        param,
        inverse=InverseSpec(
            prior_mean=mean,
            prior_std=std,
            max_iter=int(max_iter),
            fd_rel=0.05,
        ),
    )
    return twin


def _align_rates_to_imex_bhp(
    twin: DigitalTwin,
    truth: dict,
    *,
    n_pass: int = 3,
    scale_min: float = 0.5,
    scale_max: float = 3.0,
) -> float:
    """Scale rate controls so F(truth K) mean BHP ≈ IMEX mean.

    Cross-simulator Peaceman / WI mismatch otherwise drives unphysical drawdown
    and LM destroys contrast. With MIN BHP on ports, do not shrink rates below
    ``scale_min`` (default 0.5). Returns the cumulative scale factor.
    """
    if not twin.experiment.observations or not twin.experiment.controls:
        return 1.0
    wells = wells_from_truth(truth)
    k_true = truth_k_field(truth, twin.grid, wells)
    rock = Rock(k_true, np.full(twin.grid.n_cells, 0.08))
    obs = twin.experiment.assimilate_observations()
    times = np.unique(np.concatenate([o.times_s for o in obs]))
    d = stack_observations(obs)
    p_obs = float(np.mean(d.values))
    p_init = float(twin.physics.p_init)
    total = 1.0
    for _ in range(max(int(n_pass), 1)):
        traj = twin.simulate(rock, t_end=float(times.max()), report_times=times)
        pred = predict_from_trajectory(twin.operator, twin.experiment, traj, obs)
        p_pred = float(np.mean(pred))
        dd_obs = max(p_init - p_obs, 1.0e3)
        dd_pred = max(p_init - p_pred, 1.0e3)
        scale = float(np.clip(dd_obs / dd_pred, float(scale_min), float(scale_max)))
        if abs(scale - 1.0) < 0.08:
            break
        new_controls: list[ControlSeries] = []
        for c in twin.experiment.controls:
            if c.kind == "rate":
                new_controls.append(
                    ControlSeries(c.port_name, c.kind, c.times_s, np.asarray(c.values, dtype=float) * scale)
                )
            else:
                new_controls.append(c)
        twin.experiment.controls = new_controls
        total *= scale
        # After MIN BHP, further shrinking is less useful — stop if still far but at floor
        if scale <= float(scale_min) + 1.0e-9:
            break
    return float(total)


def _inflate_shale_sigmas(
    twin: DigitalTwin,
    truth: dict,
    *,
    mode: str = "cheap",
    cheap_factor: float = 5.0,
) -> None:
    """Inflate observation σ for cross-simulator model error.

    ``cheap`` (default): scale all σ by ``cheap_factor`` — no extra forward.
    ``full``: run F(truth K) vs IMEX and use residual-based inflation.
    ``off``: leave σ unchanged.
    """
    mode = str(mode).strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return
    if mode in {"cheap", "scale", "fast"}:
        for i, obs in enumerate(twin.experiment.observations):
            twin.experiment.observations[i] = ObservationSeries(
                obs.sensor_name,
                obs.kind,
                obs.times_s,
                obs.values,
                np.maximum(obs.sigma * float(cheap_factor), 1.0e-30),
                obs.holdout,
            )
        return
    wells = wells_from_truth(truth)
    k_true = truth_k_field(truth, twin.grid, wells)
    rock = Rock(k_true, np.full(twin.grid.n_cells, 0.08))
    if not twin.experiment.observations:
        return
    times = np.unique(np.concatenate([o.times_s for o in twin.experiment.observations]))
    traj = twin.simulate(rock, t_end=float(times.max()), report_times=times)
    clean: list[ObservationSeries] = []
    for obs in twin.experiment.observations:
        sensor = next(s for s in twin.experiment.sensors if s.name == obs.sensor_name)
        vals = [
            twin.operator.sample(sensor, traj.state_at(float(t)))
            for t in obs.times_s
        ]
        clean.append(
            ObservationSeries(
                obs.sensor_name,
                obs.kind,
                obs.times_s,
                np.asarray(vals, dtype=float),
                obs.sigma,
                obs.holdout,
            )
        )
    twin.inflate_observations(rock, clean=clean, extra_cap_mult=2.5)


def invert_shale_case(
    truth_path: Path | str,
    *,
    out_path: Path | str | None = None,
    n_times: int = 5,
    max_iter: int = 12,
    time_limit_s: float | None = 900.0,
    fully_implicit: bool = False,
    free_geometry: bool = False,
    inflate_mode: str = "cheap",
) -> dict:
    """Run LM inversion and return ruler metrics."""
    tp = Path(truth_path)
    truth = json.loads(tp.read_text(encoding="utf-8"))
    case = str(truth.get("scenario", tp.stem)).upper()
    op = Path(out_path) if out_path is not None else tp.parent / f"mxshale_{case.lower()}.out"
    if not op.is_file():
        return {"case": case, "ok": False, "error": "missing IMEX .out"}

    twin = twin_from_shale_truth(
        tp,
        out_path=op,
        n_times=n_times,
        max_iter=max_iter,
        fully_implicit=fully_implicit,
        free_geometry=free_geometry,
    )
    rate_scale = _align_rates_to_imex_bhp(twin, truth)
    _inflate_shale_sigmas(twin, truth, mode=inflate_mode)
    post = twin.calibrate(max_iter=max_iter, time_limit_s=time_limit_s)
    param = twin.parameterization
    eng = decode_frac_theta(param, post.theta)
    frac = frac_mask_from_truth(truth, twin.grid)
    mat = ~frac
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    p_series = parse_grid_series(op, field="pressure", nx=nx, ny=ny, nz=nz)
    idx = np.unique(np.linspace(0, len(p_series) - 1, int(n_times)).astype(int))
    _, p_last_psi = p_series[int(idx[-1])]
    p_true = psi_to_pa(np.asarray(p_last_psi, dtype=float)).reshape(-1)
    dp_true = float(np.nanmean(p_true[mat]) - np.nanmean(p_true[frac]))
    p_inv = post.history.states[-1].pressure
    dp_inv = float(np.mean(p_inv[mat]) - np.mean(p_inv[frac]))
    di = ft_to_m(g["di_ft"])
    truth_n = len(truth.get("frac_i_planes") or [])
    truth_fcd = float(truth["frac_perm_md"]) * MD_TO_M2 * di
    k_frac = float(np.exp(post.theta[1]))
    k_mat = float(np.exp(post.theta[0]))
    shale_extra = {
        "case": case,
        "analog": True,
        "frac_theta": True,
        "k_frac_over_matrix": float(k_frac / max(k_mat, 1.0e-30)),
        "inv_x_f_m": eng.get("x_f_m"),
        "truth_x_f_m": truth_half_length_m(truth),
        "inv_F_cd_m3": eng.get("F_cd_m3"),
        "truth_F_cd_m3": truth_fcd,
        "inv_n_frac": eng.get("n_frac"),
        "truth_n_frac_planes": truth_n,
        "imex_dp_matrix_minus_frac_Pa": dp_true,
        "inv_dp_matrix_minus_frac_Pa": dp_inv,
        "dp_sign_match": bool(dp_true * dp_inv > 0.0),
        "dp_ratio": float(dp_inv / dp_true) if abs(dp_true) > 1.0 else None,
        "assimilate_nrmse": float(post.assimilate_rmse),
        "holdout_nrmse": float(post.holdout_rmse),
        "rate_scale": float(rate_scale),
        "inflate_mode": str(inflate_mode),
        "fully_implicit": bool(fully_implicit),
        "n_theta": int(param.n_params),
    }
    from reservoir_backend.twin.run_report import build_invert_report

    run_report = build_invert_report(twin, post, case_path=tp, extra=shale_extra)
    return {
        "case": case,
        "ok": True,
        **shale_extra,
        "n_times": int(n_times),
        "n_wells": len(truth.get("wells") or []),
        "notes": post.notes[:12],
        "run_report": run_report,
    }


def truth_k_field(truth: dict, grid: CartesianGrid, wells: tuple[WellTrack, ...]) -> NDArray[np.float64]:
    """Reference K from truth perm tables and frac geometry."""
    k_m = float(truth["matrix_perm_md"]["kx_geo"]) * MD_TO_M2
    k_f = float(truth["frac_perm_md"]) * MD_TO_M2
    k_srv = float(truth.get("srv_perm_md", 0.4)) * MD_TO_M2
    planes = truth.get("frac_i_planes") or []
    x_f = truth_half_length_m(truth)
    k, _, _ = paint_fracture_strips(
        grid,
        wells,
        log_k_m=float(np.log(k_m)),
        log_k_f=float(np.log(k_f)),
        log_k_srv=float(np.log(k_srv)),
        x_f_m=float(x_f),
        n_frac=max(len(planes), 1),
        frac_phase=0.0,
    )
    return k


def forecast_shale_case(
    twin: DigitalTwin,
    posterior: Posterior,
) -> tuple[Trajectory, float]:
    """Forecast with frozen posterior K using experiment controls (incl. S5 shut-in)."""
    traj = twin.forecast(posterior)
    score = twin.score_forecast(traj)
    posterior.forecast_rmse = score
    return traj, score
