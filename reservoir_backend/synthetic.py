"""Synthetic experiments whose observations come from H(F(m_true))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor, column_sensors
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.frac import (
    FractureStripParameterization,
    MD_TO_M2,
    WellTrack,
    paint_fracture_strips,
)
from reservoir_backend.inverse.parameterization import ContrastParameterization, RegionParameterization
from reservoir_backend.physics.pvt import PSI, BlackOilPVT
from reservoir_backend.physics.relperm import TableThreePhase, TableTwoPhase
from reservoir_backend.physics.rock import Rock, log_permeability
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec, PhysicsSpec, stack_observations


def layered_permeability(grid: CartesianGrid, k_lo: float, k_hi: float, z_cut: float) -> NDArray[np.float64]:
    z = grid.cell_centers()[:, 2]
    k = np.full(grid.n_cells, float(k_lo), dtype=float)
    k[z >= float(z_cut)] = float(k_hi)
    return k


def channel_permeability(
    grid: CartesianGrid,
    k_bg: float,
    k_ch: float,
    y0: float,
    half_width: float,
) -> NDArray[np.float64]:
    y = grid.cell_centers()[:, 1]
    k = np.full(grid.n_cells, float(k_bg), dtype=float)
    k[np.abs(y - float(y0)) <= float(half_width)] = float(k_ch)
    return k


def two_layer_regions(grid: CartesianGrid, z_cut: float) -> NDArray[np.int64]:
    z = grid.cell_centers()[:, 2]
    return (z >= float(z_cut)).astype(np.int64)


def channel_regions(grid: CartesianGrid, y0: float, half_width: float) -> NDArray[np.int64]:
    y = grid.cell_centers()[:, 1]
    return (np.abs(y - float(y0)) <= float(half_width)).astype(np.int64)


@dataclass
class SyntheticCase:
    grid: CartesianGrid
    twin: DigitalTwin
    k_true: NDArray[np.float64]
    theta_true: NDArray[np.float64]
    p_true_end: NDArray[np.float64] | None = None


def make_two_layer_waterflood(
    *,
    n: tuple[int, int, int] = (8, 6, 4),
    size_m: tuple[float, float, float] = (0.24, 0.18, 0.12),
    k_lo: float = 2.0e-13,
    k_hi: float = 2.0e-12,
    phi: float = 0.20,
    q_inj: float = 1.5e-7,
    p_prod: float = 1.0e5,
    t_end: float = 700.0,
    n_times: int = 6,
    noise_p: float = 2.0e3,
    noise_s: float = 0.03,
    seed: int = 3,
    holdout_sensors: tuple[str, ...] = ("Pout_top", "Sout_bot"),
    history_frac: float = 0.80,
) -> SyntheticCase:
    nx, ny, nz = n
    grid = CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, size_m[0] / nx),
        dy=np.full(ny, size_m[1] / ny),
        dz=np.full(nz, size_m[2] / nz),
    )
    z_cut = grid.origin[2] + 0.5 * size_m[2]
    k_true = layered_permeability(grid, k_lo, k_hi, z_cut)
    regions = two_layer_regions(grid, z_cut)
    param = RegionParameterization(regions, phi=phi)
    theta_true = np.array(
        [
            float(np.mean(log_permeability(k_true[regions == 0]))),
            float(np.mean(log_permeability(k_true[regions == 1]))),
        ]
    )

    inj = FlowPort.column(grid, "INJ", "injector", "rate", float(grid.dx[0] * 0.5), size_m[1] * 0.50, sw_inj=0.85)
    prod = FlowPort.column(
        grid,
        "PROD",
        "producer",
        "pressure",
        size_m[0] - float(grid.dx[-1] * 0.5),
        size_m[1] * 0.50,
    )
    times = np.linspace(0.0, float(t_end), int(n_times) + 1)[1:]
    controls = [
        ControlSeries("INJ", "rate", times, np.full(times.size, q_inj)),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.85)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, p_prod)),
    ]
    z_bot, z_top = size_m[2] * 0.22, size_m[2] * 0.78
    sensors = []
    sensors += column_sensors("Pin", "pressure", size_m[0] * 0.30, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_p, probe_diameter_m=0.006, labels=("bot", "top"))
    sensors += column_sensors("Pout", "pressure", size_m[0] * 0.70, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_p, probe_diameter_m=0.006, labels=("bot", "top"))
    sensors += column_sensors("Sin", "saturation", size_m[0] * 0.38, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_s, probe_diameter_m=0.006, labels=("bot", "top"))
    sensors += column_sensors("Sout", "saturation", size_m[0] * 0.62, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_s, probe_diameter_m=0.006, labels=("bot", "top"))
    sensors.append(
        Sensor("Pinj", "pressure", float(grid.dx[0] * 0.5), size_m[1] * 0.50, size_m[2] * 0.50, sigma=noise_p)
    )
    experiment = Experiment(
        size_m=size_m,
        sensors=sensors,
        controls=controls,
        observations=[],
        history_end_s=float(t_end) * float(history_frac),
    )
    physics = PhysicsSpec(
        sw_init=0.20,
        p_init=p_prod + 5.0e4,
        dt_init=2.0,
        dt_min=1.0e-6,
        dt_max=10.0,
        max_cfl=0.40,
        max_ds=0.12,
        implicit_transport=True,
        fully_implicit=False,
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        physics,
        param,
        inverse=InverseSpec(
            prior_mean=float(np.log(5.0e-13)),
            prior_std=1.0,
            max_iter=4,
        ),
    )
    truth_rock = Rock(k_true, np.full(grid.n_cells, phi))
    traj = twin.simulate(truth_rock, t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    observations: list[ObservationSeries] = []
    for sensor in sensors:
        vals = []
        for t in times:
            st = traj.state_at(t)
            rates, bhp = traj.rates_and_bhp_at(t)
            vals.append(twin.operator.sample(sensor, st, port_rates=rates, port_bhp=bhp))
        vals_a = np.asarray(vals, dtype=float)
        noise = rng.normal(0.0, sensor.sigma, size=vals_a.size)
        observations.append(
            ObservationSeries(
                sensor_name=sensor.name,
                kind=sensor.kind,
                times_s=times,
                values=vals_a + noise,
                sigma=np.full(times.size, sensor.sigma),
                holdout=sensor.name in holdout_sensors,
            )
        )
    experiment.observations = observations
    return SyntheticCase(grid=grid, twin=twin, k_true=k_true, theta_true=theta_true)


def make_channel_waterflood(
    *,
    n: tuple[int, int, int] = (8, 6, 4),
    size_m: tuple[float, float, float] = (0.24, 0.18, 0.12),
    k_bg: float = 2.0e-13,
    k_ch: float = 2.0e-12,
    phi: float = 0.20,
    q_inj: float = 1.5e-7,
    p_prod: float = 1.0e5,
    t_end: float = 700.0,
    n_times: int = 6,
    noise_p: float = 2.0e3,
    noise_s: float = 0.03,
    seed: int = 5,
    holdout_sensors: tuple[str, ...] = ("Pmx_out", "Sch"),
    history_frac: float = 0.85,
) -> SyntheticCase:
    """Known high-K strip in y. Structure is the region map; magnitudes are inverted."""
    nx, ny, nz = n
    grid = CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, size_m[0] / nx),
        dy=np.full(ny, size_m[1] / ny),
        dz=np.full(nz, size_m[2] / nz),
    )
    y0 = size_m[1] * 0.50
    half = size_m[1] * 0.16
    k_true = channel_permeability(grid, k_bg, k_ch, y0, half)
    regions = channel_regions(grid, y0, half)
    param = ContrastParameterization(regions, phi=phi, log_contrast_mean=float(np.log(10.0)))
    theta_true = np.array([float(np.log(k_bg)), float(np.log(k_ch / k_bg))])

    inj = FlowPort.column(grid, "INJ", "injector", "rate", float(grid.dx[0] * 0.5), y0, sw_inj=0.85)
    prod = FlowPort.column(
        grid,
        "PROD",
        "producer",
        "pressure",
        size_m[0] - float(grid.dx[-1] * 0.5),
        y0,
    )
    times = np.linspace(0.0, float(t_end), int(n_times) + 1)[1:]
    controls = [
        ControlSeries("INJ", "rate", times, np.full(times.size, q_inj)),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.85)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, p_prod)),
    ]
    y_mx = size_m[1] * 0.14
    zmid = size_m[2] * 0.50
    xin, xmid, xout = size_m[0] * 0.28, size_m[0] * 0.50, size_m[0] * 0.72
    sensors = [
        Sensor("Pch_in", "pressure", xin, y0, zmid, probe_diameter_m=0.006, sigma=noise_p),
        Sensor("Pch_out", "pressure", xout, y0, zmid, probe_diameter_m=0.006, sigma=noise_p),
        Sensor("Pmx_in", "pressure", xin, y_mx, zmid, probe_diameter_m=0.006, sigma=noise_p),
        Sensor("Pmx_out", "pressure", xout, y_mx, zmid, probe_diameter_m=0.006, sigma=noise_p),
        Sensor("Sch", "saturation", xmid, y0, zmid, probe_diameter_m=0.006, sigma=noise_s),
        Sensor("Smx", "saturation", xmid, y_mx, zmid, probe_diameter_m=0.006, sigma=noise_s),
    ]
    experiment = Experiment(
        size_m=size_m,
        sensors=sensors,
        controls=controls,
        observations=[],
        history_end_s=float(t_end) * float(history_frac),
    )
    physics = PhysicsSpec(
        sw_init=0.20,
        p_init=p_prod + 5.0e4,
        dt_init=2.0,
        dt_min=1.0e-6,
        dt_max=10.0,
        max_cfl=0.40,
        max_ds=0.12,
        implicit_transport=True,
        fully_implicit=False,
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        physics,
        param,
        inverse=InverseSpec(
            prior_mean=float(np.log(5.0e-13)),
            prior_std=1.0,
            max_iter=4,
        ),
    )
    truth_rock = Rock(k_true, np.full(grid.n_cells, phi))
    traj = twin.simulate(truth_rock, t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    observations: list[ObservationSeries] = []
    for sensor in sensors:
        vals = []
        for t in times:
            st = traj.state_at(t)
            rates, bhp = traj.rates_and_bhp_at(t)
            vals.append(twin.operator.sample(sensor, st, port_rates=rates, port_bhp=bhp))
        vals_a = np.asarray(vals, dtype=float)
        noise = rng.normal(0.0, sensor.sigma, size=vals_a.size)
        observations.append(
            ObservationSeries(
                sensor_name=sensor.name,
                kind=sensor.kind,
                times_s=times,
                values=vals_a + noise,
                sigma=np.full(times.size, sensor.sigma),
                holdout=sensor.name in holdout_sensors,
            )
        )
    experiment.observations = observations
    return SyntheticCase(grid=grid, twin=twin, k_true=k_true, theta_true=theta_true)


def evaluate_synthetic(case: SyntheticCase, posterior) -> dict[str, float]:
    rock_true = Rock(case.k_true, np.full(case.grid.n_cells, case.twin.parameterization.phi))
    # prior mismatch vs posterior using the same observation stack
    assim = case.twin.experiment.assimilate_observations()
    d = stack_observations(assim)
    prior_theta = np.full(case.twin.parameterization.n_params, np.log(1.0e-12))
    d_prior = case.twin._forward_vector(prior_theta, assim)
    d_post = case.twin._forward_vector(posterior.theta, assim)
    d_true = case.twin._forward_vector(case.theta_true, assim)
    def nrmse(pred, target=None):
        ref = d.values if target is None else target
        return float(np.sqrt(np.mean(((pred - ref) / d.sigma) ** 2)))
    k_prior = case.twin.parameterization.expand(prior_theta)
    k_post = posterior.k
    def k_err(k):
        return float(np.sqrt(np.mean((np.log(k) - np.log(case.k_true)) ** 2)))
    rid = case.twin.parameterization.region_id
    th_std = np.asarray(posterior.theta_std, dtype=float).ravel()
    log_std = th_std[np.clip(np.asarray(rid, dtype=np.int64).ravel(), 0, max(th_std.size - 1, 0))]
    k_lo_post = float(np.mean(k_post[rid == 0]))
    k_hi_post = float(np.mean(k_post[rid == 1]))
    k_lo_true = float(np.mean(case.k_true[rid == 0]))
    k_hi_true = float(np.mean(case.k_true[rid == 1]))
    return {
        "prior_data_nrmse": nrmse(d_prior),
        "posterior_data_nrmse": nrmse(d_post),
        "forward_match_nrmse": nrmse(d_post, d_true),
        "prior_logk_rmse": k_err(k_prior),
        "posterior_logk_rmse": k_err(k_post),
        "holdout_nrmse": float(posterior.holdout_rmse),
        "assimilate_nrmse": float(posterior.assimilate_rmse),
        "contrast_true": float(k_hi_true / max(k_lo_true, 1.0e-30)),
        "contrast_post": float(k_hi_post / max(k_lo_post, 1.0e-30)),
        "theta_rmse": (
            float(np.sqrt(np.mean((posterior.theta - case.theta_true) ** 2)))
            if posterior.theta.size == case.theta_true.size
            else float("nan")
        ),
        "k_true_in_2std_frac": float(
            np.mean(np.abs(np.log(k_post) - np.log(case.k_true)) <= 2.0 * np.maximum(log_std, 1.0e-12))
        ),
        "mass_rel": float(posterior.history.reports[-1].mass.relative_balance_error)
        if posterior.history.reports
        else 0.0,
    }


def make_two_layer_compositional(
    *,
    n: tuple[int, int, int] = (6, 4, 1),
    size_m: tuple[float, float, float] = (6.0, 4.0, 1.0),
    k_lo: float = 2.0e-13,
    k_hi: float = 2.0e-12,
    phi: float = 0.20,
    q_inj: float = 0.20,
    p_prod: float = 1.1e7,
    p_init: float = 1.2e7,
    t_end: float = 24.0,
    n_times: int = 4,
    noise_p: float = 2.0e4,
    noise_bhp: float = 5.0e3,
    seed: int = 3,
    holdout_sensors: tuple[str, ...] = ("Pout_hi",),
    history_frac: float = 0.80,
    has_water: bool = False,
    sw_init: float = 0.25,
    sw_inj: float = 1.0,
) -> SyntheticCase:
    """H(F_comp(m_true)) observations on a 2-region EXAMPLE fluid twin."""
    from reservoir_backend.comp.fluid import fluid_from_name

    nx, ny, nz = n
    grid = CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, size_m[0] / nx),
        dy=np.full(ny, size_m[1] / ny),
        dz=np.full(nz, size_m[2] / nz),
    )
    y_cut = grid.origin[1] + 0.5 * size_m[1]
    y = grid.cell_centers()[:, 1]
    k_true = np.full(grid.n_cells, float(k_lo), dtype=float)
    k_true[y >= y_cut] = float(k_hi)
    regions = (y >= y_cut).astype(np.int64)
    param = RegionParameterization(regions, phi=phi)
    theta_true = np.array(
        [
            float(np.mean(log_permeability(k_true[regions == 0]))),
            float(np.mean(log_permeability(k_true[regions == 1]))),
        ]
    )
    # One left/right connection in each y-region so both strips are driven,
    # without a full-face WI that makes Δp insensitive to K.
    j_lo = 0
    j_hi = ny - 1
    inj_cells = np.array([grid.index(0, j_lo, 0), grid.index(0, j_hi, 0)], dtype=np.int64)
    prod_cells = np.array([grid.index(nx - 1, j_lo, 0), grid.index(nx - 1, j_hi, 0)], dtype=np.int64)
    inj = FlowPort("INJ", "injector", "rate", inj_cells, sw_inj=float(sw_inj) if has_water else 0.0)
    prod = FlowPort("PROD", "producer", "pressure", prod_cells, sw_inj=0.0)
    times = np.linspace(0.0, float(t_end), int(n_times) + 1)[1:]
    controls = [
        ControlSeries("INJ", "rate", times, np.full(times.size, q_inj)),
        ControlSeries("INJ", "composition", times, np.full(times.size, 0.95)),
        ControlSeries("PROD", "pressure", times, np.full(times.size, p_prod)),
    ]
    zmid = size_m[2] * 0.50
    inj_xyz = grid.cell_centers()[int(inj_cells[0])]
    sensors = [
        Sensor("Pin_lo", "pressure", size_m[0] * 0.25, size_m[1] * 0.25, zmid, sigma=noise_p),
        Sensor("Pin_hi", "pressure", size_m[0] * 0.25, size_m[1] * 0.75, zmid, sigma=noise_p),
        Sensor("Pout_lo", "pressure", size_m[0] * 0.75, size_m[1] * 0.25, zmid, sigma=noise_p),
        Sensor("Pout_hi", "pressure", size_m[0] * 0.75, size_m[1] * 0.75, zmid, sigma=noise_p),
        Sensor(
            "Pinj_bhp",
            "bhp",
            float(inj_xyz[0]),
            float(inj_xyz[1]),
            float(inj_xyz[2]),
            port_name="INJ",
            sigma=float(noise_bhp),
        ),
    ]
    spec = fluid_from_name(
        "example",
        temperature_k=350.0,
        z_inj=np.array([0.95, 0.05]),
        has_water=bool(has_water),
        sw_init=float(sw_init) if has_water else 0.0,
    )
    if has_water:
        sensors.append(
            Sensor("Sw_mid", "saturation", size_m[0] * 0.35, size_m[1] * 0.25, zmid, sigma=0.04)
        )
    experiment = Experiment(
        size_m=size_m,
        sensors=sensors,
        controls=controls,
        observations=[],
        history_end_s=float(t_end) * float(history_frac),
    )
    physics = PhysicsSpec(
        model="compositional",
        fluid=spec,
        p_init=float(p_init),
        dt_init=6.0,
        dt_min=1.0e-6,
        dt_max=8.0,
        fully_implicit=False,
        implicit_transport=True,
        temperature_k=350.0,
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        physics,
        param,
        inverse=InverseSpec(
            prior_mean=float(np.log(0.5 * (k_lo + k_hi))),
            prior_std=0.45,
            max_iter=4,
        ),
    )
    truth_rock = Rock(k_true, np.full(grid.n_cells, phi))
    traj = twin.simulate(truth_rock, t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    observations: list[ObservationSeries] = []
    for sensor in sensors:
        vals = []
        for t in times:
            st = traj.state_at(t)
            rates, bhp = traj.rates_and_bhp_at(t)
            vals.append(twin.operator.sample(sensor, st, port_rates=rates, port_bhp=bhp))
        vals_a = np.asarray(vals, dtype=float)
        noise = rng.normal(0.0, sensor.sigma, size=vals_a.size)
        observations.append(
            ObservationSeries(
                sensor_name=sensor.name,
                kind=sensor.kind,
                times_s=times,
                values=vals_a + noise,
                sigma=np.full(times.size, sensor.sigma),
                holdout=sensor.name in holdout_sensors,
            )
        )
    experiment.observations = observations
    return SyntheticCase(
        grid=grid,
        twin=twin,
        k_true=k_true,
        theta_true=theta_true,
        p_true_end=traj.states[-1].pressure.copy(),
    )


def make_forecast_split_case(
    *,
    history_frac: float = 0.6,
    n_times: int = 8,
    t_end: float = 400.0,
    seed: int = 11,
) -> SyntheticCase:
    """Two-layer case with history/forecast split for §55 validation."""
    hist_end = float(t_end) * float(history_frac)
    case = make_two_layer_waterflood(
        n_times=int(n_times),
        t_end=float(t_end),
        seed=int(seed),
        history_frac=float(history_frac),
        holdout_sensors=("Pout_top",),
        n=(8, 6, 4),
    )
    twin = case.twin
    twin.experiment.history_end_s = hist_end
    return case


def evaluate_forecast(case: SyntheticCase, posterior) -> dict[str, float]:
    traj = case.twin.forecast(posterior)
    score = case.twin.score_forecast(traj)
    return {
        "forecast_rmse": float(score),
        "assimilate_rmse": float(posterior.assimilate_rmse),
        "history_end_s": float(case.twin.experiment.history_end_s or 0.0),
        "forecast_end_s": float(traj.times_s[-1]) if traj.times_s.size else 0.0,
    }


def make_shale_depletion(
    *,
    n: tuple[int, int, int] = (12, 10, 5),
    size_m: tuple[float, float, float] = (600.0, 500.0, 80.0),
    n_frac: int = 4,
    x_f_m: float = 60.0,
    n_perf: int = 6,
    phi: float = 0.08,
    t_end: float = 120.0 * 86400.0,
    n_times: int = 4,
    noise_bhp: float = 5.0e4,
    seed: int = 7,
    max_iter: int = 10,
    holdout_ports: tuple[str, ...] | None = None,
    history_frac: float = 0.85,
    fully_implicit: bool = False,
    free_geometry: bool = False,
    min_bhp_Pa: float | None = 1500.0 * PSI,
) -> SyntheticCase:
    """Small shale depletion twin: rate-controlled HW, BHP-only observations.

    Default: sequential black-oil + 4-D θ (geometry frozen). FIM and free
    ``n_frac``/phase are optional for expert / CMG-alignment runs.
    If ``holdout_ports`` is None, the last completion is held out.
    """
    nx, ny, nz = n
    grid = CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, size_m[0] / nx),
        dy=np.full(ny, size_m[1] / ny),
        dz=np.full(nz, size_m[2] / nz),
    )
    j_well = ny // 2
    k_well = nz // 2
    i0 = max(1, nx // 8)
    i1 = nx - 1 - i0
    well = WellTrack(name="HW1", j=j_well, k=k_well, i0=i0, i1=i1)
    k_m = 0.001 * MD_TO_M2
    k_f = 8000.0 * MD_TO_M2
    k_srv = 0.4 * MD_TO_M2
    theta_true_full = np.array(
        [np.log(k_m), np.log(k_f), np.log(k_srv), np.log(x_f_m), float(n_frac), 0.0],
        dtype=float,
    )
    theta_true = theta_true_full if free_geometry else theta_true_full[:4].copy()
    prior_bias = (
        np.array([0.35, -0.25, 0.30, -0.18, 0.75, 0.10])
        if free_geometry
        else np.array([0.35, -0.25, 0.30, -0.18])
    )
    prior_std = (
        np.array([0.8, 0.6, 0.8, 0.40, 0.75, 0.15])
        if free_geometry
        else np.array([0.8, 0.6, 0.8, 0.40])
    )
    param = FractureStripParameterization(
        grid,
        (well,),
        phi=phi,
        prior_mean=theta_true + prior_bias,
        prior_std=prior_std,
        frac_aperture_m=float(grid.dx[0]),
        free_geometry=bool(free_geometry),
        fixed_n_frac=float(n_frac),
        fixed_phase=0.0,
    )
    k_true, frac_mask, _ = paint_fracture_strips(
        grid,
        (well,),
        log_k_m=float(theta_true_full[0]),
        log_k_f=float(theta_true_full[1]),
        log_k_srv=float(theta_true_full[2]),
        x_f_m=float(x_f_m),
        n_frac=int(n_frac),
        frac_phase=0.0,
    )

    step = max(1, (i1 - i0 + 1) // max(int(n_perf), 1))
    centers = grid.cell_centers()
    ports: list[FlowPort] = []
    sensors: list[Sensor] = []
    for i in range(i0, i1 + 1, step):
        cell = int(grid.index(i, j_well, k_well))
        name = f"HW1_{i + 1:02d}"
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
                min_bhp_Pa=None if min_bhp_Pa is None else float(min_bhp_Pa),
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
                sigma=noise_bhp,
            )
        )
    if holdout_ports is None:
        holdout_ports = (ports[-1].name,) if ports else ()
    holdout_set = set(holdout_ports)

    times = np.linspace(0.0, float(t_end), int(n_times) + 1)[1:]
    stb_to_m3 = 0.158987294928
    q_each = -800.0 * stb_to_m3 / 86400.0 / max(len(ports), 1)
    controls = [
        ControlSeries(p.name, "rate", times, np.full(times.size, q_each)) for p in ports
    ]
    p_init = 3000.0 * PSI
    physics = PhysicsSpec(
        relperm=TableTwoPhase.cmg_seawater(),
        three_phase=TableThreePhase.cmg_seawater(),
        pvt=BlackOilPVT.cmg_seawater(p_init=p_init),
        sw_init=0.20,
        p_init=p_init,
        dt_init=86400.0,
        dt_min=3600.0,
        dt_max=30.0 * 86400.0,
        fully_implicit=bool(fully_implicit),
        kz_over_kx=0.1,
    )
    experiment = Experiment(
        size_m=size_m,
        sensors=sensors,
        controls=controls,
        observations=[],
        history_end_s=float(t_end) * float(history_frac),
    )
    twin = DigitalTwin(
        grid,
        experiment,
        ports,
        physics,
        param,
        inverse=InverseSpec(
            prior_mean=param.prior_mean,
            prior_std=param.prior_std,
            max_iter=int(max_iter),
            fd_rel=0.05,
        ),
    )
    truth_rock = Rock(k_true, np.full(grid.n_cells, phi))
    traj = twin.simulate(truth_rock, t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    observations: list[ObservationSeries] = []
    for sensor in sensors:
        vals = []
        for t in times:
            st = traj.state_at(t)
            vals.append(twin.operator.sample(sensor, st))
        vals_a = np.asarray(vals, dtype=float)
        noise = rng.normal(0.0, sensor.sigma, size=vals_a.size)
        observations.append(
            ObservationSeries(
                sensor_name=sensor.name,
                kind=sensor.kind,
                times_s=times,
                values=vals_a + noise,
                sigma=np.full(times.size, sensor.sigma),
                holdout=sensor.name in holdout_set,
            )
        )
    experiment.observations = observations
    return SyntheticCase(
        grid=grid,
        twin=twin,
        k_true=k_true,
        theta_true=theta_true,
        p_true_end=traj.states[-1].pressure.copy(),
    )

