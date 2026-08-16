"""Synthetic experiments whose observations come from H(F(m_true))."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor, column_sensors
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import RegionParameterization
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


@dataclass
class SyntheticCase:
    grid: CartesianGrid
    twin: DigitalTwin
    k_true: NDArray[np.float64]
    theta_true: NDArray[np.float64]


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
    sensors += column_sensors("Pin", "pressure", size_m[0] * 0.30, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_p, labels=("bot", "top"))
    sensors += column_sensors("Pout", "pressure", size_m[0] * 0.70, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_p, labels=("bot", "top"))
    sensors += column_sensors("Sin", "saturation", size_m[0] * 0.38, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_s, labels=("bot", "top"))
    sensors += column_sensors("Sout", "saturation", size_m[0] * 0.62, size_m[1] * 0.50, [z_bot, z_top], sigma=noise_s, labels=("bot", "top"))
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
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        physics,
        param,
        inverse=InverseSpec(
            n_ensemble=16,
            n_assimilations=4,
            prior_mean=float(np.log(5.0e-13)),
            prior_std=1.0,
            seed=7,
            inflation=1.01,
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
            idx = int(np.argmin(np.abs(traj.times_s - t)))
            rates = traj.port_rates[idx]
            vals.append(twin.operator.sample(sensor, st, port_rates=rates))
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
    d_post = case.twin._forward_vector(posterior.esmda.theta_mean, assim)
    def nrmse(pred):
        return float(np.sqrt(np.mean(((pred - d.values) / d.sigma) ** 2)))
    k_prior = case.twin.parameterization.expand(prior_theta)
    k_post = posterior.esmda.k_mean
    def k_err(k):
        return float(np.sqrt(np.mean((np.log(k) - np.log(case.k_true)) ** 2)))
    rid = case.twin.parameterization.region_id
    log_std = posterior.esmda.theta_std[rid]
    k_lo_post = float(np.mean(k_post[rid == 0]))
    k_hi_post = float(np.mean(k_post[rid == 1]))
    k_lo_true = float(np.mean(case.k_true[rid == 0]))
    k_hi_true = float(np.mean(case.k_true[rid == 1]))
    return {
        "prior_data_nrmse": nrmse(d_prior),
        "posterior_data_nrmse": nrmse(d_post),
        "prior_logk_rmse": k_err(k_prior),
        "posterior_logk_rmse": k_err(k_post),
        "holdout_nrmse": float(posterior.holdout_rmse),
        "assimilate_nrmse": float(posterior.assimilate_rmse),
        "contrast_true": float(k_hi_true / max(k_lo_true, 1.0e-30)),
        "contrast_post": float(k_hi_post / max(k_lo_post, 1.0e-30)),
        "theta_rmse": float(np.sqrt(np.mean((posterior.esmda.theta_mean - case.theta_true) ** 2))),
        "k_true_in_2std_frac": float(
            np.mean(np.abs(np.log(k_post) - np.log(case.k_true)) <= 2.0 * np.maximum(log_std, 1.0e-12))
        ),
        "mass_rel": float(posterior.history.reports[-1].mass.relative_balance_error)
        if posterior.history.reports
        else 0.0,
    }
