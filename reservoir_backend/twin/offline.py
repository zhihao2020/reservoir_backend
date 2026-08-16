"""Thin digital-twin orchestrator: simulate, calibrate, forecast."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor, State
from reservoir_backend.exceptions import TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.esmda import ESMdaResult, identifiability, run_esmda
from reservoir_backend.inverse.parameterization import (
    CoarseFieldParameterization,
    ContrastParameterization,
    RegionParameterization,
)
from reservoir_backend.observation.operator import ObservationOperator
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort, validate_port_controls
from reservoir_backend.solver.impes import Trajectory, simulate, water_mass


@dataclass
class InverseSpec:
    """Ensemble design for ES-MDA. Prefer presets / time_limit over raw knobs.

    Not HPO over K, and not a tabular AutoGluon stack. Localization in
    ``inverse.ensemble`` stays off until n_θ grows.
    """

    n_ensemble: int = 24
    n_assimilations: int = 4
    seed: int = 7
    prior_mean: float = float(np.log(1.0e-12))
    prior_std: float = 0.8
    inflation: float = 1.02
    algorithm: str = "esmda"
    time_limit_s: float | None = None
    n_workers: int | None = None
    reconstruct_members: int = 8


@dataclass
class PhysicsSpec:
    relperm: CoreyTwoPhase = field(default_factory=CoreyTwoPhase)
    three_phase: CoreyThreePhase | None = None
    capillary: object = field(default_factory=NoCapillary)
    pvt: BlackOilPVT = field(default_factory=BlackOilPVT.incompressible)
    gravity: float = 0.0
    kz_over_kx: float = 1.0
    single_phase: bool = False
    mu_single: float = 1.0e-3
    sw_init: float = 0.20
    sg_init: float = 0.0
    p_init: float = 1.0e6
    dt_init: float = 10.0
    dt_min: float = 1.0e-6
    dt_max: float = 60.0
    max_cfl: float = 0.5
    max_ds: float = 0.15
    implicit_transport: bool = False


@dataclass
class DataVector:
    values: NDArray[np.float64]
    sigma: NDArray[np.float64]
    times: NDArray[np.float64]
    names: list[str]
    kinds: list[str]
    holdout: NDArray[np.bool_]


def stack_observations(series: list[ObservationSeries]) -> DataVector:
    times: list[float] = []
    values: list[float] = []
    sigma: list[float] = []
    names: list[str] = []
    kinds: list[str] = []
    hold: list[bool] = []
    for obs in sorted(series, key=lambda s: (s.sensor_name, s.kind)):
        for i, t in enumerate(obs.times_s):
            times.append(float(t))
            values.append(float(obs.values[i]))
            sigma.append(float(obs.sigma[i]))
            names.append(obs.sensor_name)
            kinds.append(obs.kind)
            hold.append(bool(obs.holdout))
    order = np.lexsort((np.asarray(names), np.asarray(times)))
    return DataVector(
        values=np.asarray(values, dtype=float)[order],
        sigma=np.asarray(sigma, dtype=float)[order],
        times=np.asarray(times, dtype=float)[order],
        names=[names[i] for i in order],
        kinds=[kinds[i] for i in order],
        holdout=np.asarray(hold, dtype=bool)[order],
    )


def _sensor_lookup(experiment: Experiment) -> dict[str, Sensor]:
    return experiment.sensor_map()


def predict_from_trajectory(
    operator: ObservationOperator,
    experiment: Experiment,
    traj: Trajectory,
    series: list[ObservationSeries],
) -> NDArray[np.float64]:
    sensors = _sensor_lookup(experiment)
    vec = stack_observations(series)
    out = np.zeros(vec.values.size, dtype=float)
    for i, (t, name, kind) in enumerate(zip(vec.times, vec.names, vec.kinds)):
        state = traj.state_at(t)
        # nearest recorded rates
        idx = int(np.argmin(np.abs(traj.times_s - t)))
        rates = traj.port_rates[idx] if idx < len(traj.port_rates) else {}
        sensor = sensors[name]
        if sensor.kind != kind:
            sensor = Sensor(
                name=sensor.name,
                kind=kind,
                x=sensor.x,
                y=sensor.y,
                z=sensor.z,
                volume_m3=sensor.volume_m3,
                port_name=sensor.port_name,
                sigma=sensor.sigma,
            )
        out[i] = operator.sample(sensor, state, port_rates=rates)
    return out


@dataclass
class FieldStats:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    q10: NDArray[np.float64]
    q90: NDArray[np.float64]


@dataclass
class Posterior:
    esmda: ESMdaResult
    assimilate_rmse: float
    holdout_rmse: float
    forecast_rmse: float | None
    identifiability: NDArray[np.float64]
    history: Trajectory
    notes: list[str]


@dataclass
class DigitalTwin:
    grid: CartesianGrid
    experiment: Experiment
    ports: list[FlowPort]
    physics: PhysicsSpec
    parameterization: RegionParameterization | ContrastParameterization | CoarseFieldParameterization
    face_dirichlet: dict[str, float] | None = None
    face_mult_x: NDArray[np.float64] | None = None
    face_mult_y: NDArray[np.float64] | None = None
    face_mult_z: NDArray[np.float64] | None = None
    kz_ratio: NDArray[np.float64] | None = None
    inverse: InverseSpec = field(default_factory=InverseSpec)
    last_leaderboard: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        kinds: dict[str, set[str]] = {}
        for c in self.experiment.controls:
            kinds.setdefault(c.port_name, set()).add(c.kind)
        validate_port_controls(self.ports, kinds)
        self.operator = ObservationOperator(self.grid, self.experiment.sensors, self.ports)

    def initial_state(self) -> State:
        n = self.grid.n_cells
        sg = None
        if self.physics.three_phase is not None:
            sg = np.full(n, float(self.physics.sg_init))
        pressure = np.full(n, float(self.physics.p_init))
        if self.physics.gravity > 0.0:
            z = self.grid.cell_centers()[:, 2]
            rho = (
                float(self.physics.sw_init) * self.physics.pvt.rho_w_sc
                + (1.0 - float(self.physics.sw_init)) * self.physics.pvt.rho_o_sc
            )
            pressure = pressure - rho * float(self.physics.gravity) * (z - float(np.mean(z)))
        return State(
            pressure=pressure,
            sw=np.full(n, float(self.physics.sw_init)),
            sg=sg,
            time_s=0.0,
        )

    def rock_from_k(self, k: NDArray[np.float64]) -> Rock:
        k = np.asarray(k, dtype=float).ravel()
        phi = float(getattr(self.parameterization, "phi", 0.20))
        if self.kz_ratio is not None:
            kz = k * np.asarray(self.kz_ratio, dtype=float).ravel()
        else:
            kz = k * float(self.physics.kz_over_kx)
        return Rock(permeability=k, porosity=np.full(self.grid.n_cells, phi), kz=kz)

    def rock_from_theta(self, theta: NDArray[np.float64]) -> Rock:
        return self.rock_from_k(self.parameterization.expand(theta))

    def simulate(
        self,
        rock: Rock,
        *,
        controls: list[ControlSeries] | None = None,
        t_end: float | None = None,
        report_times: NDArray[np.float64] | None = None,
        state0: State | None = None,
        dt_min: float | None = None,
    ) -> Trajectory:
        controls = list(self.experiment.controls if controls is None else controls)
        if t_end is None:
            times = [c.times_s[-1] for c in controls]
            times += [o.times_s[-1] for o in self.experiment.observations]
            t_end = max(times) if times else 1.0
        if report_times is None:
            report_times = self.experiment.all_times_s()
        floor = self.physics.dt_min if dt_min is None else float(dt_min)
        try:
            return simulate(
                self.grid,
                rock,
                self.physics.relperm,
                self.ports,
                controls,
                state0 or self.initial_state(),
                float(t_end),
                capillary=self.physics.capillary,
                face_dirichlet=self.face_dirichlet,
                pvt=self.physics.pvt,
                gravity=self.physics.gravity,
                face_mult_x=self.face_mult_x,
                face_mult_y=self.face_mult_y,
                face_mult_z=self.face_mult_z,
                implicit=self.physics.implicit_transport,
                single_phase=self.physics.single_phase,
                mu_single=self.physics.mu_single,
                dt_init=self.physics.dt_init,
                dt_min=floor,
                dt_max=self.physics.dt_max,
                max_cfl=self.physics.max_cfl,
                max_ds=self.physics.max_ds,
                report_times=report_times,
                three_phase=self.physics.three_phase,
            )
        except TimeStepUnderflow:
            nxt = max(floor * 0.1, 1.0e-4)
            if nxt >= floor - 1.0e-15:
                raise
            return self.simulate(
                rock,
                controls=controls,
                t_end=t_end,
                report_times=report_times,
                state0=state0,
                dt_min=nxt,
            )

    def _forward_vector(
        self,
        theta: NDArray[np.float64],
        series: list[ObservationSeries],
        *,
        t_end: float | None = None,
        controls: list[ControlSeries] | None = None,
    ) -> NDArray[np.float64]:
        rock = self.rock_from_theta(theta)
        times = np.unique(np.concatenate([s.times_s for s in series])) if series else None
        traj = self.simulate(rock, controls=controls, t_end=t_end, report_times=times)
        return predict_from_trajectory(self.operator, self.experiment, traj, series)

    def calibrate(
        self,
        *,
        n_ensemble: int | None = None,
        n_assimilations: int | None = None,
        prior_mean: float | None = None,
        prior_std: float | None = None,
        seed: int | None = None,
        preset: str | None = None,
        time_limit_s: float | None = None,
        algorithm: str | None = None,
        n_workers: int | None = None,
        inflation: float | None = None,
    ) -> Posterior:
        history_end = self.experiment.history_end_s
        assim = []
        hold = []
        for obs in self.experiment.observations:
            times = obs.times_s
            mask = np.ones(times.size, dtype=bool)
            if history_end is not None:
                mask = times <= float(history_end) + 1.0e-12
            if not np.any(mask):
                continue
            trimmed = ObservationSeries(
                sensor_name=obs.sensor_name,
                kind=obs.kind,
                times_s=times[mask],
                values=obs.values[mask],
                sigma=obs.sigma[mask],
                holdout=obs.holdout,
            )
            if obs.holdout:
                hold.append(trimmed)
            else:
                assim.append(trimmed)
        if not assim:
            raise ValueError("no assimilating observations in the history window")
        d_obs = stack_observations(assim)
        t_hist = float(history_end) if history_end is not None else float(np.max(d_obs.times))

        knobs: dict = {}
        if preset is not None:
            from reservoir_backend.inverse.presets import knobs_for

            knobs = knobs_for(preset)
        ne = int(n_ensemble if n_ensemble is not None else knobs.get("n_ensemble", self.inverse.n_ensemble))
        na = int(
            n_assimilations
            if n_assimilations is not None
            else knobs.get("n_assimilations", self.inverse.n_assimilations)
        )
        pstd = float(prior_std if prior_std is not None else knobs.get("prior_std", self.inverse.prior_std))
        infl = float(inflation if inflation is not None else knobs.get("inflation", self.inverse.inflation))
        algo = str(algorithm or knobs.get("algorithm", self.inverse.algorithm))
        budget = time_limit_s if time_limit_s is not None else self.inverse.time_limit_s

        def fwd(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._forward_vector(theta, assim, t_end=t_hist)

        result = run_esmda(
            self.parameterization,
            fwd,
            d_obs.values,
            d_obs.sigma ** 2,
            n_ensemble=ne,
            n_assimilations=na,
            prior_mean=self.inverse.prior_mean if prior_mean is None else prior_mean,
            prior_std=pstd,
            seed=int(self.inverse.seed if seed is None else seed),
            inflation=infl,
            time_limit_s=budget,
            algorithm=algo,
            n_workers=self.inverse.n_workers if n_workers is None else n_workers,
        )
        rock = self.rock_from_k(result.k_mean)
        hist = self.simulate(rock, t_end=t_hist, report_times=d_obs.times)
        d_post = predict_from_trajectory(self.operator, self.experiment, hist, assim)
        assim_rmse = float(np.sqrt(np.mean(((d_post - d_obs.values) / d_obs.sigma) ** 2)))
        hold_rmse = float("nan")
        if hold:
            d_h = stack_observations(hold)
            pred_h = predict_from_trajectory(self.operator, self.experiment, hist, hold)
            hold_rmse = float(np.sqrt(np.mean(((pred_h - d_h.values) / d_h.sigma) ** 2)))
        prior_spread = np.std(result.prior_theta, axis=0)
        ident = identifiability(prior_spread, result.theta_std)
        notes = list(result.diagnostics.notes)
        notes.append(f"assimilation whitened RMSE={assim_rmse:.4g}")
        notes.append(f"hold-out whitened RMSE={hold_rmse:.4g}")
        return Posterior(
            esmda=result,
            assimilate_rmse=assim_rmse,
            holdout_rmse=hold_rmse,
            forecast_rmse=None,
            identifiability=ident,
            history=hist,
            notes=notes,
        )

    def calibrate_auto(
        self,
        *,
        time_limit_s: float | None = None,
        blend: bool = True,
        search: bool = True,
        n_trials: int | None = None,
    ) -> Posterior:
        """Try several assimilators / knobs; pick and maybe blend on hold-out."""
        budget = time_limit_s if time_limit_s is not None else self.inverse.time_limit_s
        if search:
            from reservoir_backend.inverse.hpo import run_hpo

            best, rows, extra = run_hpo(self, time_limit_s=budget, n_trials=n_trials, blend=blend)
        else:
            from reservoir_backend.inverse.portfolio import run_portfolio

            best, rows, extra = run_portfolio(self, time_limit_s=budget, blend=blend)
        self.last_leaderboard = [r.as_dict() for r in rows]
        best.notes = list(best.notes) + extra
        return best

    def forecast(
        self,
        posterior: Posterior,
        *,
        controls: list[ControlSeries] | None = None,
        t_end: float | None = None,
    ) -> Trajectory:
        rock = self.rock_from_k(posterior.esmda.k_mean)
        history_end = self.experiment.history_end_s
        state0 = posterior.history.states[-1].copy() if posterior.history.states else self.initial_state()
        if t_end is None:
            times = [o.times_s[-1] for o in self.experiment.observations]
            t_end = max(times) if times else state0.time_s
        if history_end is not None and state0.time_s < float(history_end):
            # restart forecast from the last history state
            pass
        return self.simulate(
            rock,
            controls=controls,
            t_end=t_end,
            state0=state0,
            report_times=self.experiment.all_times_s(),
        )

    def score_forecast(self, traj: Trajectory) -> float:
        history_end = self.experiment.history_end_s
        future = []
        for obs in self.experiment.observations:
            if history_end is None:
                continue
            mask = obs.times_s > float(history_end) + 1.0e-12
            if not np.any(mask):
                continue
            future.append(
                ObservationSeries(
                    sensor_name=obs.sensor_name,
                    kind=obs.kind,
                    times_s=obs.times_s[mask],
                    values=obs.values[mask],
                    sigma=obs.sigma[mask],
                    holdout=obs.holdout,
                )
            )
        if not future:
            return float("nan")
        d = stack_observations(future)
        pred = predict_from_trajectory(self.operator, self.experiment, traj, future)
        return float(np.sqrt(np.mean(((pred - d.values) / d.sigma) ** 2)))

    def reconstruct(
        self,
        posterior: Posterior,
        time_s: float,
        *,
        n_members: int | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Posterior mean/std of static K and of dynamic fields at ``time_s``."""
        theta = posterior.esmda.theta_ensemble
        n_use = int(n_members if n_members is not None else min(self.inverse.reconstruct_members, theta.shape[0]))
        pick = np.linspace(0, theta.shape[0] - 1, n_use).astype(int)
        pressures = []
        sws = []
        sgs = []
        t_end = max(float(time_s), float(posterior.history.times_s[-1]) if posterior.history.times_s.size else float(time_s))

        def _one(e: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
            rock = self.rock_from_theta(theta[e])
            traj = self.simulate(rock, t_end=t_end, report_times=np.array([time_s]))
            st = traj.state_at(time_s)
            sg = np.zeros_like(st.sw) if st.sg is None else st.sg
            return st.pressure, st.sw, sg

        from reservoir_backend.inverse.parallel import map_members

        packed = map_members(_one, [int(e) for e in pick], self.inverse.n_workers)
        for pressure, sw, sg in packed:
            pressures.append(pressure)
            sws.append(sw)
            sgs.append(sg)
        p = np.stack(pressures, axis=0)
        sw = np.stack(sws, axis=0)
        sg = np.stack(sgs, axis=0)
        so = 1.0 - sw - sg
        return {
            "k_mean": posterior.esmda.k_mean,
            "k_std": posterior.esmda.k_std,
            "k_q10": posterior.esmda.k_q10,
            "k_q50": posterior.esmda.k_q50,
            "k_q90": posterior.esmda.k_q90,
            "pressure_mean": np.mean(p, axis=0),
            "pressure_std": np.std(p, axis=0),
            "sw_mean": np.mean(sw, axis=0),
            "sw_std": np.std(sw, axis=0),
            "so_mean": np.mean(so, axis=0),
            "so_std": np.std(so, axis=0),
            "sg_mean": np.mean(sg, axis=0),
            "sg_std": np.std(sg, axis=0),
        }


def mass_report(grid: CartesianGrid, rock: Rock, traj: Trajectory, pvt: BlackOilPVT | None = None) -> dict[str, float]:
    if not traj.reports:
        st = traj.states[0]
        m = water_mass(grid, rock, st.sw, pressure=st.pressure, pvt=pvt)
        return {"initial_mass": m, "final_mass": m, "relative_balance_error": 0.0}
    return traj.reports[-1].mass.as_dict()
