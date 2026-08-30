"""Thin digital-twin orchestrator: simulate, calibrate, forecast."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor, State
from reservoir_backend.exceptions import TimeStepUnderflow
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.lm import identifiability, run_lm
from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble, sample_posterior_ensemble
from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
from reservoir_backend.inverse.parameterization import (
    ContrastParameterization,
    RegionParameterization,
)
from reservoir_backend.observation.operator import ObservationOperator
from reservoir_backend.physics.capillary import NoCapillary
from reservoir_backend.physics.pvt import BlackOilPVT
from reservoir_backend.physics.relperm import CoreyThreePhase, CoreyTwoPhase, TableThreePhase
from reservoir_backend.physics.rock import Rock
from reservoir_backend.ports.flow import FlowPort, validate_port_controls
from reservoir_backend.solver.impes import Trajectory, simulate, water_mass


@dataclass
class InverseSpec:
    """Low-dimensional inversion. Default LM; V1 Cf path is ES-MDA."""

    prior_mean: float | NDArray[np.float64] = float(np.log(1.0e-12))
    prior_std: float | NDArray[np.float64] = 0.8
    max_iter: int = 8
    fd_rel: float = 0.05
    time_limit_s: float | None = None
    post_ensemble_enabled: bool = False
    post_ensemble_ne: int = 8
    post_ensemble_seed: int = 0
    algorithm: str = "lm"
    ensemble_size: int = 16
    assimilation_steps: int = 4
    seed: int = 0
    alpha: NDArray[np.float64] | list[float] | None = None
    clip_innovation: bool = False
    n_workers: int | None = None


@dataclass
class PhysicsSpec:
    relperm: CoreyTwoPhase = field(default_factory=CoreyTwoPhase)
    three_phase: CoreyThreePhase | TableThreePhase | None = None
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
    sfi_outer: int = 0
    reupdate_pressure: bool = True
    upwind_type: str = "potential"
    # Unreduced liberation-ruler invert passed without TimeStepUnderflow; FIM is product default.
    fully_implicit: bool = True
    max_steps: int = 12000
    hydrostatic_init: bool = False
    model: str = "two_phase_immiscible"
    fluid: object | None = None
    temperature_k: float = 350.0
    z_init: NDArray[np.float64] | None = None
    shape_factor: float = 40.0
    phi_fracture: float = 0.02
    k_matrix_m2: float | None = None


def three_phase_for_fim(relperm, existing=None):
    # Dead-oil wrapper so invert FIM actually enters solve_fi_step.
    if existing is not None:
        return existing
    from reservoir_backend.physics.relperm import CoreyThreePhase

    mu_w = float(getattr(relperm, "mu_w", 1.0e-3))
    mu_o = float(getattr(relperm, "mu_o", 5.0e-3))
    mu_g = float(getattr(relperm, "mu_g", 2.0e-5))
    swi = float(getattr(relperm, "swi", 0.20))
    sor = float(getattr(relperm, "sor", 0.15))
    if swi + sor >= 0.99:
        sor = max(0.0, 0.98 - swi)
    return CoreyThreePhase(swi=swi, sor=sor, sgr=0.0, mu_w=mu_w, mu_o=mu_o, mu_g=mu_g)


@dataclass
class DataVector:
    values: NDArray[np.float64]
    sigma: NDArray[np.float64]
    times: NDArray[np.float64]
    names: list[str]
    kinds: list[str]
    holdout: NDArray[np.bool_]


def window_observations(
    observations: list[ObservationSeries],
    t_lo: float,
    t_hi: float,
) -> list[ObservationSeries]:
    """Keep assimilating samples in ``(t_lo, t_hi]``. Does not reuse earlier times."""
    lo = float(t_lo)
    hi = float(t_hi)
    out: list[ObservationSeries] = []
    for obs in observations:
        if obs.holdout:
            continue
        mask = (obs.times_s > lo + 1.0e-12) & (obs.times_s <= hi + 1.0e-12)
        if not np.any(mask):
            continue
        out.append(
            ObservationSeries(
                sensor_name=obs.sensor_name,
                kind=obs.kind,
                times_s=obs.times_s[mask],
                values=obs.values[mask],
                sigma=obs.sigma[mask],
                holdout=False,
            )
        )
    return out


def split_history_observations(
    observations: list[ObservationSeries],
    history_end_s: float | None,
) -> tuple[list[ObservationSeries], list[ObservationSeries]]:
    """Trim to the history window and split assimilating vs hold-out series."""
    assim: list[ObservationSeries] = []
    hold: list[ObservationSeries] = []
    for obs in observations:
        times = obs.times_s
        mask = np.ones(times.size, dtype=bool)
        if history_end_s is not None:
            mask = times <= float(history_end_s) + 1.0e-12
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
        (hold if obs.holdout else assim).append(trimmed)
    return assim, hold


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
        rates, bhp = traj.rates_and_bhp_at(float(t))
        sensor = sensors[name]
        if sensor.kind != kind:
            sensor = Sensor(
                name=sensor.name,
                kind=kind,
                x=sensor.x,
                y=sensor.y,
                z=sensor.z,
                volume_m3=sensor.volume_m3,
                probe_diameter_m=sensor.probe_diameter_m,
                port_name=sensor.port_name,
                sigma=sensor.sigma,
            )
        out[i] = operator.sample(sensor, state, port_rates=rates, port_bhp=bhp)
    return out


@dataclass
class FieldStats:
    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    q10: NDArray[np.float64]
    q90: NDArray[np.float64]


@dataclass
class Posterior:
    theta: NDArray[np.float64]
    k: NDArray[np.float64]
    theta_std: NDArray[np.float64]
    assimilate_rmse: float
    holdout_rmse: float
    forecast_rmse: float | None
    identifiability: NDArray[np.float64]
    history: Trajectory
    notes: list[str]
    n_forward: int = 0
    misfit: list[float] = field(default_factory=list)
    ensemble: PosteriorEnsemble | None = None


@dataclass
class DigitalTwin:
    grid: CartesianGrid
    experiment: Experiment
    ports: list[FlowPort]
    physics: PhysicsSpec
    parameterization: RegionParameterization | ContrastParameterization | LogConductivityParameterization
    face_dirichlet: dict[str, float] | None = None
    face_mult_x: NDArray[np.float64] | None = None
    face_mult_y: NDArray[np.float64] | None = None
    face_mult_z: NDArray[np.float64] | None = None
    kz_ratio: NDArray[np.float64] | None = None
    inverse: InverseSpec = field(default_factory=InverseSpec)
    _dpdp_ctx: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        kinds: dict[str, set[str]] = {}
        for c in self.experiment.controls:
            kinds.setdefault(c.port_name, set()).add(c.kind)
        validate_port_controls(self.ports, kinds)
        self.operator = ObservationOperator(self.grid, self.experiment.sensors, self.ports)
        self._last_dual = None
        self._lam_f = None
        self._lam_m = None
        self._last_dual_rock = None
        self._ct_f = None
        self._ct_m = None
        self._v_mix_f = None
        self._v_mix_m = None
        self._sw_f = None
        self._sg_f = None
        self._sw_m = None
        self._sg_m = None

    def uses_dpdp(self) -> bool:
        model = str(self.physics.model).lower()
        return model in {"dpdp", "compositional_dpdp", "dual", "dual_compositional"}

    def dual_rock_from_theta(self, theta: NDArray[np.float64]):
        if isinstance(self.parameterization, LogConductivityParameterization):
            return self.parameterization.dual_rock(theta)
        cf = float(np.asarray(self.parameterization.decode(theta), dtype=float).ravel()[0])
        return self.dual_rock_from_cf(cf)

    def dual_rock_from_cf(self, cf_m2: float):
        from reservoir_backend.physics.dual_rock import DualRock

        km = float(
            getattr(getattr(self.parameterization, "conductivity", None), "k_matrix_m2", None)
            or self.physics.k_matrix_m2
            or 1.0e-15
        )
        phi_m = float(getattr(self.parameterization, "phi", 0.08))
        phi_f = float(getattr(self.parameterization, "phi_fracture", self.physics.phi_fracture))
        return DualRock.from_cf(
            self.grid.n_cells,
            k_matrix_m2=km,
            phi_matrix=phi_m,
            cf_m2=float(cf_m2),
            phi_fracture=phi_f,
        )

    def dpdp_context(self):
        from reservoir_backend.solver.dpdp_context import DPDPModelContext

        if self._dpdp_ctx is None and self.physics.fluid is not None:
            self._dpdp_ctx = DPDPModelContext.build(
                self.grid,
                int(self.physics.fluid.nc),
                sensors=list(self.experiment.sensors),
            )
        return self._dpdp_ctx

    def transfer_operator(self):
        from reservoir_backend.physics.transfer import ComponentTransfer

        km = float(
            getattr(getattr(self.parameterization, "conductivity", None), "k_matrix_m2", None)
            or self.physics.k_matrix_m2
            or 1.0e-15
        )
        return ComponentTransfer(shape_factor=float(self.physics.shape_factor), k_matrix_m2=km)

    def initial_state(self) -> State:
        n = self.grid.n_cells
        if self.uses_dpdp() and self.physics.fluid is not None:
            from reservoir_backend.solver.fi_comp_dual import dual_to_state, initialize_dual_state

            dual = self.dual_rock_from_cf(float(self.physics.k_matrix_m2 or 1.0e-12))
            st = initialize_dual_state(self.grid, dual, self.physics.fluid, float(self.physics.p_init))
            return dual_to_state(self.physics.fluid, st)
        if str(self.physics.model).lower() in {"compositional", "comp", "eos"} and self.physics.fluid is not None:
            from reservoir_backend.solver.fi_comp import initialize_state

            phi = float(getattr(self.parameterization, "phi", 0.20))
            rock0 = Rock(np.full(n, 1.0e-12), np.full(n, phi))
            return initialize_state(self.grid, rock0, self.physics.fluid, float(self.physics.p_init))
        sg = None
        if self.physics.three_phase is not None or bool(self.physics.fully_implicit):
            sg = np.full(n, float(self.physics.sg_init))
        pressure = np.full(n, float(self.physics.p_init))
        if self.physics.hydrostatic_init and self.physics.gravity > 0.0:
            z = self.grid.cell_centers()[:, 2]
            sw0 = float(self.physics.sw_init)
            sg0 = float(self.physics.sg_init) if self.physics.three_phase is not None else 0.0
            so0 = max(0.0, 1.0 - sw0 - sg0)
            pvt = self.physics.pvt
            b_w = float(np.asarray(pvt.b_w(self.physics.p_init)))
            b_o = float(np.asarray(pvt.b_o(self.physics.p_init)))
            b_g = float(np.asarray(pvt.b_g(self.physics.p_init)))
            rho = (
                sw0 * pvt.rho_w_sc * b_w
                + so0 * pvt.rho_o_sc * b_o
                + sg0 * pvt.rho_g_sc * b_g
            )
            pressure = pressure - rho * float(self.physics.gravity) * (z - float(np.mean(z)))
        rs = None
        if self.physics.pvt.has_live_oil():
            rs = np.asarray(self.physics.pvt.rs(pressure), dtype=float).ravel()
        return State(
            pressure=pressure,
            sw=np.full(n, float(self.physics.sw_init)),
            sg=sg,
            rs=rs,
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
        rock: Rock | None = None,
        *,
        controls: list[ControlSeries] | None = None,
        t_end: float | None = None,
        report_times: NDArray[np.float64] | None = None,
        state0: State | DualCompositionalState | None = None,
        dt_min: float | None = None,
        parameters: NDArray[np.float64] | None = None,
        dual_rock=None,
    ) -> Trajectory:
        controls = list(self.experiment.controls if controls is None else controls)
        if t_end is None:
            times = [c.times_s[-1] for c in controls]
            times += [o.times_s[-1] for o in self.experiment.observations]
            t_end = max(times) if times else 1.0
        if report_times is None:
            report_times = self.experiment.all_times_s()
        floor = self.physics.dt_min if dt_min is None else float(dt_min)
        if self.uses_dpdp() and self.physics.fluid is not None:
            from reservoir_backend.solver.fi_comp_dual import (
                dual_from_visual_state,
                initialize_dual_state,
                simulate_dual_comp,
            )

            if dual_rock is None:
                if parameters is not None:
                    dual_rock = self.dual_rock_from_theta(parameters)
                elif rock is not None:
                    dual_rock = self.dual_rock_from_cf(float(np.mean(np.asarray(rock.permeability, dtype=float))))
                else:
                    raise ValueError("DPDP simulate needs parameters, dual_rock, or rock")
            if isinstance(state0, DualCompositionalState):
                dual0 = state0.copy()
            elif state0 is not None:
                has_moles = state0.moles is not None and state0.moles_matrix is not None
                last = self._last_dual
                if has_moles:
                    dual0 = dual_from_visual_state(self.grid, dual_rock, self.physics.fluid, state0)
                elif last is not None and abs(float(last.time_s) - float(state0.time_s)) < 1.0e-12:
                    dual0 = last.copy()
                elif float(state0.time_s) > 1.0e-15:
                    raise ValueError("lossless DPDP restart requires moles_matrix at t>0")
                else:
                    dual0 = initialize_dual_state(self.grid, dual_rock, self.physics.fluid, float(self.physics.p_init))
                    dual0.time_s = float(state0.time_s)
            else:
                dual0 = initialize_dual_state(self.grid, dual_rock, self.physics.fluid, float(self.physics.p_init))
            traj, dual = simulate_dual_comp(
                self.grid,
                dual_rock,
                self.physics.fluid,
                self.transfer_operator(),
                self.ports,
                controls,
                dual0,
                float(t_end),
                dt_init=self.physics.dt_init,
                dt_min=floor,
                dt_max=self.physics.dt_max,
                max_steps=int(self.physics.max_steps),
                report_times=report_times,
                context=self.dpdp_context(),
            )
            from reservoir_backend.comp.properties import flash_compressibility, flash_state

            pf = flash_state(self.physics.fluid, dual.fracture.pressure, dual.fracture.moles)
            pm = flash_state(self.physics.fluid, dual.matrix.pressure, dual.matrix.moles)
            self._last_dual = dual
            self._lam_f = pf.lam_l + pf.lam_v + pf.lam_w
            self._lam_m = pm.lam_l + pm.lam_v + pm.lam_w
            self._ct_f = flash_compressibility(self.physics.fluid, dual.fracture.pressure, dual.fracture.moles, pf)
            self._ct_m = flash_compressibility(self.physics.fluid, dual.matrix.pressure, dual.matrix.moles, pm)
            self._v_mix_f = pf.v_mix.copy()
            self._v_mix_m = pm.v_mix.copy()
            self._sw_f = pf.sw.copy()
            self._sg_f = pf.sv.copy()
            self._sw_m = pm.sw.copy()
            self._sg_m = pm.sv.copy()
            self._last_dual_rock = dual_rock
            return traj
        if rock is None:
            raise ValueError("simulate requires a Rock for single-continuum models")
        if str(self.physics.model).lower() in {"compositional", "comp", "eos"} and self.physics.fluid is not None:
            from reservoir_backend.solver.fi_comp import simulate_comp

            return simulate_comp(
                self.grid,
                rock,
                self.physics.fluid,
                self.ports,
                controls,
                state0 or self.initial_state(),
                float(t_end),
                dt_init=self.physics.dt_init,
                dt_min=floor,
                dt_max=self.physics.dt_max,
                max_steps=int(self.physics.max_steps),
                report_times=report_times,
            )
        fim = bool(self.physics.fully_implicit)
        implicit = bool(self.physics.implicit_transport or fim)
        three = three_phase_for_fim(self.physics.relperm, self.physics.three_phase) if fim else self.physics.three_phase
        # max_steps is a safety fuse, not an explicit-CFL step budget.
        # FIM Δt is Newton-count; do not inflate the fuse here.
        nstep = int(self.physics.max_steps)
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
                implicit=implicit,
                sfi_outer=int(self.physics.sfi_outer),
                reupdate_pressure=bool(self.physics.reupdate_pressure),
                upwind_type=str(self.physics.upwind_type),
                fully_implicit=fim,
                single_phase=self.physics.single_phase,
                mu_single=self.physics.mu_single,
                dt_init=self.physics.dt_init,
                dt_min=floor,
                dt_max=self.physics.dt_max,
                max_cfl=self.physics.max_cfl,
                max_ds=self.physics.max_ds,
                max_steps=nstep,
                report_times=report_times,
                three_phase=three,
            )
        except TimeStepUnderflow as exc:
            msg = str(exc)
            if "more than" in msg and "steps" in msg:
                raise
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

    def inflate_observations(
        self,
        rock: Rock,
        *,
        clean: list[ObservationSeries],
        history_end_s: float | None = None,
        extra_cap_mult: float | None = 1.5,
    ) -> dict[str, float]:
        """Set each σ to ``sqrt(σ_inst² + RMSE(F(rock), clean)²)``.

        ``clean`` is the noiseless gauge series. Instrument σ stays on the
        current ``experiment.observations``. Using the noisy series as clean
        folds the instrument draw into R and over-damps K.
        ``extra_cap_mult`` clips model-error extra at ``mult * σ_inst``.
        """
        from reservoir_backend.observation.error import inflate_sigma

        if not self.experiment.observations:
            raise ValueError("no observations to inflate")
        hist = self.experiment.history_end_s if history_end_s is None else float(history_end_s)
        clean_map = {s.sensor_name: s for s in clean}
        times = np.unique(np.concatenate([s.times_s for s in self.experiment.observations]))
        t_end = float(hist) if hist is not None else float(times.max())
        report = times[times <= t_end + 1.0] if hist is not None else times
        if report.size == 0:
            report = times
        traj = self.simulate(rock, t_end=float(report.max()), report_times=report)
        extras: dict[str, float] = {}
        inflated: list[ObservationSeries] = []
        for series in self.experiment.observations:
            ref = clean_map.get(series.sensor_name)
            if ref is None:
                raise ValueError(f"clean series missing {series.sensor_name}")
            sensor = next(s for s in self.experiment.sensors if s.name == series.sensor_name)
            mask = series.times_s <= t_end + 1.0 if hist is not None else np.ones(series.times_s.size, dtype=bool)
            pred = []
            clean_v = []
            for i, t in enumerate(series.times_s):
                if not mask[i]:
                    continue
                idx = int(np.argmin(np.abs(traj.times_s - t)))
                rates, bhp = traj.rates_and_bhp_at(float(t))
                pred.append(self.operator.sample(sensor, traj.state_at(float(t)), port_rates=rates, port_bhp=bhp))
                j = int(np.argmin(np.abs(ref.times_s - t)))
                clean_v.append(float(ref.values[j]))
            inst = float(np.mean(series.sigma[mask])) if np.any(mask) else float(np.mean(series.sigma))
            cap = None if extra_cap_mult is None else inst * float(extra_cap_mult)
            extra, sig = inflate_sigma(pred, clean_v, inst, extra_cap=cap)
            extras[series.sensor_name] = extra
            inflated.append(
                ObservationSeries(
                    series.sensor_name,
                    series.kind,
                    series.times_s,
                    series.values,
                    np.full(series.times_s.size, sig),
                    series.holdout,
                )
            )
        self.experiment.observations = inflated
        return extras

    def _forward_vector(
        self,
        theta: NDArray[np.float64],
        series: list[ObservationSeries],
        *,
        t_end: float | None = None,
        controls: list[ControlSeries] | None = None,
    ) -> NDArray[np.float64]:
        times = np.unique(np.concatenate([s.times_s for s in series])) if series else None
        if self.uses_dpdp():
            traj = self.simulate(parameters=theta, controls=controls, t_end=t_end, report_times=times)
        else:
            rock = self.rock_from_theta(theta)
            traj = self.simulate(rock, controls=controls, t_end=t_end, report_times=times)
        return predict_from_trajectory(self.operator, self.experiment, traj, series)

    def calibrate(
        self,
        *,
        prior_mean: float | NDArray[np.float64] | None = None,
        prior_std: float | NDArray[np.float64] | None = None,
        max_iter: int | None = None,
        time_limit_s: float | None = None,
        fd_rel: float | None = None,
    ) -> Posterior:
        """Fit θ to history observations with LM or ES-MDA."""
        history_end = self.experiment.history_end_s
        assim, hold = split_history_observations(self.experiment.observations, history_end)
        if not assim:
            raise ValueError("no assimilating observations in the history window")
        d_obs = stack_observations(assim)
        t_hist = float(history_end) if history_end is not None else float(np.max(d_obs.times))

        if prior_mean is None:
            pmean = getattr(self.parameterization, "prior_mean", self.inverse.prior_mean)
        else:
            pmean = prior_mean
        if prior_std is None:
            pstd = getattr(self.parameterization, "prior_std", self.inverse.prior_std)
        else:
            pstd = prior_std
        pmean = np.asarray(pmean, dtype=float)
        pstd = np.asarray(pstd, dtype=float)
        niter = int(self.inverse.max_iter if max_iter is None else max_iter)
        fd = float(self.inverse.fd_rel if fd_rel is None else fd_rel)
        budget = time_limit_s if time_limit_s is not None else self.inverse.time_limit_s

        algo = str(self.inverse.algorithm).strip().lower()
        if algo in {"esmda", "es-mda", "es_mda"}:
            from reservoir_backend.twin.history_match import HistoryMatchWorkflow

            return HistoryMatchWorkflow().run(
                self,
                observations=self.experiment.observations,
                parameter_prior=(pmean, pstd),
                config={
                    "ensemble_size": self.inverse.ensemble_size,
                    "assimilation_steps": self.inverse.assimilation_steps,
                    "seed": self.inverse.seed,
                    "alpha": self.inverse.alpha,
                    "clip_innovation": self.inverse.clip_innovation,
                    "n_workers": self.inverse.n_workers,
                },
            )

        def fwd(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._forward_vector(theta, assim, t_end=t_hist)

        result = run_lm(
            self.parameterization,
            fwd,
            d_obs.values,
            d_obs.sigma,
            prior_mean=pmean,
            prior_std=pstd,
            max_iter=niter,
            fd_rel=fd,
            time_limit_s=budget,
        )
        rock = self.rock_from_k(result.k)
        hist = self.simulate(rock, t_end=t_hist, report_times=d_obs.times)
        d_post = predict_from_trajectory(self.operator, self.experiment, hist, assim)
        assim_rmse = float(np.sqrt(np.mean(((d_post - d_obs.values) / d_obs.sigma) ** 2)))
        hold_rmse = float("nan")
        if hold:
            d_h = stack_observations(hold)
            pred_h = predict_from_trajectory(self.operator, self.experiment, hist, hold)
            hold_rmse = float(np.sqrt(np.mean(((pred_h - d_h.values) / d_h.sigma) ** 2)))
        prior_spread = np.broadcast_to(np.asarray(pstd, dtype=float), result.theta.shape)
        ident = identifiability(prior_spread, result.theta_std)
        notes = list(result.notes)
        notes.append(f"assimilation whitened RMSE={assim_rmse:.4g}")
        notes.append(f"hold-out whitened RMSE={hold_rmse:.4g}")
        ensemble = None
        if bool(self.inverse.post_ensemble_enabled):
            ensemble = sample_posterior_ensemble(
                self.parameterization,
                result,
                ne=int(self.inverse.post_ensemble_ne),
                seed=int(self.inverse.post_ensemble_seed),
            )
            notes.append(f"post ensemble Ne={int(self.inverse.post_ensemble_ne)}")
        return Posterior(
            theta=result.theta,
            k=result.k,
            theta_std=result.theta_std,
            assimilate_rmse=assim_rmse,
            holdout_rmse=hold_rmse,
            forecast_rmse=None,
            identifiability=ident,
            history=hist,
            notes=notes,
            n_forward=int(result.n_forward),
            misfit=list(result.misfit),
            ensemble=ensemble,
        )

    def assimilate(
        self,
        posterior: Posterior,
        new_observations: list[ObservationSeries],
        *,
        max_iter: int = 2,
        time_limit_s: float | None = None,
    ) -> Posterior:
        """Warm-start LM from an existing posterior with additional observations."""
        merged = list(self.experiment.observations)
        by_name = {o.sensor_name: o for o in merged}
        for obs in new_observations:
            if obs.sensor_name in by_name:
                old = by_name[obs.sensor_name]
                times = np.unique(np.concatenate([old.times_s, obs.times_s]))
                vals = np.zeros(times.size, dtype=float)
                sigs = np.zeros(times.size, dtype=float)
                for i, t in enumerate(times):
                    j = int(np.argmin(np.abs(old.times_s - t)))
                    vals[i] = float(old.values[j])
                    sigs[i] = float(old.sigma[j])
                    k = int(np.argmin(np.abs(obs.times_s - t)))
                    if abs(float(obs.times_s[k]) - float(t)) < 1.0e-9:
                        vals[i] = float(obs.values[k])
                        sigs[i] = float(obs.sigma[k])
                by_name[obs.sensor_name] = ObservationSeries(
                    obs.sensor_name, obs.kind, times, vals, sigs, obs.holdout or old.holdout
                )
            else:
                by_name[obs.sensor_name] = obs
        self.experiment.observations = list(by_name.values())
        return self.calibrate(
            prior_mean=posterior.theta.copy(),
            prior_std=np.maximum(posterior.theta_std, 1.0e-6),
            max_iter=int(max_iter),
            time_limit_s=time_limit_s,
        )

    def forecast(
        self,
        posterior: Posterior,
        *,
        controls: list[ControlSeries] | None = None,
        t_end: float | None = None,
    ) -> Trajectory:
        rock = self.rock_from_k(posterior.k)
        history_end = self.experiment.history_end_s
        state0 = posterior.history.states[-1].copy() if posterior.history.states else self.initial_state()
        if t_end is None:
            ends = [float(state0.time_s)]
            ends.extend(float(c.times_s[-1]) for c in (controls or self.experiment.controls) if c.times_s.size)
            ends.extend(float(o.times_s[-1]) for o in self.experiment.observations if o.times_s.size)
            t_end = max(ends)
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
    ) -> dict[str, NDArray[np.float64]]:
        """Point-estimate static K and dynamic fields at ``time_s`` from F(θ)."""
        t_end = max(float(time_s), float(posterior.history.times_s[-1]) if posterior.history.times_s.size else float(time_s))
        traj = self.simulate(self.rock_from_k(posterior.k), t_end=t_end, report_times=np.array([time_s]))
        st = traj.state_at(time_s)
        sg = np.zeros_like(st.sw) if st.sg is None else st.sg
        so = 1.0 - st.sw - sg
        return {
            "k": np.asarray(posterior.k, dtype=float),
            "pressure": np.asarray(st.pressure, dtype=float),
            "sw": np.asarray(st.sw, dtype=float),
            "so": np.asarray(so, dtype=float),
            "sg": np.asarray(sg, dtype=float),
        }


def mass_report(grid: CartesianGrid, rock: Rock, traj: Trajectory, pvt: BlackOilPVT | None = None) -> dict[str, float]:
    if not traj.reports:
        st = traj.states[0]
        m = water_mass(grid, rock, st.sw, pressure=st.pressure, pvt=pvt)
        return {"initial_mass": m, "final_mass": m, "relative_balance_error": 0.0}
    return traj.reports[-1].mass.as_dict()
