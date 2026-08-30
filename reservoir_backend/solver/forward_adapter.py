"""Wrap DigitalTwin.simulate as a ForwardModel. Parameters are explicit."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.exceptions import PhysicsConvergenceError, TimeStepUnderflow
from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.physics.rock import Rock
from reservoir_backend.solver.impes import Trajectory
from reservoir_backend.twin.offline import DigitalTwin


@dataclass
class TwinForwardAdapter:
    """Existing IMPES / FIM / compositional twin behind the ForwardModel API."""

    twin: DigitalTwin
    conductivity: FractureConductivityModel | None = None
    log_cf: LogConductivityParameterization | None = None
    _rock: Rock | None = field(default=None, init=False, repr=False)

    def initialize(self, case: DigitalTwin | None = None) -> None:
        if case is not None:
            self.twin = case
        theta0 = np.full(self.twin.parameterization.n_params, float(np.log(1.0e-12)))
        if self.conductivity is not None and self.log_cf is not None:
            cf = self.log_cf.decode(self.log_cf.encode(1.0e-12))
            self._rock = self.twin.rock_from_k(self.conductivity.permeability(cf))
        else:
            self._rock = self.twin.rock_from_theta(theta0)

    def dual_rock_from_parameters(self, parameters: NDArray[np.float64]):
        """Gate 5: C_f updates DualRock.fracture only."""
        th = np.asarray(parameters, dtype=float).ravel()
        if self.log_cf is None:
            raise ValueError("log_cf parameterization is required for DualRock")
        if self.conductivity is not None:
            phi_m = float(getattr(self.log_cf, "phi", 0.08))
            phi_f = float(getattr(self.log_cf, "phi_fracture", 0.02))
            return self.conductivity.dual_rock(self.log_cf.decode(th), phi_matrix=phi_m, phi_fracture=phi_f)
        return self.twin.dual_rock_from_theta(th)

    def _rock_from_parameters(self, parameters: NDArray[np.float64]) -> Rock:
        th = np.asarray(parameters, dtype=float).ravel()
        if self.twin.uses_dpdp():
            dual = self.dual_rock_from_parameters(th)
            return dual.fracture
        if self.conductivity is not None and self.log_cf is not None:
            cf = self.log_cf.decode(th)
            return self.twin.rock_from_k(self.conductivity.permeability(cf))
        return self.twin.rock_from_theta(th)

    def step(self, state: State, controls: list[ControlSeries], dt: float) -> State:
        if self._rock is None:
            self.initialize()
        t1 = float(state.time_s) + float(dt)
        try:
            traj = self.twin.simulate(
                self._rock,
                controls=controls,
                t_end=t1,
                report_times=np.array([t1], dtype=float),
                state0=state,
            )
        except TimeStepUnderflow as exc:
            raise PhysicsConvergenceError(str(exc)) from exc
        last = traj.states[-1]
        last.time_s = t1
        return last

    def run(
        self,
        case: DigitalTwin | None,
        parameters: NDArray[np.float64],
        observation_times: NDArray[np.float64] | None = None,
    ) -> Trajectory:
        twin = self.twin if case is None else case
        if case is not None:
            self.twin = case
        if twin.uses_dpdp():
            try:
                return twin.simulate(parameters=parameters, report_times=observation_times)
            except TimeStepUnderflow as exc:
                raise PhysicsConvergenceError(str(exc)) from exc
        rock = self._rock_from_parameters(parameters)
        self._rock = rock
        try:
            return twin.simulate(rock, report_times=observation_times)
        except TimeStepUnderflow as exc:
            raise PhysicsConvergenceError(str(exc)) from exc
