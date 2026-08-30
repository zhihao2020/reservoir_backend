"""Forward model interface. Parameters are passed explicitly, not via globals."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ControlSeries, State
from reservoir_backend.solver.impes import Trajectory


class ForwardModel(Protocol):
    def initialize(self, case: Any) -> None: ...

    def step(self, state: State, controls: list[ControlSeries], dt: float) -> State: ...

    def run(
        self,
        case: Any,
        parameters: NDArray[np.float64],
        observation_times: NDArray[np.float64] | None = None,
    ) -> Trajectory: ...
