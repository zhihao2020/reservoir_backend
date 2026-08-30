import numpy as np
import pytest

from reservoir_backend.domain.types import State
from reservoir_backend.solver.impes import Trajectory
from reservoir_backend.twin.loops import TwinLoops


def test_fast_loop_reads_last_trajectory() -> None:
    st = State(pressure=np.array([1.0e7]), sw=np.array([0.2]), time_s=5.0)
    traj = Trajectory(times_s=np.array([0.0, 5.0]), states=[st, st], reports=[], port_rates=[{}, {}])

    class _Twin:
        pass

    loops = TwinLoops(twin=_Twin(), slow_interval_s=30.0)  # type: ignore[arg-type]
    loops.last_traj = traj
    got = loops.fast_state(5.0)
    assert got.pressure[0] == pytest.approx(1.0e7)


def test_fast_step_requires_dpdp_state() -> None:
    class _Twin:
        _last_dual = None

    loops = TwinLoops(twin=_Twin())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        loops.fast_step(1.0)


def test_slow_loop_skips_inside_interval() -> None:
    class _Twin:
        def calibrate(self):
            raise AssertionError("should not assimilate yet")

    loops = TwinLoops(twin=_Twin(), slow_interval_s=30.0, last_slow_s=0.0)  # type: ignore[arg-type]
    assert loops.maybe_slow(10.0) is None
