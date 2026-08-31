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


def test_incremental_window_excludes_past_times() -> None:
    from reservoir_backend.domain.types import ObservationSeries
    from reservoir_backend.twin.offline import window_observations

    obs = ObservationSeries(
        "P1",
        "pressure",
        np.array([10.0, 30.0, 60.0, 90.0]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.full(4, 0.1),
        False,
    )
    w = window_observations([obs], 30.0, 60.0)
    assert len(w) == 1
    assert w[0].times_s == pytest.approx(np.array([60.0]))
    assert window_observations([obs], 90.0, 120.0) == []


def test_from_posterior_does_not_resample_prior() -> None:
    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.twin.offline import Posterior

    members = np.array([[1.0, 2.0, 3.0, 4.0]])
    ens = PosteriorEnsemble(
        theta_members=members.T,
        k_members=np.ones((4, 1)),
        k_mean=np.ones(1),
        k_std=np.ones(1),
        theta_mean=np.array([2.5]),
        theta_std=np.array([1.0]),
    )
    from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState

    dummy = DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([1.0e7]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([1.2e7]), np.ones((1, 2)) * 4.0e-4),
        time_s=30.0,
    )
    ens.dual_states = [dummy.copy() for _ in range(4)]
    post = Posterior(
        theta=np.array([2.5]),
        k=np.ones(1),
        theta_std=np.array([1.0]),
        assimilate_rmse=0.0,
        holdout_rmse=0.0,
        forecast_rmse=None,
        identifiability=np.array([1.0]),
        history=Trajectory(times_s=np.array([0.0, 30.0]), states=[], reports=[], port_rates=[]),
        notes=[],
        ensemble=ens,
    )

    class _Twin:
        pass

    loops = TwinLoops.from_posterior(_Twin(), post, slow_interval_s=30.0)  # type: ignore[arg-type]
    assert loops.members is not None
    assert loops.members.shape == (1, 4)
    assert loops.last_slow_s == pytest.approx(30.0)
    assert loops.members[0, 0] == pytest.approx(1.0)
    assert loops.dual_states is not None
    assert all(s is not None for s in loops.dual_states)


def test_from_posterior_requires_duals_at_t_positive() -> None:
    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.twin.offline import Posterior

    members = np.array([[1.0, 2.0]])
    ens = PosteriorEnsemble(
        theta_members=members.T,
        k_members=np.ones((2, 1)),
        k_mean=np.ones(1),
        k_std=np.ones(1),
        theta_mean=np.array([1.5]),
        theta_std=np.array([0.5]),
    )
    post = Posterior(
        theta=np.array([1.5]),
        k=np.ones(1),
        theta_std=np.array([0.5]),
        assimilate_rmse=0.0,
        holdout_rmse=0.0,
        forecast_rmse=None,
        identifiability=np.array([1.0]),
        history=Trajectory(times_s=np.array([0.0, 10.0]), states=[], reports=[], port_rates=[]),
        notes=[],
        ensemble=ens,
    )

    class _Twin:
        pass

    with pytest.raises(ValueError, match="DualState"):
        TwinLoops.from_posterior(_Twin(), post)  # type: ignore[arg-type]


def test_slow_loop_is_parameter_enkf_not_calibrate() -> None:
    import inspect

    src = inspect.getsource(TwinLoops.maybe_slow)
    assert "calibrate(" not in src
    assert "analysis_parameters" in src
    assert "forecast_parameters" in src
    assert "window_observations" in src
    assert "classify_observations" in src
    assert "replace_failed_members" in src
    assert "eta_threshold" in src
    assert "last_fast_error" in src


def test_fast_step_marks_saturations_held() -> None:
    from reservoir_backend.comp.dual_state import CompositionalContinuumState, DualCompositionalState
    from reservoir_backend.grid.cartesian import CartesianGrid
    from reservoir_backend.physics.dual_rock import DualRock
    from reservoir_backend.physics.transfer import ComponentTransfer
    from reservoir_backend.solver.dpdp_context import DPDPModelContext
    from reservoir_backend.twin.offline import PhysicsSpec

    grid = CartesianGrid.uniform((0.1, 0.1, 0.1), 0.1)
    rock = DualRock.from_cf(1, k_matrix_m2=1.0e-15, phi_matrix=0.08, cf_m2=1.0e-12, phi_fracture=0.02)
    ctx = DPDPModelContext.build(grid, n_comp=2)
    dual = DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([1.0e7]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([1.2e7]), np.ones((1, 2)) * 4.0e-4),
        time_s=0.0,
    )

    class _Twin:
        grid = None
        ports = []
        _last_dual = dual
        _last_dual_rock = rock
        _lam_f = np.array([1.0e-3])
        _lam_m = np.array([1.0e-3])
        _ct_f = np.array([2.0e-9])
        _ct_m = np.array([2.0e-9])
        _v_mix_f = np.array([1.0e-4])
        _v_mix_m = np.array([1.0e-4])
        _sw_f = np.array([0.11])
        _sg_f = np.array([0.22])
        _sw_m = np.array([0.33])
        _sg_m = np.array([0.05])
        physics = PhysicsSpec()
        experiment = type("E", (), {"controls": []})()

        def dpdp_context(self):
            return ctx

        def transfer_operator(self):
            return ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)

    _Twin.grid = grid
    loops = TwinLoops(twin=_Twin())  # type: ignore[arg-type]
    st = loops.fast_step(1.0)
    assert st.saturations_held is True
    assert st.sw[0] == pytest.approx(0.11)
    assert st.moles_matrix is not None
