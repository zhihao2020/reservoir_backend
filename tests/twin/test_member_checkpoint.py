"""Gate 1: per-member DualState checkpoints and window-only online forward."""

import numpy as np

from reservoir_backend.comp.dual_state import DualCompositionalState
from reservoir_backend.twin.loops import OnlineMemberState, TwinLoops
from reservoir_backend.twin.offline import Posterior


def test_online_member_state_holds_dual() -> None:
    from reservoir_backend.comp.dual_state import CompositionalContinuumState

    dual = DualCompositionalState(
        fracture=CompositionalContinuumState(np.array([1.0e7]), np.ones((1, 2)) * 1.0e-4),
        matrix=CompositionalContinuumState(np.array([1.2e7]), np.ones((1, 2)) * 4.0e-4),
        time_s=30.0,
    )
    st = OnlineMemberState(theta=np.array([0.1]), dual_state=dual)
    assert st.dual_state is not None
    assert st.dual_state.time_s == 30.0


def test_from_posterior_uses_per_member_dual_not_mean() -> None:
    from reservoir_backend.comp.dual_state import CompositionalContinuumState
    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.solver.impes import Trajectory

    def _dual(p, t):
        return DualCompositionalState(
            fracture=CompositionalContinuumState(np.array([p]), np.ones((1, 2)) * 1.0e-4),
            matrix=CompositionalContinuumState(np.array([p + 1.0e6]), np.ones((1, 2)) * 4.0e-4),
            time_s=t,
        )

    members = np.array([[1.0, 2.0, 3.0]])
    duals = [_dual(1.0e7, 30.0), _dual(1.1e7, 30.0), _dual(1.2e7, 30.0)]
    ens = PosteriorEnsemble(
        theta_members=members.T,
        k_members=np.ones((3, 1)),
        k_mean=np.ones(1),
        k_std=np.ones(1),
        theta_mean=np.array([2.0]),
        theta_std=np.array([1.0]),
        dual_states=duals,
    )
    post = Posterior(
        theta=np.array([2.0]),
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
        _last_dual = _dual(9.9e9, 30.0)

    loops = TwinLoops.from_posterior(_Twin(), post, slow_interval_s=30.0)  # type: ignore[arg-type]
    assert loops.dual_states is not None
    assert len(loops.dual_states) == 3
    assert loops.dual_states[0].fracture.pressure[0] == 1.0e7
    assert loops.dual_states[1].fracture.pressure[0] == 1.1e7
    assert loops.dual_states[2].fracture.pressure[0] == 1.2e7
    loops.dual_states[0].fracture.pressure[0] = 0.0
    assert duals[0].fracture.pressure[0] == 1.0e7


def test_from_posterior_without_member_states_does_not_clone_mean() -> None:
    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.solver.impes import Trajectory

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
        history=Trajectory(times_s=np.array([0.0, 30.0]), states=[], reports=[], port_rates=[]),
        notes=[],
        ensemble=ens,
    )

    class _Twin:
        _last_dual = object()

    loops = TwinLoops.from_posterior(_Twin(), post, slow_interval_s=30.0)  # type: ignore[arg-type]
    assert loops.dual_states is None


def test_two_cf_members_keep_distinct_states_into_online() -> None:
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.grid.cartesian import CartesianGrid
    from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
    from reservoir_backend.physics.conductivity import FractureConductivityModel
    from reservoir_backend.physics.transfer import ComponentTransfer
    from reservoir_backend.solver.fi_comp_dual import initialize_dual_state, simulate_dual_comp
    from reservoir_backend.twin.history_match import _forward_ensemble
    from reservoir_backend.twin.offline import InverseSpec, PhysicsSpec
    from reservoir_backend.domain.types import ControlSeries, Experiment, ObservationSeries, Sensor

    grid = CartesianGrid.uniform((0.2, 0.1, 0.1), (0.1, 0.1, 0.1))
    spec = fluid_from_name("example", temperature_k=350.0)
    cond = FractureConductivityModel(n_cells=2, fracture_mask=np.array([True, True]), k_matrix_m2=1.0e-15)
    param = LogConductivityParameterization(conductivity=cond, prior_mean=0.0, prior_std=0.8)
    theta_lo = param.encode(np.array([1.0e-13]))
    theta_hi = param.encode(np.array([5.0e-12]))
    dual0 = initialize_dual_state(grid, param.dual_rock(theta_lo), spec, 1.20e7, p_matrix=1.22e7)
    dual0.fracture.pressure = np.array([1.20e7, 1.00e7])
    dual0.matrix.pressure = np.array([1.22e7, 1.18e7])
    tr = ComponentTransfer(shape_factor=40.0, k_matrix_m2=1.0e-15)
    _, end_lo = simulate_dual_comp(grid, param.dual_rock(theta_lo), spec, tr, [], [], dual0, t_end=0.2, dt_init=0.1, dt_max=0.1, max_steps=8)
    _, end_hi = simulate_dual_comp(grid, param.dual_rock(theta_hi), spec, tr, [], [], dual0.copy(), t_end=0.2, dt_init=0.1, dt_max=0.1, max_steps=8)
    dp_lo = abs(float(end_lo.fracture.pressure[0] - end_lo.fracture.pressure[1]))
    dp_hi = abs(float(end_hi.fracture.pressure[0] - end_hi.fracture.pressure[1]))
    assert dp_hi < dp_lo
    assert end_lo.flash is not None and end_hi.flash is not None

    from reservoir_backend.inverse.post_ensemble import PosteriorEnsemble
    from reservoir_backend.solver.impes import Trajectory
    from reservoir_backend.twin.offline import Posterior

    ens = PosteriorEnsemble(
        theta_members=np.stack([theta_lo, theta_hi]),
        k_members=np.ones((2, 1)),
        k_mean=np.ones(1),
        k_std=np.ones(1),
        theta_mean=0.5 * (theta_lo + theta_hi),
        theta_std=np.ones(1),
        dual_states=[end_lo, end_hi],
        flash_caches=[end_lo.flash, end_hi.flash],
    )
    post = Posterior(
        theta=ens.theta_mean,
        k=np.ones(1),
        theta_std=np.ones(1),
        assimilate_rmse=0.0,
        holdout_rmse=0.0,
        forecast_rmse=None,
        identifiability=np.ones(1),
        history=Trajectory(times_s=np.array([0.0, 1.0]), states=[], reports=[], port_rates=[]),
        notes=[],
        ensemble=ens,
    )

    class _Twin:
        parameterization = param
        inverse = InverseSpec(ensemble_size=2)
        physics = PhysicsSpec(model="compositional_dpdp", fluid=spec)
        _last_dual = end_lo

        def uses_dpdp(self):
            return True

    _Twin.grid = grid
    loops = TwinLoops.from_posterior(_Twin(), post, slow_interval_s=30.0)  # type: ignore[arg-type]
    assert loops.dual_states[0].fracture.pressure[0] == end_lo.fracture.pressure[0]
    assert loops.dual_states[1].fracture.pressure[0] == end_hi.fracture.pressure[0]
    _ = Sensor
    _ = Experiment
    _ = ObservationSeries
    _ = ControlSeries
    _ = _forward_ensemble


def test_forward_ensemble_passes_state0() -> None:
    from reservoir_backend.domain.types import ObservationSeries
    from reservoir_backend.twin.history_match import _forward_ensemble

    seen: list[float | None] = []

    class _Twin:
        grid = type("G", (), {"n_cells": 2})()

        def _forward_vector(self, theta, series, *, t_end=None, state0=None, **kwargs):
            seen.append(None if state0 is None else float(getattr(state0, "time_s", -1.0)))
            self._last_dual = state0
            return np.array([1.0])

    duals = [type("D", (), {"time_s": 30.0, "copy": lambda self=None: type("D2", (), {"time_s": 30.0})()})() for _ in range(2)]
    for d in duals:
        d.copy = lambda d=d: d
    members = np.array([[0.0, 0.1]])
    series = [
        ObservationSeries("P1", "pressure", np.array([60.0]), np.array([1.0]), np.array([0.1]), False)
    ]
    y, failed, n, out = _forward_ensemble(_Twin(), members, series, 60.0, dual_states=duals)  # type: ignore[arg-type]
    assert failed == []
    assert n == 2
    assert seen == [30.0, 30.0]
