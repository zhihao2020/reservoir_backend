import numpy as np
import pytest

from reservoir_backend.inverse.log_conductivity import LogConductivityParameterization
from reservoir_backend.physics.conductivity import FractureConductivityModel
from reservoir_backend.solver.forward_adapter import TwinForwardAdapter
from reservoir_backend.synthetic import make_two_layer_waterflood


def test_adapter_run_on_existing_twin() -> None:
    case = make_two_layer_waterflood(n=(4, 3, 2), n_times=3, t_end=40.0)
    adapter = TwinForwardAdapter(case.twin)
    adapter.initialize(case.twin)
    traj = adapter.run(case.twin, case.theta_true, observation_times=case.twin.experiment.all_times_s())
    assert traj.states
    assert traj.times_s.size >= 1
    assert np.all(np.isfinite(traj.states[-1].pressure))


def test_adapter_step_advances_time() -> None:
    case = make_two_layer_waterflood(n=(4, 3, 2), n_times=3, t_end=40.0)
    adapter = TwinForwardAdapter(case.twin)
    adapter.initialize()
    adapter._rock = case.twin.rock_from_k(case.k_true)
    s0 = case.twin.initial_state()
    s1 = adapter.step(s0, case.twin.experiment.controls, dt=5.0)
    assert s1.time_s == pytest.approx(5.0)
    assert np.all(np.isfinite(s1.pressure))


def test_adapter_cf_path_keeps_matrix_k() -> None:
    case = make_two_layer_waterflood(n=(4, 3, 2), n_times=2, t_end=20.0)
    mask = np.zeros(case.grid.n_cells, dtype=bool)
    mask[::2] = True
    km = 1.0e-15
    cond = FractureConductivityModel(n_cells=case.grid.n_cells, fracture_mask=mask, k_matrix_m2=km)
    log_cf = LogConductivityParameterization()
    adapter = TwinForwardAdapter(case.twin, conductivity=cond, log_cf=log_cf)
    cf = np.array([5.0e-13])
    rock = adapter._rock_from_parameters(log_cf.encode(cf))
    k = np.asarray(rock.permeability, dtype=float).ravel()
    assert k[~mask] == pytest.approx(km)
    assert k[mask] == pytest.approx(5.0e-13)
