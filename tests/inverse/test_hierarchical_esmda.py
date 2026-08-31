import numpy as np

from reservoir_backend.domain.types import Experiment, Sensor
from reservoir_backend.twin.history_match import joint_phase_schedule, observation_mask_for_phase
from reservoir_backend.twin.offline import DataVector


def test_phase_schedule_two_param() -> None:
    assert joint_phase_schedule(2, 2) == ["tmf", "cf"]
    assert joint_phase_schedule(3, 2) == ["tmf", "cf", "tmf"]
    assert joint_phase_schedule(4, 2) == ["tmf", "tmf", "cf", "cf"]
    assert joint_phase_schedule(5, 2) == ["tmf", "tmf", "cf", "tmf", "cf"]
    assert joint_phase_schedule(1, 2) == ["joint"]
    assert joint_phase_schedule(4, 1) == ["joint"] * 4


def test_obs_mask_splits_fracture_pressure_and_matrix() -> None:
    class _P:
        n_params = 2

    class _T:
        parameterization = _P()
        experiment = Experiment(
            sensors=[
                Sensor("P_f_in", "pressure", 0.1, 0.1, 0.05, sigma=2e3, medium="fracture"),
                Sensor("P_m_mid", "pressure", 0.15, 0.1, 0.05, sigma=2e3, medium="matrix"),
                Sensor("S_m", "sg", 0.15, 0.1, 0.05, sigma=0.03, medium="matrix"),
            ]
        )

    d = DataVector(
        values=np.ones(3),
        sigma=np.ones(3),
        times=np.ones(3),
        names=["P_f_in", "P_m_mid", "S_m"],
        kinds=["pressure", "pressure", "gas_saturation"],
        holdout=np.zeros(3, dtype=bool),
    )
    cf = observation_mask_for_phase(_T(), d, "cf")
    tmf = observation_mask_for_phase(_T(), d, "tmf")
    assert list(cf) == [True, False, False]
    assert list(tmf) == [False, True, True]
    assert observation_mask_for_phase(_T(), d, "joint").all()
