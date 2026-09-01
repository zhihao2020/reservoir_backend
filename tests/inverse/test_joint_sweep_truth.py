"""Joint sweep must rebuild H(F(theta_true)), not reuse observations from another T_mf."""

import numpy as np

from reservoir_backend.synthetic import make_lab_v1_face_twin


def test_make_lab_v1_face_twin_tmf_true_rebuilds_observations() -> None:
    a = make_lab_v1_face_twin(
        cf_true=1.0e-12,
        tmf_true=0.5,
        ensemble_size=2,
        assimilation_steps=1,
        t_end=1.0,
        n_times=1,
    )
    b = make_lab_v1_face_twin(
        cf_true=1.0e-12,
        tmf_true=2.0,
        ensemble_size=2,
        assimilation_steps=1,
        t_end=1.0,
        n_times=1,
    )
    va = np.concatenate([o.values for o in a.twin.experiment.observations])
    vb = np.concatenate([o.values for o in b.twin.experiment.observations])
    assert va.size == vb.size
    assert not np.allclose(va, vb, rtol=0.0, atol=1.0)


def test_case_dev_has_no_sw_and_has_fracture_matrix() -> None:
    from reservoir_backend.io.case import load_case

    twin = load_case("examples/lab_v1/case_dev.yaml")
    kinds = {s.kind for s in twin.experiment.sensors}
    media = {s.medium for s in twin.experiment.sensors}
    assert "saturation" not in kinds
    assert "gas_saturation" in kinds
    assert media >= {"fracture", "matrix", "bulk"}
