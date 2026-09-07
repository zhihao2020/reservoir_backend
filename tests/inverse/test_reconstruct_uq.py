import numpy as np

from reservoir_backend.synthetic import make_lab_v1_face_twin


def test_reconstruct_returns_point_estimate_fields() -> None:
    case = make_lab_v1_face_twin(n_times=3, t_end=2.0, seed=9, with_saturation=False)
    case.twin.inverse.algorithm = "lm"
    case.twin.inverse.max_iter = 2
    post = case.twin.calibrate(max_iter=2)
    fields = case.twin.reconstruct(post, float(post.history.times_s[-1]))
    for key in ("k", "pressure", "sw", "so", "sg"):
        assert key in fields
        assert fields[key].shape == (case.grid.n_cells,)
        assert np.all(np.isfinite(fields[key]))
    assert np.allclose(fields["k"], post.k)
    assert np.allclose(fields["so"], 1.0 - fields["sw"] - fields["sg"])
