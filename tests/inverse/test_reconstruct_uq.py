import numpy as np

from reservoir_backend.synthetic import make_two_layer_waterflood


def test_reconstruct_returns_point_estimate_fields() -> None:
    case = make_two_layer_waterflood(n_times=3, t_end=120.0, seed=9, history_frac=1.0)
    post = case.twin.calibrate()
    fields = case.twin.reconstruct(post, float(post.history.times_s[-1]))
    for key in ("k", "pressure", "sw", "so", "sg"):
        assert key in fields
        assert fields[key].shape == (case.grid.n_cells,)
        assert np.all(np.isfinite(fields[key]))
    assert np.allclose(fields["k"], post.k)
    assert np.allclose(fields["so"], 1.0 - fields["sw"] - fields["sg"])
