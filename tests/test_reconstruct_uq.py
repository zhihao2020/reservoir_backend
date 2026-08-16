import numpy as np

from reservoir_backend.validation.synthetic import make_two_layer_waterflood


def test_reconstruct_returns_state_uncertainty() -> None:
    case = make_two_layer_waterflood(n_times=3, t_end=120.0, seed=9, history_frac=1.0)
    post = case.twin.calibrate(n_ensemble=8, n_assimilations=2, seed=4)
    fields = case.twin.reconstruct(post, float(post.history.times_s[-1]), n_members=4)
    for key in (
        "k_mean",
        "k_std",
        "k_q10",
        "k_q90",
        "pressure_mean",
        "pressure_std",
        "sw_mean",
        "sw_std",
        "so_mean",
        "so_std",
        "sg_mean",
        "sg_std",
    ):
        assert key in fields
        assert fields[key].shape == (case.grid.n_cells,)
        assert np.all(np.isfinite(fields[key]))
    assert np.all(fields["k_std"] >= 0.0)
    assert np.all(fields["pressure_std"] >= 0.0)
