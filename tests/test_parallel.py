import numpy as np

from reservoir_backend.inverse.esmda import run_esmda
from reservoir_backend.inverse.parallel import map_members, resolve_n_workers


def test_resolve_workers_caps_and_serial_small() -> None:
    assert resolve_n_workers(None, 1) == 1
    assert resolve_n_workers(None, 3) == 1
    assert resolve_n_workers(1, 20) == 1
    assert resolve_n_workers(4, 3) == 3
    assert resolve_n_workers(99, 10) == 10


def test_map_members_matches_serial() -> None:
    xs = list(range(8))
    assert map_members(lambda x: x * x, xs, 1) == [x * x for x in xs]
    assert map_members(lambda x: x * x, xs, 4) == [x * x for x in xs]


def test_esmda_parallel_matches_serial_mean() -> None:
    h = np.array([[1.0, 0.2], [0.0, 1.0], [0.4, 0.4]])
    obs = np.array([0.5, -0.2, 0.1])
    r = np.full(3, 0.04)

    class _P:
        n_params = 2

        def expand(self, theta):
            return np.asarray(theta, dtype=float).ravel()

        def sample_prior(self, n_ensemble, mean, std, seed):
            rng = np.random.default_rng(seed)
            return rng.normal(0.0, 1.0, size=(n_ensemble, 2))

    def fwd(theta):
        return h @ np.asarray(theta, dtype=float).ravel()

    kwargs = dict(
        parameterization=_P(),
        forward=fwd,
        obs=obs,
        r_diag=r,
        n_ensemble=16,
        n_assimilations=3,
        prior_mean=0.0,
        prior_std=1.0,
        seed=5,
        inflation=1.0,
        early_stop=False,
    )
    serial = run_esmda(**kwargs, n_workers=1)
    parallel = run_esmda(**kwargs, n_workers=4)
    assert np.allclose(serial.theta_mean, parallel.theta_mean, atol=1.0e-12)
    assert any("workers=4" in n for n in parallel.diagnostics.notes)
