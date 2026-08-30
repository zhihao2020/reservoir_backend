import numpy as np
import pytest

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp


def test_k_guess_matches_wilson_flash() -> None:
    eos = example_c1_nc10()
    z = np.array([0.55, 0.45])
    p, t = 1.2e7, 350.0
    a = flash_tp(eos, p, t, z)
    assert a.k is not None
    b = flash_tp(eos, p * 1.001, t, z, k_guess=a.k)
    assert b.vapor_frac == pytest.approx(a.vapor_frac, rel=0.05, abs=0.02)


def test_stability_bypass_single_phase() -> None:
    eos = example_c1_nc10()
    z = np.array([0.99, 0.01])
    p, t = 5.0e7, 350.0
    a = flash_tp(eos, p, t, z)
    vapor = bool(a.vapor_frac > 0.5)
    b = flash_tp(eos, p, t, z, skip_stability=True, single_vapor=vapor)
    assert b.two_phase is False
    assert (b.vapor_frac > 0.5) == vapor
