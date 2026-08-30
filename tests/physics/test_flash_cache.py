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


def test_k_guess_allowed_on_finite_difference_cells() -> None:
    from reservoir_backend.comp.fluid import fluid_from_name
    from reservoir_backend.comp.properties import flash_state, moles_from_z

    spec = fluid_from_name("example", temperature_k=350.0)
    p = np.array([1.2e7, 1.2e7])
    n = moles_from_z(spec, p, spec.z_init, np.full(2, 1.0e-6))
    props = flash_state(spec, p, n)
    assert props.k_flash is not None
    p2 = p.copy()
    p2[0] *= 1.0 + 1.0e-8
    out = flash_state(spec, p2, n, cells=np.array([0]), out=props.copy())
    assert out.k_flash is not None
    assert np.isfinite(out.v_mix[0])


def test_stability_bypass_single_phase() -> None:
    eos = example_c1_nc10()
    z = np.array([0.99, 0.01])
    p, t = 5.0e7, 350.0
    a = flash_tp(eos, p, t, z)
    vapor = bool(a.vapor_frac > 0.5)
    b = flash_tp(eos, p, t, z, skip_stability=True, single_vapor=vapor)
    assert b.two_phase is False
    assert (b.vapor_frac > 0.5) == vapor
