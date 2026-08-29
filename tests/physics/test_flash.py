"""PT flash conservation and stability on the EXAMPLE binary."""

import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp, rachford_rice
from reservoir_backend.eos.stability import wilson_k


def test_rachford_rice_material_balance() -> None:
    z = np.array([0.6, 0.4])
    k = np.array([3.0, 0.2])
    v = rachford_rice(k, z)
    x = z / (1.0 + v * (k - 1.0))
    y = k * x
    x = x / np.sum(x)
    y = y / np.sum(y)
    rec = v * y + (1.0 - v) * x
    np.testing.assert_allclose(rec, z / z.sum(), atol=1.0e-10)
    rr = np.sum(z * (k - 1.0) / (1.0 + v * (k - 1.0)))
    assert abs(rr) < 1.0e-8


def test_flash_material_balance_two_or_one_phase() -> None:
    eos = example_c1_nc10()
    z = np.array([0.70, 0.30])
    fl = flash_tp(eos, 8.0e6, 350.0, z)
    rec = fl.vapor_frac * fl.y + (1.0 - fl.vapor_frac) * fl.x
    np.testing.assert_allclose(rec, z / z.sum(), atol=1.0e-8)
    assert fl.v_mix > 0.0
    assert 0.0 <= fl.sv <= 1.0
    assert abs(fl.sl + fl.sv - 1.0) < 1.0e-12


def test_pure_methane_is_single_phase_vapor() -> None:
    eos = example_c1_nc10()
    fl = flash_tp(eos, 1.0e6, 300.0, np.array([1.0, 0.0]))
    assert fl.two_phase is False
    assert fl.vapor_frac > 0.5


def test_wilson_k_methane_lighter_than_decane() -> None:
    eos = example_c1_nc10()
    k = wilson_k(eos, 5.0e6, 350.0)
    assert k[0] > k[1]
