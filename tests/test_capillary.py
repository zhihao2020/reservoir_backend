import numpy as np

from reservoir_backend.physics.capillary import BrooksCorey, NoCapillary, VanGenuchten


def test_brooks_corey_monotone_decreasing() -> None:
    pc = BrooksCorey(entry_pressure=2.0e3, lambda_pc=2.0, swi=0.2, sor=0.2)
    sw = np.linspace(0.21, 0.79, 20)
    values = pc.pc(sw)
    assert np.all(np.diff(values) < 0.0)
    assert values[0] > values[-1]


def test_no_capillary_zero() -> None:
    assert np.all(NoCapillary().pc(np.array([0.2, 0.5, 0.8])) == 0.0)


def test_van_genuchten_positive() -> None:
    vg = VanGenuchten()
    sw = np.linspace(0.25, 0.7, 8)
    assert np.all(vg.pc(sw) >= 0.0)
