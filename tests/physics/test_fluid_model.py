import numpy as np

from reservoir_backend.physics.fluid_model import SimpleFluidModel
from reservoir_backend.physics.pvt import BlackOilPVT


def test_simple_fluid_evaluate_returns_density_and_viscosity() -> None:
    model = SimpleFluidModel(pvt=BlackOilPVT())
    props = model.evaluate(1.5e5, 293.15)
    assert props.density.shape[-1] == 2
    assert props.viscosity.shape[-1] == 2
    assert np.all(props.density > 0.0)
    assert np.all(props.viscosity > 0.0)
