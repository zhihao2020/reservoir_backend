import numpy as np
import pytest

from reservoir_backend.exceptions import InvalidPermeability
from reservoir_backend.physics.conductivity import FractureConductivityModel


def test_scalar_cf_paints_fracture_only() -> None:
    mask = np.array([False, True, True, False])
    model = FractureConductivityModel(n_cells=4, fracture_mask=mask, k_matrix_m2=1.0e-15)
    k = model.permeability(4.0e-13)
    assert k[0] == pytest.approx(1.0e-15)
    assert k[3] == pytest.approx(1.0e-15)
    assert k[1] == pytest.approx(4.0e-13)
    assert k[2] == pytest.approx(4.0e-13)


def test_rejects_nonpositive_cf() -> None:
    model = FractureConductivityModel(
        n_cells=2, fracture_mask=np.array([True, False]), k_matrix_m2=1.0e-15
    )
    with pytest.raises(InvalidPermeability):
        model.permeability(-1.0)
