import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.lumping import lump_peng_robinson


def test_binary_identity_lumping() -> None:
    eos = example_c1_nc10()
    z = np.array([0.6, 0.4])
    lumped, z2 = lump_peng_robinson(eos, z, [[0], [1]])
    assert lumped.nc == 2
    np.testing.assert_allclose(z2, z)
    np.testing.assert_allclose(lumped.tc, eos.tc)
