import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.pvt_ingest import lump_experimental_eos


def test_lump_report_keeps_names() -> None:
    eos = example_c1_nc10()
    lumped, z2, report = lump_experimental_eos(eos, np.array([0.6, 0.4]), [[0], [1]])
    assert lumped.nc == 2
    assert report.n_raw == 2
    np.testing.assert_allclose(z2, [0.6, 0.4])
