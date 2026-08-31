import numpy as np

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp
from reservoir_backend.eos.flash_backend import FastPRBackend, ReferencePRBackend
from reservoir_backend.eos.flash_corpus import build_realfluid_corpus, realfluid_corpus_path
from reservoir_backend.eos.pvt_ingest import lump_experimental_eos
from reservoir_backend.io.eos_load import load_eos_card
from reservoir_backend.eos.pr import PengRobinson


def test_lab_v1_pvt_card_loads() -> None:
    eos = load_eos_card("examples/lab_v1/pvt.yaml")
    assert eos.nc == 2
    assert eos.names[0] == "C1"


def test_lumping_six_to_four_stays_in_band() -> None:
    eos = PengRobinson(
        tc=np.array([190.564, 305.32, 369.83, 507.6, 568.7, 617.70]),
        pc=np.array([4.5992e6, 4.872e6, 4.248e6, 3.025e6, 2.49e6, 2.103e6]),
        omega=np.array([0.01142, 0.0995, 0.1523, 0.3013, 0.399, 0.490]),
        mw=np.array([16.0425e-3, 30.07e-3, 44.096e-3, 86.175e-3, 114.23e-3, 142.282e-3]),
        kij=np.zeros((6, 6)),
        names=("C1", "C2", "C3", "nC6", "nC8", "nC10"),
    )
    z = np.array([0.40, 0.10, 0.10, 0.15, 0.10, 0.15])
    lumped, z2, report = lump_experimental_eos(eos, z, [[0], [1, 2], [3], [4, 5]])
    assert report.n_raw == 6
    assert 3 <= lumped.nc <= 6
    assert lumped.nc == 4
    assert abs(float(np.sum(z2)) - 1.0) < 1.0e-12


def test_fastpr_matches_reference_on_realfluid_corpus(tmp_path) -> None:
    dest = tmp_path / "flash_states_realfluid.npz"
    path = build_realfluid_corpus(dest, temperature=350.0)
    data = np.load(path)
    p = data["pressure"]
    z = data["composition"]
    eos = load_eos_card("examples/lab_v1/pvt.yaml")
    fast = FastPRBackend().evaluate_batch(eos, p, 350.0, z)
    np.testing.assert_allclose(fast.vapor_frac, data["ref_vapor_frac"], atol=2.0e-4, rtol=2.0e-4)
    _ = example_c1_nc10
    _ = flash_tp
    _ = ReferencePRBackend
    _ = realfluid_corpus_path
