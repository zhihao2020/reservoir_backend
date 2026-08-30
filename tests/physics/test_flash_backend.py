"""Gate 0/2/3/4: FlashBackend parity, cold/warm, binary RR, metadata."""

from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.eos.example import example_c1_nc10
from reservoir_backend.eos.flash import flash_tp, rachford_rice
from reservoir_backend.eos.flash_backend import (
    FastPRBackend,
    ReferencePRBackend,
    TabulatedPRBackend,
    flash_cold_warm_pair,
    validate_backend,
)
from reservoir_backend.eos.flash_corpus import build_flash_corpus, corpus_path


def _ensure_corpus() -> Path:
    path = corpus_path()
    if not path.is_file():
        build_flash_corpus(path)
    return path


def test_flash_result_has_convergence_metadata() -> None:
    eos = example_c1_nc10()
    fl = flash_tp(eos, 8.0e6, 350.0, np.array([0.7, 0.3]))
    assert hasattr(fl, "converged")
    assert hasattr(fl, "iterations")
    assert hasattr(fl, "fugacity_error")
    assert hasattr(fl, "stability_checked")
    assert hasattr(fl, "fallback_used")
    if fl.two_phase:
        assert fl.iterations >= 1
        assert np.isfinite(fl.fugacity_error)
        if not fl.converged:
            assert fl.fugacity_error >= 0.0


def test_binary_rachford_rice_closed_form() -> None:
    z = np.array([0.55, 0.45])
    k = np.array([2.4, 0.35])
    v = rachford_rice(k, z)
    k1 = k - 1.0
    v_bin = -(z[0] * k1[0] + z[1] * k1[1]) / (k1[0] * k1[1])
    assert v == pytest.approx(float(np.clip(v_bin, 0.0, 1.0)), rel=1.0e-12, abs=1.0e-12)


def test_cold_warm_flash_parity() -> None:
    eos = example_c1_nc10()
    rng = np.random.default_rng(3)
    n_fail = 0
    for _ in range(40):
        p = float(rng.uniform(3.0e6, 2.5e7))
        z1 = float(rng.uniform(0.1, 0.9))
        z = np.array([z1, 1.0 - z1])
        cold, warm = flash_cold_warm_pair(eos, p, 350.0, z)
        if bool(cold.two_phase) != bool(warm.two_phase):
            n_fail += 1
            continue
        if abs(cold.vapor_frac - warm.vapor_frac) > 1.0e-8:
            n_fail += 1
            continue
        if float(np.max(np.abs(cold.x - warm.x))) > 1.0e-8:
            n_fail += 1
    assert n_fail == 0


def test_fast_backend_matches_reference_on_corpus() -> None:
    path = _ensure_corpus()
    data = np.load(path)
    eos = example_c1_nc10()
    t = float(data["temperature"][0])
    assert "lam_l" in data.files and "lam_v" in data.files
    assert data["lam_l"].shape == data["pressure"].shape
    assert validate_backend(FastPRBackend(), eos, data["pressure"], t, data["composition"], rtol=5.0e-7)


def test_tabulated_backend_falls_back_near_envelope() -> None:
    eos = example_c1_nc10()
    p = np.array([8.0e6, 2.5e7, 5.0e6])
    z = np.array([[0.6, 0.4], [0.2, 0.8], [0.85, 0.15]])
    ok = validate_backend(TabulatedPRBackend(), eos, p, 350.0, z, rtol=2.0e-3)
    assert ok


def test_reference_backend_is_scalar_truth() -> None:
    eos = example_c1_nc10()
    z = np.array([0.65, 0.35])
    a = flash_tp(eos, 1.0e7, 350.0, z)
    b = ReferencePRBackend().flash_tp(eos, 1.0e7, 350.0, z)
    assert a.vapor_frac == pytest.approx(b.vapor_frac)
    assert a.two_phase == b.two_phase
