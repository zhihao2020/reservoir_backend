from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmarks.saturation_inversion_benchmark import run_saturation_inversion_benchmark
from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.inversion.acoustic import invert_saturation_acoustic
from reservoir_backend.inversion.electromagnetic import invert_saturation_em
from reservoir_backend.inversion.resistivity_archie import (
    archie_sensitivity_report,
    invert_saturation_archie,
)
from reservoir_backend.inversion.saturation_fusion import fuse_saturation_estimates


ROOT = Path(__file__).resolve().parents[1]


def test_archie_scalar_formula():
    sw = invert_saturation_archie(25.0, 0.25, 0.2)
    expected = ((0.25) / ((0.2**2.0) * 25.0)) ** 0.5
    assert sw == pytest.approx(expected)


def test_archie_array_formula():
    sw_true = np.array([0.2, 0.4, 0.6])
    phi = np.full_like(sw_true, 0.25)
    rt = 0.2 / ((phi**2.0) * (sw_true**2.0))
    sw = invert_saturation_archie(rt, 0.2, phi)
    assert np.allclose(sw, sw_true)


def test_archie_rejects_nonpositive_resistivity():
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(0.0, 0.2, 0.25)
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, -0.2, 0.25)


def test_archie_rejects_invalid_porosity():
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, 0.2, 0.0)
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, 0.2, 1.2)


def test_archie_rejects_invalid_parameters():
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, 0.2, 0.25, a=0.0)
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, 0.2, 0.25, m=-1.0)
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_archie(10.0, 0.2, 0.25, n=0.0)


def test_archie_clips_low_high():
    _, report = invert_saturation_archie(np.array([1.0e-9, 1.0e9]), 0.2, 0.25, return_report=True)
    assert report["saturation_max"] <= 1.0
    assert report["saturation_min"] >= 0.0
    assert report["num_clipped_high"] == 1
    assert "num_clipped_low" in report


def test_archie_report_keys():
    _, report = invert_saturation_archie(25.0, 0.25, 0.2, return_report=True)
    expected = {
        "method",
        "success",
        "saturation",
        "raw_saturation_min",
        "raw_saturation_max",
        "saturation_min",
        "saturation_max",
        "num_clipped_low",
        "num_clipped_high",
        "warnings",
        "has_nan",
        "has_inf",
    }
    assert expected <= set(report)


def test_archie_report_no_nan_inf():
    _, report = invert_saturation_archie(np.array([25.0, 50.0]), 0.25, 0.2, return_report=True)
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_em_linear_inversion():
    sw = invert_saturation_em(np.array([1.0, 2.0]), {"model": "linear", "c0": 0.1, "c1": 0.2})
    assert np.allclose(sw, [0.3, 0.5])


def test_em_polynomial_inversion():
    sw = invert_saturation_em(np.array([1.0, 2.0]), {"model": "polynomial", "coefficients": [0.1, 0.2, 0.1]})
    assert np.allclose(sw, [0.4, 0.9])


def test_em_rejects_empty_coefficients():
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_em(np.array([1.0]), [])


def test_em_clipping_report():
    _, report = invert_saturation_em(np.array([0.0, 1.0]), [-0.2, 1.5], return_report=True)
    assert report["num_clipped_low"] == 1
    assert report["num_clipped_high"] == 1


def test_acoustic_linear_inversion():
    sw = invert_saturation_acoustic(np.array([2000.0, 2500.0]), {"model": "linear", "c0": -0.5, "c1": 0.0004})
    assert np.allclose(sw, [0.3, 0.5])


def test_acoustic_polynomial_inversion():
    sw = invert_saturation_acoustic(np.array([1.0, 2.0]), {"model": "polynomial", "coefficients": [0.1, 0.2, 0.1]})
    assert np.allclose(sw, [0.4, 0.9])


def test_acoustic_rejects_empty_coefficients():
    with pytest.raises(InvalidPhysicalValueError):
        invert_saturation_acoustic(np.array([1.0]), [])


def test_acoustic_clipping_report():
    _, report = invert_saturation_acoustic(np.array([0.0, 1.0]), [1.2, -1.5], return_report=True)
    assert report["num_clipped_low"] == 1
    assert report["num_clipped_high"] == 1


def test_fusion_equal_weights():
    fused = fuse_saturation_estimates({"archie": 0.2, "em": 0.6})
    assert fused == pytest.approx(0.4)


def test_fusion_user_weights():
    fused = fuse_saturation_estimates({"archie": 0.2, "em": 0.6}, weights={"archie": 3.0, "em": 1.0})
    assert fused == pytest.approx(0.3)


def test_fusion_confidence_weights():
    fused = fuse_saturation_estimates(
        {"archie": 0.2, "em": 0.6},
        confidence={"archie": 0.75, "em": 0.25},
    )
    assert fused == pytest.approx(0.3)


def test_fusion_uncertainty_inverse_variance_weights():
    _, report = fuse_saturation_estimates(
        {"archie": 0.2, "em": 0.6},
        uncertainties={"archie": 0.1, "em": 0.2},
        return_report=True,
    )
    assert report["normalized_weights"]["archie"] == pytest.approx(0.8)
    assert report["normalized_weights"]["em"] == pytest.approx(0.2)


def test_fusion_priority_uncertainty_over_confidence():
    fused, report = fuse_saturation_estimates(
        {"archie": 0.2, "em": 0.6},
        uncertainties={"archie": 0.1, "em": 0.2},
        confidence={"archie": 0.0, "em": 1.0},
        return_report=True,
    )
    assert report["fusion_mode"] == "uncertainty_inverse_variance"
    assert fused == pytest.approx(0.28)


def test_fusion_rejects_negative_weights():
    with pytest.raises(InvalidPhysicalValueError):
        fuse_saturation_estimates({"archie": 0.2}, weights={"archie": -1.0})


def test_fusion_rejects_all_invalid_estimates():
    with pytest.raises(InvalidPhysicalValueError):
        fuse_saturation_estimates({"archie": np.nan})


def test_fusion_report_keys():
    _, report = fuse_saturation_estimates({"archie": 0.2, "em": 0.6}, return_report=True)
    expected = {
        "method",
        "success",
        "fusion_mode",
        "used_signals",
        "dropped_signals",
        "normalized_weights",
        "saturation",
        "saturation_min",
        "saturation_max",
        "num_clipped_low",
        "num_clipped_high",
        "warnings",
        "has_nan",
        "has_inf",
    }
    assert expected <= set(report)


def test_fusion_weights_sum_to_one():
    _, report = fuse_saturation_estimates({"archie": 0.2, "em": 0.6}, return_report=True)
    assert sum(report["normalized_weights"].values()) == pytest.approx(1.0)


def test_fusion_clips_output():
    fused, report = fuse_saturation_estimates({"archie": -0.5, "em": 1.5}, return_report=True)
    assert fused == pytest.approx(0.5)
    assert report["saturation_min"] >= 0.0
    assert report["saturation_max"] <= 1.0


def test_archie_sensitivity_report_keys():
    report = archie_sensitivity_report(25.0, 0.25, 0.2)
    assert {"method", "base_saturation", "sensitivity", "relative_sensitivity", "warnings", "has_nan", "has_inf"} <= set(report)


def test_archie_sensitivity_perturbation_positive():
    with pytest.raises(InvalidPhysicalValueError):
        archie_sensitivity_report(25.0, 0.25, 0.2, perturbation=0.0)


def test_archie_sensitivity_no_nan_inf():
    report = archie_sensitivity_report(25.0, 0.25, 0.2)
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_saturation_inversion_benchmark_runs(tmp_path):
    report = run_saturation_inversion_benchmark(tmp_path)
    assert report["benchmark_name"] == "saturation_inversion_benchmark"
    assert (tmp_path / "saturation_inversion_benchmark_summary.json").exists()
    assert (tmp_path / "saturation_inversion_benchmark_summary.md").exists()


def test_saturation_inversion_benchmark_success(tmp_path):
    assert run_saturation_inversion_benchmark(tmp_path)["success"] is True


def test_saturation_inversion_benchmark_archie_error_small(tmp_path):
    assert run_saturation_inversion_benchmark(tmp_path)["archie_formula_error"] < 1.0e-12


def test_saturation_inversion_benchmark_noise_cases(tmp_path):
    noise = run_saturation_inversion_benchmark(tmp_path)["noise_sensitivity"]
    assert noise["levels"] == [0.01, 0.05, 0.10]
    assert noise["monotonic_error_growth"] is True


def test_saturation_inversion_benchmark_fusion_checked(tmp_path):
    report = run_saturation_inversion_benchmark(tmp_path)
    assert report["fusion_error"] < 0.02
    assert report["clipping_checked"] is True


def test_saturation_inversion_benchmark_reports_no_nan_inf(tmp_path):
    report = run_saturation_inversion_benchmark(tmp_path)
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_saturation_inversion_docs_updated():
    text = (ROOT / "docs" / "saturation_inversion_validation.md").read_text(encoding="utf-8")
    assert "saturation inversion hardening" in text
    assert "No Bayesian inversion implemented." in text


def test_requirement_traceability_mentions_saturation_inversion_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "saturation inversion hardening" in text
    assert "Done" in text


def test_readme_mentions_saturation_inversion_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "saturation inversion benchmark" in text.lower()


def test_no_solver_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_existing_function_benchmark_matrix_tests_still_pass():
    text = (ROOT / "specs" / "14_function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "Function hardening first." in text
    assert "Saturation inversion module" in text
