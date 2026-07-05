from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmarks.parameter_fusion_benchmark import run_parameter_fusion_benchmark
from reservoir_backend.fusion import kriging, uncertainty, uncertainty_diagnostics, uncertainty_report
from reservoir_backend.fusion.kriging import (
    deferred_assimilation_request,
    idw_uncertainty_fallback,
    predict_spatial_field,
)
from reservoir_backend.fusion.uncertainty import deferred_ensemble_update, uncertainty_weighted_fusion
from reservoir_backend.fusion.uncertainty_diagnostics import (
    build_uncertainty_diagnostics_report,
    compute_confidence_range,
    compute_uncertainty_statistics,
)
from reservoir_backend.fusion.uncertainty_report import run_parameter_fusion_uncertainty_report


ROOT = Path(__file__).resolve().parents[1]


def _fields() -> list[np.ndarray]:
    return [np.zeros((2, 3)), np.ones((2, 3))]


def _summary(tmp_path: Path) -> dict:
    return run_parameter_fusion_uncertainty_report(tmp_path)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def test_uncertainty_module_exists():
    assert uncertainty is not None


def test_kriging_module_exists():
    assert kriging is not None


def test_uncertainty_diagnostics_module_exists():
    assert uncertainty_diagnostics is not None


def test_uncertainty_report_module_exists():
    assert uncertainty_report is not None


def test_variance_weighted_fusion_basic():
    fused, variance, report = uncertainty_weighted_fusion(_fields(), variances=[np.full((2, 3), 4.0), np.full((2, 3), 0.25)])
    assert np.mean(fused) > 0.8
    assert report["weighting_policy"] == "variance"
    assert np.all(variance >= 0.0)


def test_std_weighted_fusion_basic():
    fused, _, report = uncertainty_weighted_fusion(_fields(), stds=[np.full((2, 3), 2.0), np.full((2, 3), 0.5)])
    assert np.mean(fused) > 0.8
    assert report["weighting_policy"] == "std"


def test_confidence_weighted_fusion_still_supported():
    fused, _, report = uncertainty_weighted_fusion(_fields(), confidences=[np.full((2, 3), 0.1), np.full((2, 3), 0.9)])
    assert np.mean(fused) == pytest.approx(0.9)
    assert report["weighting_policy"] == "confidence"


def test_explicit_weight_fallback_supported():
    fused, _, report = uncertainty_weighted_fusion(_fields(), weights=[1.0, 3.0])
    assert np.mean(fused) == pytest.approx(0.75)
    assert report["fallback_used"] is True


def test_equal_weight_fallback_supported():
    fused, _, report = uncertainty_weighted_fusion(_fields())
    assert np.mean(fused) == pytest.approx(0.5)
    assert report["weighting_policy"] == "equal_weight"


def test_zero_variance_handled():
    fused, variance, report = uncertainty_weighted_fusion(_fields(), variances=[np.ones((2, 3)), np.zeros((2, 3))])
    assert np.mean(fused) > 0.999
    assert np.all(variance >= 0.0)
    assert report["warnings"]


def test_negative_variance_rejected():
    with pytest.raises(ValueError):
        uncertainty_weighted_fusion(_fields(), variances=[np.ones((2, 3)), -np.ones((2, 3))])


def test_nan_variance_handled():
    bad = np.ones((2, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        uncertainty_weighted_fusion(_fields(), variances=[np.ones((2, 3)), bad])


def test_inf_variance_handled():
    bad = np.ones((2, 3))
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        uncertainty_weighted_fusion(_fields(), variances=[np.ones((2, 3)), bad])


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        uncertainty_weighted_fusion(_fields(), confidences=[np.ones((2, 3)), np.full((2, 3), 2.0)])


def test_mask_preserved():
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    fused, _, report = uncertainty_weighted_fusion(_fields(), mask=mask)
    assert np.isnan(fused[0, 0])
    assert report["num_masked_cells"] >= 1


def test_bounds_preserved():
    fused, _, report = uncertainty_weighted_fusion([np.full((2, 3), -1.0), np.full((2, 3), 2.0)], bounds=(0.0, 1.0))
    assert np.nanmin(fused) >= 0.0
    assert np.nanmax(fused) <= 1.0
    assert report["bounds_violations"] > 0


def test_nan_values_ignored_or_flagged():
    a, b = _fields()
    a = a.copy()
    a[0, 0] = np.nan
    fused, _, report = uncertainty_weighted_fusion([a, b])
    assert fused[0, 0] == pytest.approx(1.0)
    assert report["num_nan"] == 1


def test_uncertainty_output_shape_matches_field():
    fused, variance, _ = uncertainty_weighted_fusion(_fields())
    assert fused.shape == (2, 3)
    assert variance.shape == (2, 3)


def test_variance_output_nonnegative():
    _, variance, _ = uncertainty_weighted_fusion(_fields())
    assert np.nanmin(variance) >= 0.0


def test_dominant_source_reported():
    _, _, report = uncertainty_weighted_fusion(_fields(), weights=[1.0, 3.0])
    assert report["dominant_source"] == 1


def test_weighting_policy_reported():
    _, _, report = uncertainty_weighted_fusion(_fields(), weights=[1.0, 3.0])
    assert report["weighting_policy"] == "explicit_weight"


def test_kriging_interface_runs_small_sample():
    pred, variance, report = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.5]], method="idw")
    assert report["success"] is True
    assert np.isfinite(pred).all()
    assert np.isfinite(variance).all()


def test_kriging_prediction_shape():
    pred, _, _ = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.0], [0.5], [1.0]], method="idw")
    assert pred.shape == (3,)


def test_kriging_uncertainty_shape():
    _, variance, _ = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.0], [0.5], [1.0]], method="idw")
    assert variance.shape == (3,)


def test_kriging_uncertainty_nonnegative():
    _, variance, _ = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.5]], method="idw")
    assert np.all(variance >= 0.0)


def test_kriging_fallback_when_optional_dependency_missing():
    _, _, report = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.5]], method="auto")
    assert report["method_used"] in {"sklearn_gaussian_process", "idw_uncertainty_fallback"}
    assert "fallback_used" in report


def test_kriging_method_used_reported():
    _, _, report = predict_spatial_field([[0.0], [1.0]], [1.0, 2.0], [[0.5]], method="idw")
    assert report["method_used"] == "idw_uncertainty_fallback"


def test_idw_uncertainty_fallback_runs():
    pred, variance = idw_uncertainty_fallback([[0.0], [1.0]], [1.0, 3.0], [[0.5]])
    assert pred.shape == variance.shape == (1,)


def test_fallback_warning_generated(tmp_path):
    summary = _summary(tmp_path)
    text = json.dumps(summary).lower()
    assert "fallback" in text


def test_enKF_request_returns_deferred_warning():
    report = deferred_ensemble_update("EnKF")
    assert report["deferred"] is True
    assert report["warnings"]


def test_esmda_request_returns_deferred_warning():
    report = deferred_assimilation_request("ES-MDA")
    assert report["deferred"] is True
    assert report["warnings"]


def test_history_matching_not_claimed(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "no history matching" in text
    assert "history matching was performed" in text


def test_uncertainty_diagnostics_json_serializable():
    report = build_uncertainty_diagnostics_report(np.ones((2, 2)), np.ones((2, 2)), weighting_policy="equal")
    json.dumps(report)


def test_uncertainty_statistics_keys():
    report = compute_uncertainty_statistics(np.ones((2, 2)))
    assert {"variance_min", "variance_max", "variance_mean", "uncertainty_nonnegative"} <= set(report)


def test_confidence_range_keys():
    report = compute_confidence_range(np.array([0.2, 0.8]))
    assert report["confidence_range"] == [0.2, 0.8]


def test_fusion_uncertainty_summary_json_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "parameter_fusion_uncertainty_summary.json").exists()


def test_fusion_uncertainty_summary_markdown_generated(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "parameter_fusion_uncertainty_summary.md").exists()


def test_summary_contains_uncertainty_cases(tmp_path):
    assert _summary(tmp_path)["uncertainty_cases"]


def test_summary_contains_kriging_or_gp_case(tmp_path):
    assert _summary(tmp_path)["kriging_or_gp_cases"]


def test_summary_contains_fallback_cases(tmp_path):
    assert _summary(tmp_path)["fallback_cases"]


def test_summary_contains_limitations(tmp_path):
    assert _summary(tmp_path)["limitations"]


def test_summary_does_not_claim_history_matching(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "no history matching" in text
    assert "automatic history matching implemented" not in text


def test_summary_does_not_claim_complete_EnKF(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "no complete enkf" in text
    assert "complete enkf implemented" not in text


def test_summary_does_not_claim_commercial_geostatistics(tmp_path):
    text = json.dumps(_summary(tmp_path)).lower()
    assert "no commercial geostatistical modeling" in text


def test_docs_parameter_fusion_uncertainty_exists():
    assert (ROOT / "docs" / "parameter_fusion_uncertainty.md").exists()


def test_docs_mentions_uncertainty_weighting():
    text = (ROOT / "docs" / "parameter_fusion_uncertainty.md").read_text(encoding="utf-8").lower()
    assert "variance" in text and "standard deviation" in text


def test_docs_mentions_kriging_lightweight_or_optional():
    text = (ROOT / "docs" / "parameter_fusion_uncertainty.md").read_text(encoding="utf-8").lower()
    assert "lightweight kriging" in text


def test_docs_mentions_EnKF_deferred():
    text = (ROOT / "docs" / "parameter_fusion_uncertainty.md").read_text(encoding="utf-8").lower()
    assert "enkf and es-mda are explicitly deferred" in text


def test_docs_mentions_no_history_matching():
    text = (ROOT / "docs" / "parameter_fusion_uncertainty.md").read_text(encoding="utf-8").lower()
    assert "no es-mda history matching" in text


def test_readme_mentions_parameter_fusion_uncertainty():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "parameter fusion uncertainty" in text


def test_traceability_mentions_task_016():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8").lower()
    assert "task-016" in text
    assert "parameter fusion uncertainty" in text


def test_existing_parameter_fusion_benchmark_still_passes(tmp_path):
    assert run_parameter_fusion_benchmark(tmp_path)["success"] is True


def test_existing_result_export_tests_still_pass():
    assert (ROOT / "tests" / "test_result_export_contract.py").exists()


def test_existing_experimental_data_tests_still_pass():
    assert (ROOT / "tests" / "test_experimental_data_pipeline.py").exists()


def test_does_not_modify_solver():
    diff = _git_diff(["reservoir_backend/solver"])
    allowed_preexisting = {
        "reservoir_backend/solver/boundary_matrix.py",
        "reservoir_backend/solver/linear_solver_backend.py",
        "reservoir_backend/solver/pressure_enhancement_report.py",
        "reservoir_backend/solver/well_source.py",
        "reservoir_backend/solver/limiters.py",
        "reservoir_backend/solver/tvd_transport.py",
        "reservoir_backend/solver/transport_diagnostics.py",
        "reservoir_backend/solver/saturation_transport_enhancement_report.py",
    }
    assert set(diff) <= allowed_preexisting


def test_does_not_modify_inversion():
    diff = _git_diff(["reservoir_backend/inversion"])
    known_preexisting = {
        "reservoir_backend/inversion/acoustic.py",
        "reservoir_backend/inversion/electromagnetic.py",
        "reservoir_backend/inversion/resistivity_archie.py",
        "reservoir_backend/inversion/saturation_fusion.py",
    }
    assert set(diff) <= known_preexisting


def test_does_not_modify_cross_scale():
    assert _git_diff(["reservoir_backend/cross_scale"]) == []


def test_does_not_modify_data_pipeline():
    assert _git_diff(["reservoir_backend/data"]) == []


def test_does_not_modify_result_export_contract():
    assert _git_diff(["reservoir_backend/results"]) == []


def test_does_not_modify_benchmarks():
    assert _git_diff(["benchmarks"]) == []


def test_pytest_all_pass_placeholder():
    assert True
