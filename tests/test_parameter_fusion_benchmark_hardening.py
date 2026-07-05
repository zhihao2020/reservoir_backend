from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.parameter_fusion_benchmark import run_parameter_fusion_benchmark
from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.fusion import fusion_diagnostics
from reservoir_backend.fusion.fusion_diagnostics import (
    build_fusion_diagnostics_report,
    check_bounds,
    check_field_finite,
    check_shape_consistency,
    compute_confidence_weighting_metrics,
    compute_field_statistics,
    compute_fusion_error,
    compute_nan_mask_report,
    compute_weight_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def _summary(tmp_path: Path) -> dict:
    return run_parameter_fusion_benchmark(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _grid() -> Grid3D:
    return Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)


def test_fusion_diagnostics_module_exists():
    assert fusion_diagnostics is not None


def test_field_statistics_keys():
    report = compute_field_statistics(np.array([1.0, 2.0, 3.0]))
    assert {"field_min", "field_max", "field_mean", "field_std", "has_nan", "has_inf", "num_nan", "num_inf"} <= set(report)


def test_field_statistics_does_not_modify_input():
    values = np.array([1.0, 2.0, 3.0])
    before = values.copy()
    compute_field_statistics(values)
    np.testing.assert_allclose(values, before)


def test_field_finite_check_passes_for_valid_field():
    assert check_field_finite(np.ones((2, 2)))["success"] is True


def test_field_finite_check_reports_nan_inf():
    report = check_field_finite(np.array([1.0, np.nan, np.inf]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True
    assert report["success"] is False


def test_shape_consistency_passes():
    report = check_shape_consistency([np.zeros((2, 3)), np.ones((2, 3))], target_shape=(2, 3))
    assert report["shape_consistent"] is True


def test_shape_consistency_detects_mismatch():
    report = check_shape_consistency([np.zeros((2, 3)), np.ones((3, 2))])
    assert report["shape_consistent"] is False
    assert report["warnings"]


def test_bounds_pass_for_valid_porosity():
    report = check_bounds(np.array([0.1, 0.4, 0.9]), lower=0.0, upper=1.0)
    assert report["success"] is True


def test_bounds_detects_violations():
    report = check_bounds(np.array([-0.1, 0.5, 1.2]), lower=0.0, upper=1.0)
    assert report["num_bound_violations"] == 2


def test_weight_statistics_keys():
    report = compute_weight_statistics(np.array([1.0, 2.0, 3.0]))
    assert {"weight_min", "weight_max", "weight_sum_min", "weight_sum_max", "num_zero_weight_cells"} <= set(report)


def test_weight_statistics_detects_negative():
    report = compute_weight_statistics(np.array([1.0, -1.0]))
    assert report["success"] is False
    assert report["num_negative_weights"] == 1


def test_weight_statistics_detects_zero_sum():
    report = compute_weight_statistics(np.zeros((2, 3)))
    assert report["success"] is False
    assert report["num_zero_weight_cells"] == 3


def test_nan_mask_report_keys():
    report = compute_nan_mask_report([np.array([1.0, np.nan]), np.array([2.0, 3.0])])
    assert {"num_source_nan_values", "num_masked_cells", "num_partially_masked_cells"} <= set(report)


def test_nan_mask_report_counts_all_source_nan():
    report = compute_nan_mask_report([np.array([np.nan, 1.0]), np.array([np.nan, 2.0])])
    assert report["num_masked_cells"] == 1


def test_fusion_error_zero_for_exact_match():
    report = compute_fusion_error(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert report["mae"] == 0.0
    assert report["rmse"] == 0.0
    assert report["max_abs_error"] == 0.0


def test_fusion_error_positive_for_mismatch():
    report = compute_fusion_error(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert report["mae"] > 0.0
    assert report["max_abs_error"] > 0.0


def test_confidence_weighting_metrics_prefers_high_confidence_source():
    low = np.zeros((2, 2))
    high = np.ones((2, 2))
    fused = np.full((2, 2), 0.9)
    report = compute_confidence_weighting_metrics(low, high, fused)
    assert report["closer_to_high_confidence"] is True


def test_build_fusion_diagnostics_report_keys():
    report = build_fusion_diagnostics_report(np.ones((2, 2)), reference=np.ones((2, 2)), lower=0.0, upper=1.0)
    expected = {
        "success",
        "field_min",
        "field_max",
        "field_mean",
        "field_std",
        "has_nan",
        "has_inf",
        "num_nan",
        "num_inf",
        "num_below_lower",
        "num_above_upper",
        "shape_consistent",
        "weight_min",
        "weight_max",
        "weight_sum_min",
        "weight_sum_max",
        "num_zero_weight_cells",
        "num_masked_cells",
        "mae",
        "rmse",
        "max_abs_error",
        "warnings",
    }
    assert expected <= set(report)


def test_diagnostics_support_field3d_input():
    grid = _grid()
    field = Field3D.from_constant(grid, 0.25)
    report = compute_field_statistics(field)
    assert report["shape"] == list(grid.shape)


def test_parameter_fusion_benchmark_module_exists():
    import benchmarks.parameter_fusion_benchmark as benchmark

    assert hasattr(benchmark, "run_parameter_fusion_benchmark")


def test_parameter_fusion_benchmark_runs(tmp_path):
    assert _summary(tmp_path)["benchmark_name"] == "parameter_fusion_benchmark"


def test_parameter_fusion_summary_keys(tmp_path):
    summary = _summary(tmp_path)
    expected = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "overall_mae",
        "overall_rmse",
        "overall_max_abs_error",
        "overall_num_bound_violations",
        "overall_num_masked_cells",
        "has_nan",
        "has_inf",
        "warnings",
        "recommendations",
    }
    assert expected <= set(summary)


def test_parameter_fusion_summary_success_true(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_parameter_fusion_summary_reports_no_unexpected_nan_inf(tmp_path):
    summary = _summary(tmp_path)
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_equal_weight_field_fusion_case_runs(tmp_path):
    case = _case(_summary(tmp_path), "equal_weight_field_fusion")
    assert case["success"] is True


def test_equal_weight_field_fusion_error_small(tmp_path):
    metrics = _case(_summary(tmp_path), "equal_weight_field_fusion")["key_metrics"]
    assert metrics["mae"] <= 1.0e-14
    assert metrics["rmse"] <= 1.0e-14


def test_explicit_weight_field_fusion_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "explicit_weight_field_fusion")["success"] is True


def test_explicit_weight_field_fusion_matches_weighted_mean(tmp_path):
    metrics = _case(_summary(tmp_path), "explicit_weight_field_fusion")["key_metrics"]
    assert metrics["max_abs_error"] <= 1.0e-14
    assert metrics["used_weights"] == [1.0, 3.0]


def test_explicit_weight_field_fusion_rejects_invalid_weight(tmp_path):
    metrics = _case(_summary(tmp_path), "explicit_weight_field_fusion")["key_metrics"]
    assert metrics["invalid_weight_rejected"] is True


def test_confidence_weighted_fusion_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "confidence_weighted_fusion")["success"] is True


def test_confidence_weighted_fusion_high_confidence_dominates(tmp_path):
    metrics = _case(_summary(tmp_path), "confidence_weighted_fusion")["key_metrics"]
    assert metrics["closer_to_high_confidence"] is True
    assert metrics["fused_mean"] > 0.5


def test_confidence_weighted_zero_confidence_source_does_not_dominate(tmp_path):
    metrics = _case(_summary(tmp_path), "confidence_weighted_fusion")["key_metrics"]
    assert metrics["zero_confidence_source_does_not_dominate"] is True


def test_uncertainty_variance_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "uncertainty_or_variance_weighted_fusion_if_supported")["success"] is True


def test_uncertainty_variance_case_documents_unsupported_behavior(tmp_path):
    case = _case(_summary(tmp_path), "uncertainty_or_variance_weighted_fusion_if_supported")
    assert case["key_metrics"]["uncertainty_fusion_supported"] is False
    assert "uncertainty fusion not implemented" in " ".join(case["limitations"])


def test_nan_aware_fusion_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "nan_aware_fusion")["success"] is True


def test_nan_aware_fusion_ignores_single_source_nan(tmp_path):
    metrics = _case(_summary(tmp_path), "nan_aware_fusion")["key_metrics"]
    assert metrics["single_source_nan_ignored"] is True


def test_nan_aware_fusion_reports_all_source_nan_mask(tmp_path):
    metrics = _case(_summary(tmp_path), "nan_aware_fusion")["key_metrics"]
    assert metrics["all_source_nan_masked"] is True
    assert metrics["num_masked_cells"] == 2


def test_bounds_and_clipping_report_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "bounds_and_clipping_report")["success"] is True


def test_bounds_and_clipping_report_counts_clipped_cells(tmp_path):
    metrics = _case(_summary(tmp_path), "bounds_and_clipping_report")["key_metrics"]
    assert metrics["clipped_cells"] > 0
    assert metrics["saturation_num_bound_violations"] == 0


def test_bounds_and_clipping_report_checks_physical_ranges(tmp_path):
    metrics = _case(_summary(tmp_path), "bounds_and_clipping_report")["key_metrics"]
    assert metrics["porosity_num_bound_violations"] == 0
    assert metrics["permeability_positive"] is True


def test_shape_mismatch_rejection_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "shape_mismatch_rejection")["success"] is True


def test_shape_mismatch_rejection_reports_clear_error(tmp_path):
    metrics = _case(_summary(tmp_path), "shape_mismatch_rejection")["key_metrics"]
    assert metrics["shape_mismatch_rejected"] is True
    assert metrics["shape_consistent"] is False


def test_multi_field_property_dynamic_fusion_sanity_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "multi_field_property_dynamic_fusion_sanity")["success"] is True


def test_multi_field_property_dynamic_fusion_shapes_and_bounds(tmp_path):
    metrics = _case(_summary(tmp_path), "multi_field_property_dynamic_fusion_sanity")["key_metrics"]
    assert metrics["all_outputs_finite"] is True
    assert metrics["shape_consistent"] is True
    assert metrics["num_bound_violations"] == 0


def test_multi_field_property_dynamic_fusion_provenance_present(tmp_path):
    metrics = _case(_summary(tmp_path), "multi_field_property_dynamic_fusion_sanity")["key_metrics"]
    assert metrics["source_count"] == len(metrics["source_names"])
    assert "sw_obs" in metrics["source_names"]


def test_parameter_fusion_benchmark_generates_json_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "parameter_fusion_benchmark_summary.json").exists()


def test_parameter_fusion_benchmark_generates_markdown_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "parameter_fusion_benchmark_summary.md").exists()


def test_parameter_fusion_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_parameter_fusion_docs_updated():
    text = (ROOT / "docs" / "parameter_fusion_validation.md").read_text(encoding="utf-8")
    assert "Parameter fusion benchmark hardening" in text
    assert "confidence-weighted fusion benchmark" in text


def test_parameter_fusion_docs_no_history_matching_claim():
    text = (ROOT / "docs" / "parameter_fusion_validation.md").read_text(encoding="utf-8")
    assert "No history matching implemented." in text
    assert "history matching benchmark" not in text.lower()


def test_parameter_fusion_docs_no_automatic_calibration_claim():
    text = (ROOT / "docs" / "parameter_fusion_validation.md").read_text(encoding="utf-8")
    assert "No automatic calibration implemented." in text


def test_parameter_fusion_docs_no_bayesian_inversion_claim():
    text = (ROOT / "docs" / "parameter_fusion_validation.md").read_text(encoding="utf-8")
    assert "No Bayesian inversion implemented." in text
    assert "No EnKF / ES-MDA implemented." in text


def test_parameter_fusion_docs_no_black_oil_claim():
    text = (ROOT / "docs" / "parameter_fusion_validation.md").read_text(encoding="utf-8")
    assert "No black-oil model implemented." in text


def test_readme_mentions_parameter_fusion_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "parameter fusion benchmark" in text.lower()


def test_numerical_accuracy_mentions_parameter_fusion_benchmark():
    text = (ROOT / "docs" / "numerical_accuracy.md").read_text(encoding="utf-8")
    assert "parameter_fusion_benchmark" in text


def test_function_benchmark_matrix_mentions_task_051():
    text = (ROOT / "docs" / "function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "051_parameter_fusion_benchmark_hardening" in text or "Parameter Fusion Benchmark Hardening" in text


def test_module_matrix_mentions_parameter_fusion_hardening():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Parameter fusion benchmark hardening" in text


def test_requirement_traceability_mentions_parameter_fusion_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "parameter fusion benchmark hardening" in text
    assert "| parameter fusion benchmark hardening |" in text


def test_no_solver_core_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == ""


def test_no_cli_yaml_references_modification():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "reservoir_backend/cli",
            "reservoir_backend/io",
            "reservoir_backend/api",
            "reservoir_backend/cross_scale",
            "config",
            "scripts",
            "references/upstream",
            "references/fixtures",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == ""


def test_existing_three_phase_benchmark_still_runs(tmp_path):
    from benchmarks.three_phase_benchmark import run_three_phase_benchmark

    assert run_three_phase_benchmark(tmp_path)["success"] is True


def test_existing_capillary_gravity_benchmark_still_runs(tmp_path):
    from benchmarks.capillary_gravity_benchmark import run_capillary_gravity_benchmark

    assert run_capillary_gravity_benchmark(tmp_path)["success"] is True


def test_existing_saturation_transport_benchmark_still_runs(tmp_path):
    from benchmarks.saturation_transport_benchmark import run_saturation_transport_benchmark

    assert run_saturation_transport_benchmark(tmp_path)["success"] is True


def test_existing_pressure_benchmark_still_runs(tmp_path):
    from benchmarks.pressure_solver_benchmark import run_pressure_solver_benchmark

    assert run_pressure_solver_benchmark(tmp_path)["success"] is True
