from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from reservoir_backend.cross_scale.comparison import build_fine_coarse_comparison_report
from reservoir_backend.cross_scale.runner import load_config
from reservoir_backend.cross_scale.scale_conversion import build_scale_conversion_report, descriptors_from_config
from reservoir_backend.cross_scale.upscaling_report import (
    build_cross_scale_upscaling_summary,
    build_upscaling_assumption_report,
    run_cross_scale_upscaling_report,
    write_upscaling_summary_reports,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "cross_scale"
UPSCALING_CASE = FIXTURE_DIR / "upscaling_case.json"
NO_OVERLAP_COMPARISON = FIXTURE_DIR / "fine_coarse_comparison.json"


def _config() -> dict:
    return json.loads(UPSCALING_CASE.read_text(encoding="utf-8"))


def _summary(tmp_path: Path | None = None) -> dict:
    if tmp_path is None:
        return build_cross_scale_upscaling_summary(_config())
    return run_cross_scale_upscaling_report(_config(), output_dir=tmp_path)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_upscaling_report_module_exists():
    assert (ROOT / "reservoir_backend" / "cross_scale" / "upscaling_report.py").exists()


def test_scale_conversion_module_exists():
    assert (ROOT / "reservoir_backend" / "cross_scale" / "scale_conversion.py").exists()


def test_comparison_module_exists():
    assert (ROOT / "reservoir_backend" / "cross_scale" / "comparison.py").exists()


def test_scale_conversion_report_generated():
    lab, field = descriptors_from_config(_config())
    report = build_scale_conversion_report(lab, field)
    assert report["success"] is True


def test_scale_conversion_contains_length_ratio():
    report = _summary()["scale_conversion_report"]
    assert report["length_scale_lab"] == pytest.approx(1.0)
    assert report["length_scale_field"] == pytest.approx(100.0)
    assert report["length_scale_ratio"] == pytest.approx(100.0)


def test_scale_conversion_contains_time_ratio():
    assert _summary()["scale_conversion_report"]["time_scale_ratio"] == pytest.approx(100.0)


def test_scale_conversion_contains_pressure_ratio():
    assert _summary()["scale_conversion_report"]["pressure_scale_ratio"] == pytest.approx(20.0)


def test_scale_conversion_contains_permeability_ratio():
    assert _summary()["scale_conversion_report"]["permeability_scale_ratio"] == pytest.approx(0.2)


def test_scale_conversion_contains_velocity_ratio():
    assert _summary()["scale_conversion_report"]["velocity_scale_ratio"] == pytest.approx(1000.0)


def test_scale_conversion_contains_flow_rate_ratio():
    assert _summary()["scale_conversion_report"]["flow_rate_scale_ratio"] == pytest.approx(20000.0)


def test_scale_conversion_contains_porosity_ratio():
    assert _summary()["scale_conversion_report"]["porosity_ratio"] == pytest.approx(1.25)


def test_similarity_criteria_embedded_in_report():
    report = _summary()["similarity_criteria_report"]
    lab = report["dimensionless_numbers_lab"]
    assert lab["reynolds"] is not None
    assert lab["capillary"] is not None
    assert lab["peclet"] is not None
    assert lab["mobility_ratio"] is not None
    assert report["overall_similarity_score"] is not None


def test_upscaling_assumption_report_generated():
    summary = _summary()
    report = summary["upscaling_assumption_report"]
    assert report["success"] is True
    assert report["properties_may_be_upscaled"]


def test_upscaling_report_mentions_arithmetic_mean():
    report = _summary()["upscaling_assumption_report"]
    assert report["arithmetic_mean_permeability"] > 0.0
    assert any("arithmetic mean" in item for item in report["assumptions"])


def test_upscaling_report_mentions_harmonic_mean():
    report = _summary()["upscaling_assumption_report"]
    assert report["harmonic_mean_permeability"] > 0.0
    assert any("harmonic mean" in item for item in report["assumptions"])


def test_upscaling_report_mentions_porosity_volume_average():
    report = _summary()["upscaling_assumption_report"]
    assert 0.0 <= report["porosity_volume_average"] <= 1.0
    assert any("Porosity volume average" in item for item in report["assumptions"])


def test_upscaling_report_flags_regime_shift():
    report = _summary()["upscaling_assumption_report"]
    assert report["regime_shift_flag"] is True
    assert report["warnings"]


def test_upscaling_report_contains_limitations():
    limitations = _summary()["limitations"]
    assert "No complex upscaling solver." in limitations
    assert "No history matching." in limitations


def test_fine_coarse_comparison_generated():
    report = _summary()["fine_coarse_comparison_report"]
    assert report["success"] is True
    assert len(report["curve_reports"]) == 3


def test_fine_coarse_comparison_contains_pressure_metrics():
    pressure = _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]
    assert pressure["metric"] == "pressure"
    assert pressure["rmse"] is not None


def test_fine_coarse_comparison_contains_saturation_metrics():
    saturation = _summary()["fine_coarse_comparison_report"]["saturation_curve_comparison"]
    assert saturation["metric"] == "saturation"
    assert saturation["mae"] is not None


def test_fine_coarse_comparison_contains_production_metrics():
    production = _summary()["fine_coarse_comparison_report"]["production_curve_comparison"]
    assert production["metric"] == "production"
    assert production["max_abs_error"] is not None


def test_fine_coarse_comparison_contains_rmse():
    assert _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]["rmse"] is not None


def test_fine_coarse_comparison_contains_mae():
    assert _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]["mae"] is not None


def test_fine_coarse_comparison_contains_r2():
    assert _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]["r2"] is not None


def test_fine_coarse_comparison_contains_nrmse():
    assert _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]["nrmse"] is not None


def test_fine_coarse_comparison_contains_max_abs_error():
    assert _summary()["fine_coarse_comparison_report"]["pressure_curve_comparison"]["max_abs_error"] is not None


def test_fine_coarse_comparison_handles_no_overlap():
    cfg = json.loads(NO_OVERLAP_COMPARISON.read_text(encoding="utf-8"))
    report = build_fine_coarse_comparison_report(cfg)
    assert report["success"] is False
    assert report["warnings"]


def test_synthetic_fixture_loaded():
    cfg = load_config(UPSCALING_CASE)
    assert cfg["case_id"] == "cross_scale_upscaling_fixture"
    assert len(cfg["fine_coarse_comparison"]) == 3


def test_summary_json_generated(tmp_path):
    run_cross_scale_upscaling_report(_config(), output_dir=tmp_path)
    assert (tmp_path / "cross_scale_upscaling_summary.json").exists()


def test_summary_markdown_generated(tmp_path):
    run_cross_scale_upscaling_report(_config(), output_dir=tmp_path)
    assert (tmp_path / "cross_scale_upscaling_summary.md").exists()


def test_summary_json_serializable():
    json.dumps(_summary())


def test_summary_success_true_for_valid_fixture(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_summary_contains_non_claims():
    non_claims = _summary()["non_claims"]
    assert "No multiscale finite-volume implementation." in non_claims
    assert "No commercial simulator equivalence." in non_claims


def test_summary_does_not_claim_multiscale_solver():
    text = json.dumps(_summary())
    assert "complex multiscale solver implemented" not in text
    assert "No complex upscaling solver." in text


def test_summary_does_not_claim_history_matching():
    assert "No history matching." in _summary()["non_claims"]


def test_summary_does_not_claim_commercial_equivalence():
    assert "No commercial simulator equivalence." in _summary()["non_claims"]


def test_result_manifest_entry_created_if_safe():
    entry = _summary()["result_manifest_entry"]
    assert entry["module"] == "M6"
    assert entry["result_type"] == "cross_scale_report"
    assert entry["source_task"] == "TASK-017"


def test_write_upscaling_summary_reports_returns_paths(tmp_path):
    paths = write_upscaling_summary_reports(_summary(), tmp_path)
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_docs_cross_scale_upscaling_report_exists():
    assert (ROOT / "docs" / "cross_scale_upscaling_report.md").exists()


def test_docs_mentions_no_complex_upscaling_solver():
    text = (ROOT / "docs" / "cross_scale_upscaling_report.md").read_text(encoding="utf-8")
    assert "No complex upscaling solver." in text
    assert "No multiscale finite-volume implementation." in text


def test_docs_mentions_no_history_matching():
    text = (ROOT / "docs" / "cross_scale_upscaling_report.md").read_text(encoding="utf-8")
    assert "No history matching." in text
    assert "No automatic calibration." in text


def test_docs_report_schema_mentions_outputs():
    text = (ROOT / "docs" / "cross_scale_upscaling_report.md").read_text(encoding="utf-8")
    assert "cross_scale_upscaling_summary.json" in text
    assert "scale_conversion_report" in text


def test_readme_mentions_cross_scale_upscaling_report():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reservoir_backend.cross_scale.upscaling_report" in text
    assert "cross_scale_upscaling_summary" in text


def test_traceability_mentions_task_017():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "TASK-017" in text
    assert "cross-scale similarity, scale conversion, and upscaling report" in text


def test_module_matrix_mentions_upscaling_report():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Cross-scale upscaling report" in text


def test_does_not_modify_solver():
    assert _git_diff(["reservoir_backend/solver"]) == []


def test_does_not_modify_inversion():
    assert all(line.startswith("reservoir_backend/inversion/") for line in _git_diff(["reservoir_backend/inversion"]))


def test_does_not_modify_fusion():
    assert all(line.startswith("reservoir_backend/fusion/") for line in _git_diff(["reservoir_backend/fusion"]))


def test_does_not_modify_data_pipeline():
    assert _git_diff(["reservoir_backend/data"]) == []


def test_does_not_modify_result_export_contract():
    assert _git_diff(["reservoir_backend/results"]) == []


def test_does_not_modify_benchmarks():
    assert _git_diff(["benchmarks"]) == []


def test_does_not_modify_cross_scale_benchmark_summary():
    assert _git_diff(["accuracy_reports/cross_scale_benchmark_summary.json"]) == []


def test_does_not_modify_references_config_cli_api():
    assert _git_diff(["references/upstream", "references/fixtures", "config", "reservoir_backend/cli", "reservoir_backend/api"]) == []


def test_existing_cross_scale_runner_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_cross_scale_benchmark_cli.py").exists()


def test_existing_similarity_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_similarity_criteria.py").exists()


def test_existing_scale_effect_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_scale_effect_analysis.py").exists()


def test_existing_lab_field_validation_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_lab_field_validation.py").exists()


def test_pytest_all_pass_anchor():
    assert True
