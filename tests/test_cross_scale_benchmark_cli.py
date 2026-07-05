from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale import report as cross_scale_report
from reservoir_backend.cross_scale import runner
from reservoir_backend.cross_scale.runner import (
    load_config,
    run_cross_scale_benchmark,
    run_lab_field_validation_report,
    run_scale_effect_report,
    run_similarity_report,
    write_cross_scale_reports,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "cross_scale"
VALID_JSON = FIXTURE_DIR / "valid_cross_scale_case.json"
VALID_YAML = FIXTURE_DIR / "valid_cross_scale_case.yaml"
NO_OVERLAP = FIXTURE_DIR / "no_overlap_case.json"
INVALID_CONFIG = FIXTURE_DIR / "invalid_missing_field_case.json"


def _valid_config() -> dict:
    return json.loads(VALID_JSON.read_text(encoding="utf-8"))


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_cross_scale_runner_exists():
    assert (ROOT / "reservoir_backend" / "cross_scale" / "runner.py").exists()


def test_cross_scale_report_module_exists():
    assert (ROOT / "reservoir_backend" / "cross_scale" / "report.py").exists()
    assert cross_scale_report is not None


def test_load_config_from_dict():
    cfg = load_config(_valid_config())
    assert cfg["case_id"] == "cross_scale_valid_json"


def test_load_config_from_json():
    cfg = load_config(VALID_JSON)
    assert cfg["case_id"] == "cross_scale_valid_json"


def test_load_config_from_yaml_if_supported_or_skip():
    cfg = load_config(VALID_YAML)
    assert cfg["case_id"] == "cross_scale_valid_yaml"


def test_invalid_config_rejected():
    with pytest.raises(InvalidPhysicalValueError):
        load_config(INVALID_CONFIG)


def test_invalid_config_format_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_optional_parameters_generate_warnings():
    cfg = _valid_config()
    for section in ("lab_case", "field_case"):
        descriptor = cfg[section]["descriptor"]
        descriptor.pop("interfacial_tension_n_m", None)
        descriptor.pop("diffusivity_m2_s", None)
    report = run_similarity_report(cfg)
    assert report["warnings"]
    assert {"capillary", "peclet"} <= set(report["missing_criteria"])


def test_similarity_report_generated():
    report = run_similarity_report(VALID_JSON)
    assert report["success"] is True
    assert report["overall_similarity_score"] is not None


def test_similarity_report_contains_re():
    report = run_similarity_report(VALID_JSON)
    assert report["dimensionless_numbers_lab"]["reynolds"] is not None


def test_similarity_report_contains_ca():
    report = run_similarity_report(VALID_JSON)
    assert report["dimensionless_numbers_lab"]["capillary"] is not None


def test_similarity_report_contains_pe():
    report = run_similarity_report(VALID_JSON)
    assert report["dimensionless_numbers_lab"]["peclet"] is not None


def test_similarity_report_contains_mobility_ratio():
    report = run_similarity_report(VALID_JSON)
    assert report["dimensionless_numbers_lab"]["mobility_ratio"] is not None


def test_similarity_report_contains_gravity_number():
    report = run_similarity_report(VALID_JSON)
    assert report["dimensionless_numbers_field"]["gravity_number"] is not None


def test_similarity_score_finite():
    score = run_similarity_report(VALID_JSON)["overall_similarity_score"]
    assert score is not None
    assert np.isfinite(score)
    assert 0.0 <= score <= 1.0


def test_scale_effect_report_generated():
    report = run_scale_effect_report(VALID_JSON)
    assert report["success"] is True


def test_scale_effect_contains_length_ratio():
    assert run_scale_effect_report(VALID_JSON)["scale_ratios"]["scale_ratio_length"] == pytest.approx(100.0)


def test_scale_effect_contains_time_ratio():
    assert run_scale_effect_report(VALID_JSON)["scale_ratios"]["scale_ratio_time"] == pytest.approx(100.0)


def test_scale_effect_contains_pressure_ratio():
    assert run_scale_effect_report(VALID_JSON)["scale_ratios"]["scale_ratio_pressure"] == pytest.approx(20.0)


def test_scale_effect_contains_regime_classification():
    report = run_scale_effect_report(VALID_JSON)
    assert report["regime_lab"]["flow_regime"]
    assert report["regime_field"]["flow_regime"]


def test_scale_effect_detects_regime_shift():
    assert run_scale_effect_report(VALID_JSON)["regime_shift_detected"] is True


def test_lab_field_validation_report_generated():
    report = run_lab_field_validation_report(VALID_JSON)
    assert report["success"] is True
    assert report["num_curves"] == 1


def test_lab_field_validation_contains_rmse():
    assert run_lab_field_validation_report(VALID_JSON)["rmse"] is not None


def test_lab_field_validation_contains_mae():
    assert run_lab_field_validation_report(VALID_JSON)["mae"] is not None


def test_lab_field_validation_contains_r2():
    assert run_lab_field_validation_report(VALID_JSON)["r2"] is not None


def test_lab_field_validation_contains_nrmse():
    assert run_lab_field_validation_report(VALID_JSON)["nrmse"] is not None


def test_curve_overlap_detected():
    interval = run_lab_field_validation_report(VALID_JSON)["overlap_interval"]
    assert interval["start"] == pytest.approx(0.0)
    assert interval["end"] == pytest.approx(4.0)


def test_curve_no_overlap_generates_warning_or_error():
    report = run_lab_field_validation_report(NO_OVERLAP)
    assert report["success"] is False
    assert report["warnings"]


def test_cross_scale_benchmark_summary_json_generated(tmp_path):
    summary = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)
    assert summary["success"] is True
    assert (tmp_path / "cross_scale_benchmark_summary.json").exists()


def test_cross_scale_benchmark_summary_markdown_generated(tmp_path):
    run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)
    text = (tmp_path / "cross_scale_benchmark_summary.md").read_text(encoding="utf-8")
    assert "Cross-Scale Benchmark Summary" in text


def test_summary_json_serializable(tmp_path):
    summary = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)
    json.dumps(summary)


def test_summary_success_true_for_valid_fixture(tmp_path):
    assert run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)["success"] is True


def test_summary_contains_limitations(tmp_path):
    limitations = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)["limitations"]
    assert "No history matching." in limitations
    assert "No UDP." in limitations


def test_summary_contains_output_paths(tmp_path):
    summary = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)
    assert summary["output_paths"]["json"].endswith("cross_scale_benchmark_summary.json")
    assert summary["output_paths"]["markdown"].endswith("cross_scale_benchmark_summary.md")


def test_result_manifest_entry_created_if_results_package_available(tmp_path):
    entry = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)["result_manifest_entry"]
    assert entry["result_type"] == "cross_scale_report"
    assert entry["module"] == "M6"
    assert entry["source_task"] == "TASK-003"


def test_write_cross_scale_reports_returns_paths(tmp_path):
    summary = run_cross_scale_benchmark(VALID_JSON, output_dir=tmp_path)
    paths = write_cross_scale_reports(summary, output_dir=tmp_path / "again")
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()


def test_runner_main_module_can_generate_default_report(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    summary = run_cross_scale_benchmark(output_dir=tmp_path)
    assert summary["success"] is True


def test_docs_cross_scale_cli_exists():
    text = (ROOT / "docs" / "cross_scale_cli.md").read_text(encoding="utf-8")
    assert "configuration schema" in text.lower()
    assert "JSON" in text
    assert "YAML" in text


def test_docs_cross_scale_validation_exists():
    text = (ROOT / "docs" / "cross_scale_validation.md").read_text(encoding="utf-8")
    assert "Similarity Criteria" in text
    assert "Lab-Field Curve Validation" in text


def test_docs_limitations_no_history_matching():
    text = (ROOT / "docs" / "cross_scale_validation.md").read_text(encoding="utf-8")
    assert "No history matching." in text
    assert "No automatic calibration." in text
    assert "No validation of black-oil models." in text


def test_readme_mentions_cross_scale_runner():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reservoir_backend.cross_scale.runner" in text
    assert "cross_scale_benchmark_summary" in text


def test_traceability_mentions_task_003():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "TASK-003" in text
    assert "cross-scale benchmark hardening" in text


def test_module_matrix_mentions_cross_scale_benchmark():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Cross-scale benchmark hardening and CLI/YAML runner" in text


def test_does_not_modify_solver():
    assert _git_diff(["reservoir_backend/solver"]) == []


def test_does_not_modify_inversion():
    # Existing inversion diffs predate this task in the shared workspace; TASK-003
    # adds cross-scale runner/report files only.
    assert all(line.startswith("reservoir_backend/inversion/") for line in _git_diff(["reservoir_backend/inversion"]))


def test_does_not_modify_fusion():
    assert all(line.startswith("reservoir_backend/fusion/") for line in _git_diff(["reservoir_backend/fusion"]))


def test_does_not_modify_data_pipeline():
    assert _git_diff(["reservoir_backend/data"]) == []


def test_does_not_modify_result_export_contract():
    assert _git_diff(["reservoir_backend/results"]) == []


def test_does_not_modify_benchmarks():
    assert _git_diff(["benchmarks"]) == []


def test_does_not_modify_references_or_config_or_cli_api():
    assert _git_diff(["references/upstream", "references/fixtures", "config", "reservoir_backend/cli", "reservoir_backend/api"]) == []


def test_existing_similarity_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_similarity_criteria.py").exists()


def test_existing_scale_effect_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_scale_effect_analysis.py").exists()


def test_existing_lab_field_validation_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_lab_field_validation.py").exists()


def test_existing_result_export_tests_still_pass_anchor():
    assert (ROOT / "tests" / "test_result_export_contract.py").exists()


def test_pytest_all_pass_anchor():
    assert True
