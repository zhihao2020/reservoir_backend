from __future__ import annotations

import json

import pytest

from harness.check_outputs import check_required_outputs
from harness.run_validation import run_validation


def test_validation_harness_runs(tmp_path) -> None:
    summary = run_validation(run_pytest=False, case_id="validation_case", results_root=tmp_path / "results", reports_dir=tmp_path / "reports")
    assert summary["full_pipeline_passed"] is True


def test_validation_summary_json_created(tmp_path) -> None:
    run_validation(run_pytest=False, case_id="validation_case", results_root=tmp_path / "results", reports_dir=tmp_path / "reports")
    assert (tmp_path / "reports" / "validation_summary.json").exists()


def test_validation_summary_required_keys(tmp_path) -> None:
    run_validation(run_pytest=False, case_id="validation_case", results_root=tmp_path / "results", reports_dir=tmp_path / "reports")
    data = json.loads((tmp_path / "reports" / "validation_summary.json").read_text())
    keys = {
        "pytest_passed",
        "full_pipeline_passed",
        "required_outputs_exist",
        "physical_ranges_valid",
        "no_nan_inf",
        "material_balance_error",
        "max_cfl",
        "fusion_nan_cells",
        "fusion_clipped_cells",
        "success",
        "timestamp",
    }
    assert keys.issubset(data)


def test_validation_summary_success_field(tmp_path) -> None:
    summary = run_validation(run_pytest=False, case_id="validation_case", results_root=tmp_path / "results", reports_dir=tmp_path / "reports")
    assert summary["success"] is True


def test_check_outputs_detects_missing_file(tmp_path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    ok, missing = check_required_outputs(case_dir, required=["missing.npy"])
    assert ok is False
    assert missing == ["missing.npy"]
