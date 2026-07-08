from __future__ import annotations

import json
from pathlib import Path

import pytest

from reservoir_backend.workflow.industrial_case import (
    build_impes_config_from_workflow_config,
    load_industrial_case_config,
    run_industrial_case_workflow,
    validate_industrial_case_config,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "industrial_case" / "industrial_case_v0.yaml"


def test_case_config_load() -> None:
    config = load_industrial_case_config(FIXTURE)
    assert config["case"]["case_id"] == "industrial_case_fixture"
    assert config["run"]["run_id"] == "industrial_run_fixture"


def test_config_validation() -> None:
    config = load_industrial_case_config(FIXTURE)
    normalized = validate_industrial_case_config(config)
    assert normalized["grid"]["nx"] == 6


def test_invalid_config_rejected() -> None:
    config = load_industrial_case_config(FIXTURE)
    config["rock"]["porosity"] = 1.5
    with pytest.raises(ValueError, match="porosity"):
        validate_industrial_case_config(config)


def test_impes_config_created() -> None:
    config = load_industrial_case_config(FIXTURE)
    impes = build_impes_config_from_workflow_config(config)
    assert impes.grid.shape == (2, 2, 6)
    assert impes.num_steps == 6


def test_project_case_run_integration(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    assert summary["project"]["num_projects"] == 1
    assert summary["case"]["num_cases"] == 1
    assert summary["run_history"]["num_runs"] == 1


def test_impes_runner_integration(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    assert summary["success"] is True
    assert summary["impes_summary"]["num_steps"] == 6


def test_production_summary(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    production = summary["production_summary"]
    assert production["num_curve_points"] == 6
    assert production["final_total_liquid_rate"] >= 0.0


def test_water_cut_curve(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    curve = summary["water_cut_curve"]
    assert len(curve) == 6
    assert all(0.0 <= item["water_cut"] <= 1.0 for item in curve)


def test_breakthrough_time(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    assert "breakthrough_time" in summary


def test_result_manifest_path(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    manifest_path = Path(summary["result_manifest_path"])
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["results"][0]["result_type"] == "engineering_report"


def test_engineering_report_json_markdown(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    assert Path(summary["engineering_report_json"]).exists()
    markdown = Path(summary["engineering_report_markdown"])
    assert markdown.exists()
    assert "Industrial Case Workflow Summary" in markdown.read_text(encoding="utf-8")


def test_report_json_serializable(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    json.dumps(summary)


def test_no_black_oil_claim(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    text = "\n".join(summary["non_claims"] + summary["limitations"])
    assert "No black-oil solver is implemented." in text
    assert "commercial simulator equivalence" in text


def test_no_history_matching_claim(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    text = "\n".join(summary["non_claims"] + summary["limitations"])
    assert "No history matching is implemented." in text


def test_default_report_paths_written(tmp_path: Path) -> None:
    summary = run_industrial_case_workflow(FIXTURE, output_dir=tmp_path)
    assert Path(summary["engineering_report_json"]).name == "industrial_case_workflow_summary.json"
    assert Path(summary["engineering_report_markdown"]).name == "industrial_case_workflow_summary.md"
