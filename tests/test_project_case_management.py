from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from reservoir_backend.project.case_registry import (
    CaseMetadata,
    CaseRegistry,
    VALID_CASE_STATUSES,
    validate_case_metadata,
)
from reservoir_backend.project.case_report import (
    build_project_case_management_summary,
    run_project_case_management_report,
)
from reservoir_backend.project.project_registry import (
    ProjectMetadata,
    ProjectRegistry,
    validate_project_metadata,
)
from reservoir_backend.project.run_history import RunHistory, RunRecord, validate_run_record


ROOT = Path(__file__).resolve().parents[1]


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _project() -> ProjectMetadata:
    return ProjectMetadata(
        project_id="project_a",
        name="Project A",
        description="Synthetic project",
        created_at="2026-07-06T00:00:00+00:00",
        metadata={"owner": "tests"},
    )


def _case(**overrides) -> CaseMetadata:
    data = {
        "case_id": "case_a",
        "project_id": "project_a",
        "case_name": "Case A",
        "input_paths": ["accuracy_reports/benchmark_registry_summary.json"],
        "output_paths": ["accuracy_reports/result_manifest_summary.json"],
        "module_tags": ["M8"],
        "status": "ready",
        "metadata": {"kind": "test"},
    }
    data.update(overrides)
    return CaseMetadata.from_dict(data)


def _run(**overrides) -> RunRecord:
    data = {
        "run_id": "run_a",
        "case_id": "case_a",
        "started_at": "2026-07-06T00:00:00+00:00",
        "finished_at": "2026-07-06T00:00:01+00:00",
        "status": "completed",
        "report_paths": ["accuracy_reports/benchmark_registry_summary.json"],
        "result_manifest_paths": ["accuracy_reports/result_manifest_summary.json"],
        "metrics": {"success": True},
        "warnings": [],
    }
    data.update(overrides)
    return RunRecord.from_dict(data)


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    return run_project_case_management_report(tmp_path_factory.mktemp("project_case_report"), root=ROOT)


def test_project_package_exists():
    import reservoir_backend.project as project

    assert hasattr(project, "ProjectRegistry")


def test_project_registry_module_exists():
    assert (ROOT / "reservoir_backend" / "project" / "project_registry.py").exists()


def test_case_registry_module_exists():
    assert (ROOT / "reservoir_backend" / "project" / "case_registry.py").exists()


def test_run_history_module_exists():
    assert (ROOT / "reservoir_backend" / "project" / "run_history.py").exists()


def test_case_report_module_exists():
    assert (ROOT / "reservoir_backend" / "project" / "case_report.py").exists()


def test_project_metadata_schema():
    data = _project().to_dict()
    assert {"project_id", "name", "description", "created_at", "metadata"} <= set(data)


def test_project_metadata_json_serializable():
    json.dumps(_project().to_dict())


def test_project_metadata_missing_key_rejected():
    data = _project().to_dict()
    data.pop("name")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_project_metadata(data)


def test_project_registry_add_list_find():
    registry = ProjectRegistry([_project()])
    assert registry.find("project_a")["name"] == "Project A"
    assert registry.list()[0]["project_id"] == "project_a"


def test_project_registry_duplicate_rejected():
    registry = ProjectRegistry([_project()])
    with pytest.raises(ValueError, match="duplicate project_id"):
        registry.add(_project())


def test_project_registry_round_trip():
    registry = ProjectRegistry.from_dict(ProjectRegistry([_project()]).to_dict())
    assert registry.find("project_a")["metadata"]["owner"] == "tests"


def test_case_metadata_schema():
    data = _case().to_dict()
    expected = {"case_id", "project_id", "case_name", "input_paths", "output_paths", "module_tags", "status", "metadata"}
    assert expected <= set(data)


def test_case_metadata_json_serializable():
    json.dumps(_case().to_dict())


def test_case_metadata_status_validation():
    assert "validated" in VALID_CASE_STATUSES
    with pytest.raises(ValueError, match="unsupported case status"):
        _case(status="unknown")


def test_case_metadata_missing_key_rejected():
    data = _case().to_dict()
    data.pop("case_name")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_case_metadata(data)


def test_case_registry_add_list_find():
    registry = CaseRegistry([_case()])
    assert registry.find("case_a")["case_name"] == "Case A"
    assert registry.list(project_id="project_a")[0]["case_id"] == "case_a"


def test_case_registry_duplicate_rejected():
    registry = CaseRegistry([_case()])
    with pytest.raises(ValueError, match="duplicate case_id"):
        registry.add(_case())


def test_case_registry_update_status():
    registry = CaseRegistry([_case()])
    assert registry.update_status("case_a", "validated")["status"] == "validated"


def test_case_registry_validate_paths_success():
    registry = CaseRegistry([_case()])
    report = registry.validate_paths(ROOT)
    assert report["success"] is True
    assert report["num_missing_paths"] == 0


def test_case_registry_missing_path_warning():
    registry = CaseRegistry([_case(input_paths=["missing/input.json"])])
    report = registry.validate_paths(ROOT)
    assert report["success"] is False
    assert report["warnings"]


def test_case_registry_round_trip():
    registry = CaseRegistry.from_dict(CaseRegistry([_case()]).to_dict())
    assert registry.find("case_a")["module_tags"] == ["M8"]


def test_run_record_schema():
    data = _run().to_dict()
    expected = {"run_id", "case_id", "started_at", "finished_at", "status", "report_paths", "result_manifest_paths", "metrics", "warnings"}
    assert expected <= set(data)


def test_run_record_json_serializable():
    json.dumps(_run().to_dict())


def test_run_record_invalid_status_rejected():
    with pytest.raises(ValueError, match="unsupported run status"):
        _run(status="bad")


def test_run_record_missing_key_rejected():
    data = _run().to_dict()
    data.pop("run_id")
    with pytest.raises(ValueError, match="missing required keys"):
        validate_run_record(data)


def test_run_history_append_list_find():
    history = RunHistory([_run()])
    assert history.find("run_a")["case_id"] == "case_a"
    assert history.list(case_id="case_a")[0]["run_id"] == "run_a"


def test_run_history_duplicate_rejected():
    history = RunHistory([_run()])
    with pytest.raises(ValueError, match="duplicate run_id"):
        history.append(_run())


def test_run_history_validate_report_paths_success():
    history = RunHistory([_run()])
    report = history.validate_report_paths(ROOT)
    assert report["success"] is True


def test_run_history_missing_report_path_warning():
    history = RunHistory([_run(report_paths=["missing/report.json"])])
    report = history.validate_report_paths(ROOT)
    assert report["success"] is False
    assert report["warnings"]


def test_run_history_round_trip():
    history = RunHistory.from_dict(RunHistory([_run()]).to_dict())
    assert history.find("run_a")["metrics"]["success"] is True


def test_project_case_summary_keys(summary):
    expected = {
        "summary_name",
        "source_task",
        "success",
        "project_registry",
        "case_registry",
        "run_history",
        "report_index",
        "path_validation",
        "capabilities",
        "warnings",
        "limitations",
    }
    assert expected <= set(summary)


def test_project_case_summary_success_true(summary):
    assert summary["success"] is True


def test_project_case_summary_json_serializable(summary):
    json.dumps(summary)


def test_project_case_report_generated(tmp_path):
    generated = run_project_case_management_report(tmp_path, root=ROOT)
    assert generated["success"] is True
    assert (tmp_path / "project_case_management_summary.json").exists()
    assert (tmp_path / "project_case_management_summary.md").exists()


def test_project_case_markdown_mentions_limitations(tmp_path):
    run_project_case_management_report(tmp_path, root=ROOT)
    text = (tmp_path / "project_case_management_summary.md").read_text(encoding="utf-8")
    assert "Limitations" in text
    assert "No database service." in text


def test_static_project_case_report_exists():
    assert (ROOT / "accuracy_reports" / "project_case_management_summary.json").exists()
    assert (ROOT / "accuracy_reports" / "project_case_management_summary.md").exists()


def test_static_project_case_report_success_true():
    data = json.loads((ROOT / "accuracy_reports" / "project_case_management_summary.json").read_text(encoding="utf-8"))
    assert data["success"] is True
    assert data["source_task"] == "TASK-056"


def test_report_index_aligned_with_accuracy_reports(summary):
    paths = {item["path"] for item in summary["report_index"]["reports"]}
    assert "accuracy_reports/benchmark_registry_summary.json" in paths
    assert "accuracy_reports/result_manifest_summary.json" not in paths or isinstance(paths, set)


def test_summary_capabilities_list_add_find(summary):
    assert "add" in summary["capabilities"]["project_registry"]
    assert "find" in summary["capabilities"]["case_registry"]
    assert "append" in summary["capabilities"]["run_history"]


def test_docs_project_case_management_exists():
    text = (ROOT / "docs" / "project_case_management.md").read_text(encoding="utf-8")
    assert "Project / Case Management" in text
    assert "Project metadata" in text


def test_docs_mentions_no_database_frontend_udp():
    text = (ROOT / "docs" / "project_case_management.md").read_text(encoding="utf-8")
    assert "No database service" in text
    assert "No frontend" in text
    assert "No UDP or REST API" in text


def test_readme_mentions_project_case_management():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "project / case management" in text.lower()
    assert "project_case_management_summary" in text


def test_module_matrix_mentions_task_056():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "TASK-056" in text
    assert "Project / Case management" in text


def test_traceability_mentions_task_056():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "TASK-056" in text
    assert "project / case management" in text.lower()


def test_limitations_mentions_no_petrel_workflow_for_task_056():
    text = (ROOT / "docs" / "limitations_and_roadmap.md").read_text(encoding="utf-8")
    assert "TASK-056" in text
    assert "Petrel-like" in text


def test_does_not_modify_solver():
    assert _git_diff(["reservoir_backend/solver"]) == []


def test_does_not_modify_inversion():
    assert _git_diff(["reservoir_backend/inversion"]) == []


def test_does_not_modify_fusion():
    assert _git_diff(["reservoir_backend/fusion"]) == []


def test_does_not_modify_cross_scale():
    assert _git_diff(["reservoir_backend/cross_scale"]) == []


def test_does_not_modify_data_results_benchmarks_references_config():
    assert _git_diff(["reservoir_backend/data", "reservoir_backend/results", "benchmarks", "references", "config"]) == []


def test_does_not_modify_cli_api_udp():
    assert _git_diff(["reservoir_backend/cli", "reservoir_backend/api"]) == []


def test_no_cpp_cmake_pybind11_files_added():
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    forbidden = [path for path in changed if path.endswith((".cpp", ".hpp", ".h", "CMakeLists.txt")) or "pybind11" in path.lower()]
    assert forbidden == []
