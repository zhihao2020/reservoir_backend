from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from reservoir_backend.performance import profiler
from reservoir_backend.performance.performance_report import run_performance_baseline


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def performance_summary(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("performance_report")
    return run_performance_baseline(output_dir)


def _git_diff(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", *paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_performance_package_exists():
    import reservoir_backend.performance as performance

    assert hasattr(performance, "run_performance_baseline")


def test_profiler_module_exists():
    assert hasattr(profiler, "run_stage_profiles")
    assert hasattr(profiler, "summarize_profiles")


def test_performance_report_module_exists():
    import reservoir_backend.performance.performance_report as report

    assert hasattr(report, "run_performance_baseline")


def test_synthetic_cases_include_small_medium_large():
    assert [case.case_id for case in profiler.SYNTHETIC_CASES] == ["small", "medium", "large"]


def test_measure_runtime_success():
    measured = profiler.measure_runtime("toy", lambda: {"success": True, "value": 1})
    assert measured["success"] is True
    assert measured["runtime_sec"] >= 0.0
    assert measured["memory_peak_mb"] >= 0.0


def test_measure_runtime_failure_is_reported():
    measured = profiler.measure_runtime("bad", lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert measured["success"] is False
    assert "ValueError" in measured["error"]


def test_run_stage_profiles_returns_three_cases():
    profiles = profiler.run_stage_profiles()
    assert len(profiles) == 3
    assert {profile["case_id"] for profile in profiles} == {"small", "medium", "large"}


def test_each_case_contains_required_stages(performance_summary):
    required = {"pressure", "saturation_transport", "fusion", "cross_scale", "benchmark_registry"}
    for case in performance_summary["case_profiles"]:
        assert {stage["stage_name"] for stage in case["stages"]} == required


def test_pressure_stage_records_solver_stats(performance_summary):
    pressure = next(stage for stage in performance_summary["case_profiles"][0]["stages"] if stage["stage_name"] == "pressure")
    assert pressure["solver_backend"]
    assert "residual_norm" in pressure
    assert "mass_balance_error" in pressure


def test_saturation_stage_records_cfl_and_balance(performance_summary):
    saturation = next(stage for stage in performance_summary["case_profiles"][0]["stages"] if stage["stage_name"] == "saturation_transport")
    assert saturation["max_cfl"] >= 0.0
    assert np.isfinite(saturation["material_balance_error"])


def test_fusion_stage_records_array_size(performance_summary):
    fusion = next(stage for stage in performance_summary["case_profiles"][0]["stages"] if stage["stage_name"] == "fusion")
    assert fusion["array_size_bytes"] > 0
    assert fusion["fused_min"] <= fusion["fused_max"]


def test_cross_scale_stage_records_similarity(performance_summary):
    cross_scale = next(stage for stage in performance_summary["case_profiles"][0]["stages"] if stage["stage_name"] == "cross_scale")
    assert 0.0 <= cross_scale["similarity_score"] <= 1.0
    assert "rmse" in cross_scale


def test_registry_stage_records_case_counts(performance_summary):
    registry = next(stage for stage in performance_summary["case_profiles"][0]["stages"] if stage["stage_name"] == "benchmark_registry")
    assert registry["num_benchmark_summaries"] >= 6
    assert registry["num_benchmark_cases"] > 0


def test_runtime_summary_contains_stages(performance_summary):
    assert {"pressure", "saturation_transport", "fusion", "cross_scale", "benchmark_registry"} <= set(performance_summary["runtime_summary"])


def test_memory_summary_contains_stages(performance_summary):
    assert {"pressure", "saturation_transport", "fusion", "cross_scale", "benchmark_registry"} <= set(performance_summary["memory_summary"])


def test_slowest_stage_recorded(performance_summary):
    slowest = performance_summary["slowest_stage"]
    assert slowest["stage_name"] in performance_summary["runtime_summary"]
    assert slowest["runtime_sec"] >= 0.0


def test_numerical_equivalence_success(performance_summary):
    equivalence = performance_summary["numerical_equivalence"]
    assert equivalence["success"] is True
    assert equivalence["max_abs_error"] <= 1.0e-12


def test_report_success_true(performance_summary):
    assert performance_summary["success"] is True
    assert performance_summary["has_nan"] is False
    assert performance_summary["has_inf"] is False


def test_no_numba_used(performance_summary):
    assert performance_summary["runtime_environment"]["numba_used"] is False


def test_no_cpp_used(performance_summary):
    assert performance_summary["runtime_environment"]["cpp_used"] is False
    assert performance_summary["runtime_environment"]["pybind11_used"] is False


def test_numba_not_recommended_for_current_baseline(performance_summary):
    assert performance_summary["numba_recommended"] is False
    assert "not recommended" in performance_summary["numba_recommendation"]


def test_cpp_not_recommended_for_current_baseline(performance_summary):
    assert performance_summary["cpp_recommended"] is False
    assert "not recommended" in performance_summary["cpp_recommendation"]


def test_summary_json_serializable(performance_summary):
    json.dumps(performance_summary)


def test_performance_reports_generated(tmp_path):
    summary = run_performance_baseline(tmp_path)
    assert summary["success"] is True
    assert (tmp_path / "performance_baseline_summary.json").exists()
    assert (tmp_path / "performance_baseline_summary.md").exists()


def test_markdown_mentions_limitations(tmp_path):
    run_performance_baseline(tmp_path)
    text = (tmp_path / "performance_baseline_summary.md").read_text(encoding="utf-8")
    assert "Limitations" in text
    assert "does not implement C++" in text


def test_static_accuracy_report_exists():
    assert (ROOT / "accuracy_reports" / "performance_baseline_summary.json").exists()
    assert (ROOT / "accuracy_reports" / "performance_baseline_summary.md").exists()


def test_static_accuracy_report_success_true():
    data = json.loads((ROOT / "accuracy_reports" / "performance_baseline_summary.json").read_text(encoding="utf-8"))
    assert data["success"] is True


def test_docs_performance_baseline_exists():
    text = (ROOT / "docs" / "performance_baseline.md").read_text(encoding="utf-8")
    assert "Performance Baseline" in text
    assert "small / medium / large synthetic cases" in text


def test_docs_mentions_no_cpp_numba_migration_now():
    text = (ROOT / "docs" / "performance_baseline.md").read_text(encoding="utf-8")
    assert "No C++ kernels are implemented" in text
    assert "No numba kernels are introduced" in text
    assert "not recommended" in text


def test_readme_mentions_performance_baseline():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "performance baseline" in text
    assert "performance_baseline_summary" in text


def test_module_matrix_mentions_task_019():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "TASK-019" in text
    assert "performance baseline" in text


def test_traceability_mentions_task_019():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "TASK-019" in text
    assert "performance baseline" in text


def test_limitations_mentions_cpp_deferred_by_baseline():
    text = (ROOT / "docs" / "limitations_and_roadmap.md").read_text(encoding="utf-8")
    assert "TASK-019" in text
    assert "C++" in text


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
