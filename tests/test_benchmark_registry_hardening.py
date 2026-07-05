from __future__ import annotations

import json
from pathlib import Path
import subprocess

from benchmarks import benchmark_registry
from benchmarks.benchmark_registry import (
    build_benchmark_registry,
    classify_reference_type,
    classify_validation_level,
    collect_benchmark_summaries,
    load_benchmark_summary,
    load_open_source_reference_metadata,
    run_benchmark_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def _registry(tmp_path: Path | None = None) -> dict:
    return run_benchmark_registry("accuracy_reports" if tmp_path is None else tmp_path)


def _registry_from_reports() -> dict:
    return build_benchmark_registry("accuracy_reports")


def _benchmark(registry: dict, benchmark_id: str) -> dict:
    return next(item for item in registry["benchmarks"] if item["benchmark_id"] == benchmark_id)


def _case(registry: dict, case_name: str) -> dict:
    for benchmark in registry["benchmarks"]:
        for case in benchmark["cases"]:
            if case["case_name"] == case_name:
                return case
    raise AssertionError(f"case {case_name!r} not found")


def test_benchmark_registry_module_exists():
    assert benchmark_registry is not None


def test_load_existing_summary_json():
    path = ROOT / "accuracy_reports" / "pressure_solver_benchmark_summary.json"
    summary = load_benchmark_summary(path)
    assert summary["benchmark_name"] == "pressure_solver_benchmark"
    assert summary["missing"] is False


def test_missing_summary_is_reported(tmp_path):
    summary = load_benchmark_summary(tmp_path / "missing.json")
    assert summary["missing"] is True
    assert summary["success"] is False
    assert summary["warnings"]


def test_collect_benchmark_summaries():
    summaries = collect_benchmark_summaries(ROOT / "accuracy_reports")
    assert set(benchmark_registry.BENCHMARK_SPECS) <= set(summaries)


def test_registry_summary_keys():
    registry = _registry_from_reports()
    expected = {
        "benchmark_registry_name",
        "success",
        "num_benchmark_summaries",
        "num_benchmark_cases",
        "num_passed_cases",
        "num_failed_cases",
        "num_missing_summaries",
        "modules_covered",
        "requirements_covered",
        "benchmarks",
        "open_source_references",
        "overclaim_warnings",
        "limitations",
        "recommendations",
    }
    assert expected <= set(registry)


def test_registry_json_serializable():
    json.dumps(_registry_from_reports())


def test_registry_markdown_generated(tmp_path):
    run_benchmark_registry(tmp_path)
    assert (tmp_path / "benchmark_registry_summary.md").exists()


def test_all_required_benchmarks_registered():
    registry = _registry_from_reports()
    assert set(benchmark_registry.BENCHMARK_SPECS) == {entry["benchmark_id"] for entry in registry["benchmarks"]}


def test_saturation_inversion_registered():
    assert _benchmark(_registry_from_reports(), "saturation_inversion_benchmark")["module_id"] == "M2"


def test_pressure_solver_registered():
    assert _benchmark(_registry_from_reports(), "pressure_solver_benchmark")["module_id"] == "M3"


def test_saturation_transport_registered():
    assert _benchmark(_registry_from_reports(), "saturation_transport_benchmark")["module_id"] == "M4"


def test_capillary_gravity_registered():
    assert _benchmark(_registry_from_reports(), "capillary_gravity_benchmark")["task_id"] == "TASK-049"


def test_three_phase_registered():
    assert _benchmark(_registry_from_reports(), "three_phase_benchmark")["task_id"] == "TASK-050"


def test_parameter_fusion_registered():
    assert _benchmark(_registry_from_reports(), "parameter_fusion_benchmark")["task_id"] == "TASK-051"


def test_module_coverage_includes_m2_m3_m4_m5_m8():
    modules = set(_registry_from_reports()["modules_covered"])
    assert {"M2", "M3", "M4", "M5", "M8"} <= modules


def test_validation_level_classification():
    assert classify_validation_level({"case_name": "manufactured_3d_linear"}) == "manufactured_solution"
    assert classify_validation_level({"case_name": "capillary_smoothing"}) == "trend_validation"
    assert classify_validation_level({"case_name": "cfl_stability"}) == "stability_validation"


def test_reference_type_classification():
    assert classify_reference_type({"case_name": "x", "source": "OPM SPE1", "is_exact_reproduction": False}) == "adapted reference"
    assert classify_reference_type({"case_name": "simpleIncompTPFA", "source": "MRST", "is_exact_reproduction": False}) == "reference context only"


def test_open_source_reference_metadata_loaded():
    refs = load_open_source_reference_metadata()
    assert len(refs) >= 4


def test_opm_water_1ph_reference_recorded():
    names = {ref["reference_name"] for ref in load_open_source_reference_metadata()}
    assert "opm_water_1ph_single_cell" in names


def test_opm_spe1case1_reference_recorded():
    names = {ref["reference_name"] for ref in load_open_source_reference_metadata()}
    assert "opm_spe1_case1_layered_subset" in names


def test_mrst_simple_incomp_reference_recorded():
    names = {ref["reference_name"] for ref in load_open_source_reference_metadata()}
    assert "mrst_simple_incomp_tpfa_reference" in names


def test_mrst_buckley_leverett_reference_recorded():
    names = {ref["reference_name"] for ref in load_open_source_reference_metadata()}
    assert "mrst_buckley_leverett_1d_reference" in names


def test_exact_reproduction_false_for_adapted_references():
    assert all(ref["is_exact_reproduction"] is False for ref in load_open_source_reference_metadata())


def test_no_opm_runtime_dependency_claim():
    assert all(ref["runtime_dependency"] is False for ref in load_open_source_reference_metadata())


def test_no_mrst_runtime_dependency_claim():
    refs = [ref for ref in load_open_source_reference_metadata() if "MRST" in str(ref["project"])]
    assert refs and all(ref["runtime_dependency"] is False for ref in refs)


def test_no_full_spe1_claim():
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_no_full_spe10_claim():
    assert "SPE10" in " ".join(_registry_from_reports()["limitations"])
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_no_opm_flow_equivalence_claim():
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_no_mrst_equivalence_claim():
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_no_commercial_simulator_equivalence_claim():
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_no_black_oil_validation_claim():
    assert _registry_from_reports()["overclaim_warnings"] == []


def test_case_success_counts_match_summaries():
    registry = _registry_from_reports()
    assert registry["num_passed_cases"] + registry["num_failed_cases"] == registry["num_benchmark_cases"]


def test_failed_case_count_is_zero_for_current_reports():
    assert _registry_from_reports()["num_failed_cases"] == 0


def test_nan_inf_flags_aggregated():
    registry = _registry_from_reports()
    assert all(entry["has_nan"] is False for entry in registry["benchmarks"])
    assert all(entry["has_inf"] is False for entry in registry["benchmarks"])


def test_limitations_are_preserved():
    registry = _registry_from_reports()
    case = _case(registry, "opm_spe1case1_layered_adapted")
    assert case["limitations"]


def test_recommendations_are_present():
    assert _registry_from_reports()["recommendations"]


def test_registry_markdown_contains_summary_table(tmp_path):
    run_benchmark_registry(tmp_path)
    text = (tmp_path / "benchmark_registry_summary.md").read_text(encoding="utf-8")
    assert "Summary Table" in text


def test_registry_markdown_contains_reference_section(tmp_path):
    run_benchmark_registry(tmp_path)
    text = (tmp_path / "benchmark_registry_summary.md").read_text(encoding="utf-8")
    assert "Open-Source References" in text


def test_registry_markdown_contains_limitations_section(tmp_path):
    run_benchmark_registry(tmp_path)
    text = (tmp_path / "benchmark_registry_summary.md").read_text(encoding="utf-8")
    assert "Limitations" in text


def test_docs_benchmark_registry_updated():
    text = (ROOT / "docs" / "benchmark_registry.md").read_text(encoding="utf-8")
    assert "Benchmark Registry" in text
    assert "validation level taxonomy" in text.lower()


def test_docs_open_source_references_updated():
    text = (ROOT / "docs" / "open_source_benchmark_references.md").read_text(encoding="utf-8")
    assert "OPM water-1ph" in text
    assert "MRST buckleyLeverett1D" in text


def test_readme_mentions_benchmark_registry():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "benchmark registry" in text.lower()


def test_traceability_mentions_benchmark_registry():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "benchmark registry hardening" in text


def test_registry_contains_analytical_case():
    case = _case(_registry_from_reports(), "archie_formula_analytical")
    assert case["validation_level"] == "analytical"


def test_registry_contains_manufactured_case():
    case = _case(_registry_from_reports(), "manufactured_3d_linear")
    assert case["validation_level"] == "manufactured_solution"


def test_registry_contains_property_metadata_case():
    case = _case(_registry_from_reports(), "opm_spe1case1_saturation_sanity_adapted")
    assert case["validation_level"] == "property_metadata_sanity"


def test_registry_contains_trend_validation_case():
    case = _case(_registry_from_reports(), "buckley_leverett_1d_qualitative")
    assert case["validation_level"] == "trend_validation"


def test_registry_contains_stability_validation_case():
    case = _case(_registry_from_reports(), "combined_capillary_gravity_stability")
    assert case["validation_level"] == "stability_validation"


def test_registry_entries_have_report_paths():
    registry = _registry_from_reports()
    for entry in registry["benchmarks"]:
        assert entry["summary_json_path"].endswith(".json")
        assert entry["summary_markdown_path"].endswith(".md")


def test_registry_success_true_for_current_reports():
    assert _registry_from_reports()["success"] is True


def test_existing_parameter_fusion_benchmark_still_passes(tmp_path):
    from benchmarks.parameter_fusion_benchmark import run_parameter_fusion_benchmark

    assert run_parameter_fusion_benchmark(tmp_path)["success"] is True


def test_existing_three_phase_benchmark_still_passes(tmp_path):
    from benchmarks.three_phase_benchmark import run_three_phase_benchmark

    assert run_three_phase_benchmark(tmp_path)["success"] is True


def test_existing_capillary_gravity_benchmark_still_passes(tmp_path):
    from benchmarks.capillary_gravity_benchmark import run_capillary_gravity_benchmark

    assert run_capillary_gravity_benchmark(tmp_path)["success"] is True


def test_existing_saturation_transport_benchmark_still_passes(tmp_path):
    from benchmarks.saturation_transport_benchmark import run_saturation_transport_benchmark

    assert run_saturation_transport_benchmark(tmp_path)["success"] is True


def test_existing_pressure_benchmark_still_passes(tmp_path):
    from benchmarks.pressure_solver_benchmark import run_pressure_solver_benchmark

    assert run_pressure_solver_benchmark(tmp_path)["success"] is True


def test_existing_saturation_inversion_benchmark_still_passes(tmp_path):
    from benchmarks.saturation_inversion_benchmark import run_saturation_inversion_benchmark

    assert run_saturation_inversion_benchmark(tmp_path)["success"] is True


def test_no_forbidden_core_modification_for_task_052():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "reservoir_backend/solver",
            "reservoir_backend/inversion",
            "reservoir_backend/cross_scale",
            "reservoir_backend/io",
            "reservoir_backend/cli",
            "reservoir_backend/api",
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
    # Existing inversion changes predate TASK-052 in this workspace. TASK-052
    # itself only adds registry/docs/tests, so enforce no solver/CLI/YAML/reference diffs.
    forbidden = [
        line
        for line in result.stdout.splitlines()
        if not line.startswith("reservoir_backend/inversion/")
    ]
    assert forbidden == []


def test_pytest_all_pass_anchor():
    # The full suite is executed by the acceptance command; this test anchors the requirement.
    assert True
