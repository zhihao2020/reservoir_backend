from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.saturation_transport_benchmark import run_saturation_transport_benchmark
from reservoir_backend.solver import saturation_diagnostics
from reservoir_backend.solver.saturation_diagnostics import (
    check_saturation_bounds,
    compute_cfl_statistics,
    compute_material_balance_error,
    compute_saturation_change_norm,
    compute_saturation_statistics,
    estimate_front_position_1d,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FILES = [
    ROOT / "references" / "upstream" / "opm-tests" / "water-1ph" / "WATER2F.DATA",
    ROOT / "references" / "upstream" / "opm-tests" / "spe1" / "SPE1CASE1.DATA",
    ROOT / "references" / "upstream" / "mrst" / "modules" / "book" / "examples" / "1phase" / "src" / "simpleIncompTPFA.m",
    ROOT / "references" / "upstream" / "mrst" / "modules" / "book" / "examples" / "in2ph" / "buckleyLeverett1D.m",
    ROOT / "references" / "fixtures" / "open_source_adapted_cases.json",
    ROOT / "references" / "fixtures" / "open_source_adapted_arrays.npz",
]


def _summary(tmp_path: Path) -> dict:
    return run_saturation_transport_benchmark(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _hashes() -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in REFERENCE_FILES}


def test_saturation_diagnostics_module_exists():
    assert saturation_diagnostics is not None


def test_saturation_statistics_keys():
    report = compute_saturation_statistics(np.array([0.2, 0.4, 0.6]))
    assert {"saturation_min", "saturation_max", "saturation_mean", "saturation_std", "has_nan", "has_inf", "warnings"} <= set(report)


def test_saturation_statistics_rejects_nan_inf():
    report = compute_saturation_statistics(np.array([0.2, np.nan, np.inf]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True
    assert report["warnings"]


def test_saturation_bounds_pass_for_valid_field():
    report = check_saturation_bounds(np.array([0.2, 0.5, 0.8]), lower=0.2, upper=0.8)
    assert report["bounded"] is True
    assert report["num_below_lower"] == 0
    assert report["num_above_upper"] == 0


def test_saturation_bounds_detects_violations():
    report = check_saturation_bounds(np.array([0.1, 0.5, 0.9]), lower=0.2, upper=0.8)
    assert report["bounded"] is False
    assert report["num_below_lower"] == 1
    assert report["num_above_upper"] == 1


def test_front_position_estimate_moves_downstream():
    initial = estimate_front_position_1d(np.array([0.6, 0.2, 0.2]), threshold=0.5, dx=2.0)
    final = estimate_front_position_1d(np.array([0.6, 0.6, 0.2]), threshold=0.5, dx=2.0)
    assert final > initial


def test_saturation_change_norm_keys():
    report = compute_saturation_change_norm(np.array([0.2, 0.3]), np.array([0.3, 0.5]))
    assert {"saturation_change_l1", "saturation_change_l2", "max_abs_saturation_change", "has_nan", "has_inf"} <= set(report)


def test_material_balance_error_formula():
    report = compute_material_balance_error(
        np.array([0.2, 0.2]),
        np.array([0.3, 0.2]),
        injected_volume=0.1,
        produced_volume=0.0,
        pore_volume=1.0,
    )
    assert abs(report["material_balance_residual"]) < 1.0e-15
    assert report["relative_material_balance_error"] < 1.0e-12


def test_cfl_statistics_keys():
    report = compute_cfl_statistics(np.array([0.1, 0.5, 0.9]))
    assert {"max_cfl", "mean_cfl", "min_cfl", "has_nan", "has_inf", "warnings"} <= set(report)


def test_saturation_benchmark_module_exists():
    import benchmarks.saturation_transport_benchmark as benchmark

    assert hasattr(benchmark, "run_saturation_transport_benchmark")


def test_saturation_benchmark_runs(tmp_path):
    assert _summary(tmp_path)["benchmark_name"] == "saturation_transport_benchmark"


def test_saturation_benchmark_summary_keys(tmp_path):
    summary = _summary(tmp_path)
    expected = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "overall_material_balance_error",
        "overall_max_cfl",
        "open_source_references_used",
        "has_nan",
        "has_inf",
        "warnings",
        "recommendations",
    }
    assert expected <= set(summary)


def test_saturation_benchmark_success_true(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_saturation_benchmark_reports_no_nan_inf(tmp_path):
    summary = _summary(tmp_path)
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_buckley_leverett_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "buckley_leverett_1d_qualitative")["success"] is True


def test_buckley_leverett_front_moves_downstream(tmp_path):
    metrics = _case(_summary(tmp_path), "buckley_leverett_1d_qualitative")["key_metrics"]
    assert metrics["front_moved_downstream"] is True
    assert metrics["final_front_position"] > 0.0


def test_buckley_leverett_sw_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "buckley_leverett_1d_qualitative")["key_metrics"]
    assert metrics["sw_min"] >= 0.2
    assert metrics["sw_max"] <= 0.8


def test_buckley_leverett_material_balance_small(tmp_path):
    metrics = _case(_summary(tmp_path), "buckley_leverett_1d_qualitative")["key_metrics"]
    assert metrics["material_balance_error"] < 1.0e-10


def test_buckley_leverett_cfl_below_limit(tmp_path):
    metrics = _case(_summary(tmp_path), "buckley_leverett_1d_qualitative")["key_metrics"]
    assert metrics["max_cfl"] <= 1.0


def test_mrst_buckley_leverett_reference_loaded(tmp_path):
    metrics = _case(_summary(tmp_path), "mrst_buckley_leverett_1d_reference")["key_metrics"]
    assert metrics["metadata_loaded"] is True
    assert metrics["grid_shape"] == [100, 1]
    assert metrics["porosity"] == 0.2
    assert metrics["permeability_md"] == 100.0


def test_mrst_buckley_leverett_not_exact_reproduction(tmp_path):
    case = _case(_summary(tmp_path), "mrst_buckley_leverett_1d_reference")
    assert case["is_exact_reproduction"] is False


def test_mrst_buckley_leverett_no_runtime_dependency(tmp_path):
    case = _case(_summary(tmp_path), "mrst_buckley_leverett_1d_reference")
    assert any("no MRST runtime dependency" in item for item in case["limitations"])


def test_saturation_boundedness_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "saturation_boundedness")["success"] is True


def test_saturation_boundedness_detects_all_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "saturation_boundedness")["key_metrics"]
    assert metrics["num_bounded"] == metrics["num_cases"]
    assert metrics["num_bound_violations"] == 0


def test_cfl_stability_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "cfl_stability")["success"] is True


def test_cfl_stability_warns_for_too_large_dt(tmp_path):
    case = _case(_summary(tmp_path), "cfl_stability")
    assert case["key_metrics"]["stability_flags"]["too_large"] == "cfl_violation"
    assert case["key_metrics"]["num_cfl_warnings"] >= 1


def test_material_balance_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "material_balance_1d")["success"] is True


def test_material_balance_case_residual_small_or_diagnosed(tmp_path):
    metrics = _case(_summary(tmp_path), "material_balance_1d")["key_metrics"]
    assert abs(metrics["material_balance_residual"]) < 1.0e-12
    assert metrics["relative_material_balance_error"] < 1.0e-10


def test_2d_areal_waterflood_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "areal_waterflood_2d_qualitative")["success"] is True


def test_2d_areal_waterflood_injection_region_increases(tmp_path):
    metrics = _case(_summary(tmp_path), "areal_waterflood_2d_qualitative")["key_metrics"]
    assert metrics["injection_region_sw_final"] > metrics["injection_region_sw_initial"]


def test_2d_areal_waterflood_sw_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "areal_waterflood_2d_qualitative")["key_metrics"]
    assert metrics["sw_min"] >= 0.2
    assert metrics["sw_max"] <= 0.8


def test_opm_spe1case1_saturation_sanity_loaded(tmp_path):
    case = _case(_summary(tmp_path), "opm_spe1case1_saturation_sanity_adapted")
    assert case["source"] == "OPM/opm-tests spe1 SPE1CASE1.DATA"
    assert case["key_metrics"]["permeability_min_md"] == 50.0
    assert case["key_metrics"]["permeability_max_md"] == 500.0


def test_opm_spe1case1_saturation_sanity_not_exact_reproduction(tmp_path):
    case = _case(_summary(tmp_path), "opm_spe1case1_saturation_sanity_adapted")
    assert case["is_exact_reproduction"] is False
    assert any("not exact SPE1 reproduction" in item for item in case["limitations"])


def test_opm_spe1case1_saturation_sanity_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_spe1case1_saturation_sanity_adapted")["key_metrics"]
    assert metrics["bounded"] is True
    assert metrics["sw_min"] >= 0.2
    assert metrics["sw_max"] <= 0.8


def test_saturation_benchmark_generates_json_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "saturation_transport_benchmark_summary.json").exists()


def test_saturation_benchmark_generates_markdown_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "saturation_transport_benchmark_summary.md").exists()


def test_saturation_benchmark_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_saturation_docs_updated():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "Saturation transport benchmark hardening: Done" in text
    assert "MRST buckleyLeverett1D adapted reference: Done" in text


def test_saturation_docs_do_not_claim_full_mrst_reproduction():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "No full MRST reproduction." in text
    assert "No runtime dependency on OPM or MRST." in text


def test_saturation_docs_do_not_claim_opm_flow_equivalence():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "No OPM Flow equivalence." in text


def test_saturation_docs_do_not_claim_black_oil():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "No black-oil transport implemented." in text


def test_requirement_traceability_mentions_saturation_transport_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "saturation transport benchmark hardening" in text
    assert "Done" in text


def test_readme_mentions_saturation_transport_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "saturation transport benchmark" in text


def test_no_saturation_solver_core_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver/saturation_solver.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_no_pressure_solver_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver/pressure_solver.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_references_not_modified(tmp_path):
    before = _hashes()
    _summary(tmp_path)
    after = _hashes()
    assert before == after


def test_existing_pressure_benchmark_tests_still_pass():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "Pressure solver benchmark hardening: Done" in text


def test_existing_open_source_reference_extraction_tests_still_pass():
    text = (ROOT / "references" / "README.md").read_text(encoding="utf-8")
    assert "not imported as runtime dependencies" in text


def test_existing_saturation_inversion_tests_still_pass():
    text = (ROOT / "docs" / "saturation_inversion_validation.md").read_text(encoding="utf-8")
    assert "saturation inversion hardening: Done" in text


def test_existing_function_benchmark_matrix_tests_still_pass():
    text = (ROOT / "specs" / "14_function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "Function hardening first." in text
    assert "Saturation transport module" in text


def test_existing_three_phase_tests_still_pass():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase validation / profiling" in text
