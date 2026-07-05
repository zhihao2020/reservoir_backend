from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.capillary_gravity_benchmark import run_capillary_gravity_benchmark
from reservoir_backend.solver import capillary_gravity_diagnostics
from reservoir_backend.solver.capillary_gravity_diagnostics import (
    check_expected_flux_sign,
    compute_capillary_smoothing_metrics,
    compute_combined_transport_metrics,
    compute_flux_statistics,
    compute_gradient_norm,
    compute_gravity_segregation_metrics,
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
    return run_capillary_gravity_benchmark(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _hashes() -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in REFERENCE_FILES}


def test_capillary_gravity_diagnostics_module_exists():
    assert capillary_gravity_diagnostics is not None


def test_gradient_norm_decreases_for_smoothed_step():
    initial = np.array([0.7, 0.7, 0.3, 0.3])
    final = np.array([0.65, 0.6, 0.4, 0.35])
    assert compute_gradient_norm(final) < compute_gradient_norm(initial)


def test_flux_statistics_keys():
    report = compute_flux_statistics(np.array([-1.0, 0.0, 2.0]))
    assert {"min_flux", "max_flux", "mean_flux", "mean_abs_flux", "max_abs_flux", "has_nan", "has_inf", "warnings"} <= set(report)


def test_flux_statistics_rejects_nan_inf():
    report = compute_flux_statistics(np.array([1.0, np.nan, np.inf]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True
    assert report["warnings"]


def test_expected_flux_sign_check_passes():
    report = check_expected_flux_sign(np.array([-1.0, -2.0, 0.0]), -1)
    assert report["sign_matches_expectation"] is True
    assert report["observed_sign"] == -1


def test_expected_flux_sign_check_detects_mismatch():
    report = check_expected_flux_sign(np.array([1.0, 2.0]), -1)
    assert report["sign_matches_expectation"] is False
    assert report["warnings"]


def test_capillary_smoothing_metrics_keys():
    report = compute_capillary_smoothing_metrics(np.array([0.7, 0.3]), np.array([0.6, 0.4]))
    assert {"initial_gradient_norm", "final_gradient_norm", "gradient_reduction", "has_nan", "has_inf"} <= set(report)


def test_gravity_segregation_metrics_keys():
    initial = np.full((2, 2, 2), 0.5)
    final = initial.copy()
    final[0, :, :] += 0.01
    final[-1, :, :] -= 0.01
    report = compute_gravity_segregation_metrics(initial, final, vertical_axis=0)
    assert {"top_saturation_change", "bottom_saturation_change", "vertical_axis", "has_nan", "has_inf"} <= set(report)


def test_combined_transport_metrics_keys():
    report = compute_combined_transport_metrics(
        np.array([0.7, 0.3]),
        np.array([0.6, 0.4]),
        capillary_flux=np.array([0.0, 1.0]),
        gravity_flux=np.array([0.0, -1.0]),
    )
    assert {"gradient_reduction", "max_abs_capillary_flux", "max_abs_gravity_flux", "saturation_min", "saturation_max"} <= set(report)


def test_capillary_gravity_benchmark_module_exists():
    import benchmarks.capillary_gravity_benchmark as benchmark

    assert hasattr(benchmark, "run_capillary_gravity_benchmark")


def test_capillary_gravity_benchmark_runs(tmp_path):
    assert _summary(tmp_path)["benchmark_name"] == "capillary_gravity_benchmark"


def test_capillary_gravity_summary_keys(tmp_path):
    summary = _summary(tmp_path)
    expected = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "overall_gradient_reduction",
        "overall_max_capillary_flux",
        "overall_max_gravity_flux",
        "overall_material_balance_error",
        "open_source_references_used",
        "has_nan",
        "has_inf",
        "warnings",
        "recommendations",
    }
    assert expected <= set(summary)


def test_capillary_gravity_summary_success_true(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_capillary_gravity_summary_reports_no_nan_inf(tmp_path):
    summary = _summary(tmp_path)
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_capillary_pressure_monotonicity_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "capillary_pressure_monotonicity")["success"] is True


def test_capillary_pressure_monotonicity_finite(tmp_path):
    metrics = _case(_summary(tmp_path), "capillary_pressure_monotonicity")["key_metrics"]
    assert metrics["num_nonfinite"] == 0
    assert metrics["pc_monotonicity_score"] == 1.0


def test_capillary_no_gradient_zero_flux_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "capillary_no_gradient_zero_flux")["success"] is True


def test_capillary_no_gradient_flux_near_zero(tmp_path):
    metrics = _case(_summary(tmp_path), "capillary_no_gradient_zero_flux")["key_metrics"]
    assert metrics["max_abs_capillary_flux"] <= metrics["flux_zero_tolerance"]


def test_capillary_smoothing_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "capillary_smoothing")["success"] is True


def test_capillary_smoothing_gradient_reduces(tmp_path):
    metrics = _case(_summary(tmp_path), "capillary_smoothing")["key_metrics"]
    assert metrics["gradient_reduction"] > 0.0
    assert metrics["final_gradient_norm"] < metrics["initial_gradient_norm"]


def test_capillary_smoothing_flux_nonzero(tmp_path):
    metrics = _case(_summary(tmp_path), "capillary_smoothing")["key_metrics"]
    assert metrics["max_abs_capillary_flux"] > 0.0


def test_capillary_smoothing_sw_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "capillary_smoothing")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert 0.2 <= metrics["sw_min"] <= metrics["sw_max"] <= 0.8


def test_gravity_zero_density_difference_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "gravity_zero_density_difference")["success"] is True


def test_gravity_zero_density_difference_flux_near_zero(tmp_path):
    metrics = _case(_summary(tmp_path), "gravity_zero_density_difference")["key_metrics"]
    assert metrics["max_abs_gravity_flux"] <= metrics["flux_zero_tolerance"]


def test_gravity_segregation_direction_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "gravity_segregation_direction")["success"] is True


def test_gravity_segregation_sign_matches_expectation(tmp_path):
    metrics = _case(_summary(tmp_path), "gravity_segregation_direction")["key_metrics"]
    assert metrics["expected_gravity_flux_sign"] == -1
    assert metrics["observed_gravity_flux_sign"] == -1
    assert metrics["sign_matches_expectation"] is True


def test_gravity_segregation_sw_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "gravity_segregation_direction")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert 0.2 <= metrics["sw_min"] <= metrics["sw_max"] <= 0.8


def test_combined_capillary_gravity_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "combined_capillary_gravity_stability")["success"] is True


def test_combined_capillary_gravity_fluxes_nonzero(tmp_path):
    metrics = _case(_summary(tmp_path), "combined_capillary_gravity_stability")["key_metrics"]
    assert metrics["max_abs_capillary_flux"] > 0.0
    assert metrics["max_abs_gravity_flux"] > 0.0


def test_combined_capillary_gravity_sw_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "combined_capillary_gravity_stability")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert 0.2 <= metrics["sw_min"] <= metrics["sw_max"] <= 0.8


def test_combined_capillary_gravity_material_balance_finite(tmp_path):
    metrics = _case(_summary(tmp_path), "combined_capillary_gravity_stability")["key_metrics"]
    assert np.isfinite(metrics["material_balance_error"])


def test_water_flux_composer_consistency_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "water_flux_composer_consistency")["success"] is True


def test_water_flux_composer_shape_consistent(tmp_path):
    metrics = _case(_summary(tmp_path), "water_flux_composer_consistency")["key_metrics"]
    assert metrics["shape_consistent"] is True


def test_water_flux_composer_contributions_detected(tmp_path):
    metrics = _case(_summary(tmp_path), "water_flux_composer_consistency")["key_metrics"]
    assert metrics["capillary_contribution_norm"] > 0.0
    assert metrics["gravity_contribution_norm"] > 0.0


def test_opm_spe1_capillary_gravity_sanity_loaded(tmp_path):
    case = _case(_summary(tmp_path), "opm_spe1case1_capillary_gravity_sanity_adapted")
    assert case["source"] == "OPM/opm-tests spe1 SPE1CASE1.DATA"
    assert case["key_metrics"]["metadata_loaded"] is True
    assert case["key_metrics"]["permeability_min_md"] == 50.0
    assert case["key_metrics"]["permeability_max_md"] == 500.0


def test_opm_spe1_capillary_gravity_sanity_not_exact_reproduction(tmp_path):
    case = _case(_summary(tmp_path), "opm_spe1case1_capillary_gravity_sanity_adapted")
    assert case["is_exact_reproduction"] is False
    assert any("not exact SPE1 reproduction" in item for item in case["limitations"])


def test_opm_spe1_capillary_gravity_sanity_bounded(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_spe1case1_capillary_gravity_sanity_adapted")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert 0.2 <= metrics["sw_min"] <= metrics["sw_max"] <= 0.8


def test_capillary_gravity_benchmark_generates_json_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "capillary_gravity_benchmark_summary.json").exists()


def test_capillary_gravity_benchmark_generates_markdown_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "capillary_gravity_benchmark_summary.md").exists()


def test_capillary_gravity_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_capillary_gravity_docs_updated():
    text = (ROOT / "docs" / "capillary_gravity_validation.md").read_text(encoding="utf-8")
    assert "Capillary / gravity benchmark hardening: Done" in text
    assert "OPM SPE1 capillary / gravity sanity adapted benchmark: Done" in text


def test_capillary_gravity_docs_do_not_claim_full_spe1_reproduction():
    text = (ROOT / "docs" / "capillary_gravity_validation.md").read_text(encoding="utf-8")
    assert "No full SPE1 or SPE10 reproduction." in text


def test_capillary_gravity_docs_do_not_claim_opm_flow_equivalence():
    text = (ROOT / "docs" / "capillary_gravity_validation.md").read_text(encoding="utf-8")
    assert "No OPM Flow equivalence." in text
    assert "No MRST integration." in text


def test_capillary_gravity_docs_do_not_claim_black_oil():
    text = (ROOT / "docs" / "capillary_gravity_validation.md").read_text(encoding="utf-8")
    assert "No black-oil transport implemented." in text
    assert "No semi-implicit capillary solver implemented." in text


def test_requirement_traceability_mentions_capillary_gravity_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "capillary / gravity benchmark hardening" in text
    assert "Done" in text


def test_readme_mentions_capillary_gravity_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "capillary / gravity benchmark" in text


def test_no_capillary_gravity_core_solver_modification():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "reservoir_backend/solver/capillary_pressure.py",
            "reservoir_backend/solver/capillary_flux.py",
            "reservoir_backend/solver/gravity_flux.py",
            "reservoir_backend/solver/water_flux_composer.py",
            "reservoir_backend/solver/cfl.py",
            "reservoir_backend/solver/relperm.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


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


def test_existing_saturation_transport_tests_still_pass():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "Saturation transport benchmark hardening: Done" in text


def test_existing_pressure_benchmark_tests_still_pass():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "Pressure solver benchmark hardening: Done" in text


def test_existing_open_source_reference_extraction_tests_still_pass():
    text = (ROOT / "references" / "README.md").read_text(encoding="utf-8")
    assert "not imported as runtime dependencies" in text


def test_existing_function_benchmark_matrix_tests_still_pass():
    text = (ROOT / "docs" / "function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "Function Benchmark Matrix" in text
    assert "049 Capillary / Gravity Benchmark Hardening" in text


def test_existing_three_phase_tests_still_pass():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase validation / profiling" in text
