from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.three_phase_benchmark import run_three_phase_benchmark
from reservoir_backend.solver import three_phase_diagnostics
from reservoir_backend.solver.three_phase_diagnostics import (
    build_three_phase_diagnostics_report,
    check_three_phase_bounds,
    compute_fractional_flow_closure_metrics,
    compute_phase_flux_statistics,
    compute_three_phase_closure_error,
    compute_three_phase_mobility_metrics,
    compute_three_phase_relperm_metrics,
    compute_three_phase_saturation_statistics,
    compute_three_phase_transport_metrics,
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


def _params(**overrides: float) -> dict[str, float]:
    params = {
        "swi": 0.2,
        "sor": 0.2,
        "sgc": 0.05,
        "krw0": 0.3,
        "kro0": 0.8,
        "krg0": 0.6,
        "nw": 2.0,
        "no": 2.0,
        "ng": 2.0,
        "mu_w": 1.0e-3,
        "mu_o": 5.0e-3,
        "mu_g": 1.0e-5,
    }
    params.update(overrides)
    return params


def _state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sw = np.array([0.25, 0.35, 0.45])
    sg = np.array([0.08, 0.12, 0.16])
    so = 1.0 - sw - sg
    return sw, so, sg


def _summary(tmp_path: Path) -> dict:
    return run_three_phase_benchmark(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _hashes() -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in REFERENCE_FILES}


def test_three_phase_diagnostics_module_exists():
    assert three_phase_diagnostics is not None


def test_saturation_statistics_keys():
    sw, so, sg = _state()
    report = compute_three_phase_saturation_statistics(sw, so, sg)
    assert {"sw_min", "so_min", "sg_min", "closure_max_abs_error", "has_nan", "has_inf", "warnings"} <= set(report)


def test_saturation_statistics_no_nan_inf_for_valid_state():
    sw, so, sg = _state()
    report = compute_three_phase_saturation_statistics(sw, so, sg)
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_saturation_statistics_reports_nan_inf():
    report = compute_three_phase_saturation_statistics(np.array([0.2, np.nan]), np.array([0.7, 0.6]), np.array([0.1, np.inf]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True
    assert report["warnings"]


def test_closure_error_small_for_exact_closure():
    sw, so, sg = _state()
    report = compute_three_phase_closure_error(sw, so, sg)
    assert report["closure_max_abs_error"] <= 1.0e-12


def test_closure_error_positive_for_mismatch():
    sw, so, sg = _state()
    report = compute_three_phase_closure_error(sw, so + 0.01, sg)
    assert report["closure_max_abs_error"] > 0.0


def test_bounds_pass_for_valid_state():
    sw, so, sg = _state()
    assert check_three_phase_bounds(sw, so, sg)["success"] is True


def test_bounds_detects_violations():
    report = check_three_phase_bounds(np.array([-0.1]), np.array([0.5]), np.array([0.6]))
    assert report["num_bound_violations"] > 0


def test_relperm_metrics_keys():
    sw, _, sg = _state()
    report = compute_three_phase_relperm_metrics(sw, sg, _params())
    assert {"krw_min", "kro_min", "krg_min", "krw_nonnegative", "has_nan", "has_inf"} <= set(report)


def test_relperm_metrics_nonnegative():
    sw, _, sg = _state()
    report = compute_three_phase_relperm_metrics(sw, sg, _params())
    assert report["krw_nonnegative"] is True
    assert report["kro_nonnegative"] is True
    assert report["krg_nonnegative"] is True


def test_mobility_metrics_keys():
    sw, _, sg = _state()
    report = compute_three_phase_mobility_metrics(sw, sg, _params())
    assert {"lambda_w_min", "lambda_o_min", "lambda_g_min", "lambda_total_min", "lambda_total_positive"} <= set(report)


def test_mobility_total_positive():
    sw, _, sg = _state()
    assert compute_three_phase_mobility_metrics(sw, sg, _params())["lambda_total_positive"] is True


def test_fractional_flow_closure_metrics_keys():
    report = compute_fractional_flow_closure_metrics(np.array([0.2]), np.array([0.3]), np.array([0.5]))
    assert {"fw_min", "fo_min", "fg_min", "fractional_flow_sum_error", "has_nan", "has_inf"} <= set(report)


def test_fractional_flow_closure_error_zero():
    report = compute_fractional_flow_closure_metrics(np.array([0.2]), np.array([0.3]), np.array([0.5]))
    assert report["fractional_flow_sum_error"] == 0.0


def test_phase_flux_statistics_keys():
    report = compute_phase_flux_statistics(np.array([1.0]), np.array([2.0]), np.array([3.0]))
    assert {"max_abs_water_flux", "max_abs_oil_flux", "max_abs_gas_flux", "water_flux_shape"} <= set(report)


def test_phase_flux_statistics_reports_nan_inf():
    report = compute_phase_flux_statistics(np.array([np.nan]), np.array([np.inf]), np.array([0.0]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True


def test_transport_metrics_keys():
    initial = {"sw": np.array([0.3]), "sg": np.array([0.1])}
    final = {"sw": np.array([0.31]), "sg": np.array([0.09])}
    report = compute_three_phase_transport_metrics(initial, final)
    assert {"closure_max_abs_error", "num_bound_violations", "saturation_change_l1", "saturation_change_l2"} <= set(report)


def test_diagnostics_report_keys():
    sw, _, sg = _state()
    report = build_three_phase_diagnostics_report(sw, sg, _params())
    expected = {
        "success",
        "sw_min",
        "sw_max",
        "so_min",
        "so_max",
        "sg_min",
        "sg_max",
        "closure_max_abs_error",
        "closure_l2_error",
        "num_bound_violations",
        "krw_min",
        "krw_max",
        "kro_min",
        "kro_max",
        "krg_min",
        "krg_max",
        "lambda_total_min",
        "lambda_total_max",
        "fractional_flow_sum_error",
        "max_abs_water_flux",
        "max_abs_oil_flux",
        "max_abs_gas_flux",
        "has_nan",
        "has_inf",
        "warnings",
    }
    assert expected <= set(report)


def test_diagnostics_report_success_true():
    sw, _, sg = _state()
    assert build_three_phase_diagnostics_report(sw, sg, _params())["success"] is True


def test_three_phase_benchmark_module_exists():
    import benchmarks.three_phase_benchmark as benchmark

    assert hasattr(benchmark, "run_three_phase_benchmark")


def test_three_phase_benchmark_runs(tmp_path):
    assert _summary(tmp_path)["benchmark_name"] == "three_phase_benchmark"


def test_three_phase_benchmark_summary_keys(tmp_path):
    summary = _summary(tmp_path)
    expected = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "overall_max_closure_error",
        "overall_num_bound_violations",
        "overall_fractional_flow_sum_error",
        "overall_max_phase_flux",
        "has_nan",
        "has_inf",
        "warnings",
        "recommendations",
    }
    assert expected <= set(summary)


def test_three_phase_benchmark_success_true(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_three_phase_benchmark_reports_no_nan_inf(tmp_path):
    summary = _summary(tmp_path)
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_three_phase_saturation_closure_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "three_phase_saturation_closure")["success"] is True


def test_three_phase_saturation_closure_error_small(tmp_path):
    metrics = _case(_summary(tmp_path), "three_phase_saturation_closure")["key_metrics"]
    assert metrics["closure_max_abs_error"] <= 1.0e-12


def test_residual_saturation_bounds_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "residual_saturation_bounds")["success"] is True


def test_residual_saturation_bounds_no_violations(tmp_path):
    metrics = _case(_summary(tmp_path), "residual_saturation_bounds")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert metrics["residual_violations"] == 0


def test_relperm_endpoint_sanity_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "three_phase_relperm_endpoint_sanity")["success"] is True


def test_relperm_endpoint_sanity_metrics(tmp_path):
    metrics = _case(_summary(tmp_path), "three_phase_relperm_endpoint_sanity")["key_metrics"]
    assert metrics["krw_endpoint"] == 0.0
    assert metrics["krg_endpoint"] == 0.0
    assert metrics["krw_monotonicity_score"] == 1.0
    assert metrics["krg_monotonicity_score"] == 1.0


def test_mobility_fractional_flow_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "phase_mobility_fractional_flow_consistency")["success"] is True


def test_mobility_fractional_flow_closure(tmp_path):
    metrics = _case(_summary(tmp_path), "phase_mobility_fractional_flow_consistency")["key_metrics"]
    assert metrics["lambda_total_positive"] is True
    assert metrics["fractional_flow_sum_error"] <= 1.0e-12


def test_phase_flux_finite_shape_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "phase_flux_finite_shape_consistency")["success"] is True


def test_phase_flux_shape_consistency(tmp_path):
    metrics = _case(_summary(tmp_path), "phase_flux_finite_shape_consistency")["key_metrics"]
    assert metrics["flux_shape_x"] == [2, 3, 5]
    assert metrics["flux_shape_y"] == [2, 4, 4]
    assert metrics["flux_shape_z"] == [3, 3, 4]


def test_phase_flux_closure_small(tmp_path):
    metrics = _case(_summary(tmp_path), "phase_flux_finite_shape_consistency")["key_metrics"]
    assert metrics["phase_flux_closure_error_max"] <= 1.0e-18


def test_three_phase_1d_transport_boundedness_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "three_phase_1d_transport_boundedness")["success"] is True


def test_three_phase_1d_transport_bounds(tmp_path):
    metrics = _case(_summary(tmp_path), "three_phase_1d_transport_boundedness")["key_metrics"]
    assert metrics["num_bound_violations"] == 0
    assert metrics["sw_min"] >= _params()["swi"]
    assert metrics["sg_min"] >= _params()["sgc"]
    assert metrics["so_min"] >= _params()["sor"]


def test_three_phase_1d_transport_cfl(tmp_path):
    metrics = _case(_summary(tmp_path), "three_phase_1d_transport_boundedness")["key_metrics"]
    assert metrics["max_cfl"] <= 1.0


def test_three_phase_3d_transport_closure_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "three_phase_3d_transport_closure")["success"] is True


def test_three_phase_3d_transport_closure_small(tmp_path):
    metrics = _case(_summary(tmp_path), "three_phase_3d_transport_closure")["key_metrics"]
    assert metrics["closure_max_abs_error"] <= 1.0e-12
    assert metrics["num_bound_violations"] == 0


def test_production_summary_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "production_summary_consistency")["success"] is True


def test_production_summary_rates_nonnegative(tmp_path):
    metrics = _case(_summary(tmp_path), "production_summary_consistency")["key_metrics"]
    assert metrics["water_rate"] >= 0.0
    assert metrics["oil_rate"] >= 0.0
    assert metrics["gas_rate"] >= 0.0


def test_production_summary_json_serializable(tmp_path):
    metrics = _case(_summary(tmp_path), "production_summary_consistency")["key_metrics"]
    assert metrics["summary_json_serializable"] is True
    json.dumps(metrics)


def test_three_phase_benchmark_generates_json_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "three_phase_benchmark_summary.json").exists()


def test_three_phase_benchmark_generates_markdown_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "three_phase_benchmark_summary.md").exists()


def test_three_phase_benchmark_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_three_phase_docs_updated():
    text = (ROOT / "docs" / "three_phase_validation.md").read_text(encoding="utf-8")
    assert "Three-phase WOG benchmark hardening: Done" in text
    assert "production summary consistency benchmark: Done" in text


def test_three_phase_docs_do_not_claim_black_oil():
    text = (ROOT / "docs" / "three_phase_validation.md").read_text(encoding="utf-8")
    assert "Current model is simplified incompressible WOG, not black-oil." in text
    assert "No PVT table implemented." in text


def test_three_phase_docs_do_not_claim_opm_flow_equivalence():
    text = (ROOT / "docs" / "three_phase_validation.md").read_text(encoding="utf-8")
    assert "No OPM Flow equivalence." in text


def test_three_phase_docs_do_not_claim_commercial_equivalence():
    text = (ROOT / "docs" / "three_phase_validation.md").read_text(encoding="utf-8")
    assert "No commercial simulator equivalence." in text


def test_requirement_traceability_mentions_three_phase_benchmark_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "three-phase WOG benchmark hardening" in text
    assert "tests/test_three_phase_benchmark_hardening.py" in text


def test_readme_mentions_three_phase_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python benchmarks/three_phase_benchmark.py" in text
    assert "Three-phase WOG benchmark hardening" in text


def test_numerical_accuracy_mentions_three_phase_benchmark():
    text = (ROOT / "docs" / "numerical_accuracy.md").read_text(encoding="utf-8")
    assert "three_phase_benchmark" in text
    assert "accuracy_reports/three_phase_benchmark_summary.json" in text


def test_function_benchmark_matrix_mentions_task_050():
    text = (ROOT / "docs" / "function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "050 Three-Phase WOG Benchmark Hardening" in text


def test_module_matrix_mentions_three_phase_benchmark():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase WOG benchmark hardening" in text


def test_no_three_phase_core_solver_modification():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "reservoir_backend/solver/three_phase_relperm.py",
            "reservoir_backend/solver/three_phase_flux.py",
            "reservoir_backend/solver/three_phase_transport.py",
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


def test_existing_capillary_gravity_benchmark_still_passes():
    text = (ROOT / "docs" / "capillary_gravity_validation.md").read_text(encoding="utf-8")
    assert "Capillary / gravity benchmark hardening: Done" in text


def test_existing_saturation_transport_tests_still_pass():
    text = (ROOT / "docs" / "saturation_transport_validation.md").read_text(encoding="utf-8")
    assert "Saturation transport benchmark hardening: Done" in text


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
    text = (ROOT / "docs" / "function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "049 Capillary / Gravity Benchmark Hardening" in text


def test_existing_three_phase_pipeline_tests_still_pass():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase validation / profiling" in text
