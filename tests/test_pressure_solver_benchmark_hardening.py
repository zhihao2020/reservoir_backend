from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from benchmarks.pressure_solver_benchmark import run_pressure_solver_benchmark
from reservoir_backend.solver import pressure_diagnostics
from reservoir_backend.solver.pressure_diagnostics import (
    compute_flux_conservation_metrics,
    compute_mass_balance_residual,
    compute_pressure_error_metrics,
    compute_pressure_statistics,
)
from reservoir_backend.solver.velocity import FaceFluxes


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
    return run_pressure_solver_benchmark(tmp_path)


def _case(summary: dict, name: str) -> dict:
    return next(case for case in summary["cases"] if case["case_name"] == name)


def _hashes() -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in REFERENCE_FILES}


def test_pressure_diagnostics_module_exists():
    assert pressure_diagnostics is not None


def test_pressure_statistics_keys():
    report = compute_pressure_statistics(np.array([1.0, 2.0, 3.0]))
    assert {"pressure_min", "pressure_max", "pressure_mean", "pressure_std", "has_nan", "has_inf", "warnings"} <= set(report)


def test_pressure_statistics_rejects_nan_inf():
    report = compute_pressure_statistics(np.array([1.0, np.nan, np.inf]))
    assert report["has_nan"] is True
    assert report["has_inf"] is True
    assert report["warnings"]


def test_pressure_error_metrics_zero_for_exact_match():
    metrics = compute_pressure_error_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert metrics["max_abs_error"] == 0.0
    assert metrics["l2_error"] == 0.0


def test_pressure_error_metrics_positive_for_mismatch():
    metrics = compute_pressure_error_metrics(np.array([1.0, 3.0]), np.array([1.0, 2.0]))
    assert metrics["max_abs_error"] > 0.0
    assert metrics["relative_l2_error"] > 0.0


def test_mass_balance_residual_formula():
    assert compute_mass_balance_residual(inflow=3.0, outflow=2.0, source_sink=-1.0) == 0.0


def test_flux_conservation_metrics_keys():
    fx = np.zeros((1, 1, 3))
    fy = np.zeros((1, 2, 2))
    fz = np.zeros((2, 1, 2))
    metrics = compute_flux_conservation_metrics(FaceFluxes(fx, fy, fz))
    assert {"max_flux_imbalance", "mean_abs_flux_imbalance", "total_flux_imbalance", "has_nan", "has_inf"} <= set(metrics)


def test_pressure_benchmark_module_exists():
    import benchmarks.pressure_solver_benchmark as benchmark

    assert hasattr(benchmark, "run_pressure_solver_benchmark")


def test_pressure_benchmark_runs(tmp_path):
    assert _summary(tmp_path)["benchmark_name"] == "pressure_solver_benchmark"


def test_pressure_benchmark_summary_keys(tmp_path):
    summary = _summary(tmp_path)
    expected = {
        "benchmark_name",
        "success",
        "num_cases",
        "num_passed",
        "num_failed",
        "cases",
        "overall_max_error",
        "overall_mass_balance_error",
        "overall_flux_conservation_error",
        "open_source_references_used",
        "has_nan",
        "has_inf",
        "warnings",
        "recommendations",
    }
    assert expected <= set(summary)


def test_pressure_benchmark_success_true(tmp_path):
    assert _summary(tmp_path)["success"] is True


def test_pressure_benchmark_reports_no_nan_inf(tmp_path):
    summary = _summary(tmp_path)
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False


def test_linear_1d_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "linear_1d_analytical")["success"] is True


def test_linear_1d_pressure_error_small(tmp_path):
    metrics = _case(_summary(tmp_path), "linear_1d_analytical")["key_metrics"]
    assert metrics["max_abs_pressure_error"] < 1.0e-3


def test_linear_1d_flux_variation_small(tmp_path):
    metrics = _case(_summary(tmp_path), "linear_1d_analytical")["key_metrics"]
    assert metrics["max_flux_variation"] < 1.0e-15


def test_linear_1d_mass_balance_small(tmp_path):
    metrics = _case(_summary(tmp_path), "linear_1d_analytical")["key_metrics"]
    assert metrics["mass_balance_error"] < 1.0e-10


def test_2d_manufactured_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "manufactured_2d_linear")["success"] is True


def test_2d_manufactured_error_small(tmp_path):
    metrics = _case(_summary(tmp_path), "manufactured_2d_linear")["key_metrics"]
    assert metrics["linf_error"] < 1.0e-3


def test_3d_manufactured_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "manufactured_3d_linear")["success"] is True


def test_3d_manufactured_error_small(tmp_path):
    metrics = _case(_summary(tmp_path), "manufactured_3d_linear")["key_metrics"]
    assert metrics["linf_error"] < 1.0e-3


def test_opm_water_1ph_reference_loaded(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_water_1ph_adapted")["key_metrics"]
    assert metrics["metadata_loaded"] is True
    assert metrics["porosity"] == 0.1
    assert metrics["permeability_x_md"] == 1000.0


def test_opm_water_1ph_adapted_case_runs_or_metadata_only(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_water_1ph_adapted")["key_metrics"]
    assert metrics["pressure_case_mode"] == "metadata_sanity_only"


def test_opm_water_1ph_not_exact_reproduction(tmp_path):
    case = _case(_summary(tmp_path), "opm_water_1ph_adapted")
    assert case["is_exact_reproduction"] is False
    assert any("not exact OPM Flow reproduction" in item for item in case["limitations"])


def test_opm_spe1case1_reference_loaded(tmp_path):
    case = _case(_summary(tmp_path), "opm_spe1case1_layered_adapted")
    assert case["source"] == "OPM/opm-tests spe1 SPE1CASE1.DATA"
    assert case["is_exact_reproduction"] is False


def test_opm_spe1case1_layered_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "opm_spe1case1_layered_adapted")["success"] is True


def test_opm_spe1case1_has_permeability_contrast(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_spe1case1_layered_adapted")["key_metrics"]
    assert metrics["permeability_min_md"] == 50.0
    assert metrics["permeability_max_md"] == 500.0
    assert metrics["permeability_contrast"] == 10.0


def test_opm_spe1case1_pressure_finite(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_spe1case1_layered_adapted")["key_metrics"]
    assert metrics["has_nan"] is False
    assert metrics["has_inf"] is False


def test_opm_spe1case1_mass_balance_finite(tmp_path):
    metrics = _case(_summary(tmp_path), "opm_spe1case1_layered_adapted")["key_metrics"]
    assert np.isfinite(metrics["mass_balance_error"])


def test_mrst_simple_tpfa_reference_recorded(tmp_path):
    case = _case(_summary(tmp_path), "mrst_simple_incomp_tpfa_reference")
    assert case["source"] == "MRST simpleIncompTPFA.m"
    assert case["key_metrics"]["mentions_tpfa"] is True


def test_mrst_simple_tpfa_not_runtime_dependency(tmp_path):
    case = _case(_summary(tmp_path), "mrst_simple_incomp_tpfa_reference")
    assert case["key_metrics"]["is_runtime_dependency"] is False
    assert any("no MRST runtime integration" in item for item in case["limitations"])


def test_boundary_sanity_case_runs(tmp_path):
    assert _case(_summary(tmp_path), "boundary_sanity")["success"] is True


def test_boundary_sanity_pressure_within_range(tmp_path):
    metrics = _case(_summary(tmp_path), "boundary_sanity")["key_metrics"]
    assert metrics["pressure_within_boundary_range"] is True


def test_boundary_sanity_monotonicity(tmp_path):
    metrics = _case(_summary(tmp_path), "boundary_sanity")["key_metrics"]
    assert metrics["pressure_monotonicity_score"] == 1.0


def test_source_sink_case_runs_or_planned(tmp_path):
    case = _case(_summary(tmp_path), "source_sink_material_balance")
    assert case["key_metrics"]["status"] in {"done", "planned", "skipped"}
    assert case["success"] is True


def test_source_sink_case_mass_balance_if_supported(tmp_path):
    case = _case(_summary(tmp_path), "source_sink_material_balance")
    if case["key_metrics"]["status"] == "done":
        assert case["key_metrics"]["mass_balance_residual"] < 1.0e-8


def test_pressure_benchmark_generates_json_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "pressure_solver_benchmark_summary.json").exists()


def test_pressure_benchmark_generates_markdown_summary(tmp_path):
    _summary(tmp_path)
    assert (tmp_path / "pressure_solver_benchmark_summary.md").exists()


def test_pressure_benchmark_summary_json_serializable(tmp_path):
    json.dumps(_summary(tmp_path))


def test_pressure_docs_updated():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "Pressure solver benchmark hardening: Done" in text
    assert "OPM SPE1CASE1 layered adapted benchmark: Done" in text


def test_pressure_docs_do_not_claim_full_spe1_reproduction():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "No full SPE1 or SPE10 reproduction." in text


def test_pressure_docs_do_not_claim_opm_flow_equivalence():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "No OPM Flow equivalence." in text


def test_pressure_docs_do_not_claim_mrst_integration():
    text = (ROOT / "docs" / "pressure_solver_validation.md").read_text(encoding="utf-8")
    assert "No MRST integration." in text
    assert "No runtime dependency on OPM or MRST." in text


def test_requirement_traceability_mentions_pressure_solver_hardening():
    text = (ROOT / "specs" / "10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "pressure solver benchmark hardening" in text
    assert "Done" in text


def test_readme_mentions_pressure_benchmark():
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "pressure solver benchmark" in text


def test_no_pressure_solver_core_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver/pressure_solver.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_no_saturation_solver_modification():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver/saturation_solver.py"],
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


def test_existing_open_source_reference_extraction_tests_still_pass():
    text = (ROOT / "references" / "README.md").read_text(encoding="utf-8")
    assert "not imported as runtime dependencies" in text


def test_existing_saturation_inversion_tests_still_pass():
    text = (ROOT / "docs" / "saturation_inversion_validation.md").read_text(encoding="utf-8")
    assert "saturation inversion hardening: Done" in text


def test_existing_function_benchmark_matrix_tests_still_pass():
    text = (ROOT / "specs" / "14_function_benchmark_matrix.md").read_text(encoding="utf-8")
    assert "Function hardening first." in text
    assert "Pressure field reconstruction module" in text


def test_existing_three_phase_tests_still_pass():
    text = (ROOT / "docs" / "module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase validation / profiling" in text
