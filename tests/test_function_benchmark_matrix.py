from pathlib import Path
import subprocess

from benchmarks.cross_scale_formula_check import run_benchmark as run_cross_scale_formula_benchmark
from benchmarks.three_phase_closure import run_benchmark as run_three_phase_closure_benchmark


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "specs" / "14_function_benchmark_matrix.md"
POLICY = ROOT / "docs" / "benchmark_selection_policy.md"
TRACEABILITY = ROOT / "specs" / "10_requirement_traceability.md"
README = ROOT / "README.md"
LIMITATIONS = ROOT / "docs" / "limitations_and_roadmap.md"
MODULE_MATRIX = ROOT / "docs" / "module_matrix.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _matrix() -> str:
    return _read(MATRIX)


def _policy() -> str:
    return _read(POLICY)


def test_function_benchmark_matrix_exists():
    assert MATRIX.exists()


def test_function_benchmark_matrix_states_current_principle():
    text = _matrix()
    assert "Function hardening first." in text
    assert "Workflow design after contract confirmation." in text
    assert "当前阶段功能优先，流程后置" in text


def test_matrix_mentions_saturation_inversion_module():
    assert "Saturation inversion module" in _matrix()


def test_matrix_mentions_pressure_reconstruction_module():
    assert "Pressure field reconstruction module" in _matrix()


def test_matrix_mentions_saturation_transport_module():
    assert "Saturation transport module" in _matrix()


def test_matrix_mentions_capillary_gravity_module():
    assert "Capillary / gravity enhancement module" in _matrix()


def test_matrix_mentions_simplified_three_phase_module():
    assert "Simplified three-phase WOG module" in _matrix()


def test_matrix_mentions_parameter_fusion_module():
    assert "Parameter field fusion module" in _matrix()


def test_matrix_mentions_cross_scale_similarity_module():
    assert "Cross-scale similarity module" in _matrix()


def test_matrix_mentions_scale_effect_module():
    assert "Scale-effect analysis module" in _matrix()


def test_matrix_mentions_lab_field_validation_module():
    assert "Lab-field validation module" in _matrix()


def test_matrix_mentions_result_reporting_module():
    assert "Result reporting module" in _matrix()


def test_matrix_mentions_future_interface_module():
    assert "Future interface module" in _matrix()


def test_matrix_mentions_future_black_oil_module():
    assert "Future black-oil extension" in _matrix()


def test_matrix_mentions_current_algorithms():
    text = _matrix()
    assert "Archie equation inversion" in text
    assert "finite-volume / TPFA pressure solve" in text
    assert "upwind finite-volume transport" in text
    assert "three-phase Corey relative permeability" in text
    assert "weighted averaging" in text
    assert "curve alignment" in text


def test_matrix_mentions_input_and_output_data():
    text = _matrix()
    assert "Input data" in text
    assert "Output data" in text
    assert "permeability field" in text
    assert "water saturation" in text
    assert "fusion report" in text


def test_matrix_mentions_candidate_benchmarks():
    text = _matrix()
    assert "Candidate benchmark" in text
    assert "1D linear pressure analytical solution" in text
    assert "Buckley-Leverett 1D qualitative benchmark" in text
    assert "known weighted average formula" in text


def test_matrix_mentions_validation_metrics():
    text = _matrix()
    assert "Validation metric" in text
    assert "absolute Sw error" in text
    assert "mass balance residual" in text
    assert "closure error" in text


def test_matrix_mentions_next_hardening_task():
    text = _matrix()
    assert "Next hardening task" in text
    assert "uncertainty-weighted inversion" in text
    assert "solver diagnostics" in text
    assert "semi-implicit transport option" in text


def test_benchmark_selection_policy_exists():
    assert POLICY.exists()


def test_policy_mentions_analytical_manufactured_benchmarks():
    text = _policy()
    assert "Analytical / Manufactured Benchmark" in text
    assert "1D linear pressure" in text


def test_policy_mentions_qualitative_physical_benchmarks():
    text = _policy()
    assert "Qualitative Physical Benchmark" in text
    assert "Buckley-Leverett front movement" in text


def test_policy_mentions_open_source_adapted_benchmarks():
    text = _policy()
    assert "Open-Source Adapted Benchmark" in text
    assert "SPE10-like heterogeneity subset" in text


def test_policy_says_no_full_spe10_yet():
    assert "No full SPE10 reproduction yet." in _policy()


def test_policy_says_no_opm_deck_parser_yet():
    assert "No OPM deck parser yet." in _policy()


def test_policy_says_no_mrst_runtime_dependency():
    assert "No MRST runtime dependency." in _policy()


def test_policy_says_no_egg_full_dataset_import():
    assert "No Egg full dataset import yet." in _policy()


def test_policy_says_no_commercial_equivalence_claim():
    assert "No commercial simulator equivalence claim." in _policy()


def test_roadmap_mentions_saturation_inversion_hardening():
    assert "046_saturation_inversion_hardening" in _matrix()


def test_roadmap_mentions_pressure_solver_hardening():
    assert "047_pressure_solver_benchmark_hardening" in _matrix()


def test_roadmap_mentions_saturation_transport_hardening():
    assert "048_saturation_transport_benchmark_hardening" in _matrix()


def test_roadmap_mentions_open_source_benchmark_adaptation():
    assert "051_open_source_benchmark_adaptation" in _matrix()


def test_requirement_traceability_mentions_function_matrix_done():
    text = _read(TRACEABILITY)
    assert "function benchmark matrix" in text
    assert "Done" in text


def test_readme_mentions_function_hardening_and_benchmarks():
    text = _read(README)
    assert "Function hardening first" in text
    assert "benchmark validation" in text


def test_docs_do_not_claim_petrel_workflow_implemented():
    text = "\n".join(_read(path) for path in [README, LIMITATIONS, MODULE_MATRIX, MATRIX])
    lowered = text.lower()
    assert "petrel workflow implemented" not in lowered
    assert "petrel-like workflow design is completed" not in lowered


def test_docs_do_not_claim_black_oil_implemented():
    text = "\n".join(_read(path) for path in [README, LIMITATIONS, MODULE_MATRIX, MATRIX]).lower()
    assert "black-oil implemented" not in text
    assert "black-oil simulator" in text


def test_docs_do_not_claim_udp_implemented():
    text = "\n".join(_read(path) for path in [README, LIMITATIONS, MODULE_MATRIX, MATRIX]).lower()
    assert "udp implemented" not in text
    assert "udp development is still deferred" in text or "udp is deferred" in text


def test_function_matrix_stage_did_not_modify_solver_files():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == ""


def test_cross_scale_formula_benchmark_success():
    report = run_cross_scale_formula_benchmark()
    assert report["success"] is True
    assert report["max_formula_error"] < 1.0e-12


def test_three_phase_closure_benchmark_success():
    report = run_three_phase_closure_benchmark()
    assert report["success"] is True
    assert report["closure_error_max"] < 1.0e-12
