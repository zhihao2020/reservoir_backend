from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SUMMARY_PATH = Path("profiling_reports/three_phase_performance_summary.json")
SOLVER_FILES = [
    Path("reservoir_backend/solver/pressure_solver.py"),
    Path("reservoir_backend/solver/velocity.py"),
    Path("reservoir_backend/solver/relperm.py"),
    Path("reservoir_backend/solver/cfl.py"),
    Path("reservoir_backend/solver/saturation_solver.py"),
    Path("reservoir_backend/solver/capillary_flux.py"),
    Path("reservoir_backend/solver/gravity_flux.py"),
    Path("reservoir_backend/solver/water_flux_composer.py"),
    Path("reservoir_backend/solver/three_phase_relperm.py"),
    Path("reservoir_backend/solver/three_phase_flux.py"),
    Path("reservoir_backend/solver/three_phase_transport.py"),
]


def test_profile_three_phase_pipeline_script_runs() -> None:
    result = _run_profile_script()
    assert result.returncode == 0


def test_three_phase_performance_summary_created() -> None:
    _ensure_summary()
    assert SUMMARY_PATH.exists()
    assert Path("profiling_reports/three_phase_performance_summary.md").exists()


def test_three_phase_performance_summary_contains_cases() -> None:
    by_case = _by_case()
    for case_id in ["demo_case", "combined_case", "three_phase_case"]:
        assert case_id in by_case


def test_three_phase_case_runtime_recorded() -> None:
    case = _by_case()["three_phase_case"]
    assert case["total_runtime_sec"] > 0.0
    assert case["total_cells"] > 0
    assert case["steps"] > 0


def test_three_phase_case_max_cfl_recorded() -> None:
    assert _by_case()["three_phase_case"]["max_cfl"] >= 0.0


def test_three_phase_case_closure_recorded() -> None:
    case = _by_case()["three_phase_case"]
    assert "closure_error_max" in case
    assert abs(float(case["closure_error_max"])) <= 1.0e-12


def test_three_phase_case_material_balance_recorded() -> None:
    case = _by_case()["three_phase_case"]
    assert abs(float(case["material_balance_error"])) <= 1.0e-8
    assert abs(float(case["water_balance_error"])) <= 1.0e-8
    assert abs(float(case["gas_balance_error"])) <= 1.0e-8
    assert abs(float(case["oil_balance_error"])) <= 1.0e-8


def test_three_phase_profile_success_true() -> None:
    summary = _summary()
    assert summary["success"] is True
    assert _by_case()["three_phase_case"]["success"] is True


def test_three_phase_profile_not_black_oil() -> None:
    summary = _summary()
    assert summary["recommend_black_oil"] is False
    assert _by_case()["three_phase_case"]["black_oil_enabled"] is False


def test_profile_does_not_modify_solver_files() -> None:
    before = {path: _sha256(path) for path in SOLVER_FILES}
    result = _run_profile_script()
    assert result.returncode == 0
    after = {path: _sha256(path) for path in SOLVER_FILES}
    assert before == after


def _run_profile_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/profile_three_phase_pipeline.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_summary() -> None:
    if not SUMMARY_PATH.exists():
        result = _run_profile_script()
        assert result.returncode == 0, result.stderr


def _summary() -> dict:
    _ensure_summary()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def _by_case() -> dict[str, dict]:
    return {case["case_id"]: case for case in _summary()["cases"]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
