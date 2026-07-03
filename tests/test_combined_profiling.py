from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


SUMMARY_PATH = Path("profiling_reports/combined_performance_summary.json")
SOLVER_FILES = [
    Path("reservoir_backend/solver/pressure_solver.py"),
    Path("reservoir_backend/solver/velocity.py"),
    Path("reservoir_backend/solver/relperm.py"),
    Path("reservoir_backend/solver/cfl.py"),
    Path("reservoir_backend/solver/saturation_solver.py"),
    Path("reservoir_backend/solver/capillary_flux.py"),
    Path("reservoir_backend/solver/gravity_flux.py"),
    Path("reservoir_backend/solver/water_flux_composer.py"),
]


def test_profile_combined_pipeline_script_runs() -> None:
    result = _run_profile_script()
    assert result.returncode == 0


def test_combined_performance_summary_created() -> None:
    _ensure_summary()
    assert SUMMARY_PATH.exists()
    assert Path("profiling_reports/combined_performance_summary.md").exists()


def test_combined_performance_summary_contains_cases() -> None:
    by_case = _by_case()
    for case_id in ["demo_case", "capillary_case", "gravity_case", "combined_case"]:
        assert case_id in by_case


def test_combined_case_runtime_recorded() -> None:
    combined = _by_case()["combined_case"]
    assert combined["total_runtime_sec"] > 0.0
    assert combined["total_cells"] > 0
    assert combined["steps"] > 0


def test_combined_case_max_cfl_recorded() -> None:
    combined = _by_case()["combined_case"]
    assert combined["max_cfl"] >= 0.0


def test_combined_case_material_balance_recorded() -> None:
    combined = _by_case()["combined_case"]
    assert abs(float(combined["material_balance_error"])) <= 1.0e-8


def test_combined_profile_success_true() -> None:
    summary = _summary()
    assert summary["success"] is True
    assert _by_case()["combined_case"]["success"] is True


def test_profile_does_not_modify_solver_files() -> None:
    before = {path: _sha256(path) for path in SOLVER_FILES}
    result = _run_profile_script()
    assert result.returncode == 0
    after = {path: _sha256(path) for path in SOLVER_FILES}
    assert before == after


def _run_profile_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/profile_combined_pipeline.py"],
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
