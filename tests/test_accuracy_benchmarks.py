from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from benchmarks import (
    buckley_leverett_1d,
    capillary_smoothing,
    combined_transport_stability,
    cross_scale_formula_check,
    gravity_segregation,
    pressure_linear_1d,
    pressure_manufactured_3d,
    three_phase_closure,
)


SUMMARY_JSON = Path("accuracy_reports/accuracy_benchmark_summary.json")
SUMMARY_MD = Path("accuracy_reports/accuracy_benchmark_summary.md")


def test_accuracy_benchmark_scripts_exist() -> None:
    for path in [
        "benchmarks/pressure_linear_1d.py",
        "benchmarks/pressure_manufactured_3d.py",
        "benchmarks/buckley_leverett_1d.py",
        "benchmarks/capillary_smoothing.py",
        "benchmarks/gravity_segregation.py",
        "benchmarks/combined_transport_stability.py",
        "benchmarks/three_phase_closure.py",
        "benchmarks/cross_scale_formula_check.py",
        "scripts/run_accuracy_benchmarks.py",
    ]:
        assert Path(path).exists()


def test_pressure_linear_benchmark_runs() -> None:
    assert pressure_linear_1d.run_benchmark()["benchmark_name"] == "pressure_linear_1d"


def test_pressure_linear_benchmark_success() -> None:
    report = pressure_linear_1d.run_benchmark()
    assert report["success"] is True
    assert report["max_pressure_error"] < 1.0e-2


def test_pressure_manufactured_benchmark_runs() -> None:
    report = pressure_manufactured_3d.run_benchmark()
    assert report["success"] is True
    assert report["relative_l2_error"] < 1.0e-10


def test_buckley_leverett_benchmark_runs() -> None:
    assert buckley_leverett_1d.run_benchmark()["success"] is True


def test_buckley_leverett_sw_bounds() -> None:
    report = buckley_leverett_1d.run_benchmark()
    assert report["sw_min"] >= 0.2
    assert report["sw_max"] <= 0.8


def test_capillary_smoothing_benchmark_runs() -> None:
    assert capillary_smoothing.run_benchmark()["success"] is True


def test_capillary_smoothing_gradient_reduces() -> None:
    report = capillary_smoothing.run_benchmark()
    assert report["final_gradient_norm"] < report["initial_gradient_norm"]
    assert report["gradient_reduction"] > 0.0


def test_gravity_segregation_benchmark_runs() -> None:
    assert gravity_segregation.run_benchmark()["success"] is True


def test_gravity_segregation_direction() -> None:
    report = gravity_segregation.run_benchmark()
    assert report["observed_gravity_flux_sign"] == report["expected_gravity_flux_sign"]
    assert report["bottom_sw_change"] > 0.0
    assert report["top_sw_change"] < 0.0


def test_combined_transport_benchmark_runs() -> None:
    report = combined_transport_stability.run_benchmark()
    assert report["success"] is True
    assert report["max_abs_capillary_flux"] > 0.0
    assert report["max_abs_gravity_flux"] > 0.0


def test_three_phase_closure_benchmark_runs() -> None:
    assert three_phase_closure.run_benchmark()["success"] is True


def test_three_phase_closure_error_small() -> None:
    report = three_phase_closure.run_benchmark()
    assert report["closure_error_max"] <= 1.0e-12
    assert abs(report["water_balance_error"]) <= 1.0e-12
    assert abs(report["gas_balance_error"]) <= 1.0e-12
    assert abs(report["oil_balance_error"]) <= 1.0e-12


def test_cross_scale_formula_benchmark_runs() -> None:
    report = cross_scale_formula_check.run_benchmark()
    assert report["success"] is True
    assert report["formula_checks_passed"] == report["num_formula_checks"]


def test_accuracy_runner_creates_json_summary() -> None:
    _run_accuracy_script()
    assert SUMMARY_JSON.exists()


def test_accuracy_runner_creates_markdown_summary() -> None:
    _run_accuracy_script()
    assert SUMMARY_MD.exists()


def test_accuracy_summary_required_keys() -> None:
    summary = _summary()
    keys = {
        "success",
        "num_benchmarks",
        "num_passed",
        "num_failed",
        "benchmarks",
        "overall_warnings",
        "has_nan",
        "has_inf",
        "recommendations",
    }
    assert keys.issubset(summary)


def test_accuracy_summary_success_true() -> None:
    assert _summary()["success"] is True


def test_accuracy_reports_no_nan_inf() -> None:
    summary = _summary()
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False
    for report in summary["benchmarks"]:
        assert report["has_nan"] is False
        assert report["has_inf"] is False


def test_interface_contract_doc_exists() -> None:
    assert Path("docs/interface_contract.md").exists()


def test_interface_contract_says_udp_deferred() -> None:
    text = Path("docs/interface_contract.md").read_text(encoding="utf-8")
    assert "UDP currently deferred" in text
    assert "command-style JSON" in text


def test_interface_contract_says_no_udp_server() -> None:
    assert "No UDP server is implemented in this stage." in Path("docs/interface_contract.md").read_text(encoding="utf-8")


def test_no_solver_modification() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_existing_cross_scale_tests_still_pass() -> None:
    assert cross_scale_formula_check.run_benchmark()["max_formula_error"] == 0.0


def test_existing_three_phase_tests_still_pass() -> None:
    report = three_phase_closure.run_benchmark()
    assert np.isclose(report["closure_error_max"], 0.0)


def _run_accuracy_script() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_accuracy_benchmarks.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _summary() -> dict:
    if not SUMMARY_JSON.exists():
        _run_accuracy_script()
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
