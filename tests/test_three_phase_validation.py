from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SUMMARY_PATH = Path("validation_reports/three_phase_validation_summary.json")


def test_validate_three_phase_pipeline_script_runs() -> None:
    result = _run_validation_script()
    assert result.returncode == 0


def test_three_phase_validation_summary_created() -> None:
    _ensure_summary()
    assert SUMMARY_PATH.exists()
    assert Path("validation_reports/three_phase_validation_summary.md").exists()


def test_three_phase_validation_summary_required_keys() -> None:
    summary = _summary()
    keys = {
        "success",
        "case_id",
        "required_outputs_exist",
        "sw_min",
        "sw_max",
        "sg_min",
        "sg_max",
        "so_min",
        "so_max",
        "closure_error_max",
        "max_cfl",
        "water_balance_error",
        "gas_balance_error",
        "oil_balance_error",
        "has_nan",
        "has_inf",
        "three_phase_enabled",
        "three_phase_transport_enabled",
        "black_oil_enabled",
        "validation_notes",
    }
    assert keys.issubset(summary)


def test_three_phase_validation_success_true() -> None:
    assert _summary()["success"] is True


def test_three_phase_required_outputs_exist() -> None:
    summary = _summary()
    assert summary["required_outputs_exist"] is True
    assert summary["missing_outputs"] == []


def test_three_phase_validation_no_nan_inf() -> None:
    summary = _summary()
    assert summary["has_nan"] is False
    assert summary["has_inf"] is False
    assert summary["no_nan_inf"] is True


def test_three_phase_validation_saturation_bounds() -> None:
    summary = _summary()
    assert summary["saturation_bounds_valid"] is True
    assert summary["sw_min"] >= 0.2
    assert summary["sg_min"] >= 0.05
    assert summary["so_min"] >= 0.2


def test_three_phase_validation_closure_error_small() -> None:
    assert float(_summary()["closure_error_max"]) <= 1.0e-12


def test_three_phase_validation_material_balance_reasonable() -> None:
    summary = _summary()
    assert summary["material_balance_reasonable"] is True
    assert max(
        abs(float(summary["water_balance_error"])),
        abs(float(summary["gas_balance_error"])),
        abs(float(summary["oil_balance_error"])),
    ) <= 1.0e-8


def test_three_phase_validation_flags() -> None:
    summary = _summary()
    assert summary["three_phase_enabled"] is True
    assert summary["three_phase_transport_enabled"] is True
    assert summary["black_oil_enabled"] is False
    assert summary["case_summary_success"] is True


def test_three_phase_dt_sensitivity_records() -> None:
    summary = _summary()
    records = summary["dt_sensitivity"]
    assert len(records) == 3
    assert [record["dt"] for record in records] == sorted([record["dt"] for record in records], reverse=True)
    assert all(record["success"] for record in records)
    assert all(record["no_nan_inf"] for record in records)
    assert records[0]["max_cfl"] >= records[1]["max_cfl"] >= records[2]["max_cfl"]
    assert summary["dt_sensitivity_success"] is True


def test_three_phase_not_black_oil_in_validation() -> None:
    summary = _summary()
    assert summary["black_oil_enabled"] is False
    assert any("Black-oil" in note for note in summary["validation_notes"])


def _run_validation_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_three_phase_pipeline.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_summary() -> None:
    if not SUMMARY_PATH.exists():
        result = _run_validation_script()
        assert result.returncode == 0, result.stderr


def _summary() -> dict:
    _ensure_summary()
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
