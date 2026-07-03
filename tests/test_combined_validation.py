from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SUMMARY_PATH = Path("validation_reports/combined_validation_summary.json")


def test_validate_combined_pipeline_script_runs() -> None:
    result = _run_validation_script()
    assert result.returncode == 0


def test_combined_validation_summary_created() -> None:
    _ensure_summary()
    assert SUMMARY_PATH.exists()
    assert Path("validation_reports/combined_validation_summary.md").exists()


def test_combined_validation_summary_required_keys() -> None:
    summary = _summary()
    keys = {
        "required_outputs_exist",
        "no_nan_inf",
        "sw_bounds_valid",
        "pc_nonnegative",
        "capillary_flux_nonzero",
        "gravity_flux_z_nonzero",
        "material_balance_error",
        "combined_transport_enabled",
        "case_summary_success",
        "dt_sensitivity",
        "dt_sensitivity_success",
        "success",
    }
    assert keys.issubset(summary)


def test_combined_validation_success_true() -> None:
    assert _summary()["success"] is True


def test_combined_required_outputs_exist() -> None:
    summary = _summary()
    assert summary["required_outputs_exist"] is True
    assert summary["missing_outputs"] == []


def test_combined_no_nan_inf() -> None:
    assert _summary()["no_nan_inf"] is True


def test_combined_sw_bounds() -> None:
    assert _summary()["sw_bounds_valid"] is True


def test_combined_flux_nonzero() -> None:
    summary = _summary()
    assert summary["capillary_flux_nonzero"] is True
    assert summary["gravity_flux_z_nonzero"] is True


def test_combined_material_balance_reasonable() -> None:
    summary = _summary()
    assert summary["material_balance_reasonable"] is True
    assert abs(float(summary["material_balance_error"])) <= 1.0e-8


def test_combined_dt_sensitivity_records() -> None:
    summary = _summary()
    records = summary["dt_sensitivity"]
    assert len(records) == 3
    assert [record["dt"] for record in records] == sorted([record["dt"] for record in records], reverse=True)
    assert all(record["success"] for record in records)
    assert all(record["no_nan_inf"] for record in records)
    assert summary["dt_sensitivity_success"] is True


def _run_validation_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/validate_combined_pipeline.py"],
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
