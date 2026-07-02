"""Run backend validation: tests, full pipeline, output checks, summary reports."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_full_pipeline_demo import run_demo
from harness.check_outputs import (
    check_case_summary,
    check_no_nan_inf,
    check_report_keys,
    check_required_outputs,
    check_sw_ranges,
    load_json,
)
from harness.generate_validation_report import write_validation_reports


def run_validation(
    *,
    run_pytest: bool = True,
    case_id: str = "demo_case",
    results_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run validation and return a summary dictionary."""
    results_root = PROJECT_ROOT / "results" if results_root is None else Path(results_root)
    reports_dir = PROJECT_ROOT / "validation_reports" if reports_dir is None else Path(reports_dir)

    pytest_passed = True
    if run_pytest:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        pytest_passed = completed.returncode == 0

    full_pipeline_passed = False
    case_dir = results_root / case_id
    try:
        result = run_demo(case_id=case_id, results_root=results_root)
        case_dir = result["case_dir"]
        full_pipeline_passed = True
    except Exception:
        full_pipeline_passed = False

    required_outputs_exist, missing_outputs = check_required_outputs(case_dir)
    summary_ok = check_case_summary(case_dir)
    no_nan_inf = required_outputs_exist and check_no_nan_inf(case_dir)
    physical_ranges_valid = required_outputs_exist and check_sw_ranges(case_dir)
    report_keys_ok, missing_report_keys = check_report_keys(case_dir) if required_outputs_exist else (False, [])

    case_summary = load_json(case_dir / "case_summary.json") if (case_dir / "case_summary.json").exists() else {}
    material_report = (
        load_json(case_dir / "material_balance_report.json")
        if (case_dir / "material_balance_report.json").exists()
        else {}
    )
    fusion_report = load_json(case_dir / "fusion_report.json") if (case_dir / "fusion_report.json").exists() else {}

    success = bool(
        pytest_passed
        and full_pipeline_passed
        and required_outputs_exist
        and summary_ok
        and no_nan_inf
        and physical_ranges_valid
        and report_keys_ok
    )
    summary: dict[str, Any] = {
        "pytest_passed": pytest_passed,
        "full_pipeline_passed": full_pipeline_passed,
        "required_outputs_exist": required_outputs_exist,
        "missing_outputs": missing_outputs,
        "physical_ranges_valid": physical_ranges_valid,
        "no_nan_inf": no_nan_inf,
        "material_balance_error": material_report.get("material_balance_error"),
        "max_cfl": case_summary.get("max_cfl"),
        "fusion_nan_cells": fusion_report.get("nan_cells_count"),
        "fusion_clipped_cells": fusion_report.get("clipped_cells"),
        "missing_report_keys": missing_report_keys,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_validation_reports(summary, reports_dir)
    return summary


def main() -> None:
    """CLI entry point."""
    summary = run_validation(run_pytest=True)
    print(f"validation success={summary['success']}")
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
