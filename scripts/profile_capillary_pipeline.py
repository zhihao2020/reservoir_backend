"""Profile demo, capillary, and capillary-gradient pipeline cases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.io.config_loader import load_case_config


CASE_CONFIGS = [
    PROJECT_ROOT / "config" / "demo_case.yaml",
    PROJECT_ROOT / "config" / "capillary_case.yaml",
    PROJECT_ROOT / "config" / "capillary_gradient_case.yaml",
]


def profile_config(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run one configured case and return profiling metadata."""
    start = time.perf_counter()
    try:
        config = load_case_config(config_path)
        case_id = str(config["case"]["case_id"])
        result = run_demo(
            case_id=case_id,
            results_root=output_root,
            use_multisignal=config["case"]["mode"] == "multisignal",
            case_config=config,
        )
        summary = result["summary"]
        return {
            "case_id": case_id,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "total_runtime_sec": time.perf_counter() - start,
            "capillary_enabled": bool(summary["capillary_enabled"]),
            "initial_saturation_type": summary["initial_saturation_type"],
            "total_cells": int(summary["total_cells"]),
            "steps": int(config["saturation"]["steps"]),
            "max_cfl": float(summary["max_cfl"]),
            "max_abs_capillary_flux": float(summary["max_abs_capillary_flux"]),
            "material_balance_error": float(summary["material_balance_error"]),
            "success": True,
        }
    except Exception as exc:
        return {
            "case_id": config_path.stem,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "total_runtime_sec": time.perf_counter() - start,
            "capillary_enabled": None,
            "initial_saturation_type": None,
            "total_cells": None,
            "steps": None,
            "max_cfl": None,
            "max_abs_capillary_flux": None,
            "material_balance_error": None,
            "success": False,
            "error": str(exc),
        }


def run_profiling(report_dir: str | Path | None = None) -> dict[str, Any]:
    """Run capillary pipeline profiling and write JSON/Markdown reports."""
    output_dir = PROJECT_ROOT / "profiling_reports" if report_dir is None else Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_output_root = output_dir / "capillary_outputs"
    cases = [profile_config(path, case_output_root) for path in CASE_CONFIGS]
    summary = {
        "cases": cases,
        "success": all(case["success"] for case in cases),
        "case_ids": [case["case_id"] for case in cases],
    }
    (output_dir / "capillary_performance_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "capillary_performance_summary.md").write_text(
        _to_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Capillary Pipeline Performance Summary",
        "",
        "| case_id | cells | steps | capillary | initial_sw | runtime_sec | max_cfl | max_abs_capillary_flux | material_balance_error | success |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['total_cells']} | {case['steps']} | "
            f"{case['capillary_enabled']} | {case['initial_saturation_type']} | "
            f"{case['total_runtime_sec']:.6f} | {_fmt(case['max_cfl'])} | "
            f"{_fmt(case['max_abs_capillary_flux'])} | {_fmt(case['material_balance_error'])} | "
            f"{case['success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def main() -> None:
    summary = run_profiling()
    print(f"capillary profiling success={summary['success']}")


if __name__ == "__main__":
    main()
