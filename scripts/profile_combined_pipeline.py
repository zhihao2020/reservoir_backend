"""Profile demo, capillary, gravity, and combined pipeline cases."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
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
    PROJECT_ROOT / "config" / "gravity_case.yaml",
    PROJECT_ROOT / "config" / "combined_case.yaml",
]


def profile_config(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run one configured pipeline case and return profiling metadata."""
    tracemalloc.start()
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
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        summary = result["summary"]
        return {
            "case_id": case_id,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "total_runtime_sec": time.perf_counter() - start,
            "total_cells": int(summary["total_cells"]),
            "steps": int(config["saturation"]["steps"]),
            "capillary_enabled": bool(summary["capillary_enabled"]),
            "gravity_enabled": bool(summary.get("gravity_enabled", False)),
            "combined_transport_enabled": bool(summary.get("combined_transport_enabled", False)),
            "max_cfl": float(summary["max_cfl"]),
            "material_balance_error": float(summary["material_balance_error"]),
            "max_abs_capillary_flux": float(summary.get("max_abs_capillary_flux", 0.0)),
            "max_abs_gravity_flux": float(summary.get("max_abs_gravity_flux", 0.0)),
            "max_total_water_flux": float(summary.get("max_total_water_flux", 0.0)),
            "max_effective_flux": float(summary.get("max_effective_flux", 0.0)),
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": True,
        }
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "case_id": config_path.stem,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "total_runtime_sec": time.perf_counter() - start,
            "total_cells": None,
            "steps": None,
            "capillary_enabled": None,
            "gravity_enabled": None,
            "combined_transport_enabled": None,
            "max_cfl": None,
            "material_balance_error": None,
            "max_abs_capillary_flux": None,
            "max_abs_gravity_flux": None,
            "max_total_water_flux": None,
            "max_effective_flux": None,
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": False,
            "error": str(exc),
        }


def run_profiling(report_dir: str | Path | None = None) -> dict[str, Any]:
    """Run combined pipeline profiling and write JSON/Markdown reports."""
    output_dir = PROJECT_ROOT / "profiling_reports" if report_dir is None else Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_output_root = output_dir / "combined_outputs"
    cases = [profile_config(path, case_output_root) for path in CASE_CONFIGS]
    by_case = {case["case_id"]: case for case in cases}
    demo_runtime = float(by_case.get("demo_case", {}).get("total_runtime_sec", 0.0) or 0.0)
    combined_runtime = float(by_case.get("combined_case", {}).get("total_runtime_sec", 0.0) or 0.0)
    runtime_ratio = None if demo_runtime <= 0.0 else combined_runtime / demo_runtime
    summary = {
        "cases": cases,
        "case_ids": [case["case_id"] for case in cases],
        "combined_vs_demo_runtime_ratio": runtime_ratio,
        "success": all(case["success"] for case in cases),
    }
    (output_dir / "combined_performance_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "combined_performance_summary.md").write_text(
        _to_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Combined Pipeline Performance Summary",
        "",
        f"- combined_vs_demo_runtime_ratio: {_fmt(summary['combined_vs_demo_runtime_ratio'])}",
        f"- success: {summary['success']}",
        "",
        "| case_id | cells | steps | capillary | gravity | combined | runtime_sec | max_cfl | mb_error | max_cap_flux | max_grav_flux | max_total_flux | max_effective_flux | success |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['total_cells']} | {case['steps']} | "
            f"{case['capillary_enabled']} | {case['gravity_enabled']} | {case['combined_transport_enabled']} | "
            f"{case['total_runtime_sec']:.6f} | {_fmt(case['max_cfl'])} | "
            f"{_fmt(case['material_balance_error'])} | {_fmt(case['max_abs_capillary_flux'])} | "
            f"{_fmt(case['max_abs_gravity_flux'])} | {_fmt(case['max_total_water_flux'])} | "
            f"{_fmt(case['max_effective_flux'])} | {case['success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def main() -> None:
    summary = run_profiling()
    print(f"combined profiling success={summary['success']}")
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
