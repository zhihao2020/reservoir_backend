"""Profile demo, combined, and simplified three-phase pipeline cases."""

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
    PROJECT_ROOT / "config" / "combined_case.yaml",
    PROJECT_ROOT / "config" / "three_phase_case.yaml",
]


def profile_config(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Run one configured case and return profiling metadata."""
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
        mode = str(config["case"]["mode"])
        three_phase_enabled = bool(summary.get("three_phase_enabled", False))
        record: dict[str, Any] = {
            "case_id": case_id,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "mode": mode,
            "total_runtime_sec": time.perf_counter() - start,
            "total_cells": int(summary["total_cells"]),
            "steps": int(config["saturation"]["steps"]),
            "three_phase_enabled": three_phase_enabled,
            "combined_transport_enabled": bool(summary.get("combined_transport_enabled", False)),
            "black_oil_enabled": bool(summary.get("black_oil_enabled", False)),
            "max_cfl": float(summary["max_cfl"]),
            "material_balance_error": _material_balance_error(summary),
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": True,
        }
        if three_phase_enabled:
            record.update(
                {
                    "sw_min": float(summary["sw_min"]),
                    "sw_max": float(summary["sw_max"]),
                    "sg_min": float(summary["sg_min"]),
                    "sg_max": float(summary["sg_max"]),
                    "so_min": float(summary["so_min"]),
                    "so_max": float(summary["so_max"]),
                    "closure_error_max": float(summary["closure_error_max"]),
                    "water_balance_error": float(summary["water_balance_error"]),
                    "gas_balance_error": float(summary["gas_balance_error"]),
                    "oil_balance_error": float(summary["oil_balance_error"]),
                }
            )
        return record
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "case_id": config_path.stem,
            "config": str(config_path.relative_to(PROJECT_ROOT)),
            "mode": None,
            "total_runtime_sec": time.perf_counter() - start,
            "total_cells": None,
            "steps": None,
            "three_phase_enabled": None,
            "combined_transport_enabled": None,
            "black_oil_enabled": False,
            "max_cfl": None,
            "material_balance_error": None,
            "memory_peak_mb": peak / (1024.0 * 1024.0),
            "success": False,
            "error": str(exc),
        }


def run_profiling(report_dir: str | Path | None = None) -> dict[str, Any]:
    """Run three-phase profiling and write JSON/Markdown reports."""
    output_dir = PROJECT_ROOT / "profiling_reports" if report_dir is None else Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_output_root = output_dir / "three_phase_outputs"
    cases = [profile_config(path, case_output_root) for path in CASE_CONFIGS]
    by_case = {case["case_id"]: case for case in cases}
    demo_runtime = float(by_case.get("demo_case", {}).get("total_runtime_sec", 0.0) or 0.0)
    combined_runtime = float(by_case.get("combined_case", {}).get("total_runtime_sec", 0.0) or 0.0)
    three_runtime = float(by_case.get("three_phase_case", {}).get("total_runtime_sec", 0.0) or 0.0)
    summary = {
        "cases": cases,
        "case_ids": [case["case_id"] for case in cases],
        "three_phase_to_demo_runtime_ratio": None if demo_runtime <= 0.0 else three_runtime / demo_runtime,
        "three_phase_to_combined_runtime_ratio": None if combined_runtime <= 0.0 else three_runtime / combined_runtime,
        "recommend_cpp": False,
        "recommend_black_oil": False,
        "recommendation_notes": [
            "No C++ migration is recommended from this small-case profiling alone.",
            "Black-oil remains a future design item, not a profiling-driven immediate step.",
        ],
        "success": all(case["success"] for case in cases),
    }
    (output_dir / "three_phase_performance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "three_phase_performance_summary.md").write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _material_balance_error(summary: dict[str, Any]) -> float:
    if "material_balance_error" in summary:
        return float(summary["material_balance_error"])
    return max(
        abs(float(summary.get("water_balance_error", 0.0))),
        abs(float(summary.get("gas_balance_error", 0.0))),
        abs(float(summary.get("oil_balance_error", 0.0))),
    )


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Three-Phase Pipeline Performance Summary",
        "",
        f"- three_phase_to_demo_runtime_ratio: {_fmt(summary['three_phase_to_demo_runtime_ratio'])}",
        f"- three_phase_to_combined_runtime_ratio: {_fmt(summary['three_phase_to_combined_runtime_ratio'])}",
        f"- recommend_cpp: {summary['recommend_cpp']}",
        f"- recommend_black_oil: {summary['recommend_black_oil']}",
        f"- success: {summary['success']}",
        "",
        "| case_id | mode | cells | steps | three_phase | combined | black_oil | runtime_sec | max_cfl | mb_error | success |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['mode']} | {case['total_cells']} | {case['steps']} | "
            f"{case['three_phase_enabled']} | {case['combined_transport_enabled']} | {case['black_oil_enabled']} | "
            f"{case['total_runtime_sec']:.6f} | {_fmt(case['max_cfl'])} | {_fmt(case['material_balance_error'])} | "
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
    print(f"three-phase profiling success={summary['success']}")
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
