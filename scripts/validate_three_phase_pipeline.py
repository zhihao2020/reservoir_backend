"""Validate the simplified three-phase WOG pipeline case."""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.io.config_loader import load_case_config


THREE_PHASE_CONFIG = PROJECT_ROOT / "config" / "three_phase_case.yaml"
REQUIRED_OUTPUTS = [
    "sw_three_phase.npy",
    "sg_three_phase.npy",
    "so_three_phase.npy",
    "three_phase_report.json",
    "case_summary.json",
]


def run_validation(
    reports_dir: str | Path | None = None,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run three-phase validation and write JSON/Markdown summaries."""
    reports = PROJECT_ROOT / "validation_reports" if reports_dir is None else Path(reports_dir)
    results = reports / "three_phase_outputs" if results_root is None else Path(results_root)
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    config = load_case_config(THREE_PHASE_CONFIG)
    case_id = str(config["case"]["case_id"])
    validation_error: str | None = None
    start = time.perf_counter()
    try:
        run_demo(case_id=case_id, results_root=results, case_config=config)
    except Exception as exc:  # pragma: no cover - failure path
        validation_error = str(exc)
    runtime = time.perf_counter() - start
    case_dir = results / case_id

    required_outputs_exist, missing_outputs = _required_outputs(case_dir)
    arrays = _load_arrays(case_dir) if required_outputs_exist else {}
    report = _load_json(case_dir / "three_phase_report.json")
    case_summary = _load_json(case_dir / "case_summary.json")

    sw = arrays.get("sw_three_phase.npy")
    sg = arrays.get("sg_three_phase.npy")
    so = arrays.get("so_three_phase.npy")
    no_nan_inf = bool(arrays) and all(np.isfinite(values).all() for values in arrays.values())
    closure_error_max = float("inf")
    bounds_valid = False
    sw_min = sw_max = sg_min = sg_max = so_min = so_max = None
    if sw is not None and sg is not None and so is not None:
        sw_min, sw_max = float(sw.min()), float(sw.max())
        sg_min, sg_max = float(sg.min()), float(sg.max())
        so_min, so_max = float(so.min()), float(so.max())
        closure_error_max = float(np.max(np.abs(sw + sg + so - 1.0)))
        swi = float(config["relperm_three_phase"]["swi"])
        sor = float(config["relperm_three_phase"]["sor"])
        sgc = float(config["relperm_three_phase"]["sgc"])
        bounds_valid = bool(
            sw_min >= swi - 1.0e-12
            and sg_min >= sgc - 1.0e-12
            and so_min >= sor - 1.0e-12
            and float(np.max(sw + sg)) <= 1.0 - sor + 1.0e-12
        )

    max_cfl = float(report.get("max_cfl", float("inf")))
    water_balance_error = float(report.get("water_balance_error", float("inf")))
    gas_balance_error = float(report.get("gas_balance_error", float("inf")))
    oil_balance_error = float(report.get("oil_balance_error", float("inf")))
    balance_reasonable = max(abs(water_balance_error), abs(gas_balance_error), abs(oil_balance_error)) <= 1.0e-8
    flags_valid = (
        bool(case_summary.get("three_phase_enabled", False))
        and bool(case_summary.get("three_phase_transport_enabled", False))
        and not bool(case_summary.get("black_oil_enabled", True))
        and bool(case_summary.get("success", False))
    )
    dt_sensitivity = run_dt_sensitivity(config, reports / "three_phase_dt_outputs")
    dt_sensitivity_success = _dt_sensitivity_success(dt_sensitivity, config)

    success = bool(
        validation_error is None
        and required_outputs_exist
        and no_nan_inf
        and bounds_valid
        and closure_error_max <= 1.0e-12
        and max_cfl <= float(config["saturation"]["max_cfl"]) + 1.0e-12
        and balance_reasonable
        and flags_valid
        and dt_sensitivity_success
    )
    summary: dict[str, Any] = {
        "success": success,
        "case_id": case_id,
        "runtime_sec": runtime,
        "required_outputs_exist": required_outputs_exist,
        "missing_outputs": missing_outputs,
        "sw_min": sw_min,
        "sw_max": sw_max,
        "sg_min": sg_min,
        "sg_max": sg_max,
        "so_min": so_min,
        "so_max": so_max,
        "saturation_bounds_valid": bounds_valid,
        "closure_error_max": closure_error_max,
        "max_cfl": max_cfl,
        "water_balance_error": water_balance_error,
        "gas_balance_error": gas_balance_error,
        "oil_balance_error": oil_balance_error,
        "material_balance_reasonable": balance_reasonable,
        "has_nan": not no_nan_inf,
        "has_inf": False if not arrays else any(np.isinf(values).any() for values in arrays.values()),
        "no_nan_inf": no_nan_inf,
        "three_phase_enabled": bool(case_summary.get("three_phase_enabled", False)),
        "three_phase_transport_enabled": bool(case_summary.get("three_phase_transport_enabled", False)),
        "black_oil_enabled": bool(case_summary.get("black_oil_enabled", True)),
        "case_summary_success": bool(case_summary.get("success", False)),
        "dt_sensitivity": dt_sensitivity,
        "dt_sensitivity_success": dt_sensitivity_success,
        "validation_notes": [
            "Simplified incompressible WOG validation only.",
            "Black-oil PVT, Rs/Rv, bubble point, and phase appearance are not implemented.",
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if validation_error is not None:
        summary["error"] = validation_error
    _write_reports(summary, reports)
    return summary


def run_dt_sensitivity(config: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    """Run base dt, dt/2, and dt/4 checks for three-phase explicit transport."""
    output_root.mkdir(parents=True, exist_ok=True)
    base_dt = float(config["saturation"]["dt"])
    records: list[dict[str, Any]] = []
    for label, dt in [("base", base_dt), ("half", base_dt / 2.0), ("quarter", base_dt / 4.0)]:
        case_config = deepcopy(config)
        case_config["saturation"]["dt"] = dt
        case_id = f"three_phase_dt_{label}"
        start = time.perf_counter()
        try:
            result = run_demo(case_id=case_id, results_root=output_root, case_config=case_config)
            case_dir = result["case_dir"]
            sw = np.load(case_dir / "sw_three_phase.npy")
            sg = np.load(case_dir / "sg_three_phase.npy")
            so = np.load(case_dir / "so_three_phase.npy")
            report = _load_json(case_dir / "three_phase_report.json")
            records.append(
                {
                    "case_id": case_id,
                    "dt": dt,
                    "max_cfl": float(report["max_cfl"]),
                    "sw_min": float(sw.min()),
                    "sw_max": float(sw.max()),
                    "sg_min": float(sg.min()),
                    "sg_max": float(sg.max()),
                    "so_min": float(so.min()),
                    "so_max": float(so.max()),
                    "closure_error_max": float(np.max(np.abs(sw + sg + so - 1.0))),
                    "water_balance_error": float(report["water_balance_error"]),
                    "gas_balance_error": float(report["gas_balance_error"]),
                    "oil_balance_error": float(report["oil_balance_error"]),
                    "total_runtime_sec": time.perf_counter() - start,
                    "no_nan_inf": bool(np.isfinite(sw).all() and np.isfinite(sg).all() and np.isfinite(so).all()),
                    "success": True,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "case_id": case_id,
                    "dt": dt,
                    "max_cfl": None,
                    "total_runtime_sec": time.perf_counter() - start,
                    "success": False,
                    "error": str(exc),
                }
            )
    return records


def _dt_sensitivity_success(records: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    if len(records) != 3 or not all(record.get("success", False) for record in records):
        return False
    swi = float(config["relperm_three_phase"]["swi"])
    sor = float(config["relperm_three_phase"]["sor"])
    sgc = float(config["relperm_three_phase"]["sgc"])
    max_cfls = [float(record["max_cfl"]) for record in records]
    if not (max_cfls[0] >= max_cfls[1] >= max_cfls[2]):
        return False
    base_closure = float(records[0]["closure_error_max"])
    base_balance = max(_max_balance_error(records[0]), 1.0e-12)
    for record in records:
        if not record.get("no_nan_inf", False):
            return False
        if float(record["sw_min"]) < swi - 1.0e-12:
            return False
        if float(record["sg_min"]) < sgc - 1.0e-12:
            return False
        if float(record["so_min"]) < sor - 1.0e-12:
            return False
        if float(record["closure_error_max"]) > max(base_closure * 10.0, 1.0e-12):
            return False
        if _max_balance_error(record) > max(base_balance * 10.0, 1.0e-8):
            return False
    return True


def _max_balance_error(record: dict[str, Any]) -> float:
    return max(
        abs(float(record.get("water_balance_error", float("inf")))),
        abs(float(record.get("gas_balance_error", float("inf")))),
        abs(float(record.get("oil_balance_error", float("inf")))),
    )


def _required_outputs(case_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_OUTPUTS if not (case_dir / name).exists()]
    return len(missing) == 0, missing


def _load_arrays(case_dir: Path) -> dict[str, np.ndarray]:
    return {name: np.load(case_dir / name) for name in REQUIRED_OUTPUTS if name.endswith(".npy")}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_reports(summary: dict[str, Any], reports: Path) -> None:
    (reports / "three_phase_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports / "three_phase_validation_summary.md").write_text(_to_markdown(summary), encoding="utf-8")


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Three-Phase Pipeline Validation Summary",
        "",
        f"- success: {summary['success']}",
        f"- required_outputs_exist: {summary['required_outputs_exist']}",
        f"- no_nan_inf: {summary['no_nan_inf']}",
        f"- saturation_bounds_valid: {summary['saturation_bounds_valid']}",
        f"- closure_error_max: {summary['closure_error_max']}",
        f"- max_cfl: {summary['max_cfl']}",
        f"- black_oil_enabled: {summary['black_oil_enabled']}",
        "",
        "## DT Sensitivity",
        "",
        "| case_id | dt | max_cfl | closure_error | water_mb | gas_mb | oil_mb | runtime_sec | success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in summary["dt_sensitivity"]:
        lines.append(
            f"| {record['case_id']} | {record['dt']:.6g} | {_fmt(record.get('max_cfl'))} | "
            f"{_fmt(record.get('closure_error_max'))} | {_fmt(record.get('water_balance_error'))} | "
            f"{_fmt(record.get('gas_balance_error'))} | {_fmt(record.get('oil_balance_error'))} | "
            f"{record['total_runtime_sec']:.6f} | {record['success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def main() -> None:
    summary = run_validation()
    print(f"three-phase validation success={summary['success']}")
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
