"""Validate the combined capillary + gravity pipeline case."""

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


COMBINED_CONFIG = PROJECT_ROOT / "config" / "combined_case.yaml"
REQUIRED_OUTPUTS = [
    "sw_simulated.npy",
    "sw_fused.npy",
    "capillary_pressure.npy",
    "capillary_flux_x.npy",
    "capillary_flux_y.npy",
    "capillary_flux_z.npy",
    "gravity_flux_x.npy",
    "gravity_flux_y.npy",
    "gravity_flux_z.npy",
    "combined_report.json",
    "case_summary.json",
]


def run_validation(
    reports_dir: str | Path | None = None,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run combined case validation and write JSON/Markdown summaries."""
    reports = PROJECT_ROOT / "validation_reports" if reports_dir is None else Path(reports_dir)
    results = reports / "combined_outputs" if results_root is None else Path(results_root)
    reports.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    config = load_case_config(COMBINED_CONFIG)
    case_id = str(config["case"]["case_id"])
    start = time.perf_counter()
    validation_error: str | None = None
    try:
        run_demo(
            case_id=case_id,
            results_root=results,
            use_multisignal=config["case"]["mode"] == "multisignal",
            case_config=config,
        )
    except Exception as exc:  # pragma: no cover - exercised by failure path only
        validation_error = str(exc)
    runtime = time.perf_counter() - start
    case_dir = results / case_id

    required_outputs_exist, missing_outputs = _required_outputs(case_dir)
    arrays = _load_arrays(case_dir) if required_outputs_exist else {}
    no_nan_inf = bool(arrays) and all(np.isfinite(values).all() for values in arrays.values())
    sw_bounds_valid = False
    pc_valid = False
    capillary_flux_nonzero = False
    gravity_flux_z_nonzero = False
    if arrays:
        swi = float(config["saturation"]["swi"])
        sor = float(config["saturation"]["sor"])
        sw = arrays["sw_simulated.npy"]
        pc = arrays["capillary_pressure.npy"]
        sw_bounds_valid = bool(sw.min() >= swi - 1.0e-12 and sw.max() <= 1.0 - sor + 1.0e-12)
        pc_valid = bool(np.isfinite(pc).all() and pc.min() >= 0.0)
        capillary_flux_nonzero = any(
            float(np.max(np.abs(arrays[name]))) > 0.0
            for name in ["capillary_flux_x.npy", "capillary_flux_y.npy", "capillary_flux_z.npy"]
        )
        gravity_flux_z_nonzero = float(np.max(np.abs(arrays["gravity_flux_z.npy"][1:-1, :, :]))) > 0.0

    combined_report = _load_json(case_dir / "combined_report.json")
    case_summary = _load_json(case_dir / "case_summary.json")
    material_balance_error = float(combined_report.get("material_balance_error", float("inf")))
    material_balance_reasonable = abs(material_balance_error) <= 1.0e-8
    combined_transport_enabled = bool(case_summary.get("combined_transport_enabled", False))
    summary_success = bool(case_summary.get("success", False))

    dt_sensitivity = run_dt_sensitivity(config, reports / "combined_dt_outputs")
    dt_sensitivity_success = _dt_sensitivity_success(dt_sensitivity, config)

    success = bool(
        validation_error is None
        and required_outputs_exist
        and no_nan_inf
        and sw_bounds_valid
        and pc_valid
        and capillary_flux_nonzero
        and gravity_flux_z_nonzero
        and material_balance_reasonable
        and combined_transport_enabled
        and summary_success
        and dt_sensitivity_success
    )
    summary: dict[str, Any] = {
        "case_id": case_id,
        "runtime_sec": runtime,
        "required_outputs_exist": required_outputs_exist,
        "missing_outputs": missing_outputs,
        "no_nan_inf": no_nan_inf,
        "sw_bounds_valid": sw_bounds_valid,
        "pc_nonnegative": pc_valid,
        "capillary_flux_nonzero": capillary_flux_nonzero,
        "gravity_flux_z_nonzero": gravity_flux_z_nonzero,
        "material_balance_error": material_balance_error,
        "material_balance_reasonable": material_balance_reasonable,
        "combined_transport_enabled": combined_transport_enabled,
        "case_summary_success": summary_success,
        "max_cfl": combined_report.get("max_cfl"),
        "max_abs_capillary_flux": combined_report.get("max_abs_capillary_flux"),
        "max_abs_gravity_flux": combined_report.get("max_abs_gravity_flux"),
        "max_total_water_flux": combined_report.get("max_total_water_flux"),
        "max_effective_flux": combined_report.get("max_effective_flux"),
        "dt_sensitivity": dt_sensitivity,
        "dt_sensitivity_success": dt_sensitivity_success,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if validation_error is not None:
        summary["error"] = validation_error
    _write_reports(summary, reports)
    return summary


def run_dt_sensitivity(config: dict[str, Any], output_root: Path) -> list[dict[str, Any]]:
    """Run base dt, dt/2, and dt/4 checks for explicit-step sensitivity."""
    output_root.mkdir(parents=True, exist_ok=True)
    base_dt = float(config["saturation"]["dt"])
    records: list[dict[str, Any]] = []
    for label, dt in [("base", base_dt), ("half", base_dt / 2.0), ("quarter", base_dt / 4.0)]:
        case_config = deepcopy(config)
        case_config["saturation"]["dt"] = dt
        case_id = f"combined_dt_{label}"
        start = time.perf_counter()
        try:
            result = run_demo(
                case_id=case_id,
                results_root=output_root,
                use_multisignal=case_config["case"]["mode"] == "multisignal",
                case_config=case_config,
            )
            case_dir = result["case_dir"]
            sw = np.load(case_dir / "sw_simulated.npy")
            report = _load_json(case_dir / "combined_report.json")
            records.append(
                {
                    "case_id": case_id,
                    "dt": dt,
                    "max_cfl": float(report.get("max_cfl", 0.0)),
                    "material_balance_error": float(report.get("material_balance_error", 0.0)),
                    "sw_simulated_min": float(sw.min()),
                    "sw_simulated_max": float(sw.max()),
                    "total_runtime_sec": time.perf_counter() - start,
                    "no_nan_inf": bool(np.isfinite(sw).all()),
                    "success": True,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "case_id": case_id,
                    "dt": dt,
                    "max_cfl": None,
                    "material_balance_error": None,
                    "sw_simulated_min": None,
                    "sw_simulated_max": None,
                    "total_runtime_sec": time.perf_counter() - start,
                    "success": False,
                    "error": str(exc),
                }
            )
    return records


def _dt_sensitivity_success(records: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    if not records or not all(record.get("success", False) for record in records):
        return False
    swi = float(config["saturation"]["swi"])
    upper = 1.0 - float(config["saturation"]["sor"])
    errors = [abs(float(record["material_balance_error"])) for record in records]
    base_error = errors[0]
    for record in records:
        if not record.get("no_nan_inf", False):
            return False
        if float(record["sw_simulated_min"]) < swi - 1.0e-12:
            return False
        if float(record["sw_simulated_max"]) > upper + 1.0e-12:
            return False
    return errors[-1] <= max(base_error * 10.0, 1.0e-8)


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
    json_path = reports / "combined_validation_summary.json"
    md_path = reports / "combined_validation_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(summary), encoding="utf-8")


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Combined Pipeline Validation Summary",
        "",
        f"- success: {summary['success']}",
        f"- required_outputs_exist: {summary['required_outputs_exist']}",
        f"- no_nan_inf: {summary['no_nan_inf']}",
        f"- sw_bounds_valid: {summary['sw_bounds_valid']}",
        f"- pc_nonnegative: {summary['pc_nonnegative']}",
        f"- capillary_flux_nonzero: {summary['capillary_flux_nonzero']}",
        f"- gravity_flux_z_nonzero: {summary['gravity_flux_z_nonzero']}",
        f"- material_balance_error: {summary['material_balance_error']}",
        f"- max_cfl: {summary['max_cfl']}",
        "",
        "## DT Sensitivity",
        "",
        "| case_id | dt | max_cfl | material_balance_error | sw_min | sw_max | runtime_sec | success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in summary["dt_sensitivity"]:
        lines.append(
            f"| {record['case_id']} | {record['dt']:.6g} | {_fmt(record['max_cfl'])} | "
            f"{_fmt(record['material_balance_error'])} | {_fmt(record['sw_simulated_min'])} | "
            f"{_fmt(record['sw_simulated_max'])} | {record['total_runtime_sec']:.6f} | {record['success']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def main() -> None:
    summary = run_validation()
    print(f"combined validation success={summary['success']}")
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
