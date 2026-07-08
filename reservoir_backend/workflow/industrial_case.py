"""Industrial case workflow v0.

This module composes existing project/case registries, result manifests, and
the lightweight IMPES loop into a file-based engineering workflow. It does not
add a new solver, black-oil model, history matching loop, REST API, or frontend.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.project.case_registry import CaseMetadata, CaseRegistry
from reservoir_backend.project.project_registry import ProjectMetadata, ProjectRegistry, json_safe, utc_timestamp
from reservoir_backend.project.run_history import RunHistory, RunRecord
from reservoir_backend.results.manifest import ResultManifest
from reservoir_backend.simulation.impes import IMPESConfig, run_impes_simulation
from reservoir_backend.solver.saturation_solver import DEFAULT_RELPERM_PARAMS


DEFAULT_INDUSTRIAL_CASE_CONFIG: dict[str, Any] = {
    "project": {
        "project_id": "industrial_project_v0",
        "name": "Industrial Workflow Demo",
        "description": "File-based industrial workflow v0 example.",
        "metadata": {"workflow": "industrial_case_v0"},
    },
    "case": {
        "case_id": "industrial_case_v0",
        "case_name": "Industrial Case Workflow v0",
        "input_paths": [],
        "output_paths": [],
        "module_tags": ["M8", "IMPES", "result_manifest"],
        "metadata": {"case_type": "synthetic_waterflood"},
    },
    "run": {
        "run_id": "industrial_run_v0",
        "metadata": {"source_task": "IND-001"},
    },
    "grid": {"nx": 8, "ny": 3, "nz": 2, "dx": 1.0, "dy": 1.0, "dz": 1.0},
    "rock": {"porosity": 0.25, "permeability_m2": 1.0e-8},
    "fluid": {"mu_w": 1.0e-3, "mu_o": 5.0e-3},
    "pressure": {"left": 100.0, "right": 0.0},
    "saturation": {
        "initial_sw": 0.2,
        "dt": 500.0,
        "num_steps": 10,
        "max_cfl": 0.8,
        "swi": 0.2,
        "sor": 0.2,
        "krw0": 1.0,
        "kro0": 1.0,
        "nw": 2.0,
        "no": 2.0,
        "injected_sw": 0.8,
        "producer_boundary": "right",
        "breakthrough_water_cut": 1.0e-6,
    },
    "outputs": {
        "output_dir": "accuracy_reports",
        "summary_json": "industrial_case_workflow_summary.json",
        "summary_markdown": "industrial_case_workflow_summary.md",
        "result_manifest": "industrial_case_workflow_result_manifest.json",
    },
}


def load_industrial_case_config(config: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load a workflow config from YAML, JSON, dict, or defaults."""
    if config is None:
        loaded: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        loaded = deepcopy(dict(config))
    else:
        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        elif suffix == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"unsupported industrial case config format: {suffix}")
    merged = deepcopy(DEFAULT_INDUSTRIAL_CASE_CONFIG)
    _deep_update(merged, loaded)
    validate_industrial_case_config(merged)
    return merged


def validate_industrial_case_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the file-based industrial workflow config."""
    data = deepcopy(dict(config))
    for section in ("project", "case", "run", "grid", "rock", "fluid", "pressure", "saturation", "outputs"):
        if section not in data or not isinstance(data[section], Mapping):
            raise ValueError(f"missing or invalid workflow config section: {section}")
    for section, keys in {
        "project": ("project_id", "name"),
        "case": ("case_id", "case_name"),
        "run": ("run_id",),
    }.items():
        for key in keys:
            if not str(data[section].get(key, "")):
                raise ValueError(f"{section}.{key} must be non-empty")
    grid = data["grid"]
    for key in ("nx", "ny", "nz"):
        if int(grid[key]) <= 1:
            raise ValueError(f"grid.{key} must be > 1")
    for key in ("dx", "dy", "dz"):
        if float(grid[key]) <= 0.0:
            raise ValueError(f"grid.{key} must be positive")
    if not 0.0 < float(data["rock"]["porosity"]) < 1.0:
        raise ValueError("rock.porosity must be in (0, 1)")
    if float(data["rock"]["permeability_m2"]) <= 0.0:
        raise ValueError("rock.permeability_m2 must be positive")
    if float(data["fluid"]["mu_w"]) <= 0.0 or float(data["fluid"]["mu_o"]) <= 0.0:
        raise ValueError("fluid viscosities must be positive")
    saturation = data["saturation"]
    if int(saturation["num_steps"]) <= 0 or float(saturation["dt"]) <= 0.0:
        raise ValueError("saturation time controls must be positive")
    if float(saturation["max_cfl"]) <= 0.0:
        raise ValueError("saturation.max_cfl must be positive")
    if not 0.0 <= float(saturation["initial_sw"]) <= 1.0:
        raise ValueError("saturation.initial_sw must be within [0, 1]")
    if float(saturation["swi"]) < 0.0 or float(saturation["sor"]) < 0.0:
        raise ValueError("residual saturations must be nonnegative")
    if float(saturation["swi"]) + float(saturation["sor"]) >= 1.0:
        raise ValueError("swi + sor must be < 1")
    if saturation.get("producer_boundary", "right") not in {"left", "right", "front", "back", "bottom", "top"}:
        raise ValueError("unsupported producer boundary")
    if not isinstance(data["case"].get("input_paths", []), list):
        raise ValueError("case.input_paths must be a list")
    return json_safe(data)


def build_impes_config_from_workflow_config(config: Mapping[str, Any]) -> IMPESConfig:
    """Convert a validated workflow config into an existing IMPESConfig."""
    data = validate_industrial_case_config(config)
    grid_data = data["grid"]
    grid = Grid3D(
        nx=int(grid_data["nx"]),
        ny=int(grid_data["ny"]),
        nz=int(grid_data["nz"]),
        dx=float(grid_data["dx"]),
        dy=float(grid_data["dy"]),
        dz=float(grid_data["dz"]),
    )
    saturation = data["saturation"]
    relperm_params = dict(DEFAULT_RELPERM_PARAMS)
    relperm_params.update(
        {
            "swi": float(saturation["swi"]),
            "sor": float(saturation["sor"]),
            "krw0": float(saturation["krw0"]),
            "kro0": float(saturation["kro0"]),
            "nw": float(saturation["nw"]),
            "no": float(saturation["no"]),
            "mu_w": float(data["fluid"]["mu_w"]),
            "mu_o": float(data["fluid"]["mu_o"]),
            "injected_sw": float(saturation["injected_sw"]),
        }
    )
    return IMPESConfig(
        grid=grid,
        phi=float(data["rock"]["porosity"]),
        kx=float(data["rock"]["permeability_m2"]),
        ky=float(data["rock"]["permeability_m2"]),
        kz=float(data["rock"]["permeability_m2"]),
        initial_sw=np.full(grid.shape, float(saturation["initial_sw"]), dtype=float),
        dt=float(saturation["dt"]),
        num_steps=int(saturation["num_steps"]),
        pressure_boundaries={key: float(value) for key, value in data["pressure"].items()},
        relperm_params=relperm_params,
        max_cfl=float(saturation["max_cfl"]),
        wells=[],
        producer_boundary=str(saturation["producer_boundary"]),
        breakthrough_water_cut=float(saturation["breakthrough_water_cut"]),
        case_id=str(data["case"]["case_id"]),
    )


def run_industrial_case_workflow(
    config: str | Path | Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the IND-001 file-based industrial workflow and write reports."""
    workflow_config = load_industrial_case_config(config)
    if output_dir is not None:
        workflow_config["outputs"]["output_dir"] = str(output_dir)
    impes_config = build_impes_config_from_workflow_config(workflow_config)
    run_result = run_impes_simulation(impes_config)

    output_root = Path(workflow_config["outputs"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / str(workflow_config["outputs"]["summary_json"])
    summary_md = output_root / str(workflow_config["outputs"]["summary_markdown"])
    manifest_path = output_root / str(workflow_config["outputs"]["result_manifest"])

    project_registry = ProjectRegistry([_project_metadata(workflow_config)])
    case_registry = CaseRegistry([_case_metadata(workflow_config, summary_json, summary_md, manifest_path)])
    result_manifest = _build_result_manifest(workflow_config, summary_json, summary_md, manifest_path, run_result)
    manifest_path.write_text(json.dumps({"results": [result_manifest.to_dict()]}, indent=2, sort_keys=True), encoding="utf-8")
    run_history = RunHistory([_run_record(workflow_config, summary_json, summary_md, manifest_path, run_result)])

    summary = _build_engineering_summary(
        config=workflow_config,
        impes_summary=run_result.summary,
        project_registry=project_registry,
        case_registry=case_registry,
        run_history=run_history,
        result_manifest_path=manifest_path,
        summary_json=summary_json,
        summary_md=summary_md,
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_markdown(summary), encoding="utf-8")
    return summary


def _project_metadata(config: Mapping[str, Any]) -> ProjectMetadata:
    project = config["project"]
    return ProjectMetadata(
        project_id=str(project["project_id"]),
        name=str(project["name"]),
        description=str(project.get("description", "")),
        created_at=str(project.get("created_at") or utc_timestamp()),
        metadata=dict(project.get("metadata", {})),
    )


def _case_metadata(config: Mapping[str, Any], summary_json: Path, summary_md: Path, manifest_path: Path) -> CaseMetadata:
    case = config["case"]
    return CaseMetadata(
        case_id=str(case["case_id"]),
        project_id=str(config["project"]["project_id"]),
        case_name=str(case["case_name"]),
        input_paths=[str(item) for item in case.get("input_paths", [])],
        output_paths=[str(summary_json), str(summary_md), str(manifest_path)],
        module_tags=[str(item) for item in case.get("module_tags", [])],
        status="validated",
        metadata=dict(case.get("metadata", {})),
    )


def _run_record(
    config: Mapping[str, Any],
    summary_json: Path,
    summary_md: Path,
    manifest_path: Path,
    run_result: Any,
) -> RunRecord:
    return RunRecord(
        run_id=str(config["run"]["run_id"]),
        case_id=str(config["case"]["case_id"]),
        finished_at=utc_timestamp(),
        status="validated",
        report_paths=[str(summary_json), str(summary_md)],
        result_manifest_paths=[str(manifest_path)],
        metrics={
            "num_steps": run_result.summary["num_steps"],
            "final_water_cut": run_result.summary["final_water_cut"],
            "breakthrough_time": run_result.summary["breakthrough_time"],
        },
        warnings=[],
    )


def _build_result_manifest(
    config: Mapping[str, Any],
    summary_json: Path,
    summary_md: Path,
    manifest_path: Path,
    run_result: Any,
) -> ResultManifest:
    return ResultManifest(
        result_id=f"{config['run']['run_id']}_engineering_report",
        case_id=str(config["case"]["case_id"]),
        run_id=str(config["run"]["run_id"]),
        module="M8",
        result_type="engineering_report",
        field_name="industrial_case_workflow_summary",
        shape=[],
        dtype="json",
        unit="none",
        path=str(summary_json),
        format="json",
        source_task="IND-001",
        source_report=str(summary_md),
        metadata={
            "result_manifest_path": str(manifest_path),
            "num_steps": run_result.summary["num_steps"],
            "production_curve_points": len(run_result.summary["production_curve"]),
        },
        warnings=[],
        limitations=_limitations(),
    )


def _build_engineering_summary(
    *,
    config: Mapping[str, Any],
    impes_summary: Mapping[str, Any],
    project_registry: ProjectRegistry,
    case_registry: CaseRegistry,
    run_history: RunHistory,
    result_manifest_path: Path,
    summary_json: Path,
    summary_md: Path,
) -> dict[str, Any]:
    production_curve = list(impes_summary["production_curve"])
    water_cut_curve = [{"time": item["time"], "water_cut": item["water_cut"]} for item in production_curve]
    return {
        "workflow_name": "industrial_case_workflow_v0",
        "source_task": "IND-001",
        "success": bool(impes_summary["success"]),
        "project": project_registry.to_dict(),
        "case": case_registry.to_dict(),
        "run_history": run_history.to_dict(),
        "case_config": {
            "project_id": config["project"]["project_id"],
            "case_id": config["case"]["case_id"],
            "run_id": config["run"]["run_id"],
            "grid": dict(config["grid"]),
            "num_steps": int(config["saturation"]["num_steps"]),
        },
        "impes_summary": json_safe(dict(impes_summary)),
        "production_summary": {
            "final_water_cut": float(impes_summary["final_water_cut"]),
            "breakthrough_time": impes_summary["breakthrough_time"],
            "num_curve_points": len(production_curve),
            "final_total_liquid_rate": float(production_curve[-1]["total_liquid_rate"]),
            "final_water_rate": float(production_curve[-1]["water_rate"]),
            "final_oil_rate": float(production_curve[-1]["oil_rate"]),
        },
        "production_curve": json_safe(production_curve),
        "water_cut_curve": json_safe(water_cut_curve),
        "breakthrough_time": impes_summary["breakthrough_time"],
        "result_manifest_path": str(result_manifest_path),
        "engineering_report_json": str(summary_json),
        "engineering_report_markdown": str(summary_md),
        "warnings": [],
        "limitations": _limitations(),
        "non_claims": [
            "No black-oil solver is implemented.",
            "No history matching is implemented.",
            "No complete EnKF or ES-MDA workflow is implemented.",
            "No frontend, REST API, or UDP service is implemented.",
            "No solver core rewrite is performed.",
        ],
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Industrial Case Workflow Summary",
        "",
        "## Implemented Scope",
        "",
        f"- workflow_name: {summary['workflow_name']}",
        f"- source_task: {summary['source_task']}",
        f"- success: {summary['success']}",
        f"- case_id: {summary['case_config']['case_id']}",
        f"- run_id: {summary['case_config']['run_id']}",
        f"- result_manifest_path: {summary['result_manifest_path']}",
        "",
        "## Production Summary",
        "",
        f"- final_water_cut: {summary['production_summary']['final_water_cut']}",
        f"- breakthrough_time: {summary['production_summary']['breakthrough_time']}",
        f"- final_total_liquid_rate: {summary['production_summary']['final_total_liquid_rate']}",
        "",
        "## Production Curve",
        "",
        "| step | time | total_liquid_rate | water_rate | oil_rate | water_cut |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["production_curve"]:
        lines.append(
            "| {step} | {time:.6g} | {total_liquid_rate:.6g} | {water_rate:.6g} | {oil_rate:.6g} | {water_cut:.6g} |".format(
                **item
            )
        )
    lines.extend(["", "## Known Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Non-Claims", ""])
    for item in summary["non_claims"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Steps", "", "- Add field-data ingestion in IND-002."])
    return "\n".join(lines) + "\n"


def _limitations() -> list[str]:
    return [
        "File-based workflow v0 only.",
        "Uses existing lightweight IMPES loop.",
        "Synthetic structured-grid case by default.",
        "No commercial simulator equivalence.",
        "No black-oil PVT behavior.",
        "No history matching or automatic calibration.",
    ]


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)
