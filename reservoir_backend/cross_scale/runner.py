from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.report import write_json_report, write_markdown_report
from reservoir_backend.cross_scale.scale_effect import build_scale_effect_report
from reservoir_backend.cross_scale.similarity import build_similarity_report
from reservoir_backend.cross_scale.validation import CurveData, validate_multiple_curve_pairs


DEFAULT_LIMITATIONS = [
    "No history matching.",
    "No automatic calibration.",
    "No complex upscaling solver.",
    "No frontend.",
    "No UDP.",
    "No commercial simulator equivalence.",
    "No black-oil validation.",
]


def load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return _validate_config(dict(config))
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported cross-scale config format: {suffix}")
    if not isinstance(data, dict):
        raise InvalidPhysicalValueError("cross-scale config must decode to a mapping")
    return _validate_config(data)


def run_similarity_report(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    cfg = load_config(config)
    lab = _descriptor(cfg, "lab_case")
    field = _descriptor(cfg, "field_case")
    return build_similarity_report(lab, field, cfg.get("similarity_weights"))


def run_scale_effect_report(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    cfg = load_config(config)
    lab = _descriptor(cfg, "lab_case")
    field = _descriptor(cfg, "field_case")
    return build_scale_effect_report(lab, field, cfg.get("thresholds"))


def run_lab_field_validation_report(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    cfg = load_config(config)
    pairs = _curve_pairs(cfg)
    base = validate_multiple_curve_pairs(pairs)
    curve_names = [report.get("curve_name") for report in base["curve_reports"]]
    successful = [report for report in base["curve_reports"] if report.get("success")]
    first = successful[0] if successful else {}
    aggregate = base["aggregate_metrics"]
    return {
        **base,
        "curve_names": curve_names,
        "overlap_interval": {
            "start": first.get("time_start"),
            "end": first.get("time_end"),
        },
        "rmse": aggregate.get("mean_rmse"),
        "mae": aggregate.get("mean_mae"),
        "mape": aggregate.get("mean_mape"),
        "r2": _mean_defined([report.get("r2") for report in successful]),
        "nrmse": aggregate.get("mean_normalized_rmse"),
        "max_absolute_error": aggregate.get("max_absolute_error"),
        "num_matched_samples": int(sum(report.get("num_points", 0) or 0 for report in successful)),
    }


def run_cross_scale_benchmark(
    config: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path = "accuracy_reports",
) -> dict[str, Any]:
    cfg = load_config(config or _default_config())
    similarity_report = run_similarity_report(cfg)
    scale_effect_report = run_scale_effect_report(cfg)
    validation_report = run_lab_field_validation_report(cfg)
    warnings = [
        *similarity_report.get("warnings", []),
        *scale_effect_report.get("warnings", []),
        *validation_report.get("warnings", []),
    ]
    has_nan = bool(similarity_report.get("has_nan") or scale_effect_report.get("has_nan") or validation_report.get("has_nan"))
    has_inf = bool(similarity_report.get("has_inf") or scale_effect_report.get("has_inf") or validation_report.get("has_inf"))
    output_path = Path(output_dir)
    json_path = output_path / "cross_scale_benchmark_summary.json"
    markdown_path = output_path / "cross_scale_benchmark_summary.md"
    summary: dict[str, Any] = {
        "benchmark_name": "cross_scale_benchmark",
        "case_id": cfg.get("case_id", "cross_scale_case"),
        "success": bool(similarity_report["success"] and scale_effect_report["success"] and validation_report["success"] and not has_nan and not has_inf),
        "similarity_report": similarity_report,
        "scale_effect_report": scale_effect_report,
        "lab_field_validation_report": validation_report,
        "result_manifest_entry": _result_manifest_entry(json_path, str(cfg.get("case_id", "cross_scale_case"))),
        "output_paths": {
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
        "warnings": warnings,
        "limitations": list(DEFAULT_LIMITATIONS),
        "has_nan": has_nan,
        "has_inf": has_inf,
    }
    write_cross_scale_reports(summary, output_dir)
    return summary


def write_cross_scale_reports(summary: Mapping[str, Any], output_dir: str | Path = "accuracy_reports") -> dict[str, str]:
    output_path = Path(output_dir)
    json_path = write_json_report(summary, output_path / "cross_scale_benchmark_summary.json")
    markdown_path = write_markdown_report(summary, output_path / "cross_scale_benchmark_summary.md")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = ["lab_case", "field_case", "curves"]
    missing = [key for key in required if key not in config]
    if missing:
        raise InvalidPhysicalValueError(f"missing cross-scale config sections: {', '.join(missing)}")
    _descriptor(config, "lab_case")
    _descriptor(config, "field_case")
    _curve_pairs(config)
    return config


def _descriptor(config: Mapping[str, Any], key: str) -> ScaleDescriptor:
    section = config.get(key)
    if not isinstance(section, Mapping):
        raise InvalidPhysicalValueError(f"{key} must be a mapping")
    descriptor_data = section.get("descriptor", section)
    if not isinstance(descriptor_data, Mapping):
        raise InvalidPhysicalValueError(f"{key}.descriptor must be a mapping")
    return ScaleDescriptor.from_dict(dict(descriptor_data))


def _curve_pairs(config: Mapping[str, Any]) -> list[tuple[CurveData, CurveData]]:
    curves = config.get("curves")
    if not isinstance(curves, list) or not curves:
        raise InvalidPhysicalValueError("cross-scale config requires at least one curve pair")
    pairs: list[tuple[CurveData, CurveData]] = []
    for item in curves:
        if not isinstance(item, Mapping) or "lab" not in item or "field" not in item:
            raise InvalidPhysicalValueError("each curve item must contain lab and field curves")
        lab_data = dict(item["lab"])
        field_data = dict(item["field"])
        if "name" not in lab_data and "name" in item:
            lab_data["name"] = item["name"]
        if "name" not in field_data and "name" in item:
            field_data["name"] = item["name"]
        pairs.append((CurveData.from_dict(lab_data), CurveData.from_dict(field_data)))
    return pairs


def _result_manifest_entry(json_path: Path, case_id: str) -> dict[str, Any]:
    try:
        from reservoir_backend.results.manifest import ResultManifest

        return ResultManifest(
            result_id="cross_scale_benchmark_summary",
            case_id=case_id,
            run_id="cross-scale-runner",
            module="M6",
            result_type="cross_scale_report",
            field_name="cross_scale_benchmark_summary",
            shape=[],
            dtype="json",
            unit="dimensionless",
            path=str(json_path),
            format="json",
            source_task="TASK-003",
            source_report=str(json_path),
            metadata={"contains": ["similarity_report", "scale_effect_report", "lab_field_validation_report"]},
            warnings=[],
            limitations=list(DEFAULT_LIMITATIONS),
        ).to_dict()
    except Exception as exc:
        return {
            "result_type": "cross_scale_report",
            "module": "M6",
            "source_task": "TASK-003",
            "path": str(json_path),
            "warning": f"result manifest package unavailable: {exc}",
        }


def _mean_defined(values: list[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    if not defined:
        return None
    return float(np.mean(defined))


def _default_config() -> dict[str, Any]:
    descriptor = {
        "length_scale_m": 1.0,
        "time_scale_s": 10.0,
        "pressure_scale_pa": 100000.0,
        "permeability_scale_m2": 1.0e-12,
        "porosity": 0.2,
        "viscosity_pa_s": 0.001,
        "density_kg_m3": 1000.0,
        "velocity_scale_m_s": 1.0e-6,
        "flow_rate_m3_s": 1.0e-9,
        "temperature_scale_k": 300.0,
        "interfacial_tension_n_m": 0.03,
        "diffusivity_m2_s": 1.0e-9,
        "delta_density_kg_m3": 100.0,
        "pressure_drop_pa": 10000.0,
        "elapsed_time_s": 100.0,
        "mobility_displacing": 2.0,
        "mobility_displaced": 1.0,
    }
    field_descriptor = dict(descriptor)
    field_descriptor.update(
        {
            "length_scale_m": 100.0,
            "time_scale_s": 1000.0,
            "pressure_scale_pa": 2000000.0,
            "permeability_scale_m2": 2.0e-13,
            "porosity": 0.25,
            "velocity_scale_m_s": 1.0e-3,
            "flow_rate_m3_s": 2.0e-5,
            "delta_density_kg_m3": 5000.0,
        }
    )
    return {
        "case_id": "cross_scale_default",
        "lab_case": {"descriptor": descriptor},
        "field_case": {"descriptor": field_descriptor},
        "curves": [
            {
                "name": "water_cut",
                "lab": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.2, 0.5, 0.8], "unit": "fraction", "source": "lab"},
                "field": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.25, 0.45, 0.9], "unit": "fraction", "source": "field"},
            }
        ],
    }


def main() -> None:
    summary = run_cross_scale_benchmark()
    print(json.dumps({"success": summary["success"], "case_id": summary["case_id"]}, indent=2))


if __name__ == "__main__":
    main()
