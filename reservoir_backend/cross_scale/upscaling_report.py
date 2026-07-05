from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.comparison import build_fine_coarse_comparison_report
from reservoir_backend.cross_scale.report import write_json_report, write_markdown_report
from reservoir_backend.cross_scale.runner import load_config, run_similarity_report
from reservoir_backend.cross_scale.scale_conversion import build_scale_conversion_report, descriptors_from_config


NON_CLAIMS = [
    "No complex upscaling solver.",
    "No multiscale finite-volume implementation.",
    "No history matching.",
    "No automatic calibration.",
    "No commercial simulator equivalence.",
    "No validation of black-oil models.",
    "No front-end.",
    "No UDP.",
]


def build_upscaling_assumption_report(
    config: Mapping[str, Any],
    scale_conversion_report: Mapping[str, Any],
    similarity_report: Mapping[str, Any],
) -> dict[str, Any]:
    properties = config.get("upscaling_properties", {})
    permeability = np.asarray(properties.get("permeability_values", [1.0e-12, 2.0e-12, 4.0e-12]), dtype=float)
    porosity = np.asarray(properties.get("porosity_values", [0.18, 0.22, 0.24]), dtype=float)
    _validate_positive_array("permeability_values", permeability)
    _validate_interval_array("porosity_values", porosity, 0.0, 1.0)
    arithmetic = float(np.mean(permeability))
    harmonic = float(len(permeability) / np.sum(1.0 / permeability))
    porosity_average = float(np.mean(porosity))
    flow_rate_scaling = float(scale_conversion_report["flow_rate_scale_ratio"])
    velocity_scaling = float(scale_conversion_report["velocity_scale_ratio"])
    regime_shift = any(
        score is not None and score < 0.05
        for name, score in similarity_report.get("criterion_scores", {}).items()
        if name in {"capillary", "peclet", "gravity_number"}
    )
    warnings = []
    if regime_shift:
        warnings.append("regime shift indicated by low capillary/Peclet/gravity similarity scores")
    return {
        "success": True,
        "properties_may_be_upscaled": ["permeability", "porosity", "flow_rate"],
        "properties_should_not_be_upscaled_directly": [
            "capillary pressure curve without validation",
            "relative permeability curve without validation",
            "history-matched parameters",
        ],
        "assumptions": [
            "Permeability arithmetic mean is a diagnostic upper-tendency estimate.",
            "Permeability harmonic mean is a diagnostic lower-tendency estimate.",
            "Porosity volume average uses equal cell volumes in the synthetic fixture.",
            "Flow-rate scaling sanity is based on field/lab scale ratios.",
        ],
        "arithmetic_mean_permeability": arithmetic,
        "harmonic_mean_permeability": harmonic,
        "porosity_volume_average": porosity_average,
        "flow_rate_scaling_sanity": flow_rate_scaling,
        "velocity_scaling_sanity": velocity_scaling,
        "regime_shift_flag": regime_shift,
        "regime_shift_meaning": "A regime shift means lab-scale trends may not transfer directly to field-scale behavior.",
        "validation_required_before_use": [
            "Validate coarse outputs against fine-grid or field observations.",
            "Check material balance and boundedness in the target simulator.",
            "Re-check similarity criteria after changing scale or fluid properties.",
        ],
        "warnings": warnings,
        "limitations": [
            "These are lightweight diagnostics, not a complete multiscale finite-volume solver.",
            "Upscaled values should not be used as calibrated reservoir properties without validation.",
        ],
    }


def build_cross_scale_upscaling_summary(config: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config(config or _default_upscaling_config())
    lab_descriptor, field_descriptor = descriptors_from_config(cfg)
    scale_report = build_scale_conversion_report(lab_descriptor, field_descriptor)
    similarity_report = run_similarity_report(cfg)
    assumption_report = build_upscaling_assumption_report(cfg, scale_report, similarity_report)
    comparison_report = build_fine_coarse_comparison_report(cfg)
    warnings = [
        *scale_report.get("warnings", []),
        *similarity_report.get("warnings", []),
        *assumption_report.get("warnings", []),
        *comparison_report.get("warnings", []),
    ]
    success = bool(
        scale_report["success"]
        and similarity_report["success"]
        and assumption_report["success"]
        and comparison_report["success"]
    )
    json_path = Path("accuracy_reports/cross_scale_upscaling_summary.json")
    return {
        "report_name": "cross_scale_upscaling_summary",
        "case_id": cfg.get("case_id", "cross_scale_upscaling_case"),
        "success": success,
        "scale_conversion_report": scale_report,
        "similarity_criteria_report": similarity_report,
        "upscaling_assumption_report": assumption_report,
        "fine_coarse_comparison_report": comparison_report,
        "result_manifest_entry": _result_manifest_entry(json_path, str(cfg.get("case_id", "cross_scale_upscaling_case"))),
        "warnings": warnings,
        "limitations": list(NON_CLAIMS),
        "non_claims": list(NON_CLAIMS),
        "has_nan": _contains_nan([scale_report, similarity_report, assumption_report, comparison_report]),
        "has_inf": _contains_inf([scale_report, similarity_report, assumption_report, comparison_report]),
    }


def write_upscaling_summary_reports(summary: Mapping[str, Any], output_dir: str | Path = "accuracy_reports") -> dict[str, str]:
    output_path = Path(output_dir)
    json_path = write_json_report(summary, output_path / "cross_scale_upscaling_summary.json")
    markdown_path = _write_upscaling_markdown(summary, output_path / "cross_scale_upscaling_summary.md")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def run_cross_scale_upscaling_report(
    config: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path = "accuracy_reports",
) -> dict[str, Any]:
    summary = build_cross_scale_upscaling_summary(config)
    write_upscaling_summary_reports(summary, output_dir)
    return summary


def _write_upscaling_markdown(summary: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = summary["scale_conversion_report"]
    assumptions = summary["upscaling_assumption_report"]
    comparison = summary["fine_coarse_comparison_report"]
    lines = [
        "# Cross-Scale Upscaling Summary",
        "",
        f"- success: {summary['success']}",
        f"- case_id: {summary['case_id']}",
        f"- similarity_score: {summary['similarity_criteria_report'].get('overall_similarity_score')}",
        f"- regime_shift_flag: {assumptions.get('regime_shift_flag')}",
        "",
        "## Scale Conversion",
        "",
        "| field | lab | field | ratio |",
        "| --- | --- | --- | --- |",
    ]
    for prefix in ("length_scale", "time_scale", "pressure_scale", "permeability_scale", "velocity_scale", "flow_rate_scale", "porosity"):
        lines.append(f"| {prefix} | {scale[f'{prefix}_lab']} | {scale[f'{prefix}_field']} | {scale[f'{prefix}_ratio']} |")
    lines.extend(
        [
            "",
            "## Upscaling Assumptions",
            "",
            f"- arithmetic mean permeability: {assumptions['arithmetic_mean_permeability']}",
            f"- harmonic mean permeability: {assumptions['harmonic_mean_permeability']}",
            f"- porosity volume average: {assumptions['porosity_volume_average']}",
            f"- flow-rate scaling sanity: {assumptions['flow_rate_scaling_sanity']}",
            "",
            "## Fine-Grid vs Coarse-Grid Comparison",
            "",
            "| metric | RMSE | MAE | R2 | NRMSE | max abs error |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in comparison["curve_reports"]:
        lines.append(
            f"| {item['metric']} | {item['rmse']} | {item['mae']} | {item['r2']} | {item['nrmse']} | {item['max_abs_error']} |"
        )
    lines.extend(["", "## Non-Claims", ""])
    lines.extend(f"- {item}" for item in summary["non_claims"])
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _result_manifest_entry(json_path: Path, case_id: str) -> dict[str, Any]:
    try:
        from reservoir_backend.results.manifest import ResultManifest

        return ResultManifest(
            result_id="cross_scale_upscaling_summary",
            case_id=case_id,
            run_id="cross-scale-upscaling-report",
            module="M6",
            result_type="cross_scale_report",
            field_name="cross_scale_upscaling_summary",
            shape=[],
            dtype="json",
            unit="dimensionless",
            path=str(json_path),
            format="json",
            source_task="TASK-017",
            source_report=str(json_path),
            metadata={"contains": ["scale_conversion_report", "upscaling_assumption_report", "fine_coarse_comparison_report"]},
            warnings=[],
            limitations=list(NON_CLAIMS),
        ).to_dict()
    except Exception as exc:
        return {"module": "M6", "result_type": "cross_scale_report", "source_task": "TASK-017", "warning": str(exc)}


def _default_upscaling_config() -> dict[str, Any]:
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
        "interfacial_tension_n_m": 0.03,
        "diffusivity_m2_s": 1.0e-9,
        "delta_density_kg_m3": 10.0,
        "pressure_drop_pa": 10000.0,
        "elapsed_time_s": 100.0,
        "mobility_displacing": 2.0,
        "mobility_displaced": 1.0,
    }
    field = dict(descriptor)
    field.update(
        {
            "length_scale_m": 100.0,
            "time_scale_s": 1000.0,
            "pressure_scale_pa": 2000000.0,
            "permeability_scale_m2": 2.0e-13,
            "porosity": 0.25,
            "velocity_scale_m_s": 0.001,
            "flow_rate_m3_s": 2.0e-5,
            "delta_density_kg_m3": 5000.0,
        }
    )
    return {
        "case_id": "cross_scale_upscaling_default",
        "lab_case": {"descriptor": descriptor},
        "field_case": {"descriptor": field},
        "curves": [
            {
                "name": "water_cut",
                "lab": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.2, 0.5, 0.8]},
                "field": {"name": "water_cut", "time": [0, 1, 2, 3], "values": [0.0, 0.25, 0.45, 0.9]},
            }
        ],
        "upscaling_properties": {
            "permeability_values": [1.0e-12, 2.0e-12, 4.0e-12],
            "porosity_values": [0.18, 0.22, 0.24],
        },
        "fine_coarse_comparison": _default_comparison_curves(),
    }


def _default_comparison_curves() -> list[dict[str, Any]]:
    return [
        _comparison("pressure", [100000, 90000, 80000, 70000], [100000, 91000, 79000, 71000], "Pa"),
        _comparison("saturation", [0.2, 0.35, 0.55, 0.7], [0.2, 0.33, 0.57, 0.68], "fraction"),
        _comparison("production", [0.0, 10.0, 22.0, 35.0], [0.0, 9.5, 23.0, 34.0], "m3"),
    ]


def _comparison(metric: str, fine_values: list[float], coarse_values: list[float], unit: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "fine": {"name": metric, "time": [0, 1, 2, 3], "values": fine_values, "unit": unit, "source": "synthetic fine grid"},
        "coarse": {"name": metric, "time": [0, 1, 2, 3], "values": coarse_values, "unit": unit, "source": "synthetic coarse grid"},
    }


def _validate_positive_array(name: str, values: np.ndarray) -> None:
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise InvalidPhysicalValueError(f"{name} must contain positive finite values")


def _validate_interval_array(name: str, values: np.ndarray, lower: float, upper: float) -> None:
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values < lower) or np.any(values > upper):
        raise InvalidPhysicalValueError(f"{name} must be finite within [{lower}, {upper}]")


def _contains_nan(items: list[Any]) -> bool:
    text = json.dumps(items, default=str)
    return "NaN" in text


def _contains_inf(items: list[Any]) -> bool:
    text = json.dumps(items, default=str)
    return "Infinity" in text or "-Infinity" in text


def main() -> None:
    summary = run_cross_scale_upscaling_report()
    print(json.dumps({"success": summary["success"], "case_id": summary["case_id"]}, indent=2))


if __name__ == "__main__":
    main()
