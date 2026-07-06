"""Report runner for lightweight synthetic twin dynamic field fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.fusion.dynamic_field_fusion import build_synthetic_twin_fusion_summary
from reservoir_backend.fusion.synthetic_twin import (
    DynamicFieldRecord,
    ProductionSeriesRecord,
    StaticFieldRecord,
    SyntheticTwinMetadata,
)


def build_synthetic_twin_fixture() -> tuple[
    SyntheticTwinMetadata,
    list[StaticFieldRecord],
    list[DynamicFieldRecord],
    list[ProductionSeriesRecord],
]:
    """Return a deterministic small synthetic twin fixture."""
    grid_shape = (2, 3, 4)
    time_steps = (0.0, 100.0, 200.0)
    metadata = SyntheticTwinMetadata(
        twin_id="synthetic_twin_f4_04",
        case_id="synthetic_dynamic_fusion",
        run_id="run_f4_04",
        created_at="2026-07-06T00:00:00+00:00",
        grid_shape=grid_shape,
        time_steps=time_steps,
        source_name="synthetic_fixture",
        metadata={"stage": "F4-04"},
    )
    base = np.arange(np.prod(grid_shape), dtype=float).reshape(grid_shape)
    k_truth = 1.0e-13 * (1.0 + base / np.max(base))
    phi_truth = np.full(grid_shape, 0.24)
    pressure_truth = np.stack([(3.0 - 0.3 * t) + 0.01 * base for t in range(len(time_steps))])
    saturation_truth = np.stack([np.clip(0.2 + 0.08 * t + 0.004 * base, 0.0, 1.0) for t in range(len(time_steps))])
    water_cut_truth = np.array([0.0, 0.15, 0.42])
    static_records = [
        StaticFieldRecord(
            "permeability",
            k_truth * 1.08,
            "m2",
            "geology_model",
            confidence=0.55,
            provenance={"kind": "static_prior"},
            truth=k_truth,
        ),
        StaticFieldRecord(
            "permeability",
            k_truth * 0.98,
            "m2",
            "dynamic_interpretation",
            confidence=0.85,
            provenance={"kind": "dynamic_proxy"},
            truth=k_truth,
        ),
        StaticFieldRecord(
            "porosity",
            phi_truth + 0.015,
            "fraction",
            "core_lab",
            confidence=0.65,
            provenance={"kind": "lab"},
            truth=phi_truth,
        ),
        StaticFieldRecord(
            "porosity",
            phi_truth - 0.005,
            "fraction",
            "seismic_proxy",
            confidence=0.75,
            provenance={"kind": "proxy"},
            truth=phi_truth,
        ),
    ]
    dynamic_records = [
        DynamicFieldRecord(
            "pressure",
            pressure_truth + 0.05,
            time_steps,
            "Pa",
            "simulation_prior",
            confidence=0.6,
            provenance={"run": "prior"},
            truth=pressure_truth,
        ),
        DynamicFieldRecord(
            "pressure",
            pressure_truth - 0.02,
            time_steps,
            "Pa",
            "sensor_assimilated",
            confidence=0.9,
            provenance={"run": "sensor"},
            truth=pressure_truth,
        ),
        DynamicFieldRecord(
            "saturation",
            np.clip(saturation_truth - 0.02, 0.0, 1.0),
            time_steps,
            "fraction",
            "simulation_prior",
            confidence=0.55,
            provenance={"run": "prior"},
            truth=saturation_truth,
        ),
        DynamicFieldRecord(
            "saturation",
            np.clip(saturation_truth + 0.01, 0.0, 1.0),
            time_steps,
            "fraction",
            "sensor_proxy",
            confidence=0.8,
            provenance={"run": "sensor"},
            truth=saturation_truth,
        ),
    ]
    production_records = [
        ProductionSeriesRecord(
            "water_cut",
            time_steps,
            water_cut_truth - 0.03,
            "fraction",
            "simulation_prior",
            confidence=0.45,
            provenance={"series": "prior"},
            truth=water_cut_truth,
        ),
        ProductionSeriesRecord(
            "water_cut",
            time_steps,
            water_cut_truth + 0.01,
            "fraction",
            "production_observation",
            confidence=0.9,
            provenance={"series": "observed"},
            truth=water_cut_truth,
        ),
    ]
    return metadata, static_records, dynamic_records, production_records


def run_synthetic_twin_report(output_dir: str | Path = "accuracy_reports") -> dict[str, object]:
    """Run the synthetic twin fixture and write JSON/Markdown reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata, static_records, dynamic_records, production_records = build_synthetic_twin_fixture()
    summary = build_synthetic_twin_fusion_summary(
        metadata=metadata,
        static_records=static_records,
        dynamic_records=dynamic_records,
        production_records=production_records,
    ).to_dict()
    summary["report_json_path"] = str(output_path / "fusion_synthetic_twin_summary.json")
    summary["report_markdown_path"] = str(output_path / "fusion_synthetic_twin_summary.md")
    summary["source_task"] = "F4-04"
    (output_path / "fusion_synthetic_twin_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "fusion_synthetic_twin_summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Synthetic Twin Dynamic Field Fusion Summary",
        "",
        f"- success: {summary['success']}",
        f"- twin_id: {summary['metadata']['twin_id']}",
        f"- static fields: {summary['diagnostics']['num_static_fields']}",
        f"- dynamic fields: {summary['diagnostics']['num_dynamic_fields']}",
        f"- production series: {summary['diagnostics']['num_production_series']}",
        f"- overall_rmse: {summary['diagnostics']['overall_rmse']}",
        f"- total_bound_violations: {summary['diagnostics']['total_bound_violations']}",
        "",
        "## Static Fields",
        "",
    ]
    for name, report in summary["static_fields"].items():
        lines.append(f"- {name}: sources={report['source_count']}, rmse={report['truth_error']['rmse']}")
    lines.extend(["", "## Dynamic Fields", ""])
    for name, report in summary["dynamic_fields"].items():
        lines.append(f"- {name}: sources={report['source_count']}, rmse={report['truth_error']['rmse']}")
    lines.extend(["", "## Production Series", ""])
    for name, report in summary["production_series"].items():
        lines.append(f"- {name}: sources={report['source_count']}, rmse={report['truth_error']['rmse']}")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "- No history matching is implemented.",
            "- No EnKF / ES-MDA is implemented.",
            "- No automatic geological model update is implemented.",
            "- No closed-loop digital twin control is implemented.",
            "- No frontend, UDP, or REST API is implemented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_synthetic_twin_report()
    print(json.dumps({"success": summary["success"], "report": summary["report_json_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
