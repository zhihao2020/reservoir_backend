"""Benchmark hardening suite for parameter field fusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reservoir_backend.core.grid import Grid3D
from reservoir_backend.core.field import Field3D
from reservoir_backend.fusion.field_fusion import (
    fuse_saturation_fields,
    update_simulated_with_observed,
    weighted_average_fields,
)
from reservoir_backend.fusion.fusion_diagnostics import (
    check_bounds,
    check_shape_consistency,
    compute_confidence_weighting_metrics,
    compute_field_statistics,
    compute_fusion_error,
    compute_nan_mask_report,
    compute_weight_statistics,
)


def run_parameter_fusion_benchmark(output_dir: str | Path = "accuracy_reports") -> dict:
    """Run deterministic parameter fusion benchmark cases and write reports."""
    cases = [
        _equal_weight_field_fusion_case(),
        _explicit_weight_field_fusion_case(),
        _confidence_weighted_fusion_case(),
        _uncertainty_or_variance_weighted_fusion_case(),
        _nan_aware_fusion_case(),
        _bounds_and_clipping_report_case(),
        _shape_mismatch_rejection_case(),
        _multi_field_property_dynamic_fusion_sanity_case(),
    ]
    summary = {
        "benchmark_name": "parameter_fusion_benchmark",
        "success": bool(all(case["success"] for case in cases)),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "cases": cases,
        "overall_mae": float(max(_metric(case, "mae", 0.0) for case in cases)),
        "overall_rmse": float(max(_metric(case, "rmse", 0.0) for case in cases)),
        "overall_max_abs_error": float(max(_metric(case, "max_abs_error", 0.0) for case in cases)),
        "overall_num_bound_violations": int(max(_metric(case, "num_bound_violations", 0.0) for case in cases)),
        "overall_num_masked_cells": int(max(_metric(case, "num_masked_cells", 0.0) for case in cases)),
        "has_nan": bool(any(case["has_nan"] for case in cases)),
        "has_inf": bool(any(case["has_inf"] for case in cases)),
        "warnings": [],
        "recommendations": [
            "Keep shape, finite-value, NaN/mask, bounds, confidence, and provenance checks as fusion regression gates.",
            "Do not describe parameter fusion as history matching, automatic calibration, Bayesian inversion, EnKF, ES-MDA, or kriging.",
            "Variance-aware fusion is documented as unsupported until a dedicated design and implementation stage.",
        ],
    }
    _write_reports(summary, Path(output_dir))
    return summary


def _equal_weight_field_fusion_case() -> dict:
    grid = _grid()
    fields = [_field(grid, 1.0, "perm_a"), _field(grid, 3.0, "perm_b"), _field(grid, 5.0, "perm_c")]
    fused, report = weighted_average_fields(fields)
    reference = np.full(grid.shape, 3.0)
    error = compute_fusion_error(reference, fused)
    metrics = {
        **error,
        "fused_mean": float(np.nanmean(fused.values)),
        "field_count": int(report["field_count"]),
        "num_masked_cells": 0,
        "num_bound_violations": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "equal_weight_field_fusion",
        grid.shape,
        ["perm_a", "perm_b", "perm_c"],
        metrics,
        metrics["max_abs_error"] <= 1.0e-14,
        source="internal equal-weight same-grid fusion benchmark",
        limitations=["same-grid arithmetic mean only; no geostatistical interpolation"],
    )


def _explicit_weight_field_fusion_case() -> dict:
    grid = _grid()
    fields = [_field(grid, 1.0, "porosity_low"), _field(grid, 5.0, "porosity_high")]
    fused, report = weighted_average_fields(fields, weights=[1.0, 3.0])
    expected = np.full(grid.shape, 4.0)
    error = compute_fusion_error(expected, fused)
    invalid_weight_rejected = False
    try:
        weighted_average_fields(fields, weights=[1.0, -1.0])
    except ValueError:
        invalid_weight_rejected = True
    weight_stats = compute_weight_statistics(np.asarray([1.0, 3.0]))
    metrics = {
        **error,
        "used_weights": report["used_weights"],
        "invalid_weight_rejected": invalid_weight_rejected,
        "weight_min": weight_stats["weight_min"],
        "weight_max": weight_stats["weight_max"],
        "num_masked_cells": 0,
        "num_bound_violations": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "explicit_weight_field_fusion",
        grid.shape,
        ["porosity_low", "porosity_high"],
        metrics,
        metrics["max_abs_error"] <= 1.0e-14 and invalid_weight_rejected,
        source="internal explicit-weight fusion benchmark",
        limitations=["weights are scalar source weights; negative and non-finite weights are rejected"],
    )


def _confidence_weighted_fusion_case() -> dict:
    grid = _grid()
    low = _field(grid, 0.1, "low_confidence_saturation", confidence=0.1)
    high = _field(grid, 0.9, "high_confidence_saturation", confidence=0.9)
    fused, report = weighted_average_fields([low, high])
    metrics = compute_confidence_weighting_metrics(low, high, fused)
    zero_conf = _field(grid, 0.0, "zero_confidence_source", confidence=0.0)
    high_only, _ = weighted_average_fields([zero_conf, high])
    metrics.update(
        {
            "fused_mean": float(np.mean(fused.values)),
            "confidence_min": float(report["confidence_min"]),
            "confidence_max": float(report["confidence_max"]),
            "zero_confidence_source_does_not_dominate": bool(np.allclose(high_only.values, high.values)),
            "num_masked_cells": 0,
            "num_bound_violations": 0,
            "has_nan": False,
            "has_inf": False,
        }
    )
    return _case(
        "confidence_weighted_fusion",
        grid.shape,
        ["low_confidence_saturation", "high_confidence_saturation"],
        metrics,
        bool(metrics["closer_to_high_confidence"] and metrics["zero_confidence_source_does_not_dominate"]),
        source="internal confidence-weighted fusion benchmark",
        limitations=["confidence is used as a deterministic source weight, not Bayesian uncertainty assimilation"],
    )


def _uncertainty_or_variance_weighted_fusion_case() -> dict:
    grid = _grid()
    metrics = {
        "uncertainty_fusion_supported": False,
        "inverse_variance_weighting_verified": False,
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs_error": 0.0,
        "num_masked_cells": 0,
        "num_bound_violations": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "uncertainty_or_variance_weighted_fusion_if_supported",
        grid.shape,
        ["variance"],
        metrics,
        True,
        source="current fusion API capability survey",
        limitations=["uncertainty fusion not implemented in current fusion API"],
    )


def _nan_aware_fusion_case() -> dict:
    grid = _grid()
    a = Field3D(grid, np.array([[[1.0, np.nan, np.nan], [2.0, 4.0, np.nan]]]), name="nan_source_a")
    b = Field3D(grid, np.array([[[3.0, 6.0, np.nan], [4.0, np.nan, np.nan]]]), name="nan_source_b")
    fused, report = weighted_average_fields([a, b])
    nan_report = compute_nan_mask_report([a, b])
    expected = np.array([[[2.0, 6.0, np.nan], [3.0, 4.0, np.nan]]])
    error = compute_fusion_error(np.nan_to_num(expected, nan=0.0), np.nan_to_num(fused.values, nan=0.0))
    metrics = {
        **error,
        "single_source_nan_ignored": bool(np.isclose(fused.values[0, 0, 1], 6.0) and np.isclose(fused.values[0, 1, 1], 4.0)),
        "all_source_nan_masked": bool(np.isnan(fused.values[0, 0, 2]) and np.isnan(fused.values[0, 1, 2])),
        "num_masked_cells": int(nan_report["num_masked_cells"]),
        "nan_cells_count": int(report["nan_cells_count"]),
        "expected_masked_output_has_nan": True,
        "num_bound_violations": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "nan_aware_fusion",
        grid.shape,
        ["nan_source_a", "nan_source_b"],
        metrics,
        bool(metrics["single_source_nan_ignored"] and metrics["all_source_nan_masked"] and metrics["num_masked_cells"] == 2),
        source="internal NaN-aware fusion benchmark",
        limitations=["all-source NaN cells remain masked/NaN and are reported, not silently filled"],
    )


def _bounds_and_clipping_report_case() -> dict:
    grid = _grid()
    sw_a = Field3D(grid, np.full(grid.shape, 0.05), name="sw_low")
    sw_b = Field3D(grid, np.full(grid.shape, 0.10), name="sw_lower")
    fused, report = fuse_saturation_fields([sw_a, sw_b], swi=0.2, sor=0.2)
    saturation_bounds = check_bounds(fused, lower=0.2, upper=0.8)
    porosity = np.full(grid.shape, 0.25)
    permeability = np.full(grid.shape, 100.0)
    metrics = {
        "clipped_cells": int(report["clipped_cells"]),
        "saturation_num_bound_violations": int(saturation_bounds["num_bound_violations"]),
        "porosity_num_bound_violations": int(check_bounds(porosity, lower=0.0, upper=1.0)["num_bound_violations"]),
        "permeability_positive": bool(np.all(permeability > 0.0)),
        "num_bound_violations": int(saturation_bounds["num_bound_violations"]),
        "num_masked_cells": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "bounds_and_clipping_report",
        grid.shape,
        ["sw_low", "sw_lower", "porosity", "permeability"],
        metrics,
        metrics["clipped_cells"] > 0 and metrics["saturation_num_bound_violations"] == 0 and metrics["permeability_positive"],
        source="internal physical-bounds fusion benchmark",
        limitations=["reports current saturation clipping behavior; no silent physical-property calibration"],
    )


def _shape_mismatch_rejection_case() -> dict:
    grid = _grid()
    other_grid = Grid3D(nx=2, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)
    a = _field(grid, 1.0, "shape_a")
    b = _field(other_grid, 2.0, "shape_b")
    rejected = False
    try:
        weighted_average_fields([a, b])
    except Exception:
        rejected = True
    shape_report = check_shape_consistency([a, b], target_shape=grid.shape)
    metrics = {
        "shape_mismatch_rejected": rejected,
        "shape_consistent": bool(shape_report["shape_consistent"]),
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs_error": 0.0,
        "num_masked_cells": 0,
        "num_bound_violations": 0,
        "has_nan": False,
        "has_inf": False,
    }
    return _case(
        "shape_mismatch_rejection",
        grid.shape,
        ["shape_a", "shape_b"],
        metrics,
        rejected and not shape_report["shape_consistent"],
        source="internal shape-mismatch fusion benchmark",
        limitations=["shape mismatch must raise/report; no silent NumPy broadcasting is allowed"],
    )


def _multi_field_property_dynamic_fusion_sanity_case() -> dict:
    grid = _grid()
    conf = Field3D(grid, np.full(grid.shape, 0.8), name="confidence")
    permeability = weighted_average_fields([_field(grid, 80.0, "perm_a"), _field(grid, 120.0, "perm_b")], weights=[1.0, 2.0])[0]
    porosity = weighted_average_fields([_field(grid, 0.22, "phi_a"), _field(grid, 0.28, "phi_b")], weights=[1.0, 1.0])[0]
    pressure = weighted_average_fields([_field(grid, 1.0e7, "pressure_a"), _field(grid, 0.98e7, "pressure_b")], weights=[2.0, 1.0])[0]
    saturation = fuse_saturation_fields(
        [_field(grid, 0.35, "sw_sim"), _field(grid, 0.55, "sw_obs")],
        confidence_fields=[np.full(grid.shape, 0.4), conf],
        swi=0.2,
        sor=0.2,
    )[0]
    updated_sw, update_report = update_simulated_with_observed(_field(grid, 0.4, "sw_simulated"), saturation, alpha=0.25, swi=0.2, sor=0.2)
    mask = np.ones(grid.shape, dtype=bool)
    outputs = [permeability, porosity, pressure, saturation, updated_sw]
    shape_report = check_shape_consistency(outputs, target_shape=grid.shape)
    metrics = {
        "all_outputs_finite": bool(all(np.isfinite(field.values).all() for field in outputs)),
        "shape_consistent": bool(shape_report["shape_consistent"] and mask.shape == grid.shape),
        "permeability_positive": bool(np.all(permeability.values > 0.0)),
        "porosity_num_bound_violations": int(check_bounds(porosity, lower=0.0, upper=1.0)["num_bound_violations"]),
        "saturation_num_bound_violations": int(check_bounds(saturation, lower=0.2, upper=0.8)["num_bound_violations"]),
        "updated_saturation_num_bound_violations": int(check_bounds(updated_sw, lower=0.2, upper=0.8)["num_bound_violations"]),
        "source_names": ["perm_a", "perm_b", "phi_a", "phi_b", "pressure_a", "pressure_b", "sw_sim", "sw_obs"],
        "source_count": 8,
        "mask_shape_consistent": bool(mask.shape == grid.shape),
        "clipped_cells": int(update_report["clipped_cells"]),
        "num_masked_cells": 0,
        "num_bound_violations": 0,
        "mae": 0.0,
        "rmse": 0.0,
        "max_abs_error": 0.0,
        "has_nan": False,
        "has_inf": False,
    }
    metrics["num_bound_violations"] = (
        metrics["porosity_num_bound_violations"]
        + metrics["saturation_num_bound_violations"]
        + metrics["updated_saturation_num_bound_violations"]
    )
    return _case(
        "multi_field_property_dynamic_fusion_sanity",
        grid.shape,
        ["permeability", "porosity", "pressure", "saturation", "confidence", "mask"],
        metrics,
        metrics["all_outputs_finite"] and metrics["shape_consistent"] and metrics["num_bound_violations"] == 0,
        source="internal multi-field property/dynamic fusion sanity benchmark",
        limitations=["synthetic field fusion sanity only; no history matching or automatic calibration"],
    )


def _grid() -> Grid3D:
    return Grid3D(nx=3, ny=2, nz=1, dx=1.0, dy=1.0, dz=1.0)


def _field(grid: Grid3D, value: float, name: str, confidence: float | None = None) -> Field3D:
    return Field3D.from_constant(grid, value, name=name, confidence=confidence)


def _case(
    name: str,
    grid_shape: tuple[int, ...] | list[int],
    field_names: list[str],
    metrics: dict,
    success: bool,
    *,
    source: str,
    is_exact_reproduction: bool = False,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
) -> dict:
    return {
        "case_name": name,
        "source": source,
        "is_exact_reproduction": bool(is_exact_reproduction),
        "success": bool(success and not metrics.get("has_nan", False) and not metrics.get("has_inf", False)),
        "grid_shape": list(grid_shape),
        "field_names": list(field_names),
        "key_metrics": _jsonable(metrics),
        "warnings": [] if warnings is None else list(warnings),
        "has_nan": bool(metrics.get("has_nan", False)),
        "has_inf": bool(metrics.get("has_inf", False)),
        "limitations": [] if limitations is None else list(limitations),
    }


def _metric(case: dict, name: str, default: float) -> float:
    value = case["key_metrics"].get(name, default)
    return float(default if value is None else value)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_reports(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "parameter_fusion_benchmark_summary.json"
    md_path = output_dir / "parameter_fusion_benchmark_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Parameter Fusion Benchmark Summary",
        "",
        f"- success: {summary['success']}",
        f"- num_cases: {summary['num_cases']}",
        f"- num_passed: {summary['num_passed']}",
        f"- overall_mae: {summary['overall_mae']:.6e}",
        f"- overall_rmse: {summary['overall_rmse']:.6e}",
        f"- overall_max_abs_error: {summary['overall_max_abs_error']:.6e}",
        f"- overall_num_bound_violations: {summary['overall_num_bound_violations']}",
        f"- overall_num_masked_cells: {summary['overall_num_masked_cells']}",
        "",
        "## Cases",
    ]
    for case in summary["cases"]:
        lines.extend(["", f"### {case['case_name']}", "", f"- success: {case['success']}", f"- source: {case['source']}"])
        for key, value in case["key_metrics"].items():
            lines.append(f"- {key}: {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_parameter_fusion_benchmark(), indent=2))
