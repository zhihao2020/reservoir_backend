"""Report runner for parameter fusion uncertainty enhancement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.fusion.kriging import deferred_assimilation_request, predict_spatial_field
from reservoir_backend.fusion.uncertainty import deferred_ensemble_update, uncertainty_weighted_fusion
from reservoir_backend.fusion.uncertainty_diagnostics import build_uncertainty_diagnostics_report


LIMITATIONS = [
    "No complete EnKF workflow.",
    "No ES-MDA history matching implemented.",
    "No automatic calibration implemented.",
    "No Bayesian inversion workflow implemented.",
    "No commercial geostatistical modeling implemented.",
    "No Petrel-like workflow implemented.",
    "No front-end integration implemented.",
    "No UDP implementation.",
]


def run_parameter_fusion_uncertainty_report(output_dir: str | Path = "accuracy_reports") -> dict[str, Any]:
    """Run uncertainty enhancement cases and write JSON/Markdown reports."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cases = [
        _variance_weighted_case(),
        _confidence_case(),
        _kriging_case(),
        _fallback_case(),
        _deferred_ensemble_case(),
    ]
    summary = {
        "report_name": "parameter_fusion_uncertainty",
        "success": bool(all(case["success"] for case in cases)),
        "num_cases": len(cases),
        "num_passed": int(sum(case["success"] for case in cases)),
        "num_failed": int(sum(not case["success"] for case in cases)),
        "uncertainty_cases": [case for case in cases if "variance" in case["case_name"] or "confidence" in case["case_name"]],
        "kriging_or_gp_cases": [case for case in cases if "kriging" in case["case_name"]],
        "fallback_cases": [case for case in cases if case["key_metrics"].get("fallback_used")],
        "cases": cases,
        "limitations": LIMITATIONS,
        "warnings": [warning for case in cases for warning in case.get("warnings", [])],
    }
    json_path = output_path / "parameter_fusion_uncertainty_summary.json"
    markdown_path = output_path / "parameter_fusion_uncertainty_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(_to_markdown(summary), encoding="utf-8")
    return summary


def _variance_weighted_case() -> dict[str, Any]:
    low_variance = np.full((2, 3), 10.0)
    high_variance = np.full((2, 3), 0.0)
    fused, variance, report = uncertainty_weighted_fusion(
        [low_variance, high_variance],
        variances=[np.full((2, 3), 4.0), np.full((2, 3), 0.25)],
        bounds=(0.0, 20.0),
    )
    diagnostics = build_uncertainty_diagnostics_report(
        fused,
        variance,
        bounds=(0.0, 20.0),
        dominant_source=report["dominant_source"],
        weighting_policy=report["weighting_policy"],
        fallback_used=report["fallback_used"],
    )
    return {
        "case_name": "variance_weighted_uncertainty_fusion",
        "success": bool(np.mean(np.abs(fused - high_variance)) < np.mean(np.abs(fused - low_variance)) and diagnostics["success"]),
        "grid_shape": list(fused.shape),
        "key_metrics": {
            "fused_mean": float(np.nanmean(fused)),
            "variance_mean": diagnostics["variance_mean"],
            "dominant_source": report["dominant_source"],
            "weighting_policy": report["weighting_policy"],
            "fallback_used": report["fallback_used"],
        },
        "diagnostics": diagnostics,
        "warnings": report["warnings"] + diagnostics["warnings"],
        "limitations": ["Variance weighting is a local fusion policy, not a geological uncertainty loop."],
    }


def _confidence_case() -> dict[str, Any]:
    low_conf = np.zeros((2, 3))
    high_conf = np.ones((2, 3))
    fused, variance, report = uncertainty_weighted_fusion(
        [low_conf, high_conf],
        confidences=[np.full((2, 3), 0.1), np.full((2, 3), 0.9)],
    )
    return {
        "case_name": "confidence_weighted_uncertainty_fusion",
        "success": bool(np.mean(fused) > 0.75 and report["weighting_policy"] == "confidence"),
        "grid_shape": list(fused.shape),
        "key_metrics": {
            "fused_mean": float(np.mean(fused)),
            "variance_mean": float(np.nanmean(variance)),
            "dominant_source": report["dominant_source"],
            "weighting_policy": report["weighting_policy"],
            "fallback_used": report["fallback_used"],
        },
        "warnings": report["warnings"],
        "limitations": ["Confidence remains a user-provided weight, not calibrated probability."],
    }


def _kriging_case() -> dict[str, Any]:
    points = np.array([[0.0], [1.0], [2.0], [3.0]])
    values = np.array([1.0, 2.0, 1.5, 3.0])
    targets = np.linspace(0.0, 3.0, 6).reshape(-1, 1)
    prediction, variance, report = predict_spatial_field(points, values, targets, method="auto")
    return {
        "case_name": "lightweight_kriging_gp_interface",
        "success": bool(np.isfinite(prediction).all() and np.isfinite(variance).all() and (variance >= 0.0).all()),
        "grid_shape": list(prediction.shape),
        "key_metrics": {
            "prediction_mean": float(np.mean(prediction)),
            "variance_mean": float(np.mean(variance)),
            "method_used": report["method_used"],
            "fallback_used": report["fallback_used"],
        },
        "warnings": report["warnings"],
        "limitations": ["This is a lightweight interface/fallback, not commercial geostatistical modeling."],
    }


def _fallback_case() -> dict[str, Any]:
    fused, variance, report = uncertainty_weighted_fusion(
        [np.ones((2, 2)), np.full((2, 2), 3.0)],
        weights=[1.0, 3.0],
    )
    return {
        "case_name": "explicit_weight_fallback_uncertainty",
        "success": bool(report["fallback_used"] and report["weighting_policy"] == "explicit_weight"),
        "grid_shape": list(fused.shape),
        "key_metrics": {
            "fused_mean": float(np.mean(fused)),
            "variance_mean": float(np.nanmean(variance)),
            "weighting_policy": report["weighting_policy"],
            "fallback_used": report["fallback_used"],
        },
        "warnings": report["warnings"],
        "limitations": ["Fallback uses explicit weights when no variance/std/confidence is provided."],
    }


def _deferred_ensemble_case() -> dict[str, Any]:
    enkf = deferred_ensemble_update("EnKF")
    esmda = deferred_assimilation_request("ES-MDA")
    return {
        "case_name": "enkf_esmda_deferred_scope",
        "success": bool(enkf["deferred"] and esmda["deferred"]),
        "grid_shape": [],
        "key_metrics": {
            "enkf_deferred": enkf["deferred"],
            "esmda_deferred": esmda["deferred"],
            "fallback_used": True,
        },
        "warnings": list(enkf["warnings"]) + list(esmda["warnings"]),
        "limitations": ["EnKF and ES-MDA are future work; no history matching was performed."],
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Parameter Fusion Uncertainty Summary",
        "",
        f"- success: `{summary['success']}`",
        f"- num_cases: `{summary['num_cases']}`",
        f"- num_passed: `{summary['num_passed']}`",
        "",
        "## Cases",
        "",
        "| Case | Success | Key Metrics |",
        "| --- | --- | --- |",
    ]
    for case in summary["cases"]:
        lines.append(f"| {case['case_name']} | {case['success']} | `{json.dumps(case['key_metrics'], sort_keys=True)}` |")
    lines.extend(["", "## Limitations", ""])
    for limitation in summary["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = run_parameter_fusion_uncertainty_report()
    print(json.dumps({"success": summary["success"], "num_cases": summary["num_cases"]}, sort_keys=True))


if __name__ == "__main__":
    main()
