from __future__ import annotations

from typing import Any, Mapping

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.validation import CurveData, validate_multiple_curve_pairs


def build_fine_coarse_comparison_report(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare synthetic or supplied fine-grid and coarse-grid curves.

    This is a report framework. It does not run a grid coarsening algorithm or a
    multiscale finite-volume solver.
    """
    comparisons = config.get("fine_coarse_comparison")
    if not isinstance(comparisons, list) or not comparisons:
        raise InvalidPhysicalValueError("fine_coarse_comparison must contain at least one curve pair")

    pair_reports: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item in comparisons:
        if not isinstance(item, Mapping):
            raise InvalidPhysicalValueError("fine_coarse_comparison entries must be mappings")
        metric_name = str(item.get("metric", item.get("name", "unknown")))
        fine_data = item.get("fine")
        coarse_data = item.get("coarse")
        if not isinstance(fine_data, Mapping) or not isinstance(coarse_data, Mapping):
            raise InvalidPhysicalValueError("each fine/coarse comparison must contain fine and coarse curves")
        fine = CurveData.from_dict(_with_name(fine_data, metric_name))
        coarse = CurveData.from_dict(_with_name(coarse_data, metric_name))
        summary = validate_multiple_curve_pairs([(fine, coarse)])
        curve_report = summary["curve_reports"][0]
        metric_report = {
            "metric": metric_name,
            "success": bool(curve_report.get("success")),
            "rmse": curve_report.get("rmse"),
            "mae": curve_report.get("mae"),
            "mape": curve_report.get("mape"),
            "r2": curve_report.get("r2"),
            "nrmse": curve_report.get("normalized_rmse"),
            "max_abs_error": curve_report.get("max_absolute_error"),
            "num_matched_samples": curve_report.get("num_points"),
            "overlap_interval": {
                "start": curve_report.get("time_start"),
                "end": curve_report.get("time_end"),
            },
            "warnings": list(curve_report.get("warnings", [])),
        }
        if metric_report["warnings"]:
            warnings.extend(f"{metric_name}: {warning}" for warning in metric_report["warnings"])
        pair_reports.append(metric_report)

    successful = [item for item in pair_reports if item["success"]]
    return {
        "success": len(successful) > 0,
        "comparison_type": "fine_grid_vs_coarse_grid",
        "source": "synthetic fixture or supplied curves",
        "pressure_curve_comparison": _find_metric(pair_reports, "pressure"),
        "saturation_curve_comparison": _find_metric(pair_reports, "saturation"),
        "production_curve_comparison": _find_metric(pair_reports, "production"),
        "curve_reports": pair_reports,
        "warnings": warnings,
        "limitations": [
            "Synthetic curves are used when real fine/coarse data are not available.",
            "This framework compares outputs; it does not generate coarse-grid states.",
        ],
    }


def _with_name(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    result = dict(data)
    result.setdefault("name", name)
    return result


def _find_metric(reports: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    for report in reports:
        if report["metric"] == metric:
            return report
    return None
