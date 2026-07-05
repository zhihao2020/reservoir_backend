"""Cross-scale analysis helpers.

The package currently contains similarity-criteria, scale-effect, and
curve-validation utilities. It is intentionally independent from numerical
solvers and CLI/YAML execution.
"""

from reservoir_backend.cross_scale.validation import (
    CurveData,
    align_curves_to_common_time,
    compute_mae,
    compute_mape,
    compute_max_absolute_error,
    compute_normalized_rmse,
    compute_r2,
    compute_rmse,
    validate_curve_data,
    validate_curve_pair,
    validate_multiple_curve_pairs,
)
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.scale_effect import (
    build_scale_effect_report,
    classify_flow_regime,
    compute_scale_ratios,
    detect_regime_shift,
)
from reservoir_backend.cross_scale.similarity import (
    CriterionResult,
    build_similarity_report,
    compute_capillary_number,
    compute_criterion_similarity_score,
    compute_dimensionless_numbers,
    compute_dimensionless_pressure,
    compute_dimensionless_time,
    compute_gravity_number,
    compute_mobility_ratio,
    compute_overall_similarity_score,
    compute_peclet_number,
    compute_reynolds_number,
)

__all__ = [
    "CriterionResult",
    "CurveData",
    "ScaleDescriptor",
    "align_curves_to_common_time",
    "build_scale_effect_report",
    "build_similarity_report",
    "classify_flow_regime",
    "compute_mae",
    "compute_mape",
    "compute_max_absolute_error",
    "compute_normalized_rmse",
    "compute_r2",
    "compute_rmse",
    "compute_capillary_number",
    "compute_criterion_similarity_score",
    "compute_dimensionless_numbers",
    "compute_dimensionless_pressure",
    "compute_dimensionless_time",
    "compute_gravity_number",
    "compute_mobility_ratio",
    "compute_overall_similarity_score",
    "compute_peclet_number",
    "compute_reynolds_number",
    "compute_scale_ratios",
    "detect_regime_shift",
    "validate_curve_data",
    "validate_curve_pair",
    "validate_multiple_curve_pairs",
]
