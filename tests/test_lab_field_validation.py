from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.scale_effect import build_scale_effect_report
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.similarity import build_similarity_report
from reservoir_backend.cross_scale.validation import (
    CurveData,
    align_curves_to_common_time,
    compute_mae,
    compute_mape,
    compute_max_absolute_error,
    compute_normalized_rmse,
    compute_r2,
    compute_rmse,
    validate_curve_pair,
    validate_multiple_curve_pairs,
)


def _curve(name: str = "water_cut", *, shift: float = 0.0, values: list[float] | None = None) -> CurveData:
    return CurveData(
        name=name,
        time=np.array([0.0, 1.0, 2.0, 3.0]) + shift,
        values=np.array(values if values is not None else [0.0, 0.2, 0.5, 0.8]),
        unit="-",
        curve_type="water cut",
        source="lab",
    )


def _target() -> CurveData:
    return CurveData(
        name="water_cut",
        time=np.array([0.0, 1.0, 2.0, 3.0]),
        values=np.array([0.0, 0.25, 0.45, 0.9]),
        unit="-",
        curve_type="water cut",
        source="simulation",
    )


def test_curve_data_from_dict() -> None:
    curve = CurveData.from_dict(
        {"name": "rate", "time": [0, 1], "values": [10, 12], "unit": "m3/s", "curve_type": "production rate"}
    )
    assert curve.name == "rate"
    assert curve.time.shape == (2,)


def test_curve_data_to_dict() -> None:
    data = _curve().to_dict()
    assert data["name"] == "water_cut"
    assert data["time"] == [0.0, 1.0, 2.0, 3.0]
    assert data["source"] == "lab"


def test_curve_data_validates_shape() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        CurveData(name="bad", time=np.array([0.0, 1.0]), values=np.array([1.0]))


def test_curve_data_rejects_nan_inf() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        CurveData(name="bad", time=np.array([0.0, 1.0]), values=np.array([np.nan, 1.0]))
    with pytest.raises(InvalidPhysicalValueError):
        CurveData(name="bad", time=np.array([0.0, np.inf]), values=np.array([0.0, 1.0]))


def test_curve_data_rejects_non_monotonic_time() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        CurveData(name="bad", time=np.array([0.0, 0.0, 1.0]), values=np.array([0.0, 1.0, 2.0]))


def test_curve_data_requires_at_least_two_points() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        CurveData(name="bad", time=np.array([0.0]), values=np.array([1.0]))


def test_align_curves_overlap_only() -> None:
    common_time, ref, target, report = align_curves_to_common_time(_curve(), _target())
    assert np.allclose(common_time, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(ref, _curve().values)
    assert report["overlap_only"] is True


def test_align_curves_interpolates_target() -> None:
    target = CurveData(name="water_cut", time=np.array([0.0, 2.0, 4.0]), values=np.array([0.0, 1.0, 2.0]))
    common_time, _, target_values, _ = align_curves_to_common_time(_curve(), target)
    assert np.allclose(common_time, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(target_values, [0.0, 0.5, 1.0, 1.5])


def test_align_curves_no_overlap_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        align_curves_to_common_time(_curve(), _curve(shift=10.0))


def test_align_curves_inputs_not_modified() -> None:
    reference = _curve()
    target = _target()
    ref_time = reference.time.copy()
    target_values = target.values.copy()
    align_curves_to_common_time(reference, target)
    assert np.allclose(reference.time, ref_time)
    assert np.allclose(target.values, target_values)


def test_rmse_formula() -> None:
    assert compute_rmse([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_mae_formula() -> None:
    assert compute_mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2.0 / 3.0)


def test_mape_formula() -> None:
    assert compute_mape([1.0, 2.0], [1.1, 1.8]) == pytest.approx(10.0)


def test_mape_handles_zero_reference() -> None:
    value = compute_mape([0.0, 1.0], [0.1, 1.1])
    assert np.isfinite(value)
    assert value > 0.0


def test_r2_formula() -> None:
    assert compute_r2([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(0.5)


def test_r2_constant_reference_warns() -> None:
    assert compute_r2([2.0, 2.0, 2.0], [2.0, 2.1, 1.9]) is None
    report = validate_curve_pair(_curve(values=[2.0, 2.0, 2.0, 2.0]), _target())
    assert report["r2"] is None
    assert any("r2 is undefined" in warning for warning in report["warnings"])


def test_normalized_rmse_range() -> None:
    assert compute_normalized_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], "range") == pytest.approx((1.0 / np.sqrt(3.0)) / 2.0)


def test_normalized_rmse_mean() -> None:
    assert compute_normalized_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], "mean") == pytest.approx((1.0 / np.sqrt(3.0)) / 2.0)


def test_normalized_rmse_std() -> None:
    expected = (1.0 / np.sqrt(3.0)) / np.std([1.0, 2.0, 3.0])
    assert compute_normalized_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], "std") == pytest.approx(expected)


def test_normalized_rmse_zero_denominator_warns() -> None:
    assert compute_normalized_rmse([1.0, 1.0], [1.0, 2.0], "range") is None
    report = validate_curve_pair(_curve(values=[1.0, 1.0, 1.0, 1.0]), _target())
    assert report["normalized_rmse"] is None
    assert any("normalized_rmse is undefined" in warning for warning in report["warnings"])


def test_max_absolute_error_formula() -> None:
    assert compute_max_absolute_error([1.0, 2.0, 3.0], [1.0, 4.0, 2.0]) == pytest.approx(2.0)


def test_validate_curve_pair_report_keys() -> None:
    report = validate_curve_pair(_curve(), _target())
    expected = {
        "curve_name",
        "curve_type",
        "reference_source",
        "target_source",
        "unit",
        "num_points",
        "time_start",
        "time_end",
        "rmse",
        "mae",
        "mape",
        "r2",
        "normalized_rmse",
        "max_absolute_error",
        "alignment_report",
        "warnings",
        "success",
        "has_nan",
        "has_inf",
    }
    assert expected.issubset(report)


def test_validate_curve_pair_success_true() -> None:
    assert validate_curve_pair(_curve(), _target())["success"] is True


def test_validate_curve_pair_alignment_report() -> None:
    alignment = validate_curve_pair(_curve(), _target())["alignment_report"]
    assert alignment["overlap_start"] == pytest.approx(0.0)
    assert alignment["overlap_end"] == pytest.approx(3.0)
    assert alignment["num_points"] == 4


def test_validate_curve_pair_warnings() -> None:
    reference = _curve(values=[0.0, 0.0, 0.0, 0.0])
    report = validate_curve_pair(reference, _target())
    assert report["warnings"]
    assert report["zero_reference_count"] == 4


def test_validate_curve_pair_no_nan_inf() -> None:
    report = validate_curve_pair(_curve(), _target())
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_validate_curve_pair_repeatability() -> None:
    assert validate_curve_pair(_curve(), _target()) == validate_curve_pair(_curve(), _target())


def test_validate_multiple_curve_pairs_keys() -> None:
    summary = validate_multiple_curve_pairs([(_curve(), _target())])
    assert {"success", "num_curves", "curve_reports", "aggregate_metrics", "warnings", "has_nan", "has_inf"}.issubset(summary)


def test_validate_multiple_curve_pairs_aggregate_metrics() -> None:
    summary = validate_multiple_curve_pairs([(_curve(), _target()), (_curve("gas_fraction"), _target())])
    aggregate = summary["aggregate_metrics"]
    assert aggregate["num_successful_curves"] == 2
    assert aggregate["num_failed_curves"] == 0
    assert aggregate["mean_rmse"] is not None
    assert aggregate["max_absolute_error"] is not None


def test_validate_multiple_curve_pairs_partial_failure() -> None:
    summary = validate_multiple_curve_pairs([(_curve(), _target()), (_curve("late"), _curve(shift=10.0))])
    assert summary["num_curves"] == 2
    assert summary["aggregate_metrics"]["num_successful_curves"] == 1
    assert summary["aggregate_metrics"]["num_failed_curves"] == 1
    assert summary["curve_reports"][1]["success"] is False


def test_validate_multiple_curve_pairs_warnings() -> None:
    summary = validate_multiple_curve_pairs([(_curve(values=[1.0, 1.0, 1.0, 1.0]), _target())])
    assert summary["warnings"]


def test_validation_module_no_solver_dependency() -> None:
    source = Path("reservoir_backend/cross_scale/validation.py").read_text(encoding="utf-8")
    for forbidden in ["reservoir_backend.solver", "reservoir_backend.inversion", "reservoir_backend.fusion", "reservoir_backend.cli", "reservoir_backend.io"]:
        assert forbidden not in source


def test_validation_module_does_not_modify_solver_files() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_similarity_and_scale_effect_tests_still_pass() -> None:
    descriptor = ScaleDescriptor.from_dict(
        {
            "length_scale_m": 1.0,
            "time_scale_s": 10.0,
            "pressure_scale_pa": 1.0e5,
            "permeability_scale_m2": 1.0e-12,
            "porosity": 0.2,
            "viscosity_pa_s": 1.0e-3,
            "density_kg_m3": 1000.0,
            "velocity_scale_m_s": 1.0e-6,
            "flow_rate_m3_s": 1.0e-9,
            "interfacial_tension_n_m": 0.03,
            "diffusivity_m2_s": 1.0e-9,
            "delta_density_kg_m3": 10.0,
            "pressure_drop_pa": 1.0e4,
            "elapsed_time_s": 100.0,
            "mobility_displacing": 2.0,
            "mobility_displaced": 1.0,
        }
    )
    assert build_similarity_report(descriptor, descriptor)["success"] is True
    assert build_scale_effect_report(descriptor, descriptor)["success"] is True


def test_cross_scale_design_doc_mentions_validation_done() -> None:
    text = Path("specs/13_cross_scale_analysis_design.md").read_text(encoding="utf-8")
    assert "044_lab_field_validation_module" in text
    assert "lab-field validation module is implemented" in text


def test_requirement_traceability_mentions_lab_field_validation_done() -> None:
    text = Path("specs/10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "| lab-field validation module |" in text
    assert "Done" in text


def test_readme_mentions_lab_field_validation() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "lab-field validation" in text
    assert "curve-to-curve comparison" in text


def test_validation_does_not_claim_history_matching() -> None:
    text = (
        Path("README.md").read_text(encoding="utf-8")
        + Path("docs/limitations_and_roadmap.md").read_text(encoding="utf-8")
        + Path("specs/13_cross_scale_analysis_design.md").read_text(encoding="utf-8")
    )
    assert "history matching is implemented" not in text
    assert "automatic calibration is implemented" not in text
