from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.similarity import (
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


def _descriptor_dict() -> dict[str, float]:
    return {
        "length_scale_m": 2.0,
        "time_scale_s": 100.0,
        "pressure_scale_pa": 1.0e6,
        "permeability_scale_m2": 1.0e-12,
        "porosity": 0.25,
        "viscosity_pa_s": 1.0e-3,
        "density_kg_m3": 1000.0,
        "velocity_scale_m_s": 1.0e-5,
        "flow_rate_m3_s": 1.0e-8,
        "temperature_scale_k": 300.0,
        "interfacial_tension_n_m": 0.03,
        "diffusivity_m2_s": 1.0e-9,
        "delta_density_kg_m3": 200.0,
        "gravity_m_s2": 9.80665,
        "pressure_drop_pa": 2.0e5,
        "elapsed_time_s": 50.0,
        "mobility_displacing": 3.0,
        "mobility_displaced": 2.0,
    }


def _descriptor() -> ScaleDescriptor:
    return ScaleDescriptor.from_dict(_descriptor_dict())


def _field_descriptor() -> ScaleDescriptor:
    data = _descriptor_dict()
    data.update(
        {
            "length_scale_m": 20.0,
            "velocity_scale_m_s": 2.0e-5,
            "pressure_drop_pa": 4.0e5,
            "elapsed_time_s": 200.0,
        }
    )
    return ScaleDescriptor.from_dict(data)


def test_scale_descriptor_from_dict() -> None:
    descriptor = _descriptor()
    assert descriptor.length_scale_m == pytest.approx(2.0)
    assert descriptor.interfacial_tension_n_m == pytest.approx(0.03)


def test_scale_descriptor_to_dict() -> None:
    data = _descriptor().to_dict()
    assert data["length_scale_m"] == pytest.approx(2.0)
    assert data["gravity_m_s2"] == pytest.approx(9.80665)


def test_scale_descriptor_validates_required_positive_fields() -> None:
    data = _descriptor_dict()
    data["length_scale_m"] = 0.0
    with pytest.raises(InvalidPhysicalValueError):
        ScaleDescriptor.from_dict(data)


def test_scale_descriptor_invalid_porosity_raises() -> None:
    data = _descriptor_dict()
    data["porosity"] = 1.5
    with pytest.raises(InvalidPhysicalValueError):
        ScaleDescriptor.from_dict(data)


def test_reynolds_number_formula() -> None:
    result = compute_reynolds_number(_descriptor())
    assert result.success is True
    assert result.value == pytest.approx(1000.0 * 1.0e-5 * 2.0 / 1.0e-3)


def test_capillary_number_formula() -> None:
    assert compute_capillary_number(_descriptor()).value == pytest.approx(1.0e-3 * 1.0e-5 / 0.03)


def test_capillary_number_missing_sigma_warns() -> None:
    data = _descriptor_dict()
    data.pop("interfacial_tension_n_m")
    result = compute_capillary_number(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert result.success is False
    assert "interfacial_tension" in result.warning


def test_peclet_number_formula() -> None:
    assert compute_peclet_number(_descriptor()).value == pytest.approx(1.0e-5 * 2.0 / 1.0e-9)


def test_peclet_number_missing_diffusivity_warns() -> None:
    data = _descriptor_dict()
    data.pop("diffusivity_m2_s")
    result = compute_peclet_number(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert "diffusivity" in result.warning


def test_mobility_ratio_formula() -> None:
    assert compute_mobility_ratio(_descriptor()).value == pytest.approx(1.5)


def test_mobility_ratio_missing_warns() -> None:
    data = _descriptor_dict()
    data.pop("mobility_displacing")
    result = compute_mobility_ratio(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert "mobility" in result.warning


def test_gravity_number_formula() -> None:
    expected = 200.0 * 9.80665 * 1.0e-12 / (1.0e-3 * 1.0e-5)
    assert compute_gravity_number(_descriptor()).value == pytest.approx(expected)


def test_gravity_number_missing_delta_density_warns() -> None:
    data = _descriptor_dict()
    data.pop("delta_density_kg_m3")
    result = compute_gravity_number(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert "delta_density" in result.warning


def test_dimensionless_pressure_formula() -> None:
    assert compute_dimensionless_pressure(_descriptor()).value == pytest.approx(0.2)


def test_dimensionless_pressure_missing_warns() -> None:
    data = _descriptor_dict()
    data.pop("pressure_drop_pa")
    result = compute_dimensionless_pressure(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert "pressure_drop" in result.warning


def test_dimensionless_time_formula() -> None:
    assert compute_dimensionless_time(_descriptor()).value == pytest.approx(50.0 * 1.0e-5 / 2.0)


def test_dimensionless_time_missing_warns() -> None:
    data = _descriptor_dict()
    data.pop("elapsed_time_s")
    result = compute_dimensionless_time(ScaleDescriptor.from_dict(data))
    assert result.value is None
    assert "elapsed_time" in result.warning


def test_compute_dimensionless_numbers_keys() -> None:
    numbers = compute_dimensionless_numbers(_descriptor())
    expected = {
        "reynolds",
        "capillary",
        "peclet",
        "mobility_ratio",
        "gravity_number",
        "dimensionless_pressure",
        "dimensionless_time",
        "warnings",
        "missing_criteria",
        "has_nan",
        "has_inf",
    }
    assert expected.issubset(numbers)


def test_compute_dimensionless_numbers_missing_criteria() -> None:
    data = _descriptor_dict()
    data.pop("diffusivity_m2_s")
    numbers = compute_dimensionless_numbers(ScaleDescriptor.from_dict(data))
    assert "peclet" in numbers["missing_criteria"]
    assert any("diffusivity" in warning for warning in numbers["warnings"])


def test_compute_dimensionless_numbers_no_nan_inf() -> None:
    numbers = compute_dimensionless_numbers(_descriptor())
    assert numbers["has_nan"] is False
    assert numbers["has_inf"] is False


def test_criterion_similarity_score_equal_is_one() -> None:
    assert compute_criterion_similarity_score(10.0, 10.0).value == pytest.approx(1.0)


def test_criterion_similarity_score_decreases_with_ratio() -> None:
    close = compute_criterion_similarity_score(10.0, 20.0).value
    far = compute_criterion_similarity_score(10.0, 1000.0).value
    assert close is not None and far is not None
    assert close > far


def test_criterion_similarity_score_in_unit_interval() -> None:
    for field_value in [0.1, 1.0, 10.0, 100.0]:
        score = compute_criterion_similarity_score(1.0, field_value).value
        assert score is not None
        assert 0.0 <= score <= 1.0


def test_overall_similarity_score_with_weights() -> None:
    lab = compute_dimensionless_numbers(_descriptor())
    field = compute_dimensionless_numbers(_field_descriptor())
    result = compute_overall_similarity_score(lab, field, weights={"reynolds": 2.0, "capillary": 0.5})
    assert result["overall_score"] is not None
    assert 0.0 <= result["overall_score"] <= 1.0
    assert set(result["criterion_scores"]).issuperset({"reynolds", "capillary"})


def test_overall_similarity_score_missing_values() -> None:
    result = compute_overall_similarity_score({"reynolds": None}, {"reynolds": None})
    assert result["overall_score"] is None
    assert result["missing_criteria"]
    assert result["warnings"]


def test_similarity_report_keys() -> None:
    report = build_similarity_report(_descriptor(), _field_descriptor())
    expected = {
        "success",
        "dimensionless_numbers_lab",
        "dimensionless_numbers_field",
        "criterion_scores",
        "overall_similarity_score",
        "missing_criteria",
        "warnings",
        "has_nan",
        "has_inf",
    }
    assert expected.issubset(report)


def test_similarity_report_success_true() -> None:
    report = build_similarity_report(_descriptor(), _field_descriptor())
    assert report["success"] is True
    assert report["overall_similarity_score"] is not None


def test_similarity_report_records_warnings() -> None:
    data = _descriptor_dict()
    data.pop("interfacial_tension_n_m")
    report = build_similarity_report(ScaleDescriptor.from_dict(data), _field_descriptor())
    assert "capillary" in report["missing_criteria"]
    assert any("interfacial_tension" in warning for warning in report["warnings"])


def test_similarity_report_no_solver_dependency() -> None:
    similarity_source = Path("reservoir_backend/cross_scale/similarity.py").read_text(encoding="utf-8")
    descriptor_source = Path("reservoir_backend/cross_scale/descriptors.py").read_text(encoding="utf-8")
    assert "reservoir_backend.solver" not in similarity_source
    assert "reservoir_backend.solver" not in descriptor_source


def test_similarity_repeatability() -> None:
    first = build_similarity_report(_descriptor(), _field_descriptor())
    second = build_similarity_report(_descriptor(), _field_descriptor())
    assert first == second


def test_similarity_module_does_not_modify_solver_files() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_cross_scale_design_doc_still_exists() -> None:
    assert Path("specs/13_cross_scale_analysis_design.md").exists()


def test_requirement_traceability_mentions_similarity_done() -> None:
    text = Path("specs/10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "similarity criteria module" in text
    assert "| similarity criteria module |" in text
    assert "Done" in text


def test_readme_mentions_similarity_criteria() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "similarity criteria" in text
    assert "scale-effect analysis" in text
