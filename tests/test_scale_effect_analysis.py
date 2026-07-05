from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.cross_scale.descriptors import ScaleDescriptor
from reservoir_backend.cross_scale.scale_effect import (
    build_scale_effect_report,
    classify_flow_regime,
    compute_scale_ratios,
    detect_regime_shift,
)
from reservoir_backend.cross_scale.similarity import build_similarity_report


def _lab_dict() -> dict[str, float]:
    return {
        "length_scale_m": 1.0,
        "time_scale_s": 10.0,
        "pressure_scale_pa": 1.0e5,
        "permeability_scale_m2": 1.0e-12,
        "porosity": 0.2,
        "viscosity_pa_s": 1.0e-3,
        "density_kg_m3": 1000.0,
        "velocity_scale_m_s": 1.0e-6,
        "flow_rate_m3_s": 1.0e-9,
        "temperature_scale_k": 300.0,
        "interfacial_tension_n_m": 0.03,
        "diffusivity_m2_s": 1.0e-9,
        "delta_density_kg_m3": 10.0,
        "gravity_m_s2": 9.80665,
        "pressure_drop_pa": 1.0e4,
        "elapsed_time_s": 100.0,
        "mobility_displacing": 2.0,
        "mobility_displaced": 1.0,
    }


def _field_dict() -> dict[str, float]:
    data = _lab_dict()
    data.update(
        {
            "length_scale_m": 100.0,
            "time_scale_s": 1000.0,
            "pressure_scale_pa": 2.0e6,
            "permeability_scale_m2": 2.0e-13,
            "porosity": 0.25,
            "velocity_scale_m_s": 1.0e-3,
            "flow_rate_m3_s": 2.0e-5,
            "temperature_scale_k": 330.0,
            "interfacial_tension_n_m": 0.03,
            "diffusivity_m2_s": 1.0e-10,
            "delta_density_kg_m3": 5000.0,
        }
    )
    return data


def _lab() -> ScaleDescriptor:
    return ScaleDescriptor.from_dict(_lab_dict())


def _field() -> ScaleDescriptor:
    return ScaleDescriptor.from_dict(_field_dict())


def test_compute_scale_ratios_keys() -> None:
    ratios = compute_scale_ratios(_lab(), _field())
    keys = {
        "scale_ratio_length",
        "scale_ratio_time",
        "scale_ratio_pressure",
        "scale_ratio_permeability",
        "scale_ratio_velocity",
        "scale_ratio_flow_rate",
        "scale_ratio_porosity",
        "scale_ratio_temperature",
        "warnings",
        "missing_ratios",
        "has_nan",
        "has_inf",
    }
    assert keys.issubset(ratios)


def test_compute_scale_ratios_values() -> None:
    ratios = compute_scale_ratios(_lab(), _field())
    assert ratios["scale_ratio_length"] == pytest.approx(100.0)
    assert ratios["scale_ratio_time"] == pytest.approx(100.0)
    assert ratios["scale_ratio_pressure"] == pytest.approx(20.0)
    assert ratios["scale_ratio_permeability"] == pytest.approx(0.2)
    assert ratios["scale_ratio_velocity"] == pytest.approx(1000.0)
    assert ratios["scale_ratio_flow_rate"] == pytest.approx(2.0e4)
    assert ratios["scale_ratio_porosity"] == pytest.approx(1.25)
    assert ratios["scale_ratio_temperature"] == pytest.approx(1.1)


def test_compute_scale_ratios_temperature_missing_warns() -> None:
    lab_data = _lab_dict()
    field_data = _field_dict()
    lab_data.pop("temperature_scale_k")
    field_data.pop("temperature_scale_k")
    ratios = compute_scale_ratios(ScaleDescriptor.from_dict(lab_data), ScaleDescriptor.from_dict(field_data))
    assert ratios["scale_ratio_temperature"] is None
    assert "scale_ratio_temperature" in ratios["missing_ratios"]
    assert ratios["warnings"]


def test_compute_scale_ratios_no_nan_inf() -> None:
    ratios = compute_scale_ratios(_lab(), _field())
    assert ratios["has_nan"] is False
    assert ratios["has_inf"] is False


def test_classify_flow_regime_keys() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 2.0, "gravity_number": 0.1})
    assert {"flow_regime", "dominant_force", "capillary_role", "gravity_role", "transport_role", "inertia_role", "warnings"}.issubset(regime)


def test_classify_capillary_dominated() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-6, "peclet": 2.0, "gravity_number": 0.1})
    assert regime["capillary_role"] == "capillary_dominated"
    assert regime["dominant_force"] == "capillary"


def test_classify_viscous_dominated() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-2, "peclet": 2.0, "gravity_number": 0.1})
    assert regime["capillary_role"] == "viscous_dominated"
    assert regime["dominant_force"] == "viscous"


def test_classify_gravity_dominated() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-2, "peclet": 2.0, "gravity_number": 2.0})
    assert regime["gravity_role"] == "gravity_dominated"
    assert regime["dominant_force"] == "gravity"


def test_classify_convection_dominated() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 20.0, "gravity_number": 0.1})
    assert regime["transport_role"] == "convection_dominated"


def test_classify_diffusion_dominated() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 0.2, "gravity_number": 0.1})
    assert regime["transport_role"] == "diffusion_dominated"


def test_classify_creeping_flow() -> None:
    regime = classify_flow_regime({"reynolds": 0.001, "capillary": 1.0e-4, "peclet": 2.0, "gravity_number": 0.1})
    assert regime["inertia_role"] == "creeping_flow"


def test_classify_inertial_effect_possible() -> None:
    regime = classify_flow_regime({"reynolds": 2.0, "capillary": 1.0e-4, "peclet": 2.0, "gravity_number": 0.1})
    assert regime["inertia_role"] == "inertial_effect_possible"


def test_classify_missing_criteria_warns() -> None:
    regime = classify_flow_regime({"reynolds": 0.1})
    assert regime["warnings"]
    assert regime["capillary_role"] == "capillary_uncertain"
    assert regime["transport_role"] == "transport_uncertain"


def test_classify_custom_thresholds() -> None:
    regime = classify_flow_regime(
        {"reynolds": 0.1, "capillary": 5.0e-4, "peclet": 5.0, "gravity_number": 0.1},
        thresholds={"viscous_dominated_ca_threshold": 1.0e-4},
    )
    assert regime["capillary_role"] == "viscous_dominated"


def test_invalid_threshold_raises() -> None:
    with pytest.raises(InvalidPhysicalValueError):
        classify_flow_regime({"reynolds": 0.1}, thresholds={"inertial_re_threshold": 0.0})


def test_detect_regime_shift_none() -> None:
    regime = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 2.0, "gravity_number": 0.1})
    shift = detect_regime_shift(regime, regime)
    assert shift["regime_shift_detected"] is False
    assert shift["shift_summary"] == []


def test_detect_regime_shift_dominant_force_changed() -> None:
    lab = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-6, "peclet": 2.0, "gravity_number": 0.1})
    field = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-2, "peclet": 2.0, "gravity_number": 2.0})
    shift = detect_regime_shift(lab, field)
    assert shift["dominant_force_changed"] is True
    assert shift["regime_shift_detected"] is True


def test_detect_regime_shift_transport_changed() -> None:
    lab = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 0.2, "gravity_number": 0.1})
    field = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-4, "peclet": 20.0, "gravity_number": 0.1})
    assert detect_regime_shift(lab, field)["transport_role_changed"] is True


def test_detect_regime_shift_summary() -> None:
    lab = classify_flow_regime({"reynolds": 0.1, "capillary": 1.0e-6, "peclet": 0.2, "gravity_number": 0.1})
    field = classify_flow_regime({"reynolds": 2.0, "capillary": 1.0e-2, "peclet": 20.0, "gravity_number": 2.0})
    summary = detect_regime_shift(lab, field)["shift_summary"]
    assert summary
    assert any("changed from" in item for item in summary)


def test_build_scale_effect_report_keys() -> None:
    report = build_scale_effect_report(_lab(), _field())
    expected = {
        "success",
        "scale_ratios",
        "dimensionless_numbers_lab",
        "dimensionless_numbers_field",
        "regime_lab",
        "regime_field",
        "regime_shift",
        "dominant_force_lab",
        "dominant_force_field",
        "regime_shift_detected",
        "warnings",
        "has_nan",
        "has_inf",
    }
    assert expected.issubset(report)


def test_build_scale_effect_report_success_true() -> None:
    assert build_scale_effect_report(_lab(), _field())["success"] is True


def test_build_scale_effect_report_contains_scale_ratios() -> None:
    ratios = build_scale_effect_report(_lab(), _field())["scale_ratios"]
    assert ratios["scale_ratio_length"] == pytest.approx(100.0)
    assert "warnings" not in ratios


def test_build_scale_effect_report_contains_regimes() -> None:
    report = build_scale_effect_report(_lab(), _field())
    assert report["regime_lab"]["dominant_force"]
    assert report["regime_field"]["dominant_force"]


def test_build_scale_effect_report_detects_shift() -> None:
    assert build_scale_effect_report(_lab(), _field())["regime_shift_detected"] is True


def test_build_scale_effect_report_warnings() -> None:
    lab_data = _lab_dict()
    field_data = _field_dict()
    lab_data.pop("diffusivity_m2_s")
    field_data.pop("temperature_scale_k")
    report = build_scale_effect_report(ScaleDescriptor.from_dict(lab_data), ScaleDescriptor.from_dict(field_data))
    assert report["warnings"]
    assert any("diffusivity" in warning or "temperature" in warning for warning in report["warnings"])


def test_build_scale_effect_report_no_nan_inf() -> None:
    report = build_scale_effect_report(_lab(), _field())
    assert report["has_nan"] is False
    assert report["has_inf"] is False


def test_scale_effect_repeatability() -> None:
    assert build_scale_effect_report(_lab(), _field()) == build_scale_effect_report(_lab(), _field())


def test_scale_effect_no_solver_dependency() -> None:
    source = Path("reservoir_backend/cross_scale/scale_effect.py").read_text(encoding="utf-8")
    assert "reservoir_backend.solver" not in source
    assert "reservoir_backend.inversion" not in source
    assert "reservoir_backend.fusion" not in source


def test_scale_effect_module_does_not_modify_solver_files() -> None:
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "reservoir_backend/solver"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_similarity_criteria_tests_still_pass() -> None:
    report = build_similarity_report(_lab(), _field())
    assert report["success"] is True


def test_cross_scale_design_doc_mentions_scale_effect_done() -> None:
    text = Path("specs/13_cross_scale_analysis_design.md").read_text(encoding="utf-8")
    assert "043_scale_effect_analysis_module" in text
    assert "scale-effect analysis module is implemented" in text


def test_requirement_traceability_mentions_scale_effect_done() -> None:
    text = Path("specs/10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "| scale-effect analysis module |" in text
    assert "Done" in text


def test_readme_mentions_scale_effect_analysis() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "scale-effect analysis" in text
    assert "similarity criteria" in text


def test_scale_effect_does_not_claim_history_matching() -> None:
    text = (
        Path("README.md").read_text(encoding="utf-8")
        + Path("docs/limitations_and_roadmap.md").read_text(encoding="utf-8")
        + Path("specs/13_cross_scale_analysis_design.md").read_text(encoding="utf-8")
    )
    assert "does not perform history matching" in text or "will not perform history matching" in text
    assert "history matching is implemented" not in text
