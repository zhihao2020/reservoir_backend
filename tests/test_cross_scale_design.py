from __future__ import annotations

from pathlib import Path


DESIGN_PATH = Path("specs/13_cross_scale_analysis_design.md")


def test_cross_scale_design_doc_exists() -> None:
    assert DESIGN_PATH.exists()


def test_cross_scale_design_contains_product_architecture_decision() -> None:
    text = _design()
    assert "Product Architecture Decision" in text
    assert "Requirements 1 and 2 are not split into two independent software products" in text


def test_cross_scale_design_says_one_backend_two_modules() -> None:
    text = _design()
    assert "one Reservoir Digital Twin Backend" in text
    assert "two first-level functional modules" in text
    assert "Module A: Experimental Data Processing and Numerical Computation" in text
    assert "Module B: Cross-scale Analysis and Comparison" in text


def test_cross_scale_design_contains_scope() -> None:
    text = _design()
    assert "laboratory experiment scale" in text
    assert "field scale" in text
    assert "engineering MVP" in text


def test_cross_scale_design_contains_cross_scale_package_boundary() -> None:
    text = _design()
    assert "reservoir_backend/cross_scale/" in text
    assert "does not call `pressure_solver` directly" in text
    assert "does not overwrite existing result files" in text


def test_cross_scale_design_contains_cross_scale_objects() -> None:
    text = _design()
    for name in ["ExperimentCase", "FieldCase", "ScaleDescriptor", "SimilarityCriteria", "ScaleEffectReport", "ValidationReport"]:
        assert name in text


def test_cross_scale_design_contains_similarity_criteria() -> None:
    assert "Similarity Criteria Design" in _design()


def test_cross_scale_design_contains_reynolds_number() -> None:
    text = _design()
    assert "Reynolds number" in text
    assert "Re = rho * v * L / mu" in text


def test_cross_scale_design_contains_capillary_number() -> None:
    text = _design()
    assert "Capillary number" in text
    assert "Ca = mu * v / sigma" in text


def test_cross_scale_design_contains_peclet_number() -> None:
    text = _design()
    assert "Peclet number" in text
    assert "Pe = v * L / D" in text


def test_cross_scale_design_contains_mobility_ratio() -> None:
    text = _design()
    assert "Mobility ratio" in text
    assert "M = lambda_displacing / lambda_displaced" in text


def test_cross_scale_design_contains_gravity_number() -> None:
    text = _design()
    assert "Gravity number" in text
    assert "Ng = delta_rho * g * k / (mu * v)" in text


def test_cross_scale_design_contains_scale_effect_analysis() -> None:
    text = _design()
    assert "Scale-Effect Analysis Design" in text
    assert "regime_shift_detected" in text


def test_cross_scale_design_contains_lab_to_field_mapping() -> None:
    text = _design()
    assert "Lab-to-Field Mapping Design" in text
    assert "mapped_flow_rate" in text


def test_cross_scale_design_contains_validation_metrics() -> None:
    text = _design()
    for metric in ["RMSE", "MAE", "MAPE", "R2", "normalized RMSE", "max absolute error"]:
        assert metric in text


def test_cross_scale_design_contains_similarity_score() -> None:
    text = _design()
    assert "Similarity Score Design" in text
    assert "overall_score in `[0, 1]`" in text


def test_cross_scale_design_contains_yaml_design() -> None:
    text = _design()
    assert "YAML Design" in text
    assert "cross_scale:" in text
    assert "lab_case:" in text
    assert "field_case:" in text


def test_cross_scale_design_contains_report_schema() -> None:
    text = _design()
    assert "Report Schema" in text
    assert "cross_scale_report.json" in text
    assert "similarity_report.json" in text
    assert "scale_effect_report.json" in text
    assert "lab_field_validation_report.json" in text


def test_cross_scale_design_contains_future_tasks() -> None:
    text = _design()
    for task in ["042_similarity_criteria_module", "043_scale_effect_analysis_module", "044_lab_field_validation_module", "045_udp_api_minimal", "046_software_requirement_acceptance_report"]:
        assert task in text


def test_cross_scale_design_says_no_history_matching() -> None:
    text = _design().lower()
    assert "full history matching" in text
    assert "does not perform automatic history matching" in text


def test_cross_scale_design_says_no_solver_modification() -> None:
    text = _design()
    assert "must not modify solver internals" in text
    assert "does not change saturation transport" in text


def test_requirement_traceability_mentions_cross_scale_design() -> None:
    text = Path("specs/10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "cross-scale analysis design" in text
    assert "similarity criteria module" in text


def test_module_matrix_mentions_cross_scale() -> None:
    text = Path("docs/module_matrix.md").read_text(encoding="utf-8")
    assert "Cross-scale analysis design" in text
    assert "Similarity criteria" in text
    assert "Lab-field validation" in text


def test_roadmap_mentions_cross_scale_next() -> None:
    text = Path("docs/limitations_and_roadmap.md").read_text(encoding="utf-8")
    assert "cross-scale analysis design completed" in text
    assert "similarity criteria module" in text
    assert "lab-field validation module" in text


def test_readme_mentions_cross_scale_design_only() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "cross-scale analysis design" in text
    assert "one backend with two first-level modules" in text
    assert "cross-scale implementation is not yet complete" in text


def _design() -> str:
    return DESIGN_PATH.read_text(encoding="utf-8")
