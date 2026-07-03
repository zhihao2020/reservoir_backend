from __future__ import annotations

from pathlib import Path


DESIGN = Path("specs/12_three_phase_flow_design.md")


def test_three_phase_design_doc_exists() -> None:
    assert DESIGN.exists()


def test_three_phase_design_contains_scope() -> None:
    text = _design()
    assert "incompressible three-phase transport" in text
    assert "structured Cartesian grid" in text


def test_three_phase_design_says_not_black_oil() -> None:
    text = _design()
    assert "not a black-oil model" in text
    assert "Bo" in text
    assert "Rs" in text
    assert "bubble point" in text


def test_three_phase_design_contains_state_variables() -> None:
    text = _design()
    for token in ["Sw", "So", "Sg", "primary variables"]:
        assert token in text


def test_three_phase_design_contains_saturation_closure() -> None:
    text = _design()
    assert "Sw + So + Sg = 1" in text
    assert "So = 1 - Sw - Sg" in text


def test_three_phase_design_contains_saturation_bounds() -> None:
    text = _design()
    for token in ["Swi", "Sor", "Sgc", "Sw_max", "Sg_max", "So < Sor"]:
        assert token in text


def test_three_phase_design_contains_relperm_design() -> None:
    text = _design()
    for token in ["krw = krw0", "krg = krg0", "kro = kro0", "Stone I", "Stone II", "Baker"]:
        assert token in text


def test_three_phase_design_contains_fractional_flow() -> None:
    text = _design()
    for token in ["lambda_w", "lambda_o", "lambda_g", "fw + fo + fg = 1"]:
        assert token in text


def test_three_phase_design_contains_flux_design() -> None:
    text = _design()
    for token in ["Fw = fw_upwind * qt", "Fo = fo_upwind * qt", "Fg = fg_upwind * qt", "advective flux"]:
        assert token in text


def test_three_phase_design_contains_transport_update() -> None:
    text = _design()
    assert "Sw_new = Sw_old" in text
    assert "Sg_new = Sg_old" in text
    assert "So_new = 1 - Sw_new - Sg_new" in text


def test_three_phase_design_contains_cfl_strategy() -> None:
    text = _design()
    assert "effective_flux = abs(qt)" in text
    assert "existing two-phase total-flux CFL logic" in text


def test_three_phase_design_contains_material_balance() -> None:
    text = _design()
    for token in [
        "water_injected_volume",
        "gas_storage_change",
        "oil_balance_error",
    ]:
        assert token in text


def test_three_phase_design_contains_yaml_design() -> None:
    text = _design()
    assert "three_phase:" in text
    assert "relperm_three_phase:" in text
    assert "primary_variables: [Sw, Sg]" in text


def test_three_phase_design_contains_future_tasks() -> None:
    text = _design()
    for task in [
        "035_three_phase_relperm",
        "036_three_phase_fractional_flow",
        "037_three_phase_transport_1d",
        "038_three_phase_transport_3d",
        "039_three_phase_pipeline_case",
        "040_three_phase_validation_and_profiling",
    ]:
        assert task in text


def test_three_phase_design_contains_test_plan() -> None:
    text = _design()
    for token in ["saturation closure", "CFL violation", "material balance", "invalid viscosity raises"]:
        assert token in text


def test_requirement_traceability_mentions_three_phase_planned() -> None:
    text = Path("specs/10_requirement_traceability.md").read_text(encoding="utf-8")
    assert "three-phase flow design" in text
    assert "three-phase relperm" in text
    assert "three-phase transport" in text
    assert "Planned" in text


def test_module_matrix_mentions_three_phase() -> None:
    text = Path("docs/module_matrix.md").read_text(encoding="utf-8")
    assert "Three-phase design" in text
    assert "Three-phase relperm" in text
    assert "Three-phase transport" in text


def test_limitations_mentions_no_black_oil() -> None:
    text = Path("docs/limitations_and_roadmap.md").read_text(encoding="utf-8").lower()
    assert "black-oil" in text
    assert "not equivalent to black-oil" in text


def _design() -> str:
    return DESIGN.read_text(encoding="utf-8")
