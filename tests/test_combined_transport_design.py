from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reservoir_backend.io.config_loader import load_case_config


DESIGN_DOC = Path("specs/11_combined_capillary_gravity_design.md")


def test_combined_design_doc_exists() -> None:
    assert DESIGN_DOC.exists()


def test_combined_design_contains_total_flux_formula() -> None:
    text = _doc_text()
    assert "Fw_total = Fw_adv + Fw_cap + Fw_grav" in text
    assert "Fw_adv = fw_upwind * total_flux" in text
    assert "Fw_cap = capillary_flux" in text
    assert "Fw_grav = gravity_flux" in text


def test_combined_design_contains_sign_convention() -> None:
    text = _doc_text()
    for phrase in [
        "flux_x > 0",
        "left -> right",
        "flux_y > 0",
        "front -> back",
        "flux_z > 0",
        "bottom -> top",
        "Pc = Po - Pw",
        "qcap_z = T_abs * Mcap * (Pc_top - Pc_bottom)",
        "gravity_flux_z < 0",
    ]:
        assert phrase in text


def test_combined_design_contains_cfl_strategy() -> None:
    text = _doc_text()
    assert "effective_flux_x = abs(total_flux_x) + abs(capillary_flux_x) + abs(gravity_flux_x)" in text
    assert "effective_flux_y = abs(total_flux_y) + abs(capillary_flux_y) + abs(gravity_flux_y)" in text
    assert "effective_flux_z = abs(total_flux_z) + abs(capillary_flux_z) + abs(gravity_flux_z)" in text
    assert "semi-implicit capillary diffusion" in text


def test_combined_design_contains_material_balance() -> None:
    text = _doc_text()
    for key in ["injected_water_volume", "produced_water_volume", "storage_change", "material_balance_error"]:
        assert key in text


def test_combined_design_contains_report_schema() -> None:
    text = _doc_text()
    for key in [
        "capillary_enabled",
        "gravity_enabled",
        "capillary_model",
        "rho_w",
        "rho_o",
        "density_difference",
        "max_advective_flux",
        "max_capillary_flux",
        "max_gravity_flux",
        "max_total_water_flux",
        "max_cfl",
        "material_balance_error",
        "capillary_flux_included",
        "gravity_flux_included",
        "has_nan",
        "has_inf",
    ]:
        assert key in text


def test_combined_design_contains_yaml_policy() -> None:
    text = _doc_text()
    assert "capillary_pressure:" in text
    assert "gravity:" in text
    assert "use_capillary: true" in text
    assert "use_gravity: true" in text
    assert "combined_case.yaml" in text


def test_combined_design_says_not_implemented_yet() -> None:
    text = _doc_text()
    assert "advance_saturation_3d_with_capillary_and_gravity" in text
    assert "combined pipeline" in text


def test_combined_design_contains_future_tasks() -> None:
    text = _doc_text()
    for task in [
        "029_combined_flux_composer",
        "030_combined_capillary_gravity_transport_3d",
        "031_combined_pipeline_case",
        "032_combined_profiling_and_validation",
    ]:
        assert task in text


def test_capillary_gravity_together_loads_when_flags_consistent() -> None:
    config = load_case_config("config/combined_case.yaml")
    assert config["capillary_pressure"]["enabled"] is True
    assert config["gravity"]["enabled"] is True
    assert config["saturation"]["use_capillary"] is True
    assert config["saturation"]["use_gravity"] is True


def test_inconsistent_combined_flags_still_raise(tmp_path: Path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_gravity"] = False
    path = tmp_path / "combined_inconsistent.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="gravity.enabled=true requires saturation.use_gravity=true"):
        load_case_config(path)


def _doc_text() -> str:
    return DESIGN_DOC.read_text(encoding="utf-8")
