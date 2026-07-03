from __future__ import annotations

import pytest
import yaml

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.io.config_loader import apply_defaults, build_case_from_config, load_case_config, validate_case_config


def test_load_demo_case_config() -> None:
    config = load_case_config("config/demo_case.yaml")
    assert config["case"]["mode"] == "archie_only"


def test_load_multisignal_case_config() -> None:
    config = load_case_config("config/multisignal_case.yaml")
    assert config["case"]["mode"] == "multisignal"
    assert config["electromagnetic"]["enabled"] is True


def test_load_capillary_case_config() -> None:
    config = load_case_config("config/capillary_case.yaml")
    assert config["case"]["case_id"] == "capillary_case"
    assert config["capillary_pressure"]["enabled"] is True
    assert config["capillary_pressure"]["model"] == "brooks_corey"
    assert config["saturation"]["use_capillary"] is True


def test_load_gravity_case_config() -> None:
    config = load_case_config("config/gravity_case.yaml")
    assert config["case"]["case_id"] == "gravity_case"
    assert config["gravity"]["enabled"] is True
    assert config["saturation"]["use_gravity"] is True
    assert config["outputs"]["save_gravity_flux"] is True


def test_load_combined_case_config() -> None:
    config = load_case_config("config/combined_case.yaml")
    assert config["case"]["case_id"] == "combined_case"
    assert config["capillary_pressure"]["enabled"] is True
    assert config["gravity"]["enabled"] is True
    assert config["saturation"]["use_capillary"] is True
    assert config["saturation"]["use_gravity"] is True
    assert config["outputs"]["save_combined_report"] is True


def test_load_three_phase_case_config() -> None:
    config = load_case_config("config/three_phase_case.yaml")
    assert config["case"]["case_id"] == "three_phase_case"
    assert config["case"]["mode"] == "three_phase"
    assert config["three_phase"]["enabled"] is True
    assert config["three_phase"]["model"] == "incompressible_wog"
    assert config["relperm_three_phase"]["swi"] == pytest.approx(0.2)
    assert config["fluid"]["mu_g"] == pytest.approx(1.0e-5)


def test_inconsistent_capillary_flags_raise(tmp_path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_capillary"] = False
    path = tmp_path / "bad_capillary_flags.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="capillary_pressure.enabled=true requires saturation.use_capillary=true"):
        load_case_config(path)


def test_inconsistent_gravity_flags_raise(tmp_path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_gravity"] = False
    path = tmp_path / "bad_gravity_flags.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="gravity.enabled=true requires saturation.use_gravity=true"):
        load_case_config(path)


def test_missing_required_key_raises() -> None:
    with pytest.raises(KeyError):
        validate_case_config({"case": {"case_id": "x"}})


def test_invalid_grid_dimension_raises() -> None:
    config = apply_defaults({"grid": {"nx": 1, "ny": 5, "nz": 3, "dx": 1.0, "dy": 1.0, "dz": 1.0}})
    with pytest.raises(InvalidPhysicalValueError):
        validate_case_config(config)


def test_invalid_porosity_raises() -> None:
    config = apply_defaults({"grid": _grid(), "rock": {"porosity": 0.0}})
    with pytest.raises(InvalidPhysicalValueError):
        validate_case_config(config)


def test_invalid_permeability_raises() -> None:
    config = apply_defaults({"grid": _grid(), "rock": {"permeability_md": -1.0}})
    with pytest.raises(InvalidPhysicalValueError):
        validate_case_config(config)


def test_unit_conversion_md_to_m2() -> None:
    config = load_case_config("config/demo_case.yaml")
    assert config["rock"]["permeability_m2"] == pytest.approx(9.869233e-14)


def test_unit_conversion_mpa_to_pa() -> None:
    config = load_case_config("config/demo_case.yaml")
    assert config["pressure"]["left_pressure_pa"] == pytest.approx(1.0e7)
    assert config["pressure"]["right_pressure_pa"] == pytest.approx(9.0e6)


def test_apply_defaults() -> None:
    config = apply_defaults({"grid": _grid(), "case": {"case_id": "minimal"}})
    assert config["rock"]["porosity"] == pytest.approx(0.2)
    assert config["case"]["case_id"] == "minimal"


def test_build_case_from_config() -> None:
    case = build_case_from_config({"grid": _grid(), "case": {"case_id": "built"}})
    assert case["case"]["case_id"] == "built"
    assert "permeability_m2" in case["rock"]


def _grid() -> dict:
    return {"nx": 6, "ny": 5, "nz": 3, "dx": 1.0, "dy": 1.0, "dz": 1.0}
