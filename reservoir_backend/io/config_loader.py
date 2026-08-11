"""YAML case configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.core.units import permeability_to_m2, pressure_to_pa
from reservoir_backend.solver.capillary_pressure import build_capillary_model_from_config
from reservoir_backend.solver.gravity_flux import build_gravity_model_from_config
from reservoir_backend.solver.three_phase_relperm import validate_three_phase_params, validate_three_phase_saturations


DEFAULT_CONFIG: dict[str, Any] = {
    "case": {"case_id": "demo_case", "output_dir": "results", "mode": "archie_only"},
    "rock": {"porosity": 0.2, "permeability_md": 100.0},
    "fluid": {"mu_w": 1.0e-3, "mu_o": 5.0e-3},
    "three_phase": {"enabled": False, "model": "incompressible_wog", "primary_variables": ["Sw", "Sg"]},
    "archie": {"a": 1.0, "m": 2.0, "n": 2.0, "rw": 0.25, "swi": 0.2, "sor": 0.2},
    "electromagnetic": {
        "enabled": False,
        "model_type": "linear",
        "coefficients": [0.05, 0.9],
        "calibration_range": [0.0, 1.0],
    },
    "acoustic": {
        "enabled": False,
        "model_type": "linear",
        "coefficients": [1.1, -2.0e-4],
        "calibration_range": [1500.0, 5000.0],
    },
    "pressure": {
        "boundary_type": "left_right_dirichlet",
        "left_pressure": 10.0,
        "right_pressure": 9.0,
        "pressure_unit": "MPa",
        "reference_pressure": 0.0,
    },
    "saturation": {
        "dt": 1000.0,
        "steps": 3,
        "max_cfl": 1.0,
        "use_capillary": False,
        "use_gravity": False,
        "swi": 0.2,
        "sor": 0.2,
        "krw0": 1.0,
        "kro0": 1.0,
        "nw": 2.0,
        "no": 2.0,
        "injected_sw": 0.8,
    },
    "initial_saturation": {
        "type": "constant",
        "value": 0.2,
        "low_sw": 0.2,
        "high_sw": 0.75,
        "split_fraction": 0.5,
        "left_sw": 0.75,
        "right_sw": 0.2,
        "background_sw": 0.2,
        "blob_sw": 0.75,
        "radius_fraction": 0.25,
    },
    "capillary_pressure": {
        "enabled": False,
        "model": "none",
        "entry_pressure_pa": 1000.0,
        "lambda_pc": 2.0,
        "p0_pa": 1000.0,
        "m": 0.5,
        "n": 2.0,
    },
    "gravity": {
        "enabled": False,
        "g": 9.80665,
        "rho_w": 1000.0,
        "rho_o": 800.0,
        "depth_axis": "z",
        "depth_positive": "down",
    },
    "injection_composition": {"injected_sw": None, "injected_sg": None},
    "fusion": {"signal_weights": [1.0, 1.0, 1.0], "dynamic_alpha": 0.5},
    "outputs": {
        "save_flux": True,
        "save_velocity": True,
        "save_reports": True,
        "save_capillary_pressure": False,
        "save_capillary_flux": False,
        "save_gravity_flux": False,
        "save_combined_report": False,
        "save_three_phase_saturations": False,
        "save_three_phase_report": False,
    },
}


def load_case_config(path: str | Path) -> dict[str, Any]:
    """Load, default, validate, and normalize a YAML case config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = apply_defaults(raw)
    validate_case_config(config)
    return normalize_units(config)


def apply_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default values without mutating the input config."""
    merged = deepcopy(DEFAULT_CONFIG)
    _deep_update(merged, deepcopy(config))
    return merged


def validate_case_config(config: dict[str, Any]) -> None:
    """Validate required keys and physical parameter ranges."""
    for section in ["case", "grid", "rock", "fluid", "archie", "pressure", "saturation"]:
        if section not in config:
            raise KeyError(f"missing required config section: {section}")
    for key in ["case_id", "output_dir", "mode"]:
        if key not in config["case"]:
            raise KeyError(f"missing required case.{key}")
    if config["case"]["mode"] not in {"archie_only", "multisignal", "three_phase"}:
        raise ValueError("case.mode must be 'archie_only', 'multisignal', or 'three_phase'")

    grid = config["grid"]
    for key in ["nx", "ny", "nz"]:
        if int(grid[key]) <= 1:
            raise InvalidPhysicalValueError(f"grid.{key} must be > 1")
    counts = {"dx": int(grid["nx"]), "dy": int(grid["ny"]), "dz": int(grid["nz"])}
    for key, count in counts.items():
        raw = grid[key]
        if isinstance(raw, (list, tuple)):
            if len(raw) != count:
                raise InvalidPhysicalValueError(
                    f"grid.{key} list length must equal grid cell count {count}"
                )
            if any(float(v) <= 0.0 for v in raw):
                raise InvalidPhysicalValueError(f"grid.{key} entries must be > 0")
        elif float(raw) <= 0.0:
            raise InvalidPhysicalValueError(f"grid.{key} must be > 0")

    if not 0.0 < float(config["rock"]["porosity"]) < 1.0:
        raise InvalidPhysicalValueError("rock.porosity must be in (0, 1)")
    if "permeability" in config["rock"]:
        if float(config["rock"]["permeability"]) < 0.0:
            raise InvalidPhysicalValueError("rock.permeability must be non-negative")
    elif float(config["rock"]["permeability_md"]) < 0.0:
        raise InvalidPhysicalValueError("rock.permeability_md must be non-negative")
    if float(config["fluid"]["mu_w"]) <= 0.0 or float(config["fluid"]["mu_o"]) <= 0.0:
        raise InvalidPhysicalValueError("fluid viscosities must be positive")

    saturation = config["saturation"]
    if float(saturation["swi"]) < 0.0 or float(saturation["sor"]) < 0.0 or float(saturation["swi"]) + float(saturation["sor"]) >= 1.0:
        raise InvalidPhysicalValueError("saturation residual saturations are invalid")
    if float(saturation["dt"]) <= 0.0 or int(saturation["steps"]) <= 0 or float(saturation["max_cfl"]) <= 0.0:
        raise InvalidPhysicalValueError("saturation time controls must be positive")
    initial = config.get("initial_saturation", {})
    initial_type = initial.get("type", "constant")
    if initial_type not in {"constant", "step_x", "linear_x", "center_blob"}:
        raise ValueError("initial_saturation.type must be constant, step_x, linear_x, or center_blob")
    for key in ["value", "low_sw", "high_sw", "left_sw", "right_sw", "background_sw", "blob_sw"]:
        if key in initial and not np_is_finite_float(initial[key]):
            raise InvalidPhysicalValueError(f"initial_saturation.{key} must be finite")
    if not 0.0 < float(initial.get("split_fraction", 0.5)) < 1.0:
        raise InvalidPhysicalValueError("initial_saturation.split_fraction must be in (0, 1)")
    if not 0.0 < float(initial.get("radius_fraction", 0.25)) <= 1.0:
        raise InvalidPhysicalValueError("initial_saturation.radius_fraction must be in (0, 1]")
    build_capillary_model_from_config(config)
    build_gravity_model_from_config(config)
    _validate_three_phase_config(config)
    capillary_enabled = bool(config.get("capillary_pressure", {}).get("enabled", False))
    gravity_enabled = bool(config.get("gravity", {}).get("enabled", False))
    use_capillary = bool(saturation.get("use_capillary", False))
    use_gravity = bool(saturation.get("use_gravity", False))
    if capillary_enabled and not use_capillary:
        raise ValueError("capillary_pressure.enabled=true requires saturation.use_capillary=true")
    if use_capillary and not capillary_enabled:
        raise ValueError("saturation.use_capillary=true requires capillary_pressure.enabled=true")
    if gravity_enabled and not use_gravity:
        raise ValueError("gravity.enabled=true requires saturation.use_gravity=true")
    if use_gravity and not gravity_enabled:
        raise ValueError("saturation.use_gravity=true requires gravity.enabled=true")


def normalize_units(config: dict[str, Any]) -> dict[str, Any]:
    """Add normalized SI-unit values to the config."""
    normalized = deepcopy(config)
    if "permeability" in normalized["rock"]:
        normalized["rock"]["permeability_m2"] = float(normalized["rock"]["permeability"])
    else:
        normalized["rock"]["permeability_m2"] = permeability_to_m2(float(normalized["rock"]["permeability_md"]), "mD")
    pressure_unit = normalized["pressure"].get("pressure_unit", "MPa")
    normalized["pressure"]["left_pressure_pa"] = pressure_to_pa(float(normalized["pressure"]["left_pressure"]), pressure_unit)
    normalized["pressure"]["right_pressure_pa"] = pressure_to_pa(float(normalized["pressure"]["right_pressure"]), pressure_unit)
    normalized["pressure"]["reference_pressure_pa"] = pressure_to_pa(
        float(normalized["pressure"].get("reference_pressure", 0.0)),
        pressure_unit,
    )
    normalized["capillary_pressure"] = build_capillary_model_from_config(normalized)
    normalized["gravity"] = build_gravity_model_from_config(normalized)
    if _three_phase_requested(normalized):
        normalized["three_phase"]["enabled"] = True
        normalized["case"]["mode"] = "three_phase"
    return normalized


def build_case_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized case dictionary for the runner."""
    normalized = normalize_units(apply_defaults(config))
    validate_case_config(normalized)
    return normalized


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def np_is_finite_float(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric == numeric and numeric not in (float("inf"), float("-inf"))


def _three_phase_requested(config: dict[str, Any]) -> bool:
    return bool(config.get("three_phase", {}).get("enabled", False)) or config.get("case", {}).get("mode") == "three_phase"


def _validate_three_phase_config(config: dict[str, Any]) -> None:
    three_phase = config.get("three_phase", {})
    if "enabled" in three_phase and not isinstance(three_phase["enabled"], bool):
        raise ValueError("three_phase.enabled must be bool")
    if not _three_phase_requested(config):
        return

    if config["case"]["mode"] == "multisignal":
        raise ValueError("three-phase pipeline does not support multisignal mode")
    if str(three_phase.get("model", "")) != "incompressible_wog":
        raise ValueError("three-phase model must be incompressible_wog")

    capillary_enabled = bool(config.get("capillary_pressure", {}).get("enabled", False))
    gravity_enabled = bool(config.get("gravity", {}).get("enabled", False))
    saturation = config["saturation"]
    if capillary_enabled or gravity_enabled or bool(saturation.get("use_capillary", False)) or bool(saturation.get("use_gravity", False)):
        raise ValueError("three-phase pipeline does not support capillary/gravity/combined transport yet")

    if "relperm_three_phase" not in config:
        raise KeyError("missing required config section: relperm_three_phase")
    relperm = config["relperm_three_phase"]
    if "no" not in relperm and False in relperm:
        relperm["no"] = relperm[False]
    required_relperm = ["swi", "sor", "sgc", "krw0", "kro0", "krg0", "nw", "no", "ng"]
    for key in required_relperm:
        if key not in relperm:
            raise KeyError(f"missing required relperm_three_phase.{key}")
    for key in ["mu_w", "mu_o", "mu_g"]:
        if key not in config["fluid"]:
            raise KeyError(f"missing required fluid.{key}")
        if float(config["fluid"][key]) <= 0.0:
            raise InvalidPhysicalValueError(f"fluid.{key} must be positive")

    params = {
        **{key: float(relperm[key]) for key in required_relperm},
        "mu_w": float(config["fluid"]["mu_w"]),
        "mu_o": float(config["fluid"]["mu_o"]),
        "mu_g": float(config["fluid"]["mu_g"]),
    }
    validate_three_phase_params(params)

    initial = config.get("initial_saturation", {})
    if "sw" not in initial or "sg" not in initial:
        raise KeyError("three-phase initial_saturation requires sw and sg")
    sw = float(initial["sw"])
    sg = float(initial["sg"])
    validate_three_phase_saturations(sw, sg, params)
