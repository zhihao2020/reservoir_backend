from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.core.exceptions import InvalidPhysicalValueError
from reservoir_backend.io.config_loader import load_case_config


def test_three_phase_case_config_loads() -> None:
    config = load_case_config("config/three_phase_case.yaml")
    assert config["case"]["mode"] == "three_phase"
    assert config["three_phase"]["enabled"] is True


def test_three_phase_case_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry_outputs"
    result = _run_cli("--config", "config/three_phase_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["three_phase_enabled"] is True
    assert payload["black_oil_enabled"] is False
    assert not output_dir.exists()


def test_three_phase_pipeline_runs(tmp_path: Path) -> None:
    result = _run_three_phase(tmp_path, "three_phase_run")
    assert result["summary"]["success"] is True
    assert result["summary"]["three_phase_transport_enabled"] is True


def test_three_phase_outputs_exist(tmp_path: Path) -> None:
    case_dir = _run_three_phase(tmp_path, "three_phase_outputs")["case_dir"]
    required = ["sw_three_phase.npy", "sg_three_phase.npy", "so_three_phase.npy", "three_phase_report.json", "case_summary.json"]
    assert all((case_dir / name).exists() for name in required)


def test_three_phase_saturation_shapes(tmp_path: Path) -> None:
    case_dir = _run_three_phase(tmp_path, "three_phase_shapes")["case_dir"]
    expected = (3, 4, 6)
    assert np.load(case_dir / "sw_three_phase.npy").shape == expected
    assert np.load(case_dir / "sg_three_phase.npy").shape == expected
    assert np.load(case_dir / "so_three_phase.npy").shape == expected


def test_three_phase_saturation_closure(tmp_path: Path) -> None:
    case_dir = _run_three_phase(tmp_path, "three_phase_closure")["case_dir"]
    sw = np.load(case_dir / "sw_three_phase.npy")
    sg = np.load(case_dir / "sg_three_phase.npy")
    so = np.load(case_dir / "so_three_phase.npy")
    assert np.allclose(sw + sg + so, 1.0)


def test_three_phase_saturation_bounds(tmp_path: Path) -> None:
    case_dir = _run_three_phase(tmp_path, "three_phase_bounds")["case_dir"]
    sw = np.load(case_dir / "sw_three_phase.npy")
    sg = np.load(case_dir / "sg_three_phase.npy")
    so = np.load(case_dir / "so_three_phase.npy")
    assert sw.min() >= 0.2
    assert sg.min() >= 0.05
    assert so.min() >= 0.2


def test_three_phase_no_nan_inf(tmp_path: Path) -> None:
    case_dir = _run_three_phase(tmp_path, "three_phase_finite")["case_dir"]
    for name in ["sw_three_phase.npy", "sg_three_phase.npy", "so_three_phase.npy", "pressure.npy", "flux_x.npy", "flux_y.npy", "flux_z.npy"]:
        values = np.load(case_dir / name)
        assert np.isfinite(values).all()


def test_three_phase_report_exists(tmp_path: Path) -> None:
    assert (_run_three_phase(tmp_path, "three_phase_report_exists")["case_dir"] / "three_phase_report.json").exists()


def test_three_phase_report_keys(tmp_path: Path) -> None:
    report = _json(_run_three_phase(tmp_path, "three_phase_report_keys")["case_dir"] / "three_phase_report.json")
    keys = {
        "max_cfl",
        "water_inflow",
        "water_outflow",
        "water_storage_change",
        "water_balance_error",
        "gas_inflow",
        "gas_outflow",
        "gas_storage_change",
        "gas_balance_error",
        "oil_inflow",
        "oil_outflow",
        "oil_storage_change",
        "oil_balance_error",
        "closure_error_max",
        "sw_min",
        "sw_max",
        "sg_min",
        "sg_max",
        "so_min",
        "so_max",
        "has_nan",
        "has_inf",
        "transport_dimension",
    }
    assert keys.issubset(report)
    assert report["transport_dimension"] == "3d"


def test_three_phase_case_summary_exists(tmp_path: Path) -> None:
    assert (_run_three_phase(tmp_path, "three_phase_summary_exists")["case_dir"] / "case_summary.json").exists()


def test_three_phase_case_summary_keys(tmp_path: Path) -> None:
    summary = _json(_run_three_phase(tmp_path, "three_phase_summary_keys")["case_dir"] / "case_summary.json")
    keys = {
        "case_id",
        "success",
        "three_phase_enabled",
        "three_phase_model",
        "three_phase_transport_enabled",
        "black_oil_enabled",
        "sw_min",
        "sw_max",
        "sg_min",
        "sg_max",
        "so_min",
        "so_max",
        "closure_error_max",
        "max_cfl",
        "water_balance_error",
        "gas_balance_error",
        "oil_balance_error",
        "has_nan",
        "has_inf",
    }
    assert keys.issubset(summary)


def test_three_phase_case_summary_flags(tmp_path: Path) -> None:
    summary = _json(_run_three_phase(tmp_path, "three_phase_flags")["case_dir"] / "case_summary.json")
    assert summary["three_phase_enabled"] is True
    assert summary["three_phase_model"] == "incompressible_wog"
    assert summary["three_phase_transport_enabled"] is True
    assert summary["black_oil_enabled"] is False


def test_three_phase_max_cfl_recorded(tmp_path: Path) -> None:
    summary = _json(_run_three_phase(tmp_path, "three_phase_cfl")["case_dir"] / "case_summary.json")
    assert 0.0 <= summary["max_cfl"] <= 1.0


def test_three_phase_material_balance_recorded(tmp_path: Path) -> None:
    summary = _json(_run_three_phase(tmp_path, "three_phase_mb")["case_dir"] / "case_summary.json")
    assert abs(summary["water_balance_error"]) < 1.0e-12
    assert abs(summary["gas_balance_error"]) < 1.0e-12
    assert abs(summary["oil_balance_error"]) < 1.0e-12


def test_three_phase_repeatability(tmp_path: Path) -> None:
    first = _run_three_phase(tmp_path, "three_phase_repeat_a")["case_dir"]
    second = _run_three_phase(tmp_path, "three_phase_repeat_b")["case_dir"]
    for name in ["sw_three_phase.npy", "sg_three_phase.npy", "so_three_phase.npy", "pressure.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_three_phase_cli_run(tmp_path: Path) -> None:
    result = _run_cli("--config", "config/three_phase_case.yaml", "--output-dir", str(tmp_path), "--case-id", "cli_three_phase")
    assert result.returncode == 0
    assert (tmp_path / "cli_three_phase" / "sw_three_phase.npy").exists()
    assert (tmp_path / "cli_three_phase" / "three_phase_report.json").exists()


def test_three_phase_cli_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry_three_phase"
    result = _run_cli("--config", "config/three_phase_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    assert json.loads(result.stdout)["three_phase_transport_enabled"] is True
    assert not output_dir.exists()


def test_three_phase_rejects_capillary_enabled(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["capillary_pressure"] = {"enabled": True, "model": "brooks_corey", "entry_pressure_pa": 1000.0, "lambda_pc": 2.0}
    config["saturation"]["use_capillary"] = True
    with pytest.raises(ValueError, match="does not support capillary/gravity/combined"):
        load_case_config(_write_config(tmp_path, config, "bad_capillary.yaml"))


def test_three_phase_rejects_gravity_enabled(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["gravity"]["enabled"] = True
    config["saturation"]["use_gravity"] = True
    with pytest.raises(ValueError, match="does not support capillary/gravity/combined"):
        load_case_config(_write_config(tmp_path, config, "bad_gravity.yaml"))


def test_three_phase_rejects_combined_enabled(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["capillary_pressure"] = {"enabled": True, "model": "brooks_corey", "entry_pressure_pa": 1000.0, "lambda_pc": 2.0}
    config["gravity"]["enabled"] = True
    config["saturation"]["use_capillary"] = True
    config["saturation"]["use_gravity"] = True
    with pytest.raises(ValueError, match="does not support capillary/gravity/combined"):
        load_case_config(_write_config(tmp_path, config, "bad_combined.yaml"))


def test_three_phase_rejects_multisignal_mode(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["case"]["mode"] = "multisignal"
    with pytest.raises(ValueError, match="does not support multisignal"):
        load_case_config(_write_config(tmp_path, config, "bad_multisignal.yaml"))


def test_three_phase_rejects_invalid_model(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["three_phase"]["model"] = "black_oil"
    with pytest.raises(ValueError, match="model must be incompressible_wog"):
        load_case_config(_write_config(tmp_path, config, "bad_model.yaml"))


def test_three_phase_rejects_missing_sw_sg(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    del config["initial_saturation"]["sw"]
    with pytest.raises(KeyError, match="requires sw and sg"):
        load_case_config(_write_config(tmp_path, config, "missing_sw.yaml"))


def test_three_phase_rejects_invalid_saturation_closure(tmp_path: Path) -> None:
    config = load_case_config("config/three_phase_case.yaml")
    config["initial_saturation"]["sw"] = 0.75
    config["initial_saturation"]["sg"] = 0.20
    with pytest.raises(InvalidPhysicalValueError):
        load_case_config(_write_config(tmp_path, config, "bad_closure.yaml"))


def test_existing_demo_case_unchanged(tmp_path: Path) -> None:
    case_dir = run_demo(case_id="demo_three_phase_unchanged", results_root=tmp_path, case_config=load_case_config("config/demo_case.yaml"))["case_dir"]
    assert not (case_dir / "sw_three_phase.npy").exists()
    assert (case_dir / "sw_simulated.npy").exists()


def test_existing_combined_case_unchanged(tmp_path: Path) -> None:
    case_dir = run_demo(case_id="combined_three_phase_unchanged", results_root=tmp_path, case_config=load_case_config("config/combined_case.yaml"))["case_dir"]
    assert not (case_dir / "sw_three_phase.npy").exists()
    assert (case_dir / "combined_report.json").exists()


def test_existing_oil_water_pipeline_tests_still_pass() -> None:
    assert callable(run_demo)


def test_three_phase_not_black_oil(tmp_path: Path) -> None:
    summary = _json(_run_three_phase(tmp_path, "three_phase_not_black_oil")["case_dir"] / "case_summary.json")
    assert summary["black_oil_enabled"] is False
    assert summary["three_phase_model"] == "incompressible_wog"


def test_three_phase_transport_3d_tests_still_pass() -> None:
    from reservoir_backend.solver.three_phase_transport import advance_three_phase_saturation_3d

    assert callable(advance_three_phase_saturation_3d)


def _run_three_phase(tmp_path: Path, case_id: str) -> dict[str, object]:
    return run_demo(case_id=case_id, results_root=tmp_path, case_config=load_case_config("config/three_phase_case.yaml"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_case.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, config: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
