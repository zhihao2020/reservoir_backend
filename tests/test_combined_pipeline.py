from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.io.config_loader import load_case_config


def test_combined_case_config_loads() -> None:
    config = load_case_config("config/combined_case.yaml")
    assert config["capillary_pressure"]["enabled"] is True
    assert config["gravity"]["enabled"] is True
    assert config["saturation"]["use_capillary"] is True
    assert config["saturation"]["use_gravity"] is True


def test_combined_pipeline_runs(tmp_path: Path) -> None:
    result = _run_combined(tmp_path, "combined_run")
    assert result["summary"]["success"] is True
    assert result["summary"]["combined_transport_enabled"] is True


def test_combined_outputs_exist(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_outputs")["case_dir"]
    required = [
        "capillary_pressure.npy",
        "capillary_flux_x.npy",
        "capillary_flux_y.npy",
        "capillary_flux_z.npy",
        "gravity_flux_x.npy",
        "gravity_flux_y.npy",
        "gravity_flux_z.npy",
        "combined_report.json",
        "sw_simulated.npy",
        "sw_fused.npy",
        "case_summary.json",
    ]
    assert all((case_dir / name).exists() for name in required)


def test_combined_case_summary_keys(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_summary")["case_dir"]
    summary = _json(case_dir / "case_summary.json")
    keys = {
        "combined_transport_enabled",
        "capillary_enabled",
        "gravity_enabled",
        "max_total_water_flux",
        "max_effective_flux",
        "max_abs_capillary_flux",
        "max_abs_gravity_flux",
        "capillary_flux_included",
        "gravity_flux_included",
    }
    assert keys.issubset(summary)
    assert summary["combined_transport_enabled"] is True


def test_combined_report_keys(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_report")["case_dir"]
    report = _json(case_dir / "combined_report.json")
    keys = {
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
        "max_effective_flux",
        "max_cfl",
        "material_balance_error",
        "capillary_flux_included",
        "gravity_flux_included",
        "has_nan",
        "has_inf",
    }
    assert keys.issubset(report)
    assert report["capillary_enabled"] is True
    assert report["gravity_enabled"] is True


def test_combined_capillary_pressure_valid(tmp_path: Path) -> None:
    pc = np.load(_run_combined(tmp_path, "combined_pc")["case_dir"] / "capillary_pressure.npy")
    assert np.isfinite(pc).all()
    assert pc.min() >= 0.0


def test_combined_capillary_flux_nonzero(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_cap_flux")["case_dir"]
    max_flux = max(
        float(np.max(np.abs(np.load(case_dir / name))))
        for name in ["capillary_flux_x.npy", "capillary_flux_y.npy", "capillary_flux_z.npy"]
    )
    assert max_flux > 0.0


def test_combined_gravity_flux_nonzero(tmp_path: Path) -> None:
    flux_z = np.load(_run_combined(tmp_path, "combined_grav_flux")["case_dir"] / "gravity_flux_z.npy")
    assert np.max(np.abs(flux_z[1:-1, :, :])) > 0.0


def test_combined_sw_bounds(tmp_path: Path) -> None:
    sw = np.load(_run_combined(tmp_path, "combined_bounds")["case_dir"] / "sw_simulated.npy")
    assert sw.min() >= 0.2
    assert sw.max() <= 0.8


def test_combined_no_nan_inf(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_finite")["case_dir"]
    for name in [
        "sw_simulated.npy",
        "sw_fused.npy",
        "capillary_pressure.npy",
        "capillary_flux_x.npy",
        "capillary_flux_y.npy",
        "capillary_flux_z.npy",
        "gravity_flux_x.npy",
        "gravity_flux_y.npy",
        "gravity_flux_z.npy",
    ]:
        assert np.isfinite(np.load(case_dir / name)).all()


def test_combined_repeatability(tmp_path: Path) -> None:
    first = _run_combined(tmp_path, "combined_repeat_a")["case_dir"]
    second = _run_combined(tmp_path, "combined_repeat_b")["case_dir"]
    for name in ["sw_simulated.npy", "sw_fused.npy", "capillary_flux_x.npy", "gravity_flux_z.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_cli_run_combined_case(tmp_path: Path) -> None:
    result = _run_cli("--config", "config/combined_case.yaml", "--output-dir", str(tmp_path), "--case-id", "cli_combined")
    assert result.returncode == 0
    assert (tmp_path / "cli_combined" / "combined_report.json").exists()


def test_cli_dry_run_combined_case(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry_combined"
    result = _run_cli("--config", "config/combined_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["combined_transport_enabled"] is True
    assert not output_dir.exists()


def test_existing_demo_case_unchanged(tmp_path: Path) -> None:
    config = load_case_config("config/demo_case.yaml")
    case_dir = run_demo(case_id="demo_combined_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert not (case_dir / "combined_report.json").exists()
    assert not (case_dir / "capillary_flux_x.npy").exists()
    assert not (case_dir / "gravity_flux_x.npy").exists()


def test_existing_multisignal_case_unchanged(tmp_path: Path) -> None:
    config = load_case_config("config/multisignal_case.yaml")
    case_dir = run_demo(
        case_id="multi_combined_unchanged",
        results_root=tmp_path,
        use_multisignal=True,
        case_config=config,
    )["case_dir"]
    assert (case_dir / "sw_signal_fused.npy").exists()
    assert not (case_dir / "combined_report.json").exists()


def test_existing_capillary_case_unchanged(tmp_path: Path) -> None:
    config = load_case_config("config/capillary_case.yaml")
    case_dir = run_demo(case_id="cap_combined_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert (case_dir / "capillary_report.json").exists()
    assert not (case_dir / "gravity_report.json").exists()
    assert not (case_dir / "combined_report.json").exists()


def test_existing_gravity_case_unchanged(tmp_path: Path) -> None:
    config = load_case_config("config/gravity_case.yaml")
    case_dir = run_demo(case_id="grav_combined_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert (case_dir / "gravity_report.json").exists()
    assert not (case_dir / "capillary_report.json").exists()
    assert not (case_dir / "combined_report.json").exists()


def test_config_inconsistent_capillary_flags_raise(tmp_path: Path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_capillary"] = False
    path = tmp_path / "bad_capillary_flags.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="capillary_pressure.enabled=true requires saturation.use_capillary=true"):
        load_case_config(path)


def test_config_inconsistent_gravity_flags_raise(tmp_path: Path) -> None:
    config = load_case_config("config/combined_case.yaml")
    config["saturation"]["use_gravity"] = False
    path = tmp_path / "bad_gravity_flags.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="gravity.enabled=true requires saturation.use_gravity=true"):
        load_case_config(path)


def test_combined_pipeline_uses_combined_solver(tmp_path: Path) -> None:
    case_dir = _run_combined(tmp_path, "combined_solver_path")["case_dir"]
    report = _json(case_dir / "combined_report.json")
    assert report["combined_transport_enabled"] is True
    assert report["capillary_flux_included"] is True
    assert report["gravity_flux_included"] is True
    assert "composer_report" in report


def _run_combined(tmp_path: Path, case_id: str) -> dict[str, object]:
    config = load_case_config("config/combined_case.yaml")
    return run_demo(case_id=case_id, results_root=tmp_path, case_config=config)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_case.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
