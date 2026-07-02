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


def test_gravity_case_config_loads() -> None:
    config = load_case_config("config/gravity_case.yaml")
    assert config["gravity"]["enabled"] is True
    assert config["saturation"]["use_gravity"] is True
    assert config["outputs"]["save_gravity_flux"] is True


def test_gravity_pipeline_runs(tmp_path) -> None:
    result = _run_gravity(tmp_path, "grav_run")
    assert result["summary"]["success"] is True
    assert result["summary"]["gravity_enabled"] is True


def test_gravity_outputs_exist_when_enabled(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_outputs")["case_dir"]
    required = ["gravity_flux_x.npy", "gravity_flux_y.npy", "gravity_flux_z.npy", "gravity_report.json"]
    assert all((case_dir / name).exists() for name in required)


def test_gravity_outputs_not_required_when_disabled(tmp_path) -> None:
    config = load_case_config("config/demo_case.yaml")
    result = run_demo(case_id="gravity_disabled", results_root=tmp_path, case_config=config)
    case_dir = result["case_dir"]
    summary = json.loads((case_dir / "case_summary.json").read_text())
    assert summary["gravity_enabled"] is False
    assert not (case_dir / "gravity_flux_x.npy").exists()
    assert not (case_dir / "gravity_flux_y.npy").exists()
    assert not (case_dir / "gravity_flux_z.npy").exists()
    assert not (case_dir / "gravity_report.json").exists()


def test_gravity_case_summary_keys(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_summary")["case_dir"]
    summary = json.loads((case_dir / "case_summary.json").read_text())
    keys = {"gravity_enabled", "rho_w", "rho_o", "density_difference", "max_abs_gravity_flux", "gravity_flux_included"}
    assert keys.issubset(summary)
    assert summary["gravity_enabled"] is True
    assert summary["max_abs_gravity_flux"] > 0.0


def test_gravity_report_keys(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_report")["case_dir"]
    report = json.loads((case_dir / "gravity_report.json").read_text())
    keys = {
        "gravity_enabled",
        "gravity_flux_included",
        "rho_w",
        "rho_o",
        "density_difference",
        "max_abs_gravity_flux",
        "max_total_water_flux",
        "material_balance_error",
        "max_cfl",
    }
    assert keys.issubset(report)
    assert report["gravity_flux_included"] is True


def test_gravity_flux_shapes(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_shapes")["case_dir"]
    assert np.load(case_dir / "gravity_flux_x.npy").shape == (3, 5, 7)
    assert np.load(case_dir / "gravity_flux_y.npy").shape == (3, 6, 6)
    assert np.load(case_dir / "gravity_flux_z.npy").shape == (4, 5, 6)


def test_gravity_z_flux_nonzero_when_enabled(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_nonzero")["case_dir"]
    flux_z = np.load(case_dir / "gravity_flux_z.npy")
    assert np.max(np.abs(flux_z[1:-1, :, :])) > 0.0


def test_gravity_x_y_flux_zero_regular_grid(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_xy_zero")["case_dir"]
    assert np.allclose(np.load(case_dir / "gravity_flux_x.npy"), 0.0)
    assert np.allclose(np.load(case_dir / "gravity_flux_y.npy"), 0.0)


def test_gravity_saturation_bounds(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_bounds")["case_dir"]
    sw = np.load(case_dir / "sw_simulated.npy")
    assert sw.min() >= 0.2
    assert sw.max() <= 0.8


def test_gravity_no_nan_inf(tmp_path) -> None:
    case_dir = _run_gravity(tmp_path, "grav_finite")["case_dir"]
    for name in ["gravity_flux_x.npy", "gravity_flux_y.npy", "gravity_flux_z.npy", "sw_simulated.npy", "sw_fused.npy"]:
        values = np.load(case_dir / name)
        assert np.isfinite(values).all()


def test_gravity_pipeline_repeatability(tmp_path) -> None:
    first = _run_gravity(tmp_path, "grav_repeat_a")["case_dir"]
    second = _run_gravity(tmp_path, "grav_repeat_b")["case_dir"]
    for name in ["gravity_flux_z.npy", "sw_simulated.npy", "sw_fused.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_cli_run_gravity_case(tmp_path) -> None:
    result = _run_cli("--config", "config/gravity_case.yaml", "--output-dir", str(tmp_path), "--case-id", "cli_grav")
    assert result.returncode == 0
    assert (tmp_path / "cli_grav" / "gravity_flux_z.npy").exists()
    assert (tmp_path / "cli_grav" / "gravity_report.json").exists()


def test_cli_dry_run_gravity_case(tmp_path) -> None:
    output_dir = tmp_path / "dry_gravity"
    result = _run_cli("--config", "config/gravity_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["gravity_enabled"] is True
    assert payload["density_difference"] == pytest.approx(200.0)
    assert not output_dir.exists()


def test_existing_demo_case_unchanged(tmp_path) -> None:
    config = load_case_config("config/demo_case.yaml")
    case_dir = run_demo(case_id="demo_gravity_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert not (case_dir / "gravity_flux_x.npy").exists()
    assert not (case_dir / "gravity_report.json").exists()


def test_existing_multisignal_case_unchanged(tmp_path) -> None:
    config = load_case_config("config/multisignal_case.yaml")
    case_dir = run_demo(
        case_id="multi_gravity_unchanged",
        results_root=tmp_path,
        use_multisignal=True,
        case_config=config,
    )["case_dir"]
    assert (case_dir / "sw_signal_fused.npy").exists()
    assert not (case_dir / "gravity_flux_z.npy").exists()


def test_existing_capillary_case_unchanged(tmp_path) -> None:
    config = load_case_config("config/capillary_case.yaml")
    case_dir = run_demo(case_id="cap_gravity_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert (case_dir / "capillary_report.json").exists()
    assert not (case_dir / "gravity_report.json").exists()


def test_capillary_and_gravity_enabled_together_requires_matching_flags(tmp_path) -> None:
    config = load_case_config("config/gravity_case.yaml")
    config["capillary_pressure"]["enabled"] = True
    config["capillary_pressure"]["model"] = "brooks_corey"
    bad = tmp_path / "bad_combined.yaml"
    bad.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="capillary_pressure.enabled=true requires saturation.use_capillary=true"):
        load_case_config(bad)


def _run_gravity(tmp_path: Path, case_id: str) -> dict[str, object]:
    config = load_case_config("config/gravity_case.yaml")
    return run_demo(case_id=case_id, results_root=tmp_path, case_config=config)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_case.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
