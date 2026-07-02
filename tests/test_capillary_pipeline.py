from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from examples.run_full_pipeline_demo import run_demo
from reservoir_backend.io.config_loader import load_case_config


def test_capillary_case_config_loads() -> None:
    config = load_case_config("config/capillary_case.yaml")
    assert config["capillary_pressure"]["enabled"] is True
    assert config["outputs"]["save_capillary_flux"] is True


def test_capillary_pipeline_runs(tmp_path) -> None:
    result = _run_capillary(tmp_path, "cap_run")
    assert result["summary"]["success"] is True
    assert result["summary"]["capillary_enabled"] is True


def test_capillary_outputs_exist_when_enabled(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_outputs")["case_dir"]
    required = [
        "capillary_pressure.npy",
        "capillary_flux_x.npy",
        "capillary_flux_y.npy",
        "capillary_flux_z.npy",
        "capillary_report.json",
    ]
    assert all((case_dir / name).exists() for name in required)


def test_capillary_outputs_not_required_when_disabled(tmp_path) -> None:
    config = load_case_config("config/demo_case.yaml")
    result = run_demo(case_id="disabled_case", results_root=tmp_path, use_multisignal=False, case_config=config)
    case_dir = result["case_dir"]
    assert result["summary"]["capillary_enabled"] is False
    assert not (case_dir / "capillary_pressure.npy").exists()
    assert not (case_dir / "capillary_flux_x.npy").exists()
    assert not (case_dir / "capillary_report.json").exists()


def test_capillary_case_summary_keys(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_summary")["case_dir"]
    summary = json.loads((case_dir / "case_summary.json").read_text())
    keys = {
        "capillary_enabled",
        "capillary_model",
        "capillary_pressure_min",
        "capillary_pressure_max",
        "max_abs_capillary_flux",
        "capillary_flux_included",
    }
    assert keys.issubset(summary)
    assert summary["capillary_enabled"] is True


def test_capillary_report_keys(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_report")["case_dir"]
    report = json.loads((case_dir / "capillary_report.json").read_text())
    keys = {
        "capillary_enabled",
        "capillary_model",
        "max_abs_capillary_flux",
        "max_advective_flux",
        "max_capillary_flux",
        "max_total_water_flux",
        "capillary_flux_included",
        "material_balance_error",
    }
    assert keys.issubset(report)
    assert report["capillary_flux_included"] is True


def test_capillary_pressure_field_valid(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_pc")["case_dir"]
    pc = np.load(case_dir / "capillary_pressure.npy")
    assert np.isfinite(pc).all()
    assert pc.min() >= 0.0


def test_capillary_flux_shapes(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_flux_shapes")["case_dir"]
    assert np.load(case_dir / "capillary_flux_x.npy").shape == (3, 5, 7)
    assert np.load(case_dir / "capillary_flux_y.npy").shape == (3, 6, 6)
    assert np.load(case_dir / "capillary_flux_z.npy").shape == (4, 5, 6)


def test_capillary_saturation_bounds(tmp_path) -> None:
    case_dir = _run_capillary(tmp_path, "cap_bounds")["case_dir"]
    sw = np.load(case_dir / "sw_simulated.npy")
    assert sw.min() >= 0.2
    assert sw.max() <= 0.8


def test_capillary_pipeline_repeatability(tmp_path) -> None:
    first = _run_capillary(tmp_path, "cap_repeat_a")["case_dir"]
    second = _run_capillary(tmp_path, "cap_repeat_b")["case_dir"]
    for name in ["capillary_pressure.npy", "capillary_flux_x.npy", "sw_simulated.npy", "sw_fused.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_cli_run_capillary_case(tmp_path) -> None:
    result = _run_cli("--config", "config/capillary_case.yaml", "--output-dir", str(tmp_path), "--case-id", "cli_cap")
    assert result.returncode == 0
    assert (tmp_path / "cli_cap" / "capillary_pressure.npy").exists()


def test_cli_dry_run_capillary_case(tmp_path) -> None:
    output_dir = tmp_path / "dry_run_outputs"
    result = _run_cli("--config", "config/capillary_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    assert json.loads(result.stdout)["capillary_enabled"] is True
    assert not output_dir.exists()


def test_existing_demo_case_unchanged(tmp_path) -> None:
    config = load_case_config("config/demo_case.yaml")
    case_dir = run_demo(case_id="demo_unchanged", results_root=tmp_path, case_config=config)["case_dir"]
    assert not (case_dir / "capillary_pressure.npy").exists()
    assert not (case_dir / "capillary_flux_x.npy").exists()


def test_existing_multisignal_case_unchanged(tmp_path) -> None:
    config = load_case_config("config/multisignal_case.yaml")
    case_dir = run_demo(
        case_id="multi_unchanged",
        results_root=tmp_path,
        use_multisignal=True,
        case_config=config,
    )["case_dir"]
    assert (case_dir / "sw_signal_fused.npy").exists()
    assert not (case_dir / "capillary_pressure.npy").exists()


def _run_capillary(tmp_path: Path, case_id: str) -> dict[str, object]:
    config = load_case_config("config/capillary_case.yaml")
    return run_demo(case_id=case_id, results_root=tmp_path, use_multisignal=False, case_config=config)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_case.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
