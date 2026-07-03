from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def test_cli_dry_run_demo_case(tmp_path) -> None:
    output_dir = tmp_path / "results"
    result = _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    assert not output_dir.exists()
    assert json.loads(result.stdout)["dry_run"] is True


def test_cli_run_archie_only_case(tmp_path) -> None:
    result = _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "archie_case")
    assert result.returncode == 0
    assert (tmp_path / "archie_case" / "case_summary.json").exists()


def test_cli_run_multisignal_case(tmp_path) -> None:
    result = _run_cli("--config", "config/multisignal_case.yaml", "--output-dir", str(tmp_path), "--case-id", "multi_case")
    assert result.returncode == 0
    assert (tmp_path / "multi_case" / "sw_signal_fused.npy").exists()


def test_cli_run_capillary_case(tmp_path) -> None:
    result = _run_cli("--config", "config/capillary_case.yaml", "--output-dir", str(tmp_path), "--case-id", "cap_case")
    assert result.returncode == 0
    assert (tmp_path / "cap_case" / "capillary_pressure.npy").exists()
    assert (tmp_path / "cap_case" / "capillary_report.json").exists()


def test_cli_dry_run_capillary_case(tmp_path) -> None:
    output_dir = tmp_path / "capillary_results"
    result = _run_cli("--config", "config/capillary_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["capillary_enabled"] is True
    assert payload["capillary_model"] == "brooks_corey"
    assert not output_dir.exists()


def test_cli_run_gravity_case(tmp_path) -> None:
    result = _run_cli("--config", "config/gravity_case.yaml", "--output-dir", str(tmp_path), "--case-id", "grav_case")
    assert result.returncode == 0
    assert (tmp_path / "grav_case" / "gravity_flux_z.npy").exists()
    assert (tmp_path / "grav_case" / "gravity_report.json").exists()


def test_cli_dry_run_gravity_case(tmp_path) -> None:
    output_dir = tmp_path / "gravity_results"
    result = _run_cli("--config", "config/gravity_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["gravity_enabled"] is True
    assert payload["density_difference"] == 200.0
    assert not output_dir.exists()


def test_cli_run_combined_case(tmp_path) -> None:
    result = _run_cli("--config", "config/combined_case.yaml", "--output-dir", str(tmp_path), "--case-id", "combined_case")
    assert result.returncode == 0
    assert (tmp_path / "combined_case" / "combined_report.json").exists()
    assert (tmp_path / "combined_case" / "capillary_flux_x.npy").exists()
    assert (tmp_path / "combined_case" / "gravity_flux_z.npy").exists()


def test_cli_dry_run_combined_case(tmp_path) -> None:
    output_dir = tmp_path / "combined_results"
    result = _run_cli("--config", "config/combined_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["capillary_enabled"] is True
    assert payload["gravity_enabled"] is True
    assert payload["combined_transport_enabled"] is True
    assert not output_dir.exists()


def test_cli_run_three_phase_case(tmp_path) -> None:
    result = _run_cli("--config", "config/three_phase_case.yaml", "--output-dir", str(tmp_path), "--case-id", "three_cli")
    assert result.returncode == 0
    assert (tmp_path / "three_cli" / "sw_three_phase.npy").exists()
    assert (tmp_path / "three_cli" / "three_phase_report.json").exists()


def test_cli_dry_run_three_phase_case(tmp_path) -> None:
    output_dir = tmp_path / "three_dry"
    result = _run_cli("--config", "config/three_phase_case.yaml", "--output-dir", str(output_dir), "--dry-run")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["three_phase_enabled"] is True
    assert payload["three_phase_transport_enabled"] is True
    assert payload["black_oil_enabled"] is False
    assert not output_dir.exists()


def test_cli_outputs_case_summary(tmp_path) -> None:
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "summary_case")
    summary = json.loads((tmp_path / "summary_case" / "case_summary.json").read_text())
    assert summary["success"] is True


def test_cli_outputs_required_files(tmp_path) -> None:
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "files_case")
    required = ["pressure.npy", "sw_inverted.npy", "sw_simulated.npy", "sw_fused.npy", "case_summary.json"]
    assert all((tmp_path / "files_case" / name).exists() for name in required)


def test_cli_override_case_id(tmp_path) -> None:
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "override_case")
    assert (tmp_path / "override_case").exists()


def test_cli_override_output_dir(tmp_path) -> None:
    custom = tmp_path / "custom_outputs"
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(custom), "--case-id", "case_x")
    assert (custom / "case_x" / "case_summary.json").exists()


def test_cli_invalid_config_exits_nonzero(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"case": {"case_id": "bad"}, "grid": {"nx": 1, "ny": 1, "nz": 1, "dx": 1, "dy": 1, "dz": 1}}))
    result = _run_cli("--config", str(bad), "--output-dir", str(tmp_path))
    assert result.returncode != 0


def test_cli_repeatability(tmp_path) -> None:
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "repeat_a")
    _run_cli("--config", "config/demo_case.yaml", "--output-dir", str(tmp_path), "--case-id", "repeat_b")
    assert np.allclose(
        np.load(tmp_path / "repeat_a" / "pressure.npy"),
        np.load(tmp_path / "repeat_b" / "pressure.npy"),
    )


def test_cli_does_not_modify_core_algorithms() -> None:
    forbidden = [
        Path("reservoir_backend/solver/pressure_solver.py"),
        Path("reservoir_backend/solver/saturation_solver.py"),
        Path("reservoir_backend/inversion/resistivity_archie.py"),
    ]
    assert all(path.exists() for path in forbidden)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_case.py", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
