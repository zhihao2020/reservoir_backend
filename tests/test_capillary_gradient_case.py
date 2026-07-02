from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from examples.run_full_pipeline_demo import build_initial_saturation_field, run_demo
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.io.config_loader import load_case_config


def test_capillary_gradient_case_config_loads() -> None:
    config = load_case_config("config/capillary_gradient_case.yaml")
    assert config["case"]["case_id"] == "capillary_gradient_case"
    assert config["initial_saturation"]["type"] == "step_x"
    assert config["capillary_pressure"]["enabled"] is True


def test_step_x_initial_saturation_created() -> None:
    config = load_case_config("config/capillary_gradient_case.yaml")
    grid = Grid3D(nx=6, ny=5, nz=3, dx=1.0, dy=1.0, dz=1.0)
    field = build_initial_saturation_field(grid, config["initial_saturation"], swi=0.2, sor=0.2)
    assert np.allclose(field.values[:, :, :3], 0.75)
    assert np.allclose(field.values[:, :, 3:], 0.2)


def test_initial_saturation_saved(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_initial")["case_dir"]
    assert (case_dir / "initial_saturation.npy").exists()


def test_capillary_gradient_pipeline_runs(tmp_path) -> None:
    result = _run_gradient(tmp_path, "grad_run")
    assert result["summary"]["success"] is True
    assert result["summary"]["initial_saturation_type"] == "step_x"


def test_capillary_gradient_outputs_exist(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_outputs")["case_dir"]
    for name in [
        "capillary_pressure.npy",
        "capillary_flux_x.npy",
        "capillary_flux_y.npy",
        "capillary_flux_z.npy",
        "capillary_report.json",
    ]:
        assert (case_dir / name).exists()


def test_capillary_flux_nonzero_for_gradient_case(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_flux_nonzero")["case_dir"]
    report = json.loads((case_dir / "capillary_report.json").read_text())
    assert report["max_abs_capillary_flux"] > 0.0
    assert np.max(np.abs(np.load(case_dir / "capillary_flux_x.npy"))) > 0.0


def test_capillary_pressure_nonuniform(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_pc_nonuniform")["case_dir"]
    pc = np.load(case_dir / "capillary_pressure.npy")
    assert pc.min() < pc.max()


def test_capillary_gradient_saturation_bounds(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_bounds")["case_dir"]
    sw = np.load(case_dir / "sw_simulated.npy")
    assert sw.min() >= 0.2
    assert sw.max() <= 0.8


def test_capillary_gradient_no_nan_inf(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_finite")["case_dir"]
    for name in [
        "capillary_pressure.npy",
        "capillary_flux_x.npy",
        "capillary_flux_y.npy",
        "capillary_flux_z.npy",
        "sw_simulated.npy",
    ]:
        values = np.load(case_dir / name)
        assert np.isfinite(values).all()


def test_capillary_gradient_summary_keys(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_summary")["case_dir"]
    summary = json.loads((case_dir / "case_summary.json").read_text())
    keys = {
        "initial_saturation_type",
        "capillary_enabled",
        "max_abs_capillary_flux",
        "max_capillary_flux",
        "max_total_water_flux",
    }
    assert keys.issubset(summary)
    assert summary["initial_saturation_type"] == "step_x"
    assert summary["max_abs_capillary_flux"] > 0.0


def test_capillary_gradient_repeatability(tmp_path) -> None:
    first = _run_gradient(tmp_path, "grad_repeat_a")["case_dir"]
    second = _run_gradient(tmp_path, "grad_repeat_b")["case_dir"]
    for name in ["initial_saturation.npy", "capillary_pressure.npy", "capillary_flux_x.npy", "sw_simulated.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_capillary_enabled_differs_from_disabled(tmp_path) -> None:
    enabled_dir = _run_gradient(tmp_path, "grad_enabled")["case_dir"]
    config = load_case_config("config/capillary_gradient_case.yaml")
    config["capillary_pressure"]["enabled"] = False
    config["capillary_pressure"]["model"] = "none"
    disabled_dir = run_demo(case_id="grad_disabled", results_root=tmp_path, case_config=config)["case_dir"]
    difference = np.max(
        np.abs(np.load(enabled_dir / "sw_simulated.npy") - np.load(disabled_dir / "sw_simulated.npy"))
    )
    assert difference > 1.0e-8


def test_capillary_smooths_gradient_front(tmp_path) -> None:
    case_dir = _run_gradient(tmp_path, "grad_smooth")["case_dir"]
    initial = np.load(case_dir / "initial_saturation.npy")
    final = np.load(case_dir / "sw_simulated.npy")
    assert _max_x_jump(final) < _max_x_jump(initial)


def test_capillary_profile_script_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/profile_capillary_pipeline.py"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert Path("profiling_reports/capillary_performance_summary.json").exists()
    assert Path("profiling_reports/capillary_performance_summary.md").exists()


def test_capillary_profile_summary_keys() -> None:
    summary_path = Path("profiling_reports/capillary_performance_summary.json")
    if not summary_path.exists():
        subprocess.run(
            [sys.executable, "scripts/profile_capillary_pipeline.py"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
    summary = json.loads(summary_path.read_text())
    by_case = {case["case_id"]: case for case in summary["cases"]}
    for case_id in ["demo_case", "capillary_case", "capillary_gradient_case"]:
        assert case_id in by_case
        for key in [
            "total_runtime_sec",
            "capillary_enabled",
            "initial_saturation_type",
            "total_cells",
            "steps",
            "max_cfl",
            "max_abs_capillary_flux",
            "material_balance_error",
            "success",
        ]:
            assert key in by_case[case_id]


def test_existing_capillary_case_still_valid(tmp_path) -> None:
    config = load_case_config("config/capillary_case.yaml")
    case_dir = run_demo(case_id="cap_existing", results_root=tmp_path, case_config=config)["case_dir"]
    assert (case_dir / "capillary_pressure.npy").exists()
    report = json.loads((case_dir / "capillary_report.json").read_text())
    assert report["capillary_enabled"] is True


def test_existing_demo_and_multisignal_still_valid(tmp_path) -> None:
    demo_config = load_case_config("config/demo_case.yaml")
    demo_dir = run_demo(case_id="demo_existing", results_root=tmp_path, case_config=demo_config)["case_dir"]
    assert not (demo_dir / "capillary_pressure.npy").exists()

    multi_config = load_case_config("config/multisignal_case.yaml")
    multi_dir = run_demo(
        case_id="multi_existing",
        results_root=tmp_path,
        use_multisignal=True,
        case_config=multi_config,
    )["case_dir"]
    assert (multi_dir / "sw_signal_fused.npy").exists()
    assert not (multi_dir / "capillary_pressure.npy").exists()


def _run_gradient(tmp_path: Path, case_id: str) -> dict[str, object]:
    config = load_case_config("config/capillary_gradient_case.yaml")
    return run_demo(case_id=case_id, results_root=tmp_path, case_config=config)


def _max_x_jump(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(values, axis=2))))
