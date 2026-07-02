from __future__ import annotations

import json
import time

import numpy as np

from examples.run_full_pipeline_demo import REQUIRED_OUTPUTS, run_demo


def test_full_pipeline_small_case_runs(tmp_path) -> None:
    result = run_demo(case_id="small_case", results_root=tmp_path)
    assert result["case_dir"].exists()


def test_full_pipeline_outputs_exist(tmp_path) -> None:
    result = run_demo(case_id="outputs_case", results_root=tmp_path)
    case_dir = result["case_dir"]
    required = [
        "pressure.npy",
        "sw_inverted.npy",
        "sw_simulated.npy",
        "sw_fused.npy",
        "production_curve.csv",
        "case_summary.json",
    ]
    assert all((case_dir / name).exists() for name in required)


def test_full_pipeline_no_nan_inf(tmp_path) -> None:
    case_dir = run_demo(case_id="finite_case", results_root=tmp_path)["case_dir"]
    for name in [
        "pressure.npy",
        "sw_inverted.npy",
        "sw_simulated.npy",
        "sw_fused.npy",
        "velocity_x.npy",
        "velocity_y.npy",
        "velocity_z.npy",
        "flux_x.npy",
        "flux_y.npy",
        "flux_z.npy",
    ]:
        array = np.load(case_dir / name)
        assert not np.isnan(array).any()
        assert not np.isinf(array).any()


def test_full_pipeline_physical_ranges(tmp_path) -> None:
    case_dir = run_demo(case_id="ranges_case", results_root=tmp_path)["case_dir"]
    for name in ["sw_inverted.npy", "sw_simulated.npy", "sw_fused.npy"]:
        array = np.load(case_dir / name)
        assert array.min() >= 0.2
        assert array.max() <= 0.8


def test_full_pipeline_pressure_reasonable(tmp_path) -> None:
    case_dir = run_demo(case_id="pressure_case", results_root=tmp_path)["case_dir"]
    pressure = np.load(case_dir / "pressure.npy")
    assert np.isfinite(pressure).all()
    assert pressure.min() < pressure.max()


def test_full_pipeline_material_balance_report(tmp_path) -> None:
    case_dir = run_demo(case_id="mb_case", results_root=tmp_path)["case_dir"]
    report = json.loads((case_dir / "material_balance_report.json").read_text())
    keys = {"injected_water_volume", "produced_water_volume", "storage_change", "material_balance_error"}
    assert keys.issubset(report)


def test_full_pipeline_fusion_report(tmp_path) -> None:
    case_dir = run_demo(case_id="fusion_case", results_root=tmp_path)["case_dir"]
    report = json.loads((case_dir / "fusion_report.json").read_text())
    keys = {"nan_cells_count", "clipped_cells", "fused_min", "fused_max"}
    assert keys.issubset(report)


def test_full_pipeline_case_summary(tmp_path) -> None:
    case_dir = run_demo(case_id="summary_case", results_root=tmp_path)["case_dir"]
    summary = json.loads((case_dir / "case_summary.json").read_text())
    keys = {"grid_shape", "modules_used", "output_files", "success"}
    assert keys.issubset(summary)
    assert summary["success"] is True
    assert set(REQUIRED_OUTPUTS).issubset(set(summary["output_files"]))


def test_full_pipeline_repeatability(tmp_path) -> None:
    first = run_demo(case_id="repeat_a", results_root=tmp_path)["case_dir"]
    second = run_demo(case_id="repeat_b", results_root=tmp_path)["case_dir"]
    for name in ["pressure.npy", "sw_inverted.npy", "sw_simulated.npy", "sw_fused.npy"]:
        assert np.allclose(np.load(first / name), np.load(second / name))


def test_full_pipeline_runtime_small(tmp_path) -> None:
    start = time.perf_counter()
    run_demo(case_id="runtime_case", results_root=tmp_path)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0


def test_full_pipeline_multisignal_mode_preserves_outputs(tmp_path) -> None:
    case_dir = run_demo(case_id="multi_case", results_root=tmp_path, use_multisignal=True)["case_dir"]
    assert (case_dir / "sw_signal_fused.npy").exists()
    assert (case_dir / "sw_fused.npy").exists()
