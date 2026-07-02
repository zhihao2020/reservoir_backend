from __future__ import annotations

import json

import numpy as np

from examples.run_full_pipeline_demo import run_demo as run_full_demo
from examples.run_multisignal_inversion_demo import run_demo as run_multisignal_demo


def test_multisignal_demo_runs(tmp_path) -> None:
    result = run_multisignal_demo(case_id="multi_case", results_root=tmp_path)
    assert result["case_dir"].exists()


def test_multisignal_outputs_exist(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="outputs_case", results_root=tmp_path)["case_dir"]
    for name in ["sw_resistivity.npy", "sw_em.npy", "sw_acoustic.npy", "sw_signal_fused.npy"]:
        assert (case_dir / name).exists()


def test_multisignal_outputs_shape(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="shape_case", results_root=tmp_path)["case_dir"]
    expected = (3, 5, 6)
    for name in ["sw_resistivity.npy", "sw_em.npy", "sw_acoustic.npy", "sw_signal_fused.npy"]:
        assert np.load(case_dir / name).shape == expected


def test_multisignal_saturation_bounds(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="bounds_case", results_root=tmp_path)["case_dir"]
    for name in ["sw_resistivity.npy", "sw_em.npy", "sw_acoustic.npy", "sw_signal_fused.npy"]:
        values = np.load(case_dir / name)
        assert values.min() >= 0.2
        assert values.max() <= 0.8


def test_multisignal_confidence_range(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="confidence_case", results_root=tmp_path)["case_dir"]
    for name in ["confidence_resistivity.npy", "confidence_em.npy", "confidence_acoustic.npy"]:
        values = np.load(case_dir / name)
        assert values.min() >= 0.0
        assert values.max() <= 1.0


def test_multisignal_fusion_report_keys(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="report_case", results_root=tmp_path)["case_dir"]
    report = json.loads((case_dir / "signal_fusion_report.json").read_text())
    keys = {"nan_cells_count", "clipped_cells", "fused_min", "fused_max"}
    assert keys.issubset(report)


def test_multisignal_summary_keys(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="summary_case", results_root=tmp_path)["case_dir"]
    summary = json.loads((case_dir / "multisignal_summary.json").read_text())
    keys = {"signal_sources", "rmse_resistivity", "rmse_em", "rmse_acoustic", "rmse_signal_fused"}
    assert keys.issubset(summary)


def test_multisignal_fused_rmse_reasonable(tmp_path) -> None:
    case_dir = run_multisignal_demo(case_id="rmse_case", results_root=tmp_path)["case_dir"]
    summary = json.loads((case_dir / "multisignal_summary.json").read_text())
    worst_single = max(summary["rmse_resistivity"], summary["rmse_em"], summary["rmse_acoustic"])
    assert summary["rmse_signal_fused"] <= worst_single + 1.0e-12


def test_full_pipeline_multisignal_mode_runs(tmp_path) -> None:
    result = run_full_demo(case_id="full_multi_case", results_root=tmp_path, use_multisignal=True)
    assert result["case_dir"].exists()
    assert result["summary"]["use_multisignal"] is True


def test_full_pipeline_multisignal_outputs_exist(tmp_path) -> None:
    case_dir = run_full_demo(case_id="full_multi_outputs", results_root=tmp_path, use_multisignal=True)["case_dir"]
    assert (case_dir / "sw_signal_fused.npy").exists()
    assert (case_dir / "sw_fused.npy").exists()
