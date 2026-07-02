"""Run a lightweight multisignal saturation inversion demo."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reservoir_backend.core.field import Field3D
from reservoir_backend.core.grid import Grid3D
from reservoir_backend.fusion.field_fusion import fuse_saturation_fields
from reservoir_backend.inversion.acoustic import AcousticInverter
from reservoir_backend.inversion.electromagnetic import ElectromagneticInverter
from reservoir_backend.inversion.resistivity_archie import ArchieInverter
from reservoir_backend.io.result_manager import ResultManager


def build_multisignal_inversion(
    grid: Grid3D,
    sw_true_values: np.ndarray,
    *,
    swi: float = 0.2,
    sor: float = 0.2,
) -> dict[str, object]:
    """Build resistivity, EM, acoustic, and fused saturation fields."""
    sw_true = Field3D(grid, np.clip(sw_true_values, swi, 1.0 - sor), name="sw_true", unit="fraction")
    phi = Field3D.from_constant(grid, 0.2, name="porosity", unit="fraction")

    archie = ArchieInverter(a=1.0, m=2.0, n=2.0, swi=swi, sor=sor)
    rt = archie.forward_resistivity(sw_true.values, rw=0.25, phi=phi.values)
    sw_resistivity = archie.invert(Field3D(grid, rt, name="Rt", unit="ohm.m"), rw=0.25, phi=phi)
    assert isinstance(sw_resistivity, Field3D)

    em_signal = (sw_true.values - 0.05) / 0.9
    em_params = {
        "model": "linear",
        "a": 0.9,
        "b": 0.05,
        "swi": swi,
        "sor": sor,
        "calibration_range": [float(np.min(em_signal)), float(np.max(em_signal))],
    }
    em_inverter = ElectromagneticInverter()
    sw_em = em_inverter.invert(Field3D(grid, em_signal, name="em_signal"), em_params)
    assert isinstance(sw_em, Field3D)

    acoustic_velocity = (1.1 - sw_true.values) / 2.0e-4
    acoustic_params = {
        "model": "linear",
        "a": -2.0e-4,
        "b": 1.1,
        "swi": swi,
        "sor": sor,
        "calibration_range": [float(np.min(acoustic_velocity)), float(np.max(acoustic_velocity))],
    }
    acoustic_inverter = AcousticInverter()
    sw_acoustic = acoustic_inverter.invert(Field3D(grid, acoustic_velocity, name="vp", unit="m/s"), acoustic_params)
    assert isinstance(sw_acoustic, Field3D)

    confidence_resistivity = Field3D(
        grid,
        sw_resistivity.confidence if sw_resistivity.confidence is not None else np.ones(grid.shape),
        name="confidence_resistivity",
        unit="fraction",
    )
    confidence_em = Field3D(
        grid,
        sw_em.confidence if sw_em.confidence is not None else np.ones(grid.shape),
        name="confidence_em",
        unit="fraction",
    )
    confidence_acoustic = Field3D(
        grid,
        sw_acoustic.confidence if sw_acoustic.confidence is not None else np.ones(grid.shape),
        name="confidence_acoustic",
        unit="fraction",
    )

    sw_signal_fused, signal_fusion_report = fuse_saturation_fields(
        [sw_resistivity, sw_em, sw_acoustic],
        confidence_fields=[confidence_resistivity, confidence_em, confidence_acoustic],
        swi=swi,
        sor=sor,
    )
    sw_signal_fused.name = "sw_signal_fused"

    return {
        "sw_true": sw_true,
        "sw_resistivity": sw_resistivity,
        "sw_em": sw_em,
        "sw_acoustic": sw_acoustic,
        "sw_signal_fused": sw_signal_fused,
        "confidence_resistivity": confidence_resistivity,
        "confidence_em": confidence_em,
        "confidence_acoustic": confidence_acoustic,
        "signal_fusion_report": signal_fusion_report,
    }


def run_demo(case_id: str = "multisignal_demo", results_root: str | Path | None = None) -> dict[str, object]:
    """Run the multisignal inversion demo and save outputs."""
    root = PROJECT_ROOT / "results" if results_root is None else Path(results_root)
    manager = ResultManager(root)
    case_dir = manager.create_case_dir(case_id)

    grid = Grid3D(nx=6, ny=5, nz=3, dx=1.0, dy=1.0, dz=1.0)
    indices = np.indices(grid.shape)
    sw_true_values = np.clip(0.25 + 0.35 * indices[2] / (grid.nx - 1), 0.2, 0.8)
    outputs = build_multisignal_inversion(grid, sw_true_values)

    for name in [
        "sw_true",
        "sw_resistivity",
        "sw_em",
        "sw_acoustic",
        "sw_signal_fused",
        "confidence_resistivity",
        "confidence_em",
        "confidence_acoustic",
    ]:
        manager.save_field(name, outputs[name])
    manager.save_json("signal_fusion_report", outputs["signal_fusion_report"])

    sw_true = outputs["sw_true"].values
    summary = {
        "case_id": case_id,
        "grid_shape": list(grid.shape),
        "signal_sources": ["resistivity", "electromagnetic", "acoustic"],
        "sw_true_min": float(np.min(sw_true)),
        "sw_true_max": float(np.max(sw_true)),
        "sw_resistivity_min": float(np.min(outputs["sw_resistivity"].values)),
        "sw_resistivity_max": float(np.max(outputs["sw_resistivity"].values)),
        "sw_em_min": float(np.min(outputs["sw_em"].values)),
        "sw_em_max": float(np.max(outputs["sw_em"].values)),
        "sw_acoustic_min": float(np.min(outputs["sw_acoustic"].values)),
        "sw_acoustic_max": float(np.max(outputs["sw_acoustic"].values)),
        "sw_signal_fused_min": float(np.min(outputs["sw_signal_fused"].values)),
        "sw_signal_fused_max": float(np.max(outputs["sw_signal_fused"].values)),
        "confidence_resistivity_mean": float(np.mean(outputs["confidence_resistivity"].values)),
        "confidence_em_mean": float(np.mean(outputs["confidence_em"].values)),
        "confidence_acoustic_mean": float(np.mean(outputs["confidence_acoustic"].values)),
        "fusion_nan_cells": int(outputs["signal_fusion_report"]["nan_cells_count"]),
        "fusion_clipped_cells": int(outputs["signal_fusion_report"]["clipped_cells"]),
        "rmse_resistivity": _rmse(outputs["sw_resistivity"].values, sw_true),
        "rmse_em": _rmse(outputs["sw_em"].values, sw_true),
        "rmse_acoustic": _rmse(outputs["sw_acoustic"].values, sw_true),
        "rmse_signal_fused": _rmse(outputs["sw_signal_fused"].values, sw_true),
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manager.save_json("multisignal_summary", summary)
    return {"case_id": case_id, "case_dir": case_dir, "summary": summary}


def _rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - reference) ** 2)))


def main() -> None:
    """CLI entry point."""
    result = run_demo()
    print(result["case_dir"])


if __name__ == "__main__":
    main()
