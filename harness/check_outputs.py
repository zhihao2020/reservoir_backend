"""Validation helpers for full pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_OUTPUTS = [
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
    "production_curve.csv",
    "material_balance_report.json",
    "fusion_report.json",
    "solver_report.json",
    "case_summary.json",
]


def check_required_outputs(case_dir: str | Path, required: list[str] | None = None) -> tuple[bool, list[str]]:
    """Return whether all required outputs exist and a list of missing files."""
    path = Path(case_dir)
    names = REQUIRED_OUTPUTS if required is None else required
    missing = [name for name in names if not (path / name).exists()]
    return len(missing) == 0, missing


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON from disk."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_case_summary(case_dir: str | Path) -> bool:
    """Check that case summary exists and reports success."""
    summary_path = Path(case_dir) / "case_summary.json"
    if not summary_path.exists():
        return False
    return bool(load_json(summary_path).get("success"))


def check_no_nan_inf(case_dir: str | Path) -> bool:
    """Check numerical output arrays are finite."""
    path = Path(case_dir)
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
        values = np.load(path / name)
        if np.isnan(values).any() or np.isinf(values).any():
            return False
    return True


def check_sw_ranges(case_dir: str | Path, swi: float = 0.2, sor: float = 0.2) -> bool:
    """Check saturation arrays stay within physical bounds."""
    path = Path(case_dir)
    lower, upper = float(swi), 1.0 - float(sor)
    for name in ["sw_inverted.npy", "sw_simulated.npy", "sw_fused.npy"]:
        values = np.load(path / name)
        if np.nanmin(values) < lower or np.nanmax(values) > upper:
            return False
    return True


def check_report_keys(case_dir: str | Path) -> tuple[bool, list[str]]:
    """Check material balance and fusion reports contain required keys."""
    path = Path(case_dir)
    missing: list[str] = []
    material = load_json(path / "material_balance_report.json")
    for key in ["injected_water_volume", "produced_water_volume", "storage_change", "material_balance_error"]:
        if key not in material:
            missing.append(f"material_balance_report.{key}")
    fusion = load_json(path / "fusion_report.json")
    for key in ["nan_cells_count", "clipped_cells", "fused_min", "fused_max"]:
        if key not in fusion:
            missing.append(f"fusion_report.{key}")
    return len(missing) == 0, missing
