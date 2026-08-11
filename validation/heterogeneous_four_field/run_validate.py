"""Heterogeneous four-field validation (software requirements 2–4).

Uses a **non-uniform** channel twin (never homogeneous k). Compares
reconstructed p / Sw / k against known truth from the synthetic twin forward
model. CMG heterogeneous cases live under validation/cmg_channel_3d and
cmg_fault_3d — do not use homogeneous SPE-like layers for acceptance.

Usage:
  python validation/heterogeneous_four_field/run_validate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    build_channel_twin,
    build_faulted_channel_twin,
    run_shape_discovery,
    run_time_series,
    mask_overlap,
)

HERE = Path(__file__).resolve().parent


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    den = float(np.linalg.norm(b.ravel())) + 1.0e-30
    return float(np.linalg.norm((a - b).ravel()) / den)


def _logk_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    lp = np.log(np.clip(pred, 1.0e-20, None))
    lt = np.log(np.clip(truth, 1.0e-20, None))
    return float(np.sqrt(np.mean((lp - lt) ** 2)))


def validate_channel() -> dict:
    twin = build_channel_twin(nx=12, ny=10, nz=4, n_times=4)
    # enforce heterogeneity
    assert float(np.std(twin.true_k)) > 0.0
    assert float(np.max(twin.true_k) / np.min(twin.true_k)) > 2.0

    history = run_time_series(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=float(np.median(twin.true_k)),
        porosity_prior=float(np.mean(twin.true_phi)),
        n_k_iterations=3,
    )
    # last time metrics vs truth fields
    last = history[-1]
    p_true = twin.pressure_series[-1]
    sw_true = twin.sw_series[-1]
    report = {
        "case": "heterogeneous_channel_twin",
        "k_contrast": float(np.max(twin.true_k) / np.min(twin.true_k)),
        "k_std": float(np.std(twin.true_k)),
        "pressure_rel_l2": _rel_l2(last.pressure, p_true),
        "sw_rel_l2": _rel_l2(last.sw, sw_true),
        "logk_rmse": _logk_rmse(last.permeability, twin.true_k),
        "k_mean_pred": float(np.mean(last.permeability)),
        "k_mean_true": float(np.mean(twin.true_k)),
        "well_pressure_match": {},
        "notes_tail": last.notes[-6:],
    }
    for name, p_obs in twin.samples[-1].well_pressure.items():
        c = twin.mesh.well_cell_id[name]
        i, j, k = twin.mesh.grid.ijk(c)
        report["well_pressure_match"][name] = {
            "obs": float(p_obs),
            "recon": float(last.pressure[k, j, i]),
            "abs_err": abs(float(last.pressure[k, j, i]) - float(p_obs)),
        }

    disc = run_shape_discovery(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=float(np.median(twin.true_k)),
        refine=True,
        refine_factor=2,
        indicator_threshold=0.30,
        n_k_iterations=2,
    )
    report["shape_dice"] = mask_overlap(disc.active_mask, twin.true_channel_mask)
    report["shape_notes"] = disc.notes
    return report


def validate_faulted() -> dict:
    twin = build_faulted_channel_twin(nx=12, ny=10, nz=4, n_times=4)
    assert twin.true_fault_mask is not None
    assert float(np.std(twin.true_k)) > 0.0
    history = run_time_series(
        twin.mesh,
        twin.samples,
        permeability_prior_m2=float(np.median(twin.true_k[twin.true_k > 1.0e-17])),
        n_k_iterations=3,
    )
    last = history[-1]
    return {
        "case": "heterogeneous_faulted_channel_twin",
        "k_contrast": float(np.max(twin.true_k) / max(np.min(twin.true_k), 1.0e-30)),
        "pressure_rel_l2": _rel_l2(last.pressure, twin.pressure_series[-1]),
        "sw_rel_l2": _rel_l2(last.sw, twin.sw_series[-1]),
        "logk_rmse": _logk_rmse(last.permeability, twin.true_k),
        "fault_active_mean_k": float(np.mean(last.permeability[twin.true_fault_mask])),
        "channel_mean_k": float(np.mean(last.permeability[twin.true_channel_mask])),
    }


def main() -> int:
    reports = [validate_channel(), validate_faulted()]
    # hard requirements for heterogeneous acceptance
    ok = True
    for r in reports:
        if r.get("k_contrast", 1.0) < 2.0:
            ok = False
            r["fail"] = "homogeneity_not_allowed"
        # well pressures must match sensors tightly
        for name, m in r.get("well_pressure_match", {}).items():
            if m["abs_err"] > 1.0:
                ok = False
                r["fail"] = f"well_pressure_mismatch_{name}"
    out = HERE / "validation_report.json"
    payload = {"ok": ok, "reports": reports}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
