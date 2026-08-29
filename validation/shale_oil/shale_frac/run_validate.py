"""Validate LM inversion on the synthetic shale fracture twin (no CMG)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.inverse.frac import decode_frac_theta, paint_fracture_strips  # noqa: E402
from reservoir_backend.synthetic import make_shale_depletion  # noqa: E402

HERE = Path(__file__).resolve().parent


def _k_ratio(k: np.ndarray, frac_mask: np.ndarray) -> float:
    mat = ~frac_mask
    return float(np.mean(k[frac_mask]) / max(float(np.mean(k[mat])), 1.0e-30))


def main() -> int:
    case = make_shale_depletion(n_times=4, max_iter=8, t_end=90.0 * 86400.0)
    twin = case.twin
    param = twin.parameterization
    n_frac_true = float(param.fixed_n_frac if param.n_params == 4 else round(case.theta_true[4]))
    phase_true = float(param.fixed_phase if param.n_params == 4 else case.theta_true[5])
    _, frac_mask, _ = paint_fracture_strips(
        twin.grid,
        param.wells,
        log_k_m=float(case.theta_true[0]),
        log_k_f=float(case.theta_true[1]),
        log_k_srv=float(case.theta_true[2]),
        x_f_m=float(np.exp(case.theta_true[3])),
        n_frac=int(round(n_frac_true)),
        frac_phase=phase_true,
    )
    post = twin.calibrate(max_iter=8)
    eng = decode_frac_theta(param, post.theta)
    port = twin.ports[0].name
    obs_p = float(case.twin.experiment.observations[0].values[-1])
    pred_p = float(
        twin.operator.sample(
            case.twin.experiment.sensors[0],
            post.history.states[-1],
        )
    )
    report = {
        "case": "shale_horizontal_frac_strips",
        "n_cells": twin.grid.n_cells,
        "n_frac_cells": int(np.sum(frac_mask)),
        "n_completions": len(twin.ports),
        "n_theta": int(param.n_params),
        "fully_implicit": bool(twin.physics.fully_implicit),
        "truth_k_contrast": float(np.max(case.k_true) / max(np.min(case.k_true), 1.0e-30)),
        "well_pressure_abs_err_Pa": abs(pred_p - obs_p),
        "k_ratio_post": _k_ratio(post.k, frac_mask),
        "inv_n_frac": eng.get("n_frac"),
        "truth_n_frac": n_frac_true,
        "assimilate_nrmse": float(post.assimilate_rmse),
        "frac_theta": True,
        "notes": post.notes,
    }
    dest = HERE / "validation_report.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(HERE / "truth_frac_mask.npy", frac_mask)
    np.save(HERE / "k_post.npy", post.k)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
