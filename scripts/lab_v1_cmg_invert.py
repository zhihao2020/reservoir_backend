"""M2b/M2c: invert from GEM gauges, then optionally score hidden fields.

Invert never opens hidden/. --score is a separate step after θ̂ exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    attach_cmg_observations,
    export_blocked_reason,
    forward_at_theta,
    invert_from_cmg_observations,
    load_alignment_spec,
    load_hidden_truth,
    load_twin_case,
    reconstruction_report,
    theta_true_from_twin,
    write_comparison_plot,
    write_grid_csv,
)
from reservoir_backend.twin.lab_v1 import physical_from_theta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_invert")
    p.add_argument("--case", type=Path, default=None, help="YAML case; default M2 case_dev")
    p.add_argument("--score", action="store_true", help="open hidden/ after invert; never during ES-MDA")
    p.add_argument("--workers", type=int, default=None, help="ensemble forwards in parallel")
    p.add_argument("--cf-m2", type=float, default=None, help="scoring truth C_f; default spec theta_true")
    p.add_argument("--tmf", type=float, default=None, help="scoring truth beta_mf; default spec theta_true")
    p.add_argument("--k-m2", type=float, default=None, dest="k_m2", help="scoring truth k for log_conductivity")
    args = p.parse_args(argv)
    export = Path(args.export)
    if not (export / "observations.csv").is_file():
        print(json.dumps({"blocked": export_blocked_reason(export)}, indent=2), flush=True)
        return 2

    twin = load_twin_case(args.case)
    spec = None if args.case is not None else load_alignment_spec()
    # Invert path: observations only. Hidden 3-D is never passed in.
    if args.workers is not None:
        twin.inverse.n_workers = int(args.workers)
    post = invert_from_cmg_observations(export, twin=twin)
    phys_post = physical_from_theta(twin, np.asarray(post.theta, dtype=float).ravel())
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "m2b_sparse_observation_inversion",
        "cf_p50": phys_post["cf_m2"],
        "tmf_p50": phys_post["tmf_multiplier"],
        "holdout_rmse": post.holdout_rmse,
        "holdout_rmse_is_whitened": True,
        "assimilate_rmse": post.assimilate_rmse,
        "misfit": [float(x) for x in post.misfit],
        "hidden_used": False,
        "n_forward": post.n_forward,
    }
    (dest / "invert.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.save(dest / "theta.npy", np.asarray(post.theta, dtype=float))
    print(json.dumps(payload, indent=2), flush=True)

    if not args.score:
        return 0

    hidden = export / "hidden"
    if not (hidden / "pressure.npy").is_file():
        print(json.dumps({"blocked": "missing hidden/ for M2c scoring"}, indent=2), flush=True)
        return 2

    truth = load_hidden_truth(hidden)
    theta_true = theta_true_from_twin(
        twin, spec, cf_m2=args.cf_m2, tmf_multiplier=args.tmf, k_m2=args.k_m2
    )
    prior_theta = np.asarray(twin.parameterization.prior_mean, dtype=float).ravel()
    score_twin = load_twin_case(args.case)
    attach_cmg_observations(score_twin, export)
    prior_fields = forward_at_theta(score_twin, prior_theta, truth.times_s)
    post_fields = forward_at_theta(score_twin, np.asarray(post.theta, dtype=float).ravel(), truth.times_s)
    report = reconstruction_report(
        prior=prior_fields,
        posterior=post_fields,
        truth=truth,
        phys_prior=physical_from_theta(score_twin, prior_theta),
        phys_post=phys_post,
        phys_true=physical_from_theta(score_twin, theta_true),
        holdout_rmse=post.holdout_rmse,
    )
    (dest / "reconstruction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    packed = {
        "times_s": truth.times_s,
        "prior_p": prior_fields["pressure"],
        "post_p": post_fields["pressure"],
        "cmg_p": truth.pressure,
    }
    if "sg" in prior_fields and "sg" in post_fields and truth.sg is not None:
        packed["prior_sg"] = prior_fields["sg"]
        packed["post_sg"] = post_fields["sg"]
        packed["cmg_sg"] = truth.sg
    if "pressure_matrix" in prior_fields and "pressure_matrix" in post_fields and truth.pressure_matrix is not None:
        packed["prior_pm"] = prior_fields["pressure_matrix"]
        packed["post_pm"] = post_fields["pressure_matrix"]
        packed["cmg_pm"] = truth.pressure_matrix
    np.savez(dest / "fields.npz", **packed)
    write_grid_csv(score_twin, dest / "grid.csv")
    write_comparison_plot(truth, prior_fields, post_fields, dest / "pressure_compare.png")
    print(
        json.dumps(
            {
                "improvement_pressure": report["improvement_pressure"],
                "improvement_sg": report.get("improvement_sg"),
                "gate3": report.get("gate3"),
                "parameters": report["parameters"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
