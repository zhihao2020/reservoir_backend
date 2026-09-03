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
    reconstruction_report,
    theta_true_from_spec,
    write_comparison_plot,
    write_grid_csv,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1, physical_from_theta


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, default=ROOT / "examples" / "lab_v1" / "cmg_gem" / "export")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_invert")
    p.add_argument("--score", action="store_true", help="open hidden/ after invert; never during ES-MDA")
    args = p.parse_args(argv)
    export = Path(args.export)
    if not (export / "observations.csv").is_file():
        print(json.dumps({"blocked": export_blocked_reason(export)}, indent=2), flush=True)
        return 2

    twin = load_lab_v1(dev=True)
    spec = load_alignment_spec()
    # Invert path: observations only.
    post = invert_from_cmg_observations(export, twin=twin)
    phys_post = physical_from_theta(twin, np.asarray(post.theta, dtype=float).ravel())
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "m2b_sparse_observation_inversion",
        "cf_p50": phys_post["cf_m2"],
        "tmf_p50": phys_post["tmf_multiplier"],
        "holdout_rmse": post.holdout_rmse,
        "hidden_used": False,
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
    theta_true = theta_true_from_spec(twin, spec)
    prior_theta = np.asarray(twin.parameterization.prior_mean, dtype=float).ravel()
    # Fresh twin so scoring does not mutate invert observations.
    score_twin = load_lab_v1(dev=True)
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
    np.savez(dest / "fields.npz", prior_p=prior_fields["pressure"], post_p=post_fields["pressure"], cmg_p=truth.pressure)
    write_grid_csv(score_twin, dest / "grid.csv")
    write_comparison_plot(truth, prior_fields, post_fields, dest / "pressure_compare.png")
    print(json.dumps({"improvement_pressure": report["improvement_pressure"], "parameters": report["parameters"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
