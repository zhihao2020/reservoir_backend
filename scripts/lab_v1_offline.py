"""Offline ES-MDA on lab_v1. Default --dev; do not run 30³ ensembles in CI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.history_match import HistoryMatchWorkflow
from reservoir_backend.twin.lab_v1 import (
    CF_PRIOR_FACTOR,
    CF_TRUE_M2,
    TMF_PRIOR_FACTOR,
    TMF_TRUE,
    cf_from_theta,
    generate_truth,
    load_lab_v1,
    offline_gates,
    physical_from_theta,
)
from reservoir_backend.twin.offline import predict_from_trajectory, split_history_observations, stack_observations


def _rmse(twin, hist, series) -> float:
    if not series:
        return float("nan")
    d = stack_observations(series)
    pred = predict_from_trajectory(twin.operator, twin.experiment, hist, series)
    return float(np.sqrt(np.mean(((pred - d.values) / np.maximum(d.sigma, 1.0e-12)) ** 2)))


def run_offline(
    *,
    dev: bool = True,
    case: str = "B",
    noise: bool = False,
    cf_true: float = CF_TRUE_M2,
    ensemble_size: int | None = None,
    out: Path | None = None,
    tiny: bool = False,
) -> dict:
    if tiny:
        from reservoir_backend.synthetic import make_lab_v1_face_twin

        syn = make_lab_v1_face_twin(
            cf_true=cf_true,
            ensemble_size=int(ensemble_size or 8),
            assimilation_steps=5,
            with_saturation=str(case).upper() != "A",
            noise_p=2.0e3 if noise else 0.0,
            noise_s=0.03 if noise else 0.0,
        )
        twin = syn.twin
        truth = {
            "cf_true": float(cf_true),
            "tmf_true": float(physical_from_theta(twin, syn.theta_true)["tmf_multiplier"]),
            "theta_true": syn.theta_true.tolist(),
        }
        prior_phys = physical_from_theta(twin, np.asarray(twin.parameterization.prior_mean, dtype=float))
        prior_cf = float(prior_phys["cf_m2"])
        prior_tmf = float(prior_phys["tmf_multiplier"])
    else:
        twin = load_lab_v1(dev=dev)
        truth = generate_truth(twin, cf_true=cf_true, tmf_true=TMF_TRUE, noise=noise, case=case)
    if not tiny:
        prior_cf = float(cf_true) * float(CF_PRIOR_FACTOR)
        prior_tmf = float(TMF_TRUE) * float(TMF_PRIOR_FACTOR)
        n_th = int(twin.parameterization.n_params)
        if n_th >= 2:
            twin.parameterization.prior_mean = twin.parameterization.encode(
                np.array([prior_cf, prior_tmf], dtype=float)
            )
        else:
            twin.parameterization.prior_mean = twin.parameterization.encode(np.array([prior_cf], dtype=float))
        twin.inverse.prior_mean = np.asarray(twin.parameterization.prior_mean, dtype=float)
    if ensemble_size is not None:
        twin.inverse.ensemble_size = int(ensemble_size)
    post = HistoryMatchWorkflow().run(twin)
    phys_members = [
        physical_from_theta(twin, post.ensemble.theta_members[j]) for j in range(post.ensemble.theta_members.shape[0])
    ]
    cf_members = np.array([p["cf_m2"] for p in phys_members], dtype=float)
    tmf_members = np.array([p["tmf_multiplier"] for p in phys_members], dtype=float)
    q = np.quantile(cf_members, [0.05, 0.50, 0.95])
    q_t = np.quantile(tmf_members, [0.05, 0.50, 0.95])
    assim, hold = split_history_observations(twin.experiment.observations, twin.experiment.history_end_s)
    # Prior forward at prior mean for hold-out ratio.
    prior_theta = np.asarray(twin.parameterization.prior_mean, dtype=float).ravel()
    t_end = float(twin.experiment.history_end_s or 6.0)
    times = stack_observations(assim).times
    hist_prior = twin.simulate(parameters=prior_theta, t_end=t_end, report_times=times)
    hold_prior = _rmse(twin, hist_prior, hold)
    hold_post = float(post.holdout_rmse)
    ratio = hold_post / max(hold_prior, 1.0e-12) if np.isfinite(hold_prior) else float("nan")
    last = post.history.states[-1]
    report = {
        "gate": "lab_v1_offline",
        "dev": bool(dev),
        "case": case,
        "noise": bool(noise),
        "cf_true": float(cf_true),
        "tmf_true": float(truth.get("tmf_true", TMF_TRUE)),
        "cf_prior": prior_cf,
        "tmf_prior": prior_tmf,
        "cf_p05": float(q[0]),
        "cf_p50": float(q[1]),
        "cf_p95": float(q[2]),
        "tmf_p05": float(q_t[0]),
        "tmf_p50": float(q_t[1]),
        "tmf_p95": float(q_t[2]),
        "cf_mean": float(np.mean(cf_members)),
        "cf_std": float(np.std(cf_members, ddof=1)) if cf_members.size > 1 else 0.0,
        "ensemble_size": int(post.ensemble.theta_members.shape[0]),
        "n_forward": int(post.n_forward),
        "assimilate_rmse": float(post.assimilate_rmse),
        "holdout_rmse_prior": hold_prior,
        "holdout_rmse_posterior": hold_post,
        "holdout_rmse_ratio": ratio,
        "misfit": [float(x) for x in post.misfit],
        "notes": list(post.notes),
    }
    report["gates"] = offline_gates(report)
    dest = Path(out or (ROOT / "results" / "lab_v1" / "offline"))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "posterior.json").write_text(
        json.dumps(
            {
                "theta": post.theta.tolist(),
                "theta_members": post.ensemble.theta_members.tolist(),
                "cf_members": cf_members.tolist(),
                "tmf_members": tmf_members.tolist(),
                "cf_p50": float(q[1]),
                "tmf_p50": float(q_t[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.save(dest / "cf_ensemble.npy", cf_members)
    np.save(dest / "tmf_ensemble.npy", tmf_members)
    np.savez(dest / "pressure_mean.npz", pressure=last.pressure)
    np.savez(dest / "pressure_std.npz", pressure=np.zeros_like(last.pressure))
    np.savez(dest / "sw_mean.npz", sw=last.sw)
    np.savez(dest / "sw_std.npz", sw=np.zeros_like(last.sw))
    sg = last.sg if last.sg is not None else np.zeros_like(last.sw)
    np.savez(dest / "sg_mean.npz", sg=sg)
    np.savez(dest / "sg_std.npz", sg=np.zeros_like(sg))
    (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (dest / "residuals.csv").open("w", encoding="utf-8") as fh:
        fh.write("assimilate_rmse,holdout_rmse\n")
        fh.write(f"{post.assimilate_rmse},{post.holdout_rmse}\n")
    np.save(dest / "theta_members.npy", post.ensemble.theta_members)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", action="store_true", default=True)
    p.add_argument("--product", action="store_true", help="use 30³ case.yaml (slow)")
    p.add_argument("--case", choices=["A", "B", "C"], default="B")
    p.add_argument("--noise", action="store_true")
    p.add_argument("--cf-true", type=float, default=CF_TRUE_M2)
    p.add_argument("--ne", type=int, default=None)
    p.add_argument("--tiny", action="store_true", help="3×2×1 face-port twin (CI / recovery gate)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.product:
        args.dev = False
    report = run_offline(
        dev=bool(args.dev),
        case=str(args.case),
        noise=bool(args.noise),
        cf_true=float(args.cf_true),
        ensemble_size=args.ne,
        out=args.out,
        tiny=bool(args.tiny),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["gates"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
