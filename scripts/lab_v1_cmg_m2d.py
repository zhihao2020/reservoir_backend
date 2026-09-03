"""M2d: four GEM truths (T1–T4) plus optional noise / sparse-H variants.

Does not touch the frozen Case B export under examples/lab_v1/cmg_gem/export.
Inversion never opens hidden/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    attach_cmg_observations,
    filter_sparse_sensors,
    find_gem_exe,
    forward_at_theta,
    forward_equivalence_report,
    init_flash_report,
    invert_from_cmg_observations,
    load_alignment_spec,
    load_hidden_truth,
    parse_gem_out_maps,
    patch_gem_deck,
    perturb_observation_series,
    reconstruction_report,
    robustness_case,
    robustness_cases,
    run_gem,
    sample_observations_from_hidden,
    theta_true_from_spec,
    write_comparison_plot,
    write_grid_csv,
    write_hidden_truth,
)
from reservoir_backend.twin.lab_v1 import (
    load_lab_v1,
    physical_from_theta,
    spatial_holdout,
    write_controls_csv,
    write_observations_csv,
)

DECK_SRC = ROOT / "examples" / "lab_v1" / "cmg_gem" / "lab_v1_dev.dat"
DEFAULT_OUT = ROOT / "results" / "lab_v1" / "cmg_m2d"
FREEZE = ROOT / "examples" / "lab_v1" / "cmg_gem" / "m2d_gate.json"


def _variant_name(truth: str, *, noise: bool, sparse: bool) -> str:
    parts = [str(truth)]
    if noise:
        parts.append("noise")
    if sparse:
        parts.append("sparse")
    return "_".join(parts)


def pack_export(
    hidden: Path,
    export: Path,
    *,
    noise: bool,
    sparse: bool,
    seed: int,
) -> int:
    twin = load_lab_v1(dev=True)
    truth = load_hidden_truth(hidden)
    held = spatial_holdout(list(twin.experiment.sensors), seed=int(seed))
    series = sample_observations_from_hidden(twin, truth, holdout=held)
    if sparse:
        series = filter_sparse_sensors(series)
    if noise:
        series = perturb_observation_series(series, seed=int(seed))
    dest = Path(export)
    dest.mkdir(parents=True, exist_ok=True)
    write_observations_csv(dest / "observations.csv", series)
    write_controls_csv(
        dest / "controls.csv",
        t_end=float(np.max(truth.times_s)) if truth.times_s.size else 60.0,
        q_inj=3.0e-4,
        p_prod=1.18e7,
    )
    hidden_dest = dest / "hidden"
    hidden_dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "pressure.npy",
        "sg.npy",
        "so.npy",
        "sw.npy",
        "pressure_fracture.npy",
        "pressure_matrix.npy",
        "meta.json",
    ):
        src = Path(hidden) / name
        if src.is_file():
            shutil.copy2(src, hidden_dest / name)
    write_grid_csv(twin, hidden_dest / "grid.csv")
    return len(series)


def run_one_gem(case: dict, work: Path, timeout_s: float) -> dict:
    deck = work / "lab_v1_dev.dat"
    patch_gem_deck(
        DECK_SRC,
        deck,
        cf_md=float(case["cf_md"]),
        sigmamf=float(case["sigmamf"]),
        wi_md_m=float(case["wi_md_m"]),
    )
    rec = run_gem(deck, work, timeout_s=float(timeout_s))
    (work / "run.json").write_text(json.dumps({k: rec[k] for k in rec if k not in {"stdout_tail", "stderr_tail"}}, indent=2), encoding="utf-8")
    if not rec.get("ok"):
        return rec
    outs = rec.get("out_files") or []
    twin = load_lab_v1(dev=True)
    truth = parse_gem_out_maps(outs[0], nx=twin.grid.nx, ny=twin.grid.ny, nz=twin.grid.nz)
    hidden = work / "hidden"
    write_hidden_truth(hidden, truth)
    write_grid_csv(twin, hidden / "grid.csv")
    flash = init_flash_report(outs[0])
    rec["hidden"] = str(hidden)
    rec["init_flash"] = flash
    rec["n_times"] = int(truth.times_s.size)
    (work / "init_flash.json").write_text(json.dumps(flash, indent=2), encoding="utf-8")
    return rec


def run_m2a(export: Path, out: Path, case: dict) -> dict:
    twin = load_lab_v1(dev=True)
    spec = load_alignment_spec()
    truth = load_hidden_truth(Path(export) / "hidden")
    theta = theta_true_from_spec(twin, spec, cf_m2=float(case["cf_m2"]), tmf_multiplier=float(case["tmf"]))
    ours = forward_at_theta(twin, theta, truth.times_s)
    report = forward_equivalence_report(ours, truth)
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "forward_equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_invert(export: Path, out: Path, case: dict, *, workers: int | None) -> dict:
    twin = load_lab_v1(dev=True)
    spec = load_alignment_spec()
    if workers is not None:
        twin.inverse.n_workers = int(workers)
    post = invert_from_cmg_observations(export, twin=twin)
    phys_post = physical_from_theta(twin, np.asarray(post.theta, dtype=float).ravel())
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "m2d_sparse_observation_inversion",
        "truth": case["name"],
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
    truth = load_hidden_truth(Path(export) / "hidden")
    theta_true = theta_true_from_spec(
        twin, spec, cf_m2=float(case["cf_m2"]), tmf_multiplier=float(case["tmf"])
    )
    prior_theta = np.asarray(twin.parameterization.prior_mean, dtype=float).ravel()
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
    np.savez(
        dest / "fields.npz",
        times_s=truth.times_s,
        prior_p=prior_fields["pressure"],
        post_p=post_fields["pressure"],
        cmg_p=truth.pressure,
    )
    write_grid_csv(score_twin, dest / "grid.csv")
    write_comparison_plot(truth, prior_fields, post_fields, dest / "pressure_compare.png")
    payload["reconstruction"] = {
        "improvement_pressure": report["improvement_pressure"],
        "improvement_sg": report.get("improvement_sg"),
        "gate3": report.get("gate3"),
        "parameters": report["parameters"],
    }
    return payload


def _summarize_row(name: str, case: dict, rec: dict) -> dict:
    recon = rec.get("reconstruction") or {}
    gate3 = recon.get("gate3") or {}
    return {
        "name": name,
        "cf_true": case["cf_m2"],
        "tmf_true": case["tmf"],
        "hidden_used": rec.get("hidden_used", False),
        "cf_p50": rec.get("cf_p50"),
        "tmf_p50": rec.get("tmf_p50"),
        "cf_rel_error": (recon.get("parameters") or {}).get("cf_rel_error"),
        "tmf_rel_error": (recon.get("parameters") or {}).get("tmf_rel_error"),
        "improvement_pressure": recon.get("improvement_pressure"),
        "improvement_sg": recon.get("improvement_sg"),
        "gate3_pass": gate3.get("pass"),
        "assimilate_rmse": rec.get("assimilate_rmse"),
        "m2a_pass": rec.get("m2a_pass"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--truths", nargs="+", default=["T1", "T2", "T3", "T4"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--noise", action="store_true", help="also run T2 with N(0,σ) gauges")
    p.add_argument("--sparse", action="store_true", help="also run T2 with 5-channel H")
    p.add_argument("--skip-invert", action="store_true")
    p.add_argument("--skip-gem", action="store_true", help="reuse existing GEM hidden/")
    args = p.parse_args(argv)

    if find_gem_exe() is None and not args.skip_gem:
        print(json.dumps({"blocked": "GEM executable not found"}, indent=2), flush=True)
        return 2

    variants: list[dict] = []
    for name in args.truths:
        variants.append({"truth": name, "noise": False, "sparse": False})
    extras_on = "T2" if "T2" in args.truths else args.truths[0]
    if args.noise:
        variants.append({"truth": extras_on, "noise": True, "sparse": False})
    if args.sparse:
        variants.append({"truth": extras_on, "noise": False, "sparse": True})

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    freeze: dict = {
        "gate": "m2d_robustness",
        "variants": [],
        "n_pass_gate3": 0,
        "n_hidden_used": 0,
    }

    gem_done: dict[str, dict] = {}
    for truth_name in {v["truth"] for v in variants}:
        case = robustness_case(truth_name)
        gem_dir = root / truth_name / "gem"
        if args.skip_gem and (gem_dir / "hidden" / "pressure.npy").is_file():
            gem_done[truth_name] = {"ok": True, "hidden": str(gem_dir / "hidden")}
            print(json.dumps({"reuse_gem": truth_name}, indent=2), flush=True)
            continue
        print(json.dumps({"gem": truth_name, "cf_m2": case["cf_m2"], "tmf": case["tmf"]}, indent=2), flush=True)
        rec = run_one_gem(case, gem_dir, args.timeout)
        gem_done[truth_name] = rec
        if not rec.get("ok"):
            print(json.dumps({"gem_failed": truth_name, "rec": {k: rec.get(k) for k in ("ok", "blocked", "returncode")}}, indent=2), flush=True)
            return 1

    rows = []
    rc = 0
    for var in variants:
        case = robustness_case(var["truth"])
        name = _variant_name(var["truth"], noise=var["noise"], sparse=var["sparse"])
        dest = root / name
        hidden = Path(gem_done[var["truth"]]["hidden"])
        export = dest / "export"
        n_ch = pack_export(hidden, export, noise=var["noise"], sparse=var["sparse"], seed=int(args.seed))
        print(json.dumps({"packed": name, "n_channels": n_ch}, indent=2), flush=True)
        m2a = run_m2a(export, dest / "m2a", case)
        print(json.dumps({"m2a": name, "pass": m2a.get("pass"), "metrics": m2a.get("metrics")}, indent=2), flush=True)
        row = {
            "name": name,
            "noise": var["noise"],
            "sparse": var["sparse"],
            "m2a_pass": bool(m2a.get("pass")),
            "hidden_used": False,
        }
        if args.skip_invert:
            rows.append(row)
            continue
        inv = run_invert(export, dest / "invert", case, workers=args.workers)
        inv["m2a_pass"] = row["m2a_pass"]
        summary = _summarize_row(name, case, inv)
        summary["noise"] = var["noise"]
        summary["sparse"] = var["sparse"]
        rows.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        if summary.get("hidden_used"):
            rc = 1
        if not summary.get("gate3_pass"):
            rc = 1

    freeze["variants"] = rows
    freeze["n_pass_gate3"] = int(sum(1 for r in rows if r.get("gate3_pass")))
    freeze["n_hidden_used"] = int(sum(1 for r in rows if r.get("hidden_used")))
    scored = [r for r in rows if "gate3_pass" in r]
    freeze["pass"] = bool(scored) and freeze["n_hidden_used"] == 0 and freeze["n_pass_gate3"] == len(scored)
    (root / "summary.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    if scored:
        FREEZE.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
