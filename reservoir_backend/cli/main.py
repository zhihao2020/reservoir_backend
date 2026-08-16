"""CLI around the laboratory digital-twin workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reservoir_backend.io.case import load_case
from reservoir_backend.physics.rock import Rock
from reservoir_backend.twin.offline import mass_report
from reservoir_backend.validation.synthetic import evaluate_synthetic, make_two_layer_waterflood


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_fields(folder: Path, fields: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in fields.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.dtype != object:
            np.save(folder / f"{name}.npy", arr)


def cmd_validate(case: Path, output: Path | None) -> int:
    twin = load_case(case)
    size = twin.grid.size_m()
    vol = twin.grid.total_volume()
    expected = size[0] * size[1] * size[2]
    report = {
        "nx": twin.grid.nx,
        "ny": twin.grid.ny,
        "nz": twin.grid.nz,
        "n_cells": twin.grid.n_cells,
        "size_m": list(size),
        "volume_m3": vol,
        "volume_error": abs(vol - expected) / max(expected, 1.0e-30),
        "n_ports": len(twin.ports),
        "n_sensors": len(twin.experiment.sensors),
        "n_theta": twin.parameterization.n_params,
        "n_ensemble": twin.inverse.n_ensemble,
        "n_workers": twin.inverse.n_workers,
        "capillary": getattr(twin.physics.capillary, "name", type(twin.physics.capillary).__name__),
        "three_phase": twin.physics.three_phase is not None,
    }
    print(json.dumps(report, indent=2))
    if output:
        _write_json(output / "validate.json", report)
    return 0


def cmd_simulate(case: Path, output: Path | None) -> int:
    twin = load_case(case)
    k0 = 1.0e-12
    rock = Rock.uniform(twin.grid.n_cells, k=k0, phi=float(getattr(twin.parameterization, "phi", 0.2)))
    traj = twin.simulate(rock)
    mb = mass_report(twin.grid, rock, traj)
    last = traj.states[-1]
    payload = {
        "times_s": traj.times_s.tolist(),
        "mass_balance": mb,
        "n_steps": len(traj.reports),
        "sw_final_mean": float(np.mean(last.sw)),
        "p_final_mean": float(np.mean(last.pressure)),
        "so_final_mean": float(np.mean(last.so())),
    }
    print(json.dumps(payload, indent=2))
    if output:
        _save_fields(
            output,
            {
                "pressure": last.pressure,
                "sw": last.sw,
                "so": last.so(),
                "sg": np.zeros_like(last.sw) if last.sg is None else last.sg,
            },
        )
        _write_json(output / "simulate.json", payload)
    return 0


def cmd_invert(
    case: Path,
    output: Path | None,
    *,
    preset: str | None = None,
    time_limit: float | None = None,
    auto: bool = False,
) -> int:
    twin = load_case(case)
    if not twin.experiment.observations:
        raise SystemExit("invert needs experiment.observations (or a synthetic case)")
    if auto:
        post = twin.calibrate_auto(time_limit_s=time_limit, search=True)
    else:
        post = twin.calibrate(preset=preset, time_limit_s=time_limit)
    t_rec = float(post.history.times_s[-1])
    fields = twin.reconstruct(post, t_rec)
    payload = {
        "assimilate_rmse": post.assimilate_rmse,
        "holdout_rmse": post.holdout_rmse,
        "theta_mean": post.esmda.theta_mean.tolist(),
        "theta_std": post.esmda.theta_std.tolist(),
        "identifiability": post.identifiability.tolist(),
        "mismatch": post.esmda.diagnostics.data_mismatch,
        "quadratic_mismatch": post.esmda.diagnostics.quadratic_mismatch,
        "ensemble_spread": post.esmda.diagnostics.ensemble_spread,
        "failed_members": post.esmda.diagnostics.failed_members,
        "notes": post.notes,
        "leaderboard": getattr(twin, "last_leaderboard", []),
        "mass_balance": mass_report(twin.grid, twin.rock_from_theta(post.esmda.theta_mean), post.history),
    }
    print(json.dumps(payload, indent=2))
    if output:
        _save_fields(output, fields)
        _write_json(output / "invert.json", payload)
    return 0


def cmd_forecast(case: Path, output: Path | None) -> int:
    twin = load_case(case)
    if not twin.experiment.observations:
        raise SystemExit("forecast needs experiment.observations")
    post = twin.calibrate()
    traj = twin.forecast(post)
    score = twin.score_forecast(traj)
    last = traj.states[-1]
    payload = {
        "forecast_rmse": score,
        "assimilate_rmse": post.assimilate_rmse,
        "holdout_rmse": post.holdout_rmse,
        "times_s": traj.times_s.tolist(),
        "mass_balance": mass_report(twin.grid, twin.rock_from_theta(post.esmda.theta_mean), traj),
    }
    print(json.dumps(payload, indent=2))
    if output:
        _save_fields(output, {"forecast_pressure": last.pressure, "forecast_sw": last.sw, "forecast_so": last.so()})
        _write_json(output / "forecast.json", payload)
    return 0


def cmd_apply(case: Path, output: Path | None, *, demo: bool = False) -> int:
    """Lab invert you can actually run. Not a CMG field matcher."""
    import yaml

    from reservoir_backend.apply import attach_two_layer_demo, plot_posterior_fields, write_observation_csv

    output = output or Path("results/apply")
    output.mkdir(parents=True, exist_ok=True)
    twin = load_case(case)
    cfg = yaml.safe_load(Path(case).read_text(encoding="utf-8")) or {}
    hold = list((cfg.get("experiment") or {}).get("holdout_sensors") or [])
    k_true = None
    if not twin.experiment.observations:
        if not demo:
            raise SystemExit(
                "no observations in the case. Put a CSV in experiment.observations, "
                "or run: reservoir apply config/lab_apply.yaml --demo --output results/lab"
            )
        k_true = attach_two_layer_demo(twin, holdout=hold)
        write_observation_csv(output / "observations.csv", twin)
    post = twin.calibrate()
    t_rec = float(post.history.times_s[-1])
    fields = twin.reconstruct(post, t_rec)
    if k_true is not None:
        fields["k_true"] = k_true
    plots = plot_posterior_fields(twin.grid, fields, output / "figures", k_true=k_true)
    payload = {
        "use": "lab 300 mm invert — posterior K and F(m_post) fields, not CMG cell maps",
        "n_cells": twin.grid.n_cells,
        "n_theta": twin.parameterization.n_params,
        "assimilate_rmse": post.assimilate_rmse,
        "holdout_rmse": post.holdout_rmse,
        "theta_mean": post.esmda.theta_mean.tolist(),
        "theta_std": post.esmda.theta_std.tolist(),
        "identifiability": post.identifiability.tolist(),
        "k_mean_md": (float(np.mean(post.esmda.k_mean)) / 9.869233e-16),
        "demo": bool(demo and k_true is not None),
        "figures": [str(p) for p in plots],
        "next": "replace observations.csv with lab gauges (same columns) and drop --demo",
    }
    print(json.dumps(payload, indent=2))
    _save_fields(output, fields)
    _write_json(output / "apply.json", payload)
    return 0


def cmd_synthetic(output: Path | None) -> int:
    case = make_two_layer_waterflood()
    post = case.twin.calibrate(n_ensemble=12, n_assimilations=3, seed=5)
    fc = case.twin.forecast(post)
    post.forecast_rmse = case.twin.score_forecast(fc)
    metrics = evaluate_synthetic(case, post)
    metrics["forecast_rmse"] = float(post.forecast_rmse)
    fields = case.twin.reconstruct(post, float(post.history.times_s[-1]), n_members=4)
    print(json.dumps(metrics, indent=2))
    if output:
        _write_json(output / "synthetic.json", metrics)
        _save_fields(output, {"k_true": case.k_true, **fields})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reservoir", description="Laboratory multiphase inverse twin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "simulate", "forecast"):
        p = sub.add_parser(name)
        p.add_argument("case", type=Path)
        p.add_argument("--output", type=Path, default=None)
    inv_p = sub.add_parser("invert")
    inv_p.add_argument("case", type=Path)
    inv_p.add_argument("--output", type=Path, default=None)
    inv_p.add_argument("--preset", choices=["fast", "balanced", "strict"], default=None)
    inv_p.add_argument("--time-limit", type=float, default=None, help="seconds; stops MDA / portfolio")
    inv_p.add_argument("--auto", action="store_true", help="try a small invert portfolio, pick by hold-out")
    ap = sub.add_parser("apply", help="lab invert: demo or observations CSV → posterior fields")
    ap.add_argument("case", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--demo", action="store_true", help="if no observations, generate lab-consistent two-layer data")
    p = sub.add_parser("synthetic")
    p.add_argument("--output", type=Path, default=None)
    hs = sub.add_parser("harness", help="multi-CMG invert suite / search / journal")
    hsub = hs.add_subparsers(dest="harness_cmd", required=True)
    hs_suite = hsub.add_parser("suite", help="run catalog cases")
    hs_suite.add_argument("--cases", default=None, help="comma ids, default ready cases")
    hs_suite.add_argument("--fast", action="store_true", help="probes on all; invert only lab_layers")
    hs_suite.add_argument("--all-invert", action="store_true", help="invert every ready case (slow)")
    hs_suite.add_argument("--no-invert", action="store_true", help="probes only")
    hs_suite.add_argument("--output", type=Path, default=None)
    hs_search = hsub.add_parser("search", help="beam search invert knobs (not K)")
    hs_search.add_argument("--case", default="lab_layers")
    hs_search.add_argument("--time-limit", type=float, default=300.0)
    hs_search.add_argument("--output", type=Path, default=None)
    hs_j = hsub.add_parser("journal", help="show breakthroughs")
    hs_j.add_argument("--case", default=None)
    hs_j.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args.case, args.output)
    if args.cmd == "simulate":
        return cmd_simulate(args.case, args.output)
    if args.cmd == "invert":
        return cmd_invert(
            args.case,
            args.output,
            preset=args.preset,
            time_limit=args.time_limit,
            auto=args.auto,
        )
    if args.cmd == "forecast":
        return cmd_forecast(args.case, args.output)
    if args.cmd == "apply":
        return cmd_apply(args.case, args.output, demo=args.demo)
    if args.cmd == "synthetic":
        return cmd_synthetic(args.output)
    if args.cmd == "harness":
        return cmd_harness(args)
    return 2


def cmd_harness(args) -> int:
    from reservoir_backend.validation.cmg_harness.journal import Journal, breakthroughs
    from reservoir_backend.validation.cmg_harness.run_one import run_suite
    from reservoir_backend.validation.cmg_harness.search import run_search

    if args.harness_cmd == "suite":
        ids = [x.strip() for x in args.cases.split(",")] if args.cases else None
        fast = bool(args.fast or not args.all_invert)
        if args.all_invert:
            fast = False
        report = run_suite(ids, invert=not args.no_invert, fast=fast, journal=Journal())
        print(json.dumps(report, indent=2, default=str))
        if args.output:
            _write_json(args.output / "harness_suite.json", report)
        return 0
    if args.harness_cmd == "search":
        report = run_search(args.case, time_limit_s=args.time_limit, journal=Journal())
        print(json.dumps(report, indent=2, default=str))
        if args.output:
            _write_json(args.output / "harness_search.json", report)
        return 0
    if args.harness_cmd == "journal":
        hits = breakthroughs(Journal(), threshold=args.threshold, case=args.case)
        print(json.dumps(hits, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
