"""CLI around the laboratory digital-twin workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reservoir_backend.cli.reporting import emit_invert_artifacts
from reservoir_backend.io.case import load_case
from reservoir_backend.io.well_history import score_well_history, write_well_history_csv
from reservoir_backend.twin.offline import Posterior, mass_report
from reservoir_backend.twin.run_report import build_forecast_report, write_run_report
from reservoir_backend.solver.trajectory import Trajectory


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
        "parameterization": type(twin.parameterization).__name__,
        "parameterization_class": type(twin.parameterization).__name__,
        "max_iter": twin.inverse.max_iter,
        "capillary": getattr(twin.physics.capillary, "name", type(twin.physics.capillary).__name__),
        "three_phase": twin.physics.three_phase is not None,
        "implicit_transport": bool(twin.physics.implicit_transport),
    }
    print(json.dumps(report, indent=2))
    if output:
        _write_json(output / "validate.json", report)
    return 0


def _prior_theta(twin) -> np.ndarray:
    theta = np.asarray(getattr(twin.parameterization, "prior_mean", twin.inverse.prior_mean), dtype=float).ravel()
    n = int(twin.parameterization.n_params)
    if theta.size == n:
        return theta
    if theta.size == 1 and n == 1:
        return theta
    out = np.zeros(n, dtype=float)
    out[: min(n, theta.size)] = theta[: min(n, theta.size)]
    return out


def cmd_simulate(case: Path, output: Path | None) -> int:
    twin = load_case(case)
    theta = _prior_theta(twin)
    if twin.uses_dpdp():
        traj = twin.simulate(parameters=theta)
        rock = twin.rock_from_theta(theta)
    else:
        rock = twin.rock_from_theta(theta)
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
        "theta": theta.tolist(),
    }
    hist = score_well_history(twin.experiment.observations, traj)
    if hist is not None:
        payload["well_history"] = hist
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
        _write_json(output / "run.json", payload)
        if hist is not None:
            _write_json(output / "well_history.json", hist)
        write_well_history_csv(output / "well_history.csv", traj, twin.ports)
    return 0


cmd_run = cmd_simulate


def cmd_invert(
    case: Path,
    output: Path | None,
    *,
    time_limit: float | None = None,
    self_check: bool = False,
    write_field: bool = False,
) -> int:
    twin = load_case(case)
    k_true = None
    if not twin.experiment.observations:
        if not self_check:
            raise SystemExit("invert needs experiment.observations (or use --self-check)")
        from reservoir_backend.twin.apply import attach_cf_demo

        k_true = attach_cf_demo(twin)
    post = twin.calibrate(time_limit_s=time_limit)
    t_rec = float(post.history.times_s[-1])
    fields = twin.reconstruct(post, t_rec)
    extra: dict = {}
    if self_check:
        if k_true is None:
            raise SystemExit("--self-check requires a case with no observations")
        from reservoir_backend.twin.apply import field_nrmse
        from reservoir_backend.twin.offline import predict_from_trajectory, stack_observations

        series = twin.experiment.observations
        stacked = stack_observations(series)
        times = np.unique(np.concatenate([o.times_s for o in series]))
        t_end = float(times[-1])
        from reservoir_backend.twin.offline import encode_physical_theta

        theta_true = encode_physical_theta(
            twin.parameterization, cf_m2=float(np.mean(k_true)), tmf_multiplier=1.0
        )
        true_hist = twin.simulate(parameters=theta_true, t_end=t_end, report_times=times)
        post_hist = twin.simulate(parameters=post.theta, t_end=t_end, report_times=times)
        d_true = predict_from_trajectory(twin.operator, twin.experiment, true_hist, series)
        d_post = predict_from_trajectory(twin.operator, twin.experiment, post_hist, series)
        extra["self_check"] = {
            "forward_match_nrmse": float(np.sqrt(np.mean(((d_post - d_true) / stacked.sigma) ** 2))),
            "posterior_logk_rmse": float(np.sqrt(np.mean((np.log(post.k) - np.log(k_true)) ** 2))),
            "k_vs_expand_max": float(np.max(np.abs(post.k - twin.parameterization.expand(post.theta)))),
            "sw_field_nrmse": field_nrmse(post_hist.states[-1].sw, true_hist.states[-1].sw),
            "p_field_nrmse": field_nrmse(post_hist.states[-1].pressure, true_hist.states[-1].pressure),
            "comparison": "F(m_post) vs F(m_true); not CMG",
        }
    if write_field:
        from reservoir_backend.twin.field import pressure_field

        pf = pressure_field(twin, posterior=post)
        extra["field_shape"] = [int(pf.pressure.shape[0]), int(pf.pressure.shape[1])]
        if output:
            pf.save(output)
    emit_invert_artifacts(
        twin, post, output, case_path=case, extra=extra, fields=fields
    )
    return 0


def cmd_reconstruct(
    case: Path,
    output: Path | None,
    *,
    series: Path | None = None,
    k_path: Path | None = None,
    report_times: Path | None = None,
    probes: Path | None = None,
) -> int:
    """Batch invert (or skip if --k) then write full-grid p(t)."""
    from reservoir_backend.twin.field import _read_probes_csv, pressure_field

    k = None if k_path is None else np.load(k_path)
    times = None
    if report_times is not None:
        suffix = report_times.suffix.lower()
        if suffix == ".npy":
            times = np.load(report_times)
        elif suffix == ".npz":
            packed = np.load(report_times)
            times = packed["times_s"] if "times_s" in packed.files else packed[packed.files[0]]
        else:
            times = np.loadtxt(report_times, dtype=float, delimiter=",")
    probe_list = None if probes is None else _read_probes_csv(probes)
    out = pressure_field(
        case,
        probes=probe_list,
        series=series,
        k=k,
        report_times=times,
        output=output,
    )
    payload = {
        "n_times": int(out.times_s.size),
        "n_cells": int(out.pressure.shape[1]),
        "shape": [int(out.pressure.shape[0]), int(out.pressure.shape[1])],
        "inverted": bool(k is None and out.posterior is not None),
        "times_s": out.times_s.tolist(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _theta_from_posterior(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float).ravel()
    payload = json.loads(path.read_text(encoding="utf-8"))
    post = payload.get("posterior") or payload
    if "theta" not in post:
        raise SystemExit(f"{path} has no posterior.theta")
    return np.asarray(post["theta"], dtype=float).ravel()


def cmd_forecast(case: Path, output: Path | None, *, posterior: Path | None = None) -> int:
    twin = load_case(case)
    if posterior is None:
        raise SystemExit(
            "forecast does not invert; pass --posterior invert.json (or theta.npy) from apply/invert"
        )
    theta = _theta_from_posterior(posterior)
    dummy = Posterior(
        theta=theta,
        k=twin.parameterization.expand(theta),
        theta_std=np.zeros_like(theta),
        assimilate_rmse=float("nan"),
        holdout_rmse=float("nan"),
        forecast_rmse=None,
        identifiability=np.zeros_like(theta),
        history=Trajectory(np.array([0.0]), [], [], []),
        notes=["forecast from frozen posterior; no invert"],
        n_forward=0,
        misfit=[],
    )
    traj = twin.forward_from_posterior(dummy)
    score = twin.score_forecast(traj)
    dummy.forecast_rmse = score
    last = traj.states[-1]
    report = build_forecast_report(twin, dummy, forecast_rmse=score, case_path=case, traj=traj)
    print(json.dumps(report, indent=2))
    if output:
        _save_fields(output, {"forecast_pressure": last.pressure, "forecast_sw": last.sw, "forecast_so": last.so()})
        write_run_report(output, report)
    return 0


def cmd_apply(case: Path, output: Path | None, *, demo: bool = False) -> int:
    """Lab invert you can actually run. Not a CMG field matcher."""
    import yaml

    from reservoir_backend.twin.apply import (
        accept_demo,
        attach_cf_demo,
        plot_posterior_fields,
        write_observation_csv,
    )

    output = output or Path("results/apply")
    output.mkdir(parents=True, exist_ok=True)
    twin = load_case(case)
    cfg = yaml.safe_load(Path(case).read_text(encoding="utf-8")) or {}
    hold = list((cfg.get("experiment") or {}).get("holdout_sensors") or [])
    k_true = None
    if not twin.experiment.observations:
        if not demo:
            raise SystemExit(
                "no observations in the case. Put a CSV in experiment.observations "
                "(see examples/lab/observations_template.csv), "
                "or run: reservoir apply examples/lab/lab_cf.yaml --demo --output results/lab "
                "(compositional DPDP: examples/lab_v1/case_dev.yaml --demo)"
            )
        k_true = attach_cf_demo(twin, holdout=hold)
        write_observation_csv(output / "observations.csv", twin)
    post = twin.calibrate()
    traj = twin.forward_from_posterior(post)
    post.forecast_rmse = twin.score_forecast(traj)
    last = traj.states[-1]
    sg = np.zeros_like(last.sw) if last.sg is None else last.sg
    fields = {
        "k": np.asarray(post.k, dtype=float),
        "pressure": np.asarray(last.pressure, dtype=float),
        "sw": np.asarray(last.sw, dtype=float),
        "so": np.asarray(1.0 - last.sw - sg, dtype=float),
        "sg": np.asarray(sg, dtype=float),
        "forecast_pressure": last.pressure,
        "forecast_sw": last.sw,
        "forecast_so": last.so(),
    }
    if k_true is not None:
        fields["k_true"] = k_true
    plots = plot_posterior_fields(twin.grid, fields, output / "figures", k_true=k_true)
    extra = {
        "use": "history invert (log Cf, log Tmf) then compositional DPDP forward; not a CMG cell map",
        "demo": bool(demo and k_true is not None),
        "n_theta": int(twin.parameterization.n_params),
        "parameterization": type(twin.parameterization).__name__,
        "probe_diameter_m": [float(s.probe_diameter_m) for s in twin.experiment.sensors],
        "forecast_rmse": None if post.forecast_rmse is None else float(post.forecast_rmse),
        "holdout_rmse": float(post.holdout_rmse),
        "figures": [str(p) for p in plots],
    }
    if k_true is not None:
        extra["acceptance"] = accept_demo(twin, post, k_true)
    emit_invert_artifacts(
        twin, post, output, case_path=case, extra=extra, fields=fields
    )
    invert_json = output / "invert.json"
    if invert_json.is_file():
        (output / "apply.json").write_text(invert_json.read_text(encoding="utf-8"), encoding="utf-8")
    if k_true is not None and not bool(extra.get("acceptance", {}).get("pass", True)):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="reservoir",
        description=(
            "Case-driven laboratory twin. "
            "Forward: run <case.yaml>. Invert history: apply."
        ),
        epilog="example: python -m reservoir_backend run examples/lab/lab_cf.yaml",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_help = {
        "run": "forward a case (grid / wells / fluid / schedule live in the YAML)",
        "simulate": "same as run",
        "validate": "load a case and print geometry / ports / parameterization",
    }
    for name in ("run", "validate", "simulate"):
        p = sub.add_parser(name, help=run_help[name])
        p.add_argument("case", type=Path, help="case YAML; carries grid, wells, fluid, schedule")
        p.add_argument("--output", type=Path, default=None, help="optional directory for JSON / fields")
    fc_p = sub.add_parser("forecast", help="freeze θ from --posterior and run compositional F")
    fc_p.add_argument("case", type=Path)
    fc_p.add_argument("--output", type=Path, default=None)
    fc_p.add_argument("--posterior", type=Path, default=None, help="invert.json or theta.npy; does not invert")
    inv_p = sub.add_parser("invert")
    inv_p.add_argument("case", type=Path)
    inv_p.add_argument("--output", type=Path, default=None)
    inv_p.add_argument("--time-limit", type=float, default=None, help="seconds; stops LM / ES-MDA")
    inv_p.add_argument("--self-check", action="store_true", help="with no observations, generate a DPDP demo and verify inversion")
    inv_p.add_argument(
        "--write-field",
        action="store_true",
        help="after invert, write full-grid p(t) at observation times (pressure.npy)",
    )
    rec_p = sub.add_parser("reconstruct", help="probes + p(t) -> invert K (or --k) -> full-grid p(t)")
    rec_p.add_argument("case", type=Path)
    rec_p.add_argument("--output", type=Path, default=None)
    rec_p.add_argument("--series", type=Path, default=None, help="observation CSV: time_s,sensor,kind,value,sigma")
    rec_p.add_argument("--probes", type=Path, default=None, help="probe CSV: name,x,y,z")
    rec_p.add_argument("--k", dest="k_path", type=Path, default=None, help="cell K .npy; skip invert")
    rec_p.add_argument("--report-times", dest="report_times", type=Path, default=None, help="times .npy/.csv")
    ap = sub.add_parser("apply", help="history invert then compositional forward of the full schedule")
    ap.add_argument("case", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--demo", action="store_true", help="if no observations, generate DPDP-consistent synthetic gauges")
    rp = sub.add_parser("replay", help="replay experiments/EXP00N controls and sensors (no UDP)")
    rp.add_argument("experiment", type=Path)
    rp.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "validate":
        return cmd_validate(args.case, args.output)
    if args.cmd in {"run", "simulate"}:
        return cmd_simulate(args.case, args.output)
    if args.cmd == "invert":
        return cmd_invert(
            args.case,
            args.output,
            time_limit=args.time_limit,
            self_check=args.self_check,
            write_field=args.write_field,
        )
    if args.cmd == "reconstruct":
        return cmd_reconstruct(
            args.case,
            args.output,
            series=args.series,
            k_path=args.k_path,
            report_times=args.report_times,
            probes=args.probes,
        )
    if args.cmd == "forecast":
        return cmd_forecast(args.case, args.output, posterior=args.posterior)
    if args.cmd == "apply":
        return cmd_apply(args.case, args.output, demo=args.demo)
    if args.cmd == "replay":
        from reservoir_backend.runtime.replay import replay_experiment

        report = replay_experiment(args.experiment, output=args.output)
        print(json.dumps(report, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
