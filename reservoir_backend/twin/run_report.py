"""Standard invert / forecast run reports (check.txt §57, §61)."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend import __version__
from reservoir_backend.twin.offline import DigitalTwin, Posterior, mass_report, predict_from_trajectory, stack_observations


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _case_hash(case_path: Path | None) -> str | None:
    if case_path is None or not case_path.is_file():
        return None
    digest = hashlib.sha256(case_path.read_bytes()).hexdigest()
    return digest[:16]


def _grid_summary(twin: DigitalTwin) -> dict[str, Any]:
    g = twin.grid
    return {
        "nx": g.nx,
        "ny": g.ny,
        "nz": g.nz,
        "n_cells": g.n_cells,
        "size_m": list(g.size_m()),
        "volume_m3": float(g.total_volume()),
    }


def _physics_summary(twin: DigitalTwin) -> dict[str, Any]:
    p = twin.physics
    return {
        "model": str(p.model),
        "fully_implicit": bool(p.fully_implicit),
        "implicit_transport": bool(p.implicit_transport),
        "p_init_Pa": float(p.p_init),
        "sw_init": float(p.sw_init),
        "capillary": getattr(p.capillary, "name", type(p.capillary).__name__),
    }


def _parameterization_summary(twin: DigitalTwin) -> dict[str, Any]:
    param = twin.parameterization
    out: dict[str, Any] = {
        "class": type(param).__name__,
        "n_params": int(param.n_params),
    }
    pm = getattr(param, "prior_mean", None)
    if pm is not None:
        out["prior_mean"] = np.asarray(pm, dtype=float).tolist()
    ps = getattr(param, "prior_std", None)
    if ps is not None:
        out["prior_std"] = np.asarray(ps, dtype=float).tolist()
    return out


def observation_residuals(
    twin: DigitalTwin,
    posterior: Posterior,
    *,
    series: list | None = None,
) -> list[dict[str, Any]]:
    """Whitened residuals per assimilating observation point."""
    from reservoir_backend.domain.types import ObservationSeries

    if series is None:
        series = twin.experiment.assimilate_observations()
    if not series:
        return []
    d_obs = stack_observations(series)
    pred = predict_from_trajectory(twin.operator, twin.experiment, posterior.history, series)
    rows: list[dict[str, Any]] = []
    for i in range(d_obs.values.size):
        sig = max(float(d_obs.sigma[i]), 1.0e-30)
        resid = float(pred[i] - d_obs.values[i])
        rows.append(
            {
                "time_s": float(d_obs.times[i]),
                "sensor": d_obs.names[i],
                "kind": d_obs.kinds[i],
                "observed": float(d_obs.values[i]),
                "predicted": float(pred[i]),
                "sigma": sig,
                "residual": resid,
                "whitened": resid / sig,
            }
        )
    return rows


def build_invert_report(
    twin: DigitalTwin,
    posterior: Posterior,
    *,
    case_path: Path | str | None = None,
    git_commit: str | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified invert diagnostics dict."""
    cp = Path(case_path) if case_path is not None else None
    rock = twin.rock_from_theta(posterior.theta)
    residuals = observation_residuals(twin, posterior)
    report: dict[str, Any] = {
        "run_type": "invert",
        "run_config": {
            "case_path": str(cp) if cp else None,
            "case_hash": _case_hash(cp),
            "seed": seed,
        },
        "software": {
            "package": "reservoir-backend",
            "version": __version__,
            "git_commit": git_commit if git_commit is not None else _git_commit(),
        },
        "grid": _grid_summary(twin),
        "physics": _physics_summary(twin),
        "parameterization": _parameterization_summary(twin),
        "prior": {
            "mean": np.asarray(getattr(twin.parameterization, "prior_mean", twin.inverse.prior_mean), dtype=float).tolist(),
            "std": np.asarray(getattr(twin.parameterization, "prior_std", twin.inverse.prior_std), dtype=float).tolist(),
        },
        "posterior": {
            "theta": posterior.theta.tolist(),
            "theta_std": posterior.theta_std.tolist(),
            "identifiability": posterior.identifiability.tolist(),
            "k_mean_md": float(np.mean(posterior.k)) / 9.869233e-16,
        },
        "metrics": {
            "assimilate_rmse": float(posterior.assimilate_rmse),
            "holdout_rmse": float(posterior.holdout_rmse),
            "forecast_rmse": None if posterior.forecast_rmse is None else float(posterior.forecast_rmse),
            "n_forward": int(posterior.n_forward),
            "misfit": list(posterior.misfit),
        },
        "mass_balance": mass_report(twin.grid, rock, posterior.history, pvt=twin.physics.pvt),
        "notes": list(posterior.notes),
        "observation_residuals_summary": {
            "n_points": len(residuals),
            "whitened_rmse": float(posterior.assimilate_rmse),
            "max_abs_whitened": float(max((abs(r["whitened"]) for r in residuals), default=0.0)),
        },
    }
    if posterior.ensemble is not None:
        ens = posterior.ensemble
        report["ensemble"] = {
            "ne": int(ens.theta_members.shape[0]),
            "theta_std_mean": float(np.mean(ens.theta_std)),
            "k_std_mean_md": float(np.mean(ens.k_std)) / 9.869233e-16,
            "k_std_max_md": float(np.max(ens.k_std)) / 9.869233e-16,
        }
    if extra:
        report.update(extra)
    return report


def build_forecast_report(
    twin: DigitalTwin,
    posterior: Posterior,
    *,
    forecast_rmse: float,
    case_path: Path | str | None = None,
    traj=None,
) -> dict[str, Any]:
    cp = Path(case_path) if case_path is not None else None
    rock = twin.rock_from_theta(posterior.theta)
    mb = mass_report(twin.grid, rock, traj, pvt=twin.physics.pvt) if traj is not None else {}
    return {
        "run_type": "forecast",
        "run_config": {"case_path": str(cp) if cp else None, "case_hash": _case_hash(cp)},
        "software": {"package": "reservoir-backend", "version": __version__, "git_commit": _git_commit()},
        "metrics": {
            "forecast_rmse": float(forecast_rmse),
            "assimilate_rmse": float(posterior.assimilate_rmse),
            "holdout_rmse": float(posterior.holdout_rmse),
        },
        "mass_balance": mb,
        "history_end_s": twin.experiment.history_end_s,
    }


def write_run_report(
    output_dir: Path | str,
    report: dict[str, Any],
    *,
    twin: DigitalTwin | None = None,
    posterior: Posterior | None = None,
) -> Path:
    """Write invert.json (or forecast.json) and residuals.csv when applicable."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = "forecast.json" if report.get("run_type") == "forecast" else "invert.json"
    dest = out / name
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if twin is not None and posterior is not None and report.get("run_type") == "invert":
        rows = observation_residuals(twin, posterior)
        if rows:
            csv_path = out / "residuals.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        if posterior.ensemble is not None:
            np.save(out / "k_mean.npy", posterior.ensemble.k_mean)
            np.save(out / "k_std.npy", posterior.ensemble.k_std)
            np.save(out / "k_post.npy", posterior.k)
    return dest
