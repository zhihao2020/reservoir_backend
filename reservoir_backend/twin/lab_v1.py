"""V1 lab-case helpers: sensors, spatial holdout, synthetic truth, metrics.

Product geometry lives in ``examples/lab_v1/``. Scripts and tests reuse this
module so 30³ is the spec while CI uses ``case_dev.yaml``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ObservationSeries, Sensor
from reservoir_backend.io.case import load_case
from reservoir_backend.twin.offline import DigitalTwin

ROOT = Path(__file__).resolve().parents[2]
LAB_V1 = ROOT / "examples" / "lab_v1"

CF_TRUE_M2 = 1.0e-12
TMF_TRUE = 2.0
CF_PRIOR_FACTOR = 0.3
TMF_PRIOR_FACTOR = 0.5
NOISELESS_CF_TOL = 0.05
NOISELESS_TMF_TOL = 0.10
NOISY_CF_TOL = 0.15
NOISY_TMF_TOL = 0.25
HOLDOUT_RMSE_RATIO = 0.70
SIGMA_P = 2.0e3
SIGMA_P_FRACTURE_M1B = 30.0
SIGMA_S = 0.03
D_CF_MIN = 2.0
FAIL_RATE_MAX = 0.05
SAT_KINDS = frozenset({"saturation", "gas_saturation", "oil_saturation"})


def case_path(*, dev: bool = False) -> Path:
    name = "case_dev.yaml" if dev else "case.yaml"
    return LAB_V1 / name


def zone_of_x(x: float) -> str:
    if float(x) < 0.10:
        return "inlet"
    if float(x) < 0.20:
        return "middle"
    return "outlet"


def product_sensor_rows() -> list[dict[str, Any]]:
    """18 pressure + 75 bulk saturation sensors on the 30 cm cube."""
    rows: list[dict[str, Any]] = []
    n = 1
    for x in (0.05, 0.15, 0.25):
        for y in (0.075, 0.15, 0.225):
            for z in (0.10, 0.20):
                rows.append(
                    {
                        "sensor_id": f"P{n:03d}",
                        "kind": "pressure",
                        "x_m": x,
                        "y_m": y,
                        "z_m": z,
                        "continuum": "bulk",
                        "sigma": SIGMA_P,
                    }
                )
                n += 1
    n = 1
    for x in (0.05, 0.10, 0.15, 0.20, 0.25):
        for y in (0.05, 0.10, 0.15, 0.20, 0.25):
            for z in (0.075, 0.15, 0.225):
                rows.append(
                    {
                        "sensor_id": f"S{n:03d}",
                        "kind": "sw",
                        "x_m": x,
                        "y_m": y,
                        "z_m": z,
                        "continuum": "bulk",
                        "sigma": SIGMA_S,
                    }
                )
                n += 1
    return rows


def dev_sensor_rows(*, sigma_p_fracture: float = SIGMA_P_FRACTURE_M1B) -> list[dict[str, Any]]:
    """M1b 30 cm / 4×4×2 contract: fracture-P, matrix-P, Sg. Three x-zones.

    ``sigma_p_fracture=80`` is algorithmic identifiability (M1b), not the
    instrument contract. M1c must call this with ``SIGMA_P`` (2 kPa).
    """
    y0, z0 = 0.15, 0.15
    rows: list[dict[str, Any]] = []
    for tag, x in (("in", 0.05), ("mid", 0.15), ("out", 0.25)):
        rows.append(
            {
                "sensor_id": f"P_f_{tag}",
                "kind": "pressure",
                "x_m": x,
                "y_m": y0,
                "z_m": z0,
                "continuum": "fracture",
                "sigma": float(sigma_p_fracture),
            }
        )
    for tag, x in (("in", 0.05), ("mid", 0.15), ("out", 0.25)):
        rows.append(
            {
                "sensor_id": f"P_m_{tag}",
                "kind": "pressure",
                "x_m": x,
                "y_m": y0,
                "z_m": z0,
                "continuum": "matrix",
                "sigma": SIGMA_P,
            }
        )
    rows.append({"sensor_id": "S_f_mid", "kind": "sg", "x_m": 0.15, "y_m": y0, "z_m": z0, "continuum": "fracture", "sigma": SIGMA_S})
    rows.append({"sensor_id": "S_m_mid", "kind": "sg", "x_m": 0.15, "y_m": y0, "z_m": z0, "continuum": "matrix", "sigma": SIGMA_S})
    for tag, x in (("in", 0.05), ("mid", 0.15), ("out", 0.25)):
        rows.append(
            {
                "sensor_id": f"S_bulk_{tag}",
                "kind": "sg",
                "x_m": x,
                "y_m": y0,
                "z_m": z0,
                "continuum": "bulk",
                "sigma": SIGMA_S,
            }
        )
    return rows


def m1c_sensor_rows() -> list[dict[str, Any]]:
    """Instrument-R layout: 9 fracture-P at 2 kPa plus matrix-P and Sg."""
    z0 = 0.15
    rows: list[dict[str, Any]] = []
    for tag, x in (("in", 0.05), ("mid", 0.15), ("out", 0.25)):
        for i, y in enumerate((0.075, 0.15, 0.225)):
            rows.append(
                {
                    "sensor_id": f"P_f_{tag}{i}",
                    "kind": "pressure",
                    "x_m": x,
                    "y_m": y,
                    "z_m": z0,
                    "continuum": "fracture",
                    "sigma": SIGMA_P,
                }
            )
        rows.append(
            {
                "sensor_id": f"P_m_{tag}",
                "kind": "pressure",
                "x_m": x,
                "y_m": 0.15,
                "z_m": z0,
                "continuum": "matrix",
                "sigma": SIGMA_P,
            }
        )
    rows.append({"sensor_id": "S_f_mid", "kind": "sg", "x_m": 0.15, "y_m": 0.15, "z_m": z0, "continuum": "fracture", "sigma": SIGMA_S})
    rows.append({"sensor_id": "S_m_mid", "kind": "sg", "x_m": 0.15, "y_m": 0.15, "z_m": z0, "continuum": "matrix", "sigma": SIGMA_S})
    for tag, x in (("in", 0.05), ("mid", 0.15), ("out", 0.25)):
        rows.append(
            {
                "sensor_id": f"S_bulk_{tag}",
                "kind": "sg",
                "x_m": x,
                "y_m": 0.15,
                "z_m": z0,
                "continuum": "bulk",
                "sigma": SIGMA_S,
            }
        )
    return rows


def sensors_from_rows(rows: Iterable[dict[str, Any]]) -> list[Sensor]:
    out: list[Sensor] = []
    for r in rows:
        out.append(
            Sensor(
                name=str(r["sensor_id"]),
                kind=str(r["kind"]),
                x=float(r["x_m"]),
                y=float(r["y_m"]),
                z=float(r["z_m"]),
                sigma=float(r["sigma"]),
                medium=str(r["continuum"]),
            )
        )
    return out


def write_sensors_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sensor_id", "kind", "x_m", "y_m", "z_m", "continuum", "sigma"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


def write_controls_csv(path: Path, *, t_end: float, q_inj: float, p_prod: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "port", "kind", "value"])
        for t in (0.0, float(t_end)):
            w.writerow([t, "INJ", "rate", q_inj])
            w.writerow([t, "INJ", "composition", 0.95])
            w.writerow([t, "PROD", "pressure", p_prod])


def spatial_holdout(sensors: list[Sensor], *, frac: float = 0.20, seed: int = 3) -> set[str]:
    """Keep ~``frac`` of each inlet/middle/outlet group as hold-out."""
    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[str]] = {"inlet": [], "middle": [], "outlet": []}
    for s in sensors:
        buckets[zone_of_x(s.x)].append(s.name)
    held: set[str] = set()
    for names in buckets.values():
        if not names:
            continue
        n_hold = max(1, int(round(frac * len(names))))
        n_hold = min(n_hold, max(len(names) - 1, 0)) if len(names) > 1 else 0
        if n_hold <= 0:
            continue
        pick = rng.choice(np.asarray(names), size=n_hold, replace=False)
        held.update(str(x) for x in np.asarray(pick).ravel())
    return held


def set_inject_rate(twin: DigitalTwin, q_inj: float) -> None:
    for c in twin.experiment.controls:
        if c.port_name == "INJ" and c.kind == "rate":
            c.values[:] = float(q_inj)


def history_report_times(t_end: float, n_times: int = 5) -> NDArray[np.float64]:
    """Early sample plus ``n_times`` points in (0, t_end]. First obs at t=3 s buries C_f."""
    t_end = float(t_end)
    n_times = max(int(n_times), 1)
    early = min(0.5, 0.08 * t_end)
    rest = np.linspace(0.0, t_end, n_times + 1)[1:]
    return np.unique(np.concatenate((np.array([early], dtype=float), rest)))


def _sample_sensor(twin: DigitalTwin, sensor: Sensor, traj, t: float) -> float:
    st = traj.state_at(t)
    rates, bhp = traj.rates_and_bhp_at(t)
    return float(twin.operator.sample(sensor, st, port_rates=rates, port_bhp=bhp))


def generate_truth(
    twin: DigitalTwin,
    *,
    cf_true: float = CF_TRUE_M2,
    tmf_true: float = TMF_TRUE,
    noise: bool = False,
    case: str = "B",
    seed: int = 3,
    outlier_frac: float = 0.05,
    n_times: int = 5,
) -> dict[str, Any]:
    """Run F(θ_true), sample H, optionally add noise / outliers. Cases A/B/C."""
    case = str(case).strip().upper()
    if case not in {"A", "B", "C"}:
        raise ValueError("case must be A (P), B (P+S), or C (P+S+outliers)")
    n_th = int(twin.parameterization.n_params)
    if n_th >= 2:
        theta_true = twin.parameterization.encode(np.array([float(cf_true), float(tmf_true)], dtype=float))
    else:
        theta_true = twin.parameterization.encode(np.array([float(cf_true)], dtype=float))
    t_end = float(twin.experiment.history_end_s or 6.0)
    times = history_report_times(t_end, n_times=int(n_times))
    traj = twin.simulate(parameters=theta_true, t_end=t_end, report_times=times)
    last = traj.states[-1]
    held = spatial_holdout(list(twin.experiment.sensors), seed=seed)
    rng = np.random.default_rng(int(seed))
    observations: list[ObservationSeries] = []
    for sensor in twin.experiment.sensors:
        if case == "A" and sensor.kind != "pressure":
            continue
        t_use = np.asarray(times, dtype=float)
        vals = np.array([_sample_sensor(twin, sensor, traj, float(t)) for t in t_use], dtype=float)
        sig = np.full(t_use.size, float(sensor.sigma), dtype=float)
        noise_vec = np.zeros_like(vals)
        if noise:
            noise_vec = rng.normal(0.0, sig)
        if case == "C":
            n_out = max(1, int(round(outlier_frac * vals.size)))
            idx = rng.choice(vals.size, size=min(n_out, vals.size), replace=False)
            noise_vec[idx] += 12.0 * sig[idx]
            if t_use.size > 1:
                drop = int(rng.integers(0, t_use.size))
                keep = np.ones(t_use.size, dtype=bool)
                keep[drop] = False
                t_use = t_use[keep]
                vals = vals[keep]
                sig = sig[keep]
                noise_vec = noise_vec[keep]
        observations.append(
            ObservationSeries(
                sensor_name=sensor.name,
                kind=sensor.kind,
                times_s=t_use,
                values=vals + noise_vec,
                sigma=sig,
                holdout=sensor.name in held,
            )
        )
    twin.experiment.observations = observations
    twin.experiment.history_end_s = t_end
    return {
        "cf_true": float(cf_true),
        "tmf_true": float(tmf_true),
        "theta_true": theta_true.tolist(),
        "case": case,
        "noise": bool(noise),
        "holdout_sensors": sorted(held),
        "n_sensors": len(twin.experiment.sensors),
        "n_obs_channels": len(observations),
        "times_s": np.asarray(times, dtype=float).tolist(),
        "pressure": np.asarray(last.pressure, dtype=float),
        "sw": np.asarray(last.sw, dtype=float),
        "sg": None if last.sg is None else np.asarray(last.sg, dtype=float),
        "history": traj,
    }


def write_observations_csv(path: Path, series: list[ObservationSeries]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "sensor", "kind", "value", "sigma", "holdout"])
        for obs in series:
            hold = 1 if obs.holdout else 0
            for t, v, s in zip(obs.times_s, obs.values, np.broadcast_to(obs.sigma, obs.times_s.shape)):
                if not np.isfinite(v):
                    continue
                w.writerow([float(t), obs.sensor_name, obs.kind, float(v), float(s), hold])


def write_truth_bundle(folder: Path, twin: DigitalTwin, truth: dict[str, Any]) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in truth.items() if k not in {"pressure", "sw", "sg", "history"}}
    (folder / "truth_parameters.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez(folder / "truth_pressure.npz", pressure=truth["pressure"])
    sat = {"sw": truth["sw"]}
    if truth.get("sg") is not None:
        sat["sg"] = truth["sg"]
    np.savez(folder / "truth_saturation.npz", **sat)
    write_observations_csv(folder / "observations.csv", list(twin.experiment.observations))
    write_controls_csv(
        folder / "controls.csv",
        t_end=float(twin.experiment.history_end_s or 6.0),
        q_inj=2.0e-4,
        p_prod=1.18e7,
    )


def cf_from_theta(twin: DigitalTwin, theta: NDArray[np.float64]) -> float:
    return float(physical_from_theta(twin, theta)["cf_m2"])


def physical_from_theta(twin: DigitalTwin, theta: NDArray[np.float64]) -> dict[str, float]:
    from reservoir_backend.twin.offline import physical_from_theta as _phys

    return _phys(twin.parameterization, theta)


def offline_gates(report: dict[str, Any]) -> dict[str, Any]:
    """Noiseless: Cf <5%, Tmf <10%, holdout_ratio <1. Noisy: 15%/25%/0.7.

    Optional M1 stability / detectability: ``fail_rate`` < 5% and ``d_cf`` ≥ 2.
    """
    cf_true = float(report["cf_true"])
    cf_p50 = float(report["cf_p50"])
    cf_rel = abs(cf_p50 - cf_true) / max(abs(cf_true), 1.0e-30)
    tmf_true = float(report.get("tmf_true", 1.0))
    tmf_p50 = float(report.get("tmf_p50", tmf_true))
    tmf_rel = abs(tmf_p50 - tmf_true) / max(abs(tmf_true), 1.0e-30)
    noisy = bool(report.get("noise", False))
    ratio = float(report.get("holdout_rmse_ratio", 1.0))
    if not noisy:
        cf_ok = cf_rel < NOISELESS_CF_TOL
        tmf_ok = tmf_rel < NOISELESS_TMF_TOL
        rmse_ok = ratio < 1.0
    else:
        cf_ok = cf_rel < NOISY_CF_TOL
        tmf_ok = tmf_rel < NOISY_TMF_TOL
        rmse_ok = ratio < HOLDOUT_RMSE_RATIO
    fail_ok = True
    if "fail_rate" in report:
        fail_ok = float(report["fail_rate"]) < FAIL_RATE_MAX and not bool(report.get("repeated_fail", False))
    detect_ok = True
    if "d_cf" in report and report["d_cf"] is not None:
        detect_ok = float(report["d_cf"]) >= D_CF_MIN
    return {
        "cf_rel_error": cf_rel,
        "tmf_rel_error": tmf_rel,
        "cf_ok": bool(cf_ok),
        "tmf_ok": bool(tmf_ok),
        "holdout_rmse_ratio": ratio,
        "rmse_ok": bool(rmse_ok),
        "fail_ok": bool(fail_ok),
        "detect_ok": bool(detect_ok),
        "pass": bool(cf_ok and tmf_ok and rmse_ok and fail_ok and detect_ok),
    }


def cf_detectability(
    twin: DigitalTwin,
    theta: NDArray[np.float64],
    series: list[ObservationSeries] | None = None,
    *,
    rel: float = 0.05,
) -> float:
    """Whitened SNR of a ``rel`` perturbation in C_f: sqrt(dy^T R^{-1} dy)."""
    from reservoir_backend.twin.offline import stack_observations

    series = series if series is not None else [o for o in twin.experiment.observations if not o.holdout]
    if not series:
        series = list(twin.experiment.observations)
    if not series:
        return 0.0
    d = stack_observations(series)
    t_end = float(twin.experiment.history_end_s or float(np.max(d.times)))
    theta0 = np.asarray(theta, dtype=float).ravel()
    phys = physical_from_theta(twin, theta0)
    cf = float(phys["cf_m2"])
    n_th = int(twin.parameterization.n_params)
    if n_th >= 2:
        tmf = float(phys["tmf_multiplier"])
        theta_hi = twin.parameterization.encode(np.array([cf * (1.0 + float(rel)), tmf], dtype=float))
    else:
        theta_hi = twin.parameterization.encode(np.array([cf * (1.0 + float(rel))], dtype=float))
    y0 = np.asarray(twin._forward_vector(theta0, series, t_end=t_end), dtype=float)
    y1 = np.asarray(twin._forward_vector(theta_hi, series, t_end=t_end), dtype=float)
    dy = (y1 - y0) / np.maximum(d.sigma, 1.0e-12)
    return float(np.sqrt(np.sum(dy * dy)))


def ensemble_pressure_std(ensemble) -> NDArray[np.float64] | None:
    """Fracture-pressure std across member DualState, or None if not available."""
    duals = getattr(ensemble, "dual_states", None) if ensemble is not None else None
    if not duals:
        return None
    rows = []
    for d in duals:
        if d is None:
            continue
        frac = getattr(d, "fracture", None)
        p = getattr(frac, "pressure", None) if frac is not None else getattr(d, "pressure", None)
        if p is not None:
            rows.append(np.asarray(p, dtype=float).ravel())
    if len(rows) < 2:
        return None
    return np.std(np.stack(rows, axis=0), axis=0, ddof=1)


def sensor_information(
    twin: DigitalTwin,
    *,
    cf_ref: float,
    factors: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> list[dict[str, Any]]:
    """Finite-difference I_j = (dy_j/dm)^2 / sigma_j^2 on log C_f."""
    t_end = float(twin.experiment.history_end_s or 6.0)
    times = twin.experiment.all_times_s()
    if times.size == 0:
        times = np.array([t_end], dtype=float)
    ys: list[NDArray[np.float64]] = []
    logs = []
    for fac in factors:
        cf = float(cf_ref) * float(fac)
        theta = twin.parameterization.encode(np.array([cf], dtype=float))
        logs.append(float(theta[0]))
        traj = twin.simulate(parameters=theta, t_end=t_end, report_times=times)
        cols = []
        for sensor in twin.experiment.sensors:
            cols.append(np.array([_sample_sensor(twin, sensor, traj, float(t)) for t in times], dtype=float).mean())
        ys.append(np.asarray(cols, dtype=float))
    y = np.stack(ys, axis=0)
    m = np.asarray(logs, dtype=float)
    dydm = np.zeros(y.shape[1], dtype=float)
    if m.size >= 2:
        dydm = (y[-1] - y[0]) / max(float(m[-1] - m[0]), 1.0e-12)
    rows = []
    for j, sensor in enumerate(twin.experiment.sensors):
        sig = max(float(sensor.sigma), 1.0e-30)
        info = float((dydm[j] / sig) ** 2)
        rows.append(
            {
                "sensor_id": sensor.name,
                "kind": sensor.kind,
                "zone": zone_of_x(sensor.x),
                "pressure_sensitivity": float(dydm[j]) if sensor.kind == "pressure" else 0.0,
                "saturation_sensitivity": float(dydm[j]) if sensor.kind != "pressure" else 0.0,
                "information": info,
            }
        )
    rows.sort(key=lambda r: -float(r["information"]))
    n = len(rows)
    for i, row in enumerate(rows):
        if i < max(1, n // 3):
            row["rank_band"] = "very_informative"
        elif i < max(2, 2 * n // 3):
            row["rank_band"] = "moderately_informative"
        else:
            row["rank_band"] = "almost_insensitive"
    return rows


def load_lab_v1(*, dev: bool = False) -> DigitalTwin:
    return load_case(case_path(dev=dev))
