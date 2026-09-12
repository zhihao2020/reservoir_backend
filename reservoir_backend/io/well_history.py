"""Well-history ruler (nRMSE) plus PLACEHOLDER-friendly EXAMPLE CSV export.

nRMSE scores rates/BHP vs a CSV. Not field L2 / cell-K / Dice.

The EXAMPLE export schema keeps GEM rate/BHP columns empty until a real
``.gem`` arrives. Does not touch solver residual formulas. Does not invent
Jiyang Tc/Pc.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from reservoir_backend.domain.types import ObservationSeries
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.solver.trajectory import Trajectory

WELL_KINDS = frozenset({"bhp", "q_oil", "q_gas", "q_water", "q_inj"})

SECONDS_PER_DAY = 86400.0

# Ours columns plus empty GEM placeholders for gate ③ nRMSE once a real card exists.
WELL_HISTORY_COLUMNS = (
    "time_s",
    "time_day",
    "well",
    "role",
    "control",
    "q_oil_m3_s",
    "q_gas_m3_s",
    "q_water_m3_s",
    "q_inj_m3_s",
    "q_mol_s",
    "bhp_pa",
    "gem_q_oil_m3_s",
    "gem_q_gas_m3_s",
    "gem_q_water_m3_s",
    "gem_bhp_pa",
)


def well_history_series(observations: list[ObservationSeries]) -> list[ObservationSeries]:
    return [o for o in observations if str(o.kind) in WELL_KINDS]


def predict_well_value(traj: Trajectory, t: float, well: str, kind: str) -> float | None:
    """Read F well report at or before ``t``. ``None`` if that key was not exported."""
    rates, bhp = traj.rates_and_bhp_at(float(t))
    if kind == "bhp":
        if well not in bhp:
            return None
        return float(bhp[well])
    key = str(well) + ":" + str(kind)
    if key in rates:
        return float(rates[key])
    return None


def score_well_history(
    observations: list[ObservationSeries],
    traj: Trajectory,
    *,
    holdout: bool = False,
) -> dict[str, Any] | None:
    """nRMSE and MAE keyed by well×kind. Default uses holdout=0 rows only."""
    series = well_history_series(observations)
    chosen = [o for o in series if bool(o.holdout) == bool(holdout)]
    if not chosen:
        return None
    nrmse: dict[str, float] = {}
    mae: dict[str, float] = {}
    n_samples: dict[str, int] = {}
    missing: list[str] = []
    all_resid: list[float] = []
    for obs in chosen:
        well = str(obs.sensor_name)
        kind = str(obs.kind)
        key = well + ":" + kind
        pred: list[float] = []
        obs_v: list[float] = []
        sig_v: list[float] = []
        for t, val, sig in zip(obs.times_s, obs.values, obs.sigma):
            yhat = predict_well_value(traj, float(t), well, kind)
            if yhat is None:
                continue
            pred.append(float(yhat))
            obs_v.append(float(val))
            sig_v.append(float(sig))
        if not pred:
            missing.append(key)
            continue
        err = np.asarray(pred, dtype=float) - np.asarray(obs_v, dtype=float)
        sigma = np.asarray(sig_v, dtype=float)
        nrmse[key] = float(np.sqrt(np.mean((err / sigma) ** 2)))
        mae[key] = float(np.mean(np.abs(err)))
        n_samples[key] = int(err.size)
        all_resid.extend((err / sigma).tolist())
    overall = float(np.sqrt(np.mean(np.square(all_resid)))) if all_resid else None
    return {
        "compare": "well rates/BHP only; not field L2 / cell-K / Dice",
        "holdout": bool(holdout),
        "nrmse": nrmse,
        "mae": mae,
        "n_samples": n_samples,
        "overall_nrmse": overall,
        "missing": missing,
    }


def write_well_history_csv(
    path: str | Path,
    traj: Trajectory,
    ports: list[FlowPort] | None = None,
) -> Path:
    """Write rates + BHP per well at each stored report time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str]
    meta: dict[str, tuple[str, str]] = {}
    if ports:
        names = [p.name for p in ports]
        meta = {p.name: (p.role, p.control) for p in ports}
    else:
        seen: list[str] = []
        for rec in traj.port_bhp or traj.port_rates:
            for key in rec:
                if ":" in key:
                    continue
                if key not in seen:
                    seen.append(key)
        names = seen
    times = traj.times_s
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(WELL_HISTORY_COLUMNS))
        writer.writeheader()
        for i, t in enumerate(times):
            rates = traj.port_rates[i] if i < len(traj.port_rates) else {}
            bhp = traj.port_bhp[i] if i < len(traj.port_bhp) else {}
            for name in names:
                role, control = meta.get(name, ("", ""))
                writer.writerow(
                    {
                        "time_s": f"{float(t):.9g}",
                        "time_day": f"{float(t) / SECONDS_PER_DAY:.9g}",
                        "well": name,
                        "role": role,
                        "control": control,
                        "q_oil_m3_s": f"{float(rates.get(name + ':q_oil', 0.0)):.9g}",
                        "q_gas_m3_s": f"{float(rates.get(name + ':q_gas', 0.0)):.9g}",
                        "q_water_m3_s": f"{float(rates.get(name + ':q_water', 0.0)):.9g}",
                        "q_inj_m3_s": f"{float(rates.get(name + ':q_inj', 0.0)):.9g}",
                        "q_mol_s": f"{float(rates.get(name, 0.0)):.9g}",
                        "bhp_pa": f"{float(bhp.get(name, 0.0)):.9g}",
                        "gem_q_oil_m3_s": "",
                        "gem_q_gas_m3_s": "",
                        "gem_q_water_m3_s": "",
                        "gem_bhp_pa": "",
                    }
                )
    return path


def read_well_history_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def export_case_well_history(
    case: str | Path,
    output: str | Path,
    *,
    t_end: float | None = None,
    dt_init: float | None = None,
    dt_max: float | None = None,
) -> Path:
    """Load an EXAMPLE case, run the existing forward, emit well-history CSV.

    Thin helper so ``simulate --output`` or ``run <case>`` can emit the
    same schema without touching ``fi_comp`` residuals. ``t_end`` / ``dt_*``
    override the case schedule for short scaffolding tests.
    """
    from dataclasses import replace

    from reservoir_backend.io.case import load_case

    twin = load_case(case)
    if twin.physics.fluid is None:
        raise ValueError("case has no compositional fluid")
    if dt_init is not None or dt_max is not None:
        physics = twin.physics
        twin.physics = replace(
            physics,
            dt_init=float(physics.dt_init if dt_init is None else dt_init),
            dt_max=float(physics.dt_max if dt_max is None else dt_max),
        )
    raw = getattr(twin.parameterization, "prior_mean", twin.inverse.prior_mean)
    theta = np.asarray(raw, dtype=float).ravel()
    n = int(twin.parameterization.n_params)
    if theta.size != n:
        filled = np.zeros(n, dtype=float)
        if theta.size == 1:
            filled[:] = float(theta[0])
        else:
            filled[: min(n, theta.size)] = theta[: min(n, theta.size)]
        theta = filled
    if twin.uses_dpdp():
        traj = twin.simulate(parameters=theta, t_end=t_end)
    else:
        rock = twin.rock_from_theta(theta)
        traj = twin.simulate(rock, t_end=t_end)
    return write_well_history_csv(output, traj, twin.ports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reservoir_backend.io.well_history",
        description="Emit PLACEHOLDER-friendly well-history CSV from an EXAMPLE case",
    )
    parser.add_argument("case", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--t-end", type=float, default=None, help="override schedule end (s)")
    args = parser.parse_args(argv)
    export_case_well_history(args.case, args.output, t_end=args.t_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
