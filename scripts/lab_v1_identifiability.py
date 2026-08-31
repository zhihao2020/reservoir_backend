"""Joint identifiability of log Cf and log T_mf on a lab_v1 twin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.synthetic import make_lab_v1_face_twin
from reservoir_backend.twin.lab_v1 import load_lab_v1, zone_of_x
from reservoir_backend.twin.offline import stack_observations


def _y(twin, theta, series, t_end):
    return np.asarray(twin._forward_vector(theta, series, t_end=t_end), dtype=float)


def run_identifiability(*, tiny: bool = True, h: float = 0.05) -> dict:
    if tiny:
        syn = make_lab_v1_face_twin(ensemble_size=4, assimilation_steps=2)
        twin = syn.twin
        theta0 = np.asarray(syn.theta_true, dtype=float)
    else:
        twin = load_lab_v1(dev=True)
        theta0 = np.asarray(twin.parameterization.prior_mean, dtype=float).ravel()
        if theta0.size == 1:
            theta0 = np.array([float(theta0[0]), 0.0])
    series = [o for o in twin.experiment.observations if not o.holdout] or list(twin.experiment.observations)
    if not series:
        raise SystemExit("no observations; generate truth first")
    d = stack_observations(series)
    t_end = float(twin.experiment.history_end_s or d.times.max())
    n_th = int(twin.parameterization.n_params)
    s = np.zeros((d.values.size, n_th), dtype=float)
    for j in range(n_th):
        e = np.zeros(n_th)
        e[j] = float(h)
        yp = _y(twin, theta0 + e, series, t_end)
        ym = _y(twin, theta0 - e, series, t_end)
        s[:, j] = (yp - ym) / (2.0 * float(h) * np.maximum(d.sigma, 1.0e-12))
    gram = s.T @ s
    sv = np.linalg.svd(s, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1.0e-30)) if sv.size else float("inf")
    corr = float("nan")
    if n_th >= 2 and np.std(s[:, 0]) > 0 and np.std(s[:, 1]) > 0:
        corr = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    names = list(d.names)
    kinds = list(d.kinds)
    smap = {s.name: s for s in twin.experiment.sensors}
    press_mask = np.array([k == "pressure" for k in kinds])
    sat_mask = ~press_mask
    rows = []
    for i, name in enumerate(names):
        sen = smap.get(name)
        rows.append(
            {
                "sensor": name,
                "kind": kinds[i],
                "zone": zone_of_x(sen.x) if sen is not None else "",
                "dlogCf": float(s[i, 0]) if n_th > 0 else 0.0,
                "dlogTmf": float(s[i, 1]) if n_th > 1 else 0.0,
            }
        )
    summary = {
        "singular_values": sv.tolist(),
        "condition_number": cond,
        "column_correlation": corr,
        "pressure_information": float(np.sum(s[press_mask] ** 2)) if np.any(press_mask) else 0.0,
        "saturation_information": float(np.sum(s[sat_mask] ** 2)) if np.any(sat_mask) else 0.0,
        "n_obs": int(s.shape[0]),
        "joint_ok": bool(np.isfinite(corr) and abs(corr) < 0.9 and sv.size > 1 and sv[-1] > 1.0e-8),
    }
    dest = ROOT / "results" / "lab_v1" / "identifiability"
    dest.mkdir(parents=True, exist_ok=True)
    import csv

    with (dest / "sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["sensor"])
        w.writeheader()
        w.writerows(rows)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["rows"] = rows
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true", default=True)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--h", type=float, default=0.05)
    args = p.parse_args(argv)
    rec = run_identifiability(tiny=not args.dev, h=float(args.h))
    print(json.dumps({k: rec[k] for k in rec if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
