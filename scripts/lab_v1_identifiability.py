"""Joint identifiability and C_f detectability on tiny (M1a) or case_dev (M1b)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.synthetic import make_lab_v1_face_twin
from reservoir_backend.twin.lab_v1 import (
    D_CF_MIN,
    TMF_TRUE,
    cf_detectability,
    generate_truth,
    load_lab_v1,
    m1c_sensor_rows,
    sensors_from_rows,
    set_inject_rate,
    zone_of_x,
)
from reservoir_backend.twin.offline import stack_observations


def _y(twin, theta, series, t_end):
    return np.asarray(twin._forward_vector(theta, series, t_end=t_end), dtype=float)


def run_identifiability(
    *,
    tiny: bool = True,
    h: float = 0.05,
    sigma_p_fracture: float | None = None,
    q_inj: float | None = None,
    n_times: int | None = None,
    m1c: bool = False,
) -> dict:
    if tiny:
        syn = make_lab_v1_face_twin(ensemble_size=4, assimilation_steps=2)
        twin = syn.twin
        theta0 = np.asarray(syn.theta_true, dtype=float)
    else:
        twin = load_lab_v1(dev=True)
        if sigma_p_fracture is not None:
            from dataclasses import replace

            twin.experiment.sensors = [
                replace(sen, sigma=float(sigma_p_fracture))
                if sen.kind == "pressure" and sen.medium == "fracture"
                else sen
                for sen in twin.experiment.sensors
            ]
        if m1c:
            twin.experiment.sensors = sensors_from_rows(m1c_sensor_rows())
        if q_inj is not None:
            set_inject_rate(twin, float(q_inj))
        generate_truth(twin, tmf_true=TMF_TRUE, case="B", n_times=int(n_times or (20 if m1c else 5)))
        theta0 = np.asarray(twin.parameterization.encode(np.array([1.0e-12, TMF_TRUE])), dtype=float)
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
    sv = np.linalg.svd(s, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1.0e-30)) if sv.size else float("inf")
    corr = float("nan")
    if n_th >= 2 and np.std(s[:, 0]) > 0 and np.std(s[:, 1]) > 0:
        corr = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    names = list(d.names)
    kinds = list(d.kinds)
    smap = {sen.name: sen for sen in twin.experiment.sensors}
    press_mask = np.array([k == "pressure" for k in kinds])
    sat_mask = ~press_mask
    rows = []
    for i, name in enumerate(names):
        sen = smap.get(name)
        rows.append(
            {
                "sensor": name,
                "kind": kinds[i],
                "medium": str(getattr(sen, "medium", "")) if sen is not None else "",
                "zone": zone_of_x(sen.x) if sen is not None else "",
                "dlogCf": float(s[i, 0]) if n_th > 0 else 0.0,
                "dlogTmf": float(s[i, 1]) if n_th > 1 else 0.0,
            }
        )
    d_cf = cf_detectability(twin, theta0, series)
    summary = {
        "tiny": bool(tiny),
        "singular_values": sv.tolist(),
        "condition_number": cond,
        "column_correlation": corr,
        "pressure_information": float(np.sum(s[press_mask] ** 2)) if np.any(press_mask) else 0.0,
        "saturation_information": float(np.sum(s[sat_mask] ** 2)) if np.any(sat_mask) else 0.0,
        "n_obs": int(s.shape[0]),
        "d_cf": d_cf,
        "d_cf_ok": bool(d_cf >= D_CF_MIN),
        "joint_ok": bool(np.isfinite(corr) and abs(corr) < 0.9 and sv.size > 1 and sv[-1] > 1.0e-8),
    }
    dest = ROOT / "results" / "lab_v1" / "identifiability"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["sensor"])
        w.writeheader()
        w.writerows(rows)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["rows"] = rows
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tiny", action="store_true", help="M1a diagnostic 4×2×1 twin")
    p.add_argument("--dev", action="store_true", help="M1b 30 cm / 4×4×2 (default)")
    p.add_argument("--h", type=float, default=0.05)
    p.add_argument("--sigma-p-fracture", type=float, default=None, help="override fracture-P sigma (M1c uses 2000)")
    p.add_argument("--q-inj", type=float, default=None, help="override injector rate (M1c excitation)")
    p.add_argument("--n-times", type=int, default=None)
    p.add_argument("--m1c", action="store_true", help="instrument-R denser fracture-P layout")
    args = p.parse_args(argv)
    tiny = bool(args.tiny)
    rec = run_identifiability(
        tiny=tiny,
        h=float(args.h),
        sigma_p_fracture=args.sigma_p_fracture,
        q_inj=args.q_inj,
        n_times=args.n_times,
        m1c=bool(args.m1c),
    )
    print(json.dumps({k: rec[k] for k in rec if k != "rows"}, indent=2))
    return 0 if rec["joint_ok"] and rec["d_cf_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
