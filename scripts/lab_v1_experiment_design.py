"""M1c experimental-design gate. Deterministic forwards only. No ES-MDA."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.experiment_design import (
    D_CF_MIN,
    D_TMF_MIN,
    SIGMA_P,
    STEADY_DP_PA,
    YAML_PATH,
    cf_detectability_bound,
    default_candidates,
    evaluate_design,
    independent_samples,
    load_experiment_design_yaml,
    select_designs,
    two_gauge_delta_sigma,
)


def _markdown_table(rows: list[dict]) -> str:
    cols = ["name", "h", "d_cf", "d_tmf", "cond", "dp_max", "n_pv", "min_dt", "solver", "feasible"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" if c in {"name", "h", "solver"} else "---:" for c in cols) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            if c in {"d_cf", "d_tmf", "cond", "dp_max", "n_pv", "min_dt"} and isinstance(v, (int, float)):
                cells.append(f"{v:.4g}")
            elif c == "feasible":
                cells.append("yes" if v else "no")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bounds(t_end_s: float, tau_s: float | None, sigma_pa: float) -> dict[str, float]:
    n = independent_samples(t_end_s, tau_s)
    return {
        "steady_d_cf_bound": cf_detectability_bound(STEADY_DP_PA, sigma_pa, n_indep=1.0),
        "timeseries_d_cf_bound": cf_detectability_bound(STEADY_DP_PA, sigma_pa, n_indep=n),
        "two_gauge_delta_sigma_pa": two_gauge_delta_sigma(sigma_pa),
        "n_indep": n,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "experiment_design")
    p.add_argument("--yaml", type=Path, default=YAML_PATH, help="instrument + envelope + staged designs")
    p.add_argument("--catalog", action="store_true", help="ignore YAML designs; use the code catalog")
    p.add_argument("--include-long", action="store_true", help="also forward evaluate=false designs")
    p.add_argument("--design", type=str, default=None, help="evaluate a single named design")
    args = p.parse_args(argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    if args.catalog:
        env, designs = None, default_candidates()
    else:
        env, designs = load_experiment_design_yaml(args.yaml)
    chosen = select_designs(designs, include_long=bool(args.include_long), name=args.design)

    rows = []
    for design in chosen:
        rec = evaluate_design(design, envelope=env)
        d = rec.as_dict()
        rows.append(d)
        print(json.dumps(d), flush=True)
        (dest / "partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "name",
        "h",
        "d_cf",
        "d_tmf",
        "cond",
        "corr",
        "lambda_min",
        "p_max",
        "dp_max",
        "n_pv",
        "t_end_s",
        "q_max_used",
        "n_obs",
        "min_dt",
        "solver",
        "feasible",
        "d_cf_ok",
        "d_tmf_ok",
        "joint_ok",
        "infeasible_reasons",
        "note",
    ]
    with (dest / "pareto.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["infeasible_reasons"] = ",".join(row.get("infeasible_reasons") or [])
            w.writerow(out)
    (dest / "pareto.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (dest / "pareto.md").write_text(_markdown_table(rows) + "\n", encoding="utf-8")

    feasible = [r for r in rows if r["feasible"]]
    winners = [
        r
        for r in feasible
        if r["d_cf_ok"] and r.get("joint_ok")
    ]
    tau = None
    if chosen:
        tau = chosen[0].instrument.tau_s
    sigma = chosen[0].instrument.pressure_sigma_pa if chosen else SIGMA_P
    t_max = 1800.0 if env is None else float(env.t_max_s)
    bounds = _bounds(t_max, tau, float(sigma))
    summary = {
        "n_designs": len(rows),
        "n_feasible": len(feasible),
        "n_identifiable": len(winners),
        "d_cf_min_target": D_CF_MIN,
        "d_tmf_min_target": D_TMF_MIN,
        "best_feasible_d_cf": max((r["d_cf"] for r in feasible), default=None),
        "argmax_u": max(feasible, key=lambda r: r["d_cf"])["name"] if feasible else None,
        "steady_dp_pa": STEADY_DP_PA,
        **bounds,
        "conclusion": (
            "found a feasible design with D_Cf>2"
            if winners
            else "no laboratory-feasible design reaches D_Cf,5%>2; do not retune ES-MDA"
        ),
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(_markdown_table(rows), flush=True)
    return 0 if winners else 1


if __name__ == "__main__":
    raise SystemExit(main())
