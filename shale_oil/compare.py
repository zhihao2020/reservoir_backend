"""Compare reconstructed fields (results/shale_oil) against CMG truth (cmg_fields.csv).

Prints RMSE + relative error for p / sw / so / sg / phi / k, per quantity and
averaged over all time steps.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "shale_oil"
TRUTH = ROOT / "shale_oil" / "cmg_fields.csv"


def _load_truth():
    # cmg_fields.csv: time_s,cell,i,j,k,p_pa,sw,sg,so,phi,k_m2  (405000 rows,
    # written per time step in cell order 0..3374)
    times = []
    cols = {n: [] for n in ("p_pa", "sw", "sg", "so", "phi", "k_m2")}
    cur_t = None
    with TRUTH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t = float(row["time_s"])
            if cur_t is None or t != cur_t:
                cur_t = t
                times.append(cur_t)
            for n in cols:
                cols[n].append(float(row[n]))
    n_t = len(times)
    n_c = len(cols["p_pa"]) // n_t
    out = {n: np.array(cols[n], dtype=float).reshape(n_t, n_c) for n in cols}
    return np.array(times), out


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    times, truth = _load_truth()
    n_t = times.size
    print(f"CMG truth: {n_t} 步 x {truth['p_pa'].shape[1]} 单元")

    recon = {}
    for name in ("p", "sw", "so", "sg", "phi", "k"):
        recon[name] = np.load(RESULTS / f"{name}.npy")
    print(f"重构: {recon['p'].shape[0]} 步 x {recon['p'].shape[1]} 单元")

    # map recon quantity -> truth column + unit
    mapping = [
        ("p", "p_pa", "Pa"),
        ("sw", "sw", ""),
        ("so", "so", ""),
        ("sg", "sg", ""),
        ("phi", "phi", ""),
        ("k", "k_m2", "m2"),
    ]

    print("\n=== 逐场对比（全部 120 步平均）===")
    print(f"{'量':<4} {'RMSE':>14} {'相对误差':>12} {'真值范围':>20}")
    for name, tcol, unit in mapping:
        a = recon[name]  # (n_t, n_c)
        b = truth[tcol]  # (n_t, n_c)
        rmse = _rmse(a, b)
        # relative error: RMSE / (mean |truth|), or RMSE / range for [0,1] sats
        denom = float(np.mean(np.abs(b)))
        if denom > 0:
            rel = rmse / denom
        else:
            rel = float("nan")
        rng = f"[{float(b.min()):.4g}, {float(b.max()):.4g}]"
        print(f"{name:<4} {rmse:>14.4e} {rel:>11.2%} {rng:>20} {unit}")

    # per-probe pressure comparison (first/last step) for a spot check
    print("\n=== 首步 / 末步压力场对比 ===")
    for ti in (0, n_t - 1):
        p_recon = recon["p"][ti]
        p_truth = truth["p_pa"][ti]
        rmse = _rmse(p_recon, p_truth)
        print(f"  t={times[ti]:.0f}s: 压力 RMSE = {rmse:.4e} Pa  "
              f"(真值均值 {float(np.mean(p_truth)):.4e} Pa)")


if __name__ == "__main__":
    main()
