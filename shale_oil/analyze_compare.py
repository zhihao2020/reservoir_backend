"""Analyse and export the reconstruction-vs-CMG comparison (CSV + figures).

- comparison_summary.csv : per-field metrics (truth variation + recon RMSE)
- comparison_per_time.csv : per-field RMSE at every time step
- comparison_fields.png   : spatial slices (middle-z) of p / so / k, truth vs recon
- comparison_rmse.png     : RMSE over time for every field
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "shale_oil"
TRUTH = ROOT / "shale_oil" / "cmg_fields.csv"
OUT = ROOT / "shale_oil" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    ("p", "p_pa", "Pa"),
    ("sw", "sw", "-"),
    ("so", "so", "-"),
    ("sg", "sg", "-"),
    ("phi", "phi", "-"),
    ("k", "k_m2", "m2"),
]


def load_truth():
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
    return np.array(times), {n: np.array(cols[n]).reshape(n_t, n_c) for n in cols}


def load_recon():
    return {name: np.load(RESULTS / f"{name}.npy") for name, _, _ in FIELDS}


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    times, truth = load_truth()
    recon = load_recon()
    n_t = times.size

    # ---- per-field summary (spatial + temporal) --------------------------
    summary_rows = []
    for name, tcol, unit in FIELDS:
        a = recon[name]          # (n_t, n_c)
        b = truth[tcol]          # (n_t, n_c)
        rmse = _rmse(a, b)
        mean_abs = float(np.mean(np.abs(b)))
        rng = float(b.max() - b.min())
        # truth variation: spatial std averaged over time, temporal std averaged over space
        spatial_std = float(np.mean(np.std(b, axis=1)))   # per time, avg
        temporal_std = float(np.mean(np.std(b, axis=0)))  # per cell, avg
        rel_mean = rmse / mean_abs if mean_abs > 0 else float("nan")
        rel_range = rmse / rng if rng > 0 else float("nan")
        summary_rows.append(
            {
                "field": name,
                "truth_mean": float(np.mean(b)),
                "truth_range": rng,
                "truth_spatial_std": spatial_std,
                "truth_temporal_std": temporal_std,
                "recon_rmse": rmse,
                "rel_error_vs_mean": rel_mean,
                "rel_error_vs_range": rel_range,
            }
        )

    with (OUT / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    # ---- per-time RMSE ---------------------------------------------------
    per_time_rows = []
    for ti in range(n_t):
        row = {"time_s": times[ti]}
        for name, tcol, _ in FIELDS:
            row[name] = _rmse(recon[name][ti], truth[tcol][ti])
        per_time_rows.append(row)
    with (OUT / "comparison_per_time.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_time_rows[0]))
        w.writeheader()
        w.writerows(per_time_rows)

    # ---- figures ---------------------------------------------------------
    # Figure 1: middle-z slices (truth vs recon) for p / so / k
    z = recon["p"].shape[1]  # flat per step
    nz = 15
    n_c = 15 * 15 * 15
    zi = nz // 2
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (name, tcol, _) in enumerate([("p", "p_pa", ""), ("so", "so", ""), ("k", "k_m2", "")]):
        ti = n_t - 1
        tr = truth[tcol][ti].reshape(nz, 15, 15)[zi]
        rc = recon[name][ti].reshape(nz, 15, 15)[zi]
        vmin, vmax = float(tr.min()), float(tr.max())
        ax0, ax1 = axes[0, col], axes[1, col]
        im0 = ax0.imshow(tr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax0.set_title(f"truth {name} (z={zi})")
        im1 = ax1.imshow(rc, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title(f"recon {name} (z={zi})")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.suptitle("Middle-z slice: truth vs reconstructed (last time step)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_fields.png", dpi=110)
    plt.close(fig)

    # Figure 2: RMSE over time (per field)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, (name, _, unit) in zip(axes.ravel(), FIELDS):
        rmse_t = [r[name] for r in per_time_rows]
        ax.plot(times / 86400.0, rmse_t, lw=1.5)
        ax.set_title(f"{name} RMSE ({unit})")
        ax.set_xlabel("time (day)")
        ax.set_ylabel("RMSE")
        ax.ticklabel_format(style="scientific", scilimits=(-2, 3), axis="y")
    fig.suptitle("Reconstruction RMSE over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_rmse.png", dpi=110)
    plt.close(fig)

    # ---- console report --------------------------------------------------
    print("field  truth_mean  truth_range  spat_std  temp_std  rmse  rel_mean  rel_range")
    for r in summary_rows:
        print(
            f"{r['field']:<5} {r['truth_mean']:>10.4e} {r['truth_range']:>12.4e} "
            f"{r['truth_spatial_std']:>9.4e} {r['truth_temporal_std']:>9.4e} "
            f"{r['recon_rmse']:>9.4e} {r['rel_error_vs_mean']:>8.1%} {r['rel_error_vs_range']:>8.1%}"
        )
    print(f"\nExported: {OUT}")


if __name__ == "__main__":
    main()
