"""Figures for the CMG two-layer invert ruler. No IMEX rerun."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(VAL)]

from cmg_io.grid_parse import parse_grid_series, psi_to_pa
from run_invert_eval import _cmg_to_our, _depths, _grid, diverse_sensors

MD = 9.869233e-16
FIG = HERE / "figures"
TRUTH = HERE / "truth_lab_layers.json"
REPORT = HERE / "invert_eval_report.json"
OUT = HERE / "lab_layers.out"


def _reshape(grid, k):
    return np.asarray(k, dtype=float).reshape(grid.nz, grid.ny, grid.nx)


def _k_md(path, grid):
    return _reshape(grid, np.load(path)) / MD


def main() -> None:
    FIG.mkdir(exist_ok=True)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    grid = _grid(truth)
    jmid = grid.ny // 2
    k_true = _k_md(HERE / "k_true.npy", grid)
    k_a = _k_md(HERE / "k_post_self.npy", grid)
    k_b = _k_md(HERE / "k_post_cmg_obs.npy", grid)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)
    vmin, vmax = 30.0, 700.0
    titles = ["CMG / truth K", "A posterior (self-consistent)", "B posterior (CMG gauges)"]
    for ax, field, title in zip(axes, (k_true, k_a, k_b), titles):
        im = ax.imshow(
            field[:, jmid, :],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("i (x)")
        ax.set_ylabel("k (z, 0=bottom)")
        ax.axhline(2.5, color="w", ls="--", lw=0.8, alpha=0.8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="k (md)")
    fig.savefig(FIG / "k_xz_compare.png", dpi=140)
    plt.close(fig)

    snaps = report["cmg_experiment_design"]["snapshots"]
    days = [s["day"] for s in snaps]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.plot(days, [s["p_std_psi"] for s in snaps], "o-", label="CMG p std (psi)")
    ax.plot(days, [s["sw_std"] * 400 for s in snaps], "s--", label="CMG Sw std ×400")
    ax.axvspan(0.25, 0.50, color="0.85", label="B history (0.25–0.5 d)")
    ax.axvline(1.0, color="C3", ls=":", label="B forecast t=1 d")
    ax.set_xlabel("time (day)")
    ax.set_ylabel("field spread")
    ax.legend(fontsize=8)
    ax.set_title("CMG time series: spatial Δp / ΔSw stay informative")
    fig.savefig(FIG / "cmg_timeseries_field.png", dpi=140)
    plt.close(fig)

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    sensors, hold = diverse_sensors(grid, p_sigma=psi_to_pa(30.0), s_sigma=0.05, with_rate=False)
    from reservoir_backend.domain.types import State
    from reservoir_backend.observation.operator import ObservationOperator

    op = ObservationOperator(grid, sensors)
    t_p = np.array([t for t, _ in p_series])
    rows_p = {s.name: [] for s in sensors if s.kind == "pressure"}
    rows_s = {s.name: [] for s in sensors if s.kind == "saturation"}
    times = []
    for t, parr in p_series:
        sw = next((b for ts, b in sw_series if abs(ts - t) < 1e-9), None)
        if sw is None:
            continue
        times.append(t)
        st = State(pressure=(_cmg_to_our(parr) * 6894.757293168).ravel(), sw=_cmg_to_our(sw).ravel())
        for s in sensors:
            val = op.sample(s, st)
            if s.kind == "pressure":
                rows_p[s.name].append(val / 6894.757293168)
            else:
                rows_s[s.name].append(val)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, constrained_layout=True)
    for name, ys in rows_p.items():
        ax0.plot(times, ys, lw=1.2, label=name + (" *" if name in hold else ""))
    ax0.axvspan(0.25, 0.50, color="0.9")
    ax0.set_ylabel("CMG pressure (psi)")
    ax0.set_title("Gauge time series (* = hold-out). History is the shaded window.")
    ax0.legend(ncols=3, fontsize=7)
    for name, ys in rows_s.items():
        ax1.plot(times, ys, lw=1.2, label=name + (" *" if name in hold else ""))
    ax1.axvspan(0.25, 0.50, color="0.9")
    ax1.set_xlabel("time (day)")
    ax1.set_ylabel("CMG Sw")
    ax1.legend(ncols=3, fontsize=7)
    fig.savefig(FIG / "cmg_gauge_timeseries.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    a_m = report["A_self_consistent_diverse"]["esmda_mismatch"]
    b_m = report["B_cross_cmg_observations"]["esmda_mismatch"]
    ax.plot(np.arange(len(a_m)), a_m, "o-", label="A self-consistent")
    ax.plot(np.arange(len(b_m)), b_m, "s-", label="B CMG obs + inflated R")
    ax.set_xlabel("ES-MDA step")
    ax.set_ylabel("whitened misfit")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Assimilation uses 3 time snapshots, not a single map")
    fig.savefig(FIG / "misfit_steps.png", dpi=140)
    plt.close(fig)

    b = report.get("B_cross_cmg_observations") or {}
    if "gauges_cmg" in b and "gauges_post" in b:
        g_cmg = b["gauges_cmg"]
        g_post = b["gauges_post"]
        t_cmg = np.asarray(g_cmg["times_day"], dtype=float)
        t_post = np.asarray(g_post["times_s"], dtype=float) / 86400.0
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, constrained_layout=True)
        for name, ys in g_cmg["series"].items():
            if name.startswith("P"):
                ax0.plot(t_cmg, np.asarray(ys) / 6894.757293168, "o", ms=5, label=name + " CMG")
                if name in g_post["series"]:
                    ax0.plot(t_post, np.asarray(g_post["series"][name]) / 6894.757293168, "-", lw=1.2, label=name + " F(k_post)")
            elif name.startswith("S"):
                ax1.plot(t_cmg, ys, "o", ms=5, label=name + " CMG")
                if name in g_post["series"]:
                    ax1.plot(t_post, g_post["series"][name], "-", lw=1.2, label=name + " F(k_post)")
        ax0.axvspan(0.25, 1.0, color="0.9")
        ax1.axvspan(0.25, 1.0, color="0.9")
        ax0.set_ylabel("pressure (psi)")
        ax0.set_title("B: CMG gauges vs our F(k_post)  (shade = history)")
        ax0.legend(ncols=2, fontsize=7)
        ax1.set_xlabel("time (day)")
        ax1.set_ylabel("Sw")
        ax1.legend(ncols=2, fontsize=7)
        fig.savefig(FIG / "b_cmg_vs_pred.png", dpi=140)
        plt.close(fig)

    print("wrote", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
