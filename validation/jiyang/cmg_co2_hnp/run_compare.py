"""GEM well-history truth vs product compositional F.

Does not spawn GEM. Uses the already-run .out extract as H.
Gate: injector BHP (rate well) and producer q_oil (BHP wells).
Producer BHP is the control, not the score.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASE = ROOT / "examples" / "jiyang" / "jiyang_co2_hnp.yaml"
OBS = ROOT / "examples" / "jiyang" / "fixtures" / "jiyang_co2_hnp_obs.csv"
REGIONS = ROOT / "examples" / "jiyang" / "jiyang_frac_regions.npy"
FIG = HERE / "figures"
MD = 9.869233e-16

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 10,
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def _truth_rock(n_cells: int, phi: float):
    from reservoir_backend.physics.rock import Rock

    rid = np.load(REGIONS).astype(np.int64).ravel()
    if rid.size != n_cells:
        raise ValueError(f"region_map {rid.size} != n_cells {n_cells}")
    k = np.where(rid == 1, 5.0, 0.05) * MD
    return Rock(k, np.full(n_cells, phi)), rid


PHASES = (
    (0.0, 91.0, "#eeeeee", "衰竭"),
    (91.0, 121.0, "#ffe6cc", "注"),
    (121.0, 152.0, "#e6f2ff", "焖"),
    (152.0, 456.0, "#e8f5e9", "采"),
)
WELLS = ["INJ", "P1", "P2", "P3", "P4"]


def _gem_series(kind: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    src = HERE / "well_history.csv"
    path = src if src.is_file() else OBS
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["kind"] != kind:
            continue
        well = r.get("well") or str(r.get("sensor", "")).split("_")[0]
        by[well].append((float(r["time_s"]), float(r["value"])))
    out = {}
    for well, pairs in by.items():
        pairs.sort()
        t = np.array([p[0] for p in pairs], dtype=float)
        v = np.array([p[1] for p in pairs], dtype=float)
        out[well] = (t, v)
    return out


def _gem_bhp() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _gem_series("bhp")


def _prod_series(traj, wells: list[str], which: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t = np.asarray(traj.times_s, dtype=float)
    out = {}
    for w in wells:
        v = []
        for i in range(t.size):
            rates, bhp = traj.rates_and_bhp_at(float(t[i]))
            if which == "bhp":
                v.append(float(bhp.get(w, np.nan)))
            elif which in {"q_oil", "q_gas", "q_inj"}:
                v.append(float(rates.get(w + ":" + which, np.nan)))
            else:
                v.append(float(rates.get(w, np.nan)))
        out[w] = (t, np.asarray(v, dtype=float))
    return out


def _pred_at_gem(traj, gem_times: np.ndarray, well: str, kind: str) -> np.ndarray:
    pred = []
    for tt in np.asarray(gem_times, dtype=float):
        rates, bhp = traj.rates_and_bhp_at(float(tt))
        if kind == "bhp":
            pred.append(float(bhp.get(well, np.nan)))
        else:
            pred.append(float(rates.get(well + ":" + kind, rates.get(well, np.nan))))
    return np.asarray(pred, dtype=float)


def _prod_bhp(traj, wells: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _prod_series(traj, wells, "bhp")


def _nrmse(pred: np.ndarray, truth: np.ndarray, sigma: float) -> float:
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(((pred - truth) / max(sigma, 1.0)) ** 2)))


def _shade_phases(ax) -> None:
    for t0, t1, color, label in PHASES:
        ax.axvspan(t0, t1, color=color, lw=0, label=label)


def plot_bhp(gem, prod, title: str, path: Path) -> None:
    fig, axes = plt.subplots(len(WELLS), 1, figsize=(8.4, 9.6), sharex=True)
    for i, (w, ax) in enumerate(zip(WELLS, axes)):
        _shade_phases(ax)
        if w in prod and prod[w][0].size:
            ax.plot(prod[w][0] / 86400.0, prod[w][1] / 1.0e6, "-", color="#d35400", lw=1.7, drawstyle="steps-post", label="产品 F", zorder=3)
        if w in gem and gem[w][0].size:
            ax.plot(gem[w][0] / 86400.0, gem[w][1] / 1.0e6, "o", color="#1f4e79", ms=6, mew=0.6, mec="white", label="GEM", zorder=4)
        ax.set_ylabel(f"{w}\nMPa")
        ax.set_xlim(0.0, 460.0)
        ax.grid(True, axis="y", alpha=0.35)
        ax.set_axisbelow(True)
        if i == 0:
            h, lab = ax.get_legend_handles_labels()
            # one handle per label (phases + series)
            uniq = dict(zip(lab, h))
            ax.legend(uniq.values(), uniq.keys(), loc="upper right", ncol=3, fontsize=8, framealpha=0.92)
    axes[-1].set_xlabel("时间 / d")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_rates(gem_inj, gem_oil, prod_inj, prod_oil, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), sharex=True)
    _shade_phases(axes[0])
    if "INJ" in prod_inj and prod_inj["INJ"][0].size:
        axes[0].plot(
            prod_inj["INJ"][0] / 86400.0,
            np.maximum(prod_inj["INJ"][1], 0.0) * 86400.0,
            "-",
            color="#d35400",
            lw=1.7,
            drawstyle="steps-post",
            label="产品 F",
            zorder=3,
        )
    if "INJ" in gem_inj and gem_inj["INJ"][0].size:
        axes[0].plot(gem_inj["INJ"][0] / 86400.0, gem_inj["INJ"][1] * 86400.0, "o", color="#1f4e79", ms=6, mew=0.6, mec="white", label="GEM", zorder=4)
    axes[0].set_ylabel("INJ 地面气 m³/d")
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=8)
    _shade_phases(axes[1])
    for w, color in zip(("P1", "P2", "P3", "P4"), ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")):
        if w in prod_oil and prod_oil[w][0].size:
            axes[1].plot(prod_oil[w][0] / 86400.0, prod_oil[w][1] * 86400.0, "-", color=color, lw=1.2, drawstyle="steps-post", alpha=0.85)
        if w in gem_oil and gem_oil[w][0].size:
            axes[1].plot(gem_oil[w][0] / 86400.0, gem_oil[w][1] * 86400.0, "o", color=color, ms=5, label=w)
    axes[1].set_ylabel("产油 m³/d  线=F  点=GEM")
    axes[1].set_xlabel("时间 / d")
    axes[1].set_xlim(0.0, 460.0)
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("井史：注入率 / 产油率")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_k(grid, k, rid, path: Path) -> None:
    kx = np.asarray(k, dtype=float).reshape(grid.nz, grid.ny, grid.nx)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(np.log10(kx[2] / MD), origin="lower", cmap="viridis")
    ax.set_title("log10 K (md), layer k=3")
    ax.set_xlabel("I")
    ax.set_ylabel("J")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    from reservoir_backend.io.case import load_case

    twin = load_case(CASE)
    phi = float(getattr(twin.parameterization, "phi", 0.06))
    rock, rid = _truth_rock(twin.grid.n_cells, phi)
    gem = _gem_bhp()
    wells = WELLS
    report: dict = {
        "n_cells": twin.grid.n_cells,
        "n_obs": sum(s.values.size for s in twin.experiment.observations),
        "model": twin.physics.model,
        "fluid_nc": int(twin.physics.fluid.eos.nc) if twin.physics.fluid is not None else None,
    }

    prod = {}
    if not args.skip_forward:
        t0 = time.perf_counter()
        t_end = float(twin.experiment.history_end_s or 39398400.0)
        from reservoir_backend.solver.fi_comp import simulate_comp

        # Keep every accepted step. DigitalTwin.simulate() would collapse to
        # control/obs times (~7 points) and the time series would look like a polyline.
        traj = simulate_comp(
            twin.grid,
            rock,
            twin.physics.fluid,
            twin.ports,
            twin.experiment.controls,
            twin.initial_state(),
            t_end,
            dt_init=twin.physics.dt_init,
            dt_min=twin.physics.dt_min,
            dt_max=twin.physics.dt_max,
            max_steps=int(twin.physics.max_steps),
            report_times=None,
        )
        report["forward_s"] = time.perf_counter() - t0
        report["n_steps"] = len(traj.reports)
        report["n_plot_times"] = int(np.asarray(traj.times_s).size)
        if traj.reports:
            report["mass_rel"] = traj.reports[-1].mass.relative_balance_error
        prod = _prod_bhp(traj, wells)
        prod_oil = _prod_series(traj, wells, "q_oil")
        prod_inj = _prod_series(traj, wells, "q_inj")
        np.savez(HERE / "traj_bhp.npz", times=np.asarray(traj.times_s, dtype=float), **{f"bhp_{w}": prod[w][1] for w in wells if w in prod})
        plot_bhp(gem, prod, "井底流压（采井是控制，注入井才是观测）", FIG / "gem_vs_f_bhp.png")
        plot_rates(_gem_series("q_inj"), _gem_series("q_oil"), prod_inj, prod_oil, FIG / "gem_vs_f_rates.png")
        plot_k(twin.grid, rock.permeability, rid, FIG / "k_true_layer.png")
        gem_oil = _gem_series("q_oil")
        inj_bhp = {}
        if "INJ" in gem:
            tg, vg = gem["INJ"]
            inj_bhp["INJ"] = _nrmse(_pred_at_gem(traj, tg, "INJ", "bhp"), vg, 1.0e5)
        q_oil = {}
        for w in ("P1", "P2", "P3", "P4"):
            if w not in gem_oil:
                continue
            tg, vg = gem_oil[w]
            q_oil[w] = _nrmse(_pred_at_gem(traj, tg, w, "q_oil"), vg, 1.0e-6)
        bhp_diag = {}
        for w, (tg, vg) in gem.items():
            if w == "INJ":
                continue
            bhp_diag[w] = _nrmse(_pred_at_gem(traj, tg, w, "bhp"), vg, 1.0e5)
        report["inj_bhp_nrmse_sigma1e5"] = inj_bhp
        report["q_oil_nrmse_sigma1e-6"] = q_oil
        report["producer_bhp_diagnostic_sigma1e5"] = bhp_diag

    if args.invert and twin.experiment.observations:
        t0 = time.perf_counter()
        post = twin.calibrate(time_limit_s=1800)
        report["invert_s"] = time.perf_counter() - t0
        report["theta"] = post.theta.tolist()
        report["assimilate_rmse"] = post.assimilate_rmse
        report["holdout_rmse"] = post.holdout_rmse
        k_lo = float(np.mean(post.k[rid == 0]))
        k_hi = float(np.mean(post.k[rid == 1]))
        report["contrast_post"] = k_hi / max(k_lo, 1.0e-30)
        post_bhp = _prod_bhp(post.history, wells)
        plot_bhp(gem, post_bhp, "GEM 井史 vs 后验 F(m_post) BHP", FIG / "gem_vs_post_bhp.png")
        plot_k(twin.grid, post.k, rid, FIG / "k_post_layer.png")
        if post.history.reports:
            report["post_mass_rel"] = post.history.reports[-1].mass.relative_balance_error

    FIG.mkdir(parents=True, exist_ok=True)
    (HERE / "compare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
