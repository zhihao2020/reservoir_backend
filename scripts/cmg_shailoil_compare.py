"""Compare inverted software fields against GEM shailoil year-1 truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.units import MD_TO_M2

DAY_S = 86400.0
NX = NY = NZ = 15


def _rmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def _nrmse(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    span = float(np.nanmax(b[m]) - np.nanmin(b[m]))
    if span <= 0.0:
        span = max(abs(float(np.nanmean(b[m]))), 1.0)
    return _rmse(a, b) / span


def _reshape(flat):
    return np.asarray(flat, dtype=float).reshape(NZ, NY, NX)


def _probe_cells(case_dir: Path) -> list[tuple[str, int]]:
    import csv

    rows = []
    with (case_dir / "probes.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            x, y, z = float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
            i = min(NX - 1, max(0, int(x / 0.02)))
            j = min(NY - 1, max(0, int(y / 0.02)))
            k = min(NZ - 1, max(0, int(z / 0.02)))
            rows.append((row["id"], k * NY * NX + j * NX + i))
    return rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, default=ROOT / "results" / "shailoil_cmg" / "case")
    p.add_argument("--invert", type=Path, default=ROOT / "results" / "shailoil_cmg" / "invert")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "cmg_vs_shailoil")
    args = p.parse_args(argv)
    case_dir = Path(args.case)
    invert_dir = Path(args.invert)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    truth = np.load(case_dir / "cmg_truth.npz")
    fields = np.load(invert_dir / "fields.npz")
    t_cmg = np.asarray(truth["times_s"], dtype=float)
    t_sw = np.asarray(fields["time"], dtype=float)
    if t_cmg.shape != t_sw.shape or not np.allclose(t_cmg, t_sw, atol=1.0):
        raise SystemExit(f"time mismatch CMG {t_cmg} vs software {t_sw}")

    p_cmg = np.asarray(truth["pressure"], dtype=float)
    p_sw = np.asarray(fields["p"], dtype=float)
    sg_cmg = np.asarray(truth["sg"], dtype=float)
    sg_sw = np.asarray(fields["sg"], dtype=float)
    days = t_cmg / DAY_S

    per_time = []
    for it, day in enumerate(days):
        per_time.append(
            {
                "day": float(day),
                "p_rmse_pa": _rmse(p_sw[it], p_cmg[it]),
                "p_nrmse": _nrmse(p_sw[it], p_cmg[it]),
                "sg_rmse": _rmse(sg_sw[it], sg_cmg[it]),
            }
        )
    probes = _probe_cells(case_dir)
    probe_series = {}
    for name, cell in probes:
        probe_series[name] = {
            "cmg_pa": [float(p_cmg[it, cell]) for it in range(days.size)],
            "sw_pa": [float(p_sw[it, cell]) for it in range(days.size)],
        }
    probe_p_rmse = _rmse(
        [probe_series[n]["sw_pa"][i] for n, _ in probes for i in range(days.size)],
        [probe_series[n]["cmg_pa"][i] for n, _ in probes for i in range(days.size)],
    )
    k_md = np.asarray(fields["k"], dtype=float)[0] / MD_TO_M2
    phi = np.asarray(fields["phi"], dtype=float)[0]
    metrics = {
        "n_times": int(days.size),
        "days": [float(d) for d in days],
        "per_time": per_time,
        "p_rmse_pa_mean": float(np.nanmean([r["p_rmse_pa"] for r in per_time])),
        "p_nrmse_mean": float(np.nanmean([r["p_nrmse"] for r in per_time])),
        "sg_rmse_mean": float(np.nanmean([r["sg_rmse"] for r in per_time])),
        "probe_pressure_rmse_pa": float(probe_p_rmse),
        "k_md_mean": float(np.mean(k_md)),
        "k_md_deck": 0.03,
        "phi_mean": float(np.mean(phi)),
        "phi_deck": 0.05,
        "it_end": int(np.argmax(days)),
    }
    (outdir / "compare_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    _style()
    it = int(metrics["it_end"])
    show = [n for n, _ in probes if n in {"P_5_5_5", "P_8_5_5", "P_11_11_11", "P_5_11_11", "P_8_8_5", "P_11_5_11"}]
    if len(show) < 4:
        show = [n for n, _ in probes[:6]]

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    colors = ["#1d3557", "#2a9d8f", "#e76f51", "#457b9d", "#9b2226", "#e9c46a"]
    for n, name in enumerate(show):
        c = colors[n % len(colors)]
        ax.plot(days, np.asarray(probe_series[name]["cmg_pa"]) / 1.0e6, "o", color=c, ms=5, label=f"{name} CMG")
        ax.plot(days, np.asarray(probe_series[name]["sw_pa"]) / 1.0e6, "-", color=c, lw=1.5, label=f"{name} software")
    ax.set_xlabel("time / day")
    ax.set_ylabel("pressure / MPa")
    ax.set_title("shailoil year-1 probe pressure  |  CMG vs inversion")
    ax.legend(ncols=2, fontsize=7, frameon=False)
    fig.savefig(outdir / "01_probe_pressure.png")
    plt.close(fig)

    kmid = NZ // 2
    sl_xy = np.s_[kmid, :, :]
    jmid = NY // 2
    sl_xz = np.s_[:, jmid, :]
    pcmg = _reshape(p_cmg[it]) / 1.0e6
    psw = _reshape(p_sw[it]) / 1.0e6
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), constrained_layout=True)
    for row, sl, ylab in ((0, sl_xy, "j (y)"), (1, sl_xz, "k (z)")):
        fields_p = {
            "CMG p / MPa": pcmg[sl],
            "software p / MPa": psw[sl],
            "diff / MPa": pcmg[sl] - psw[sl],
        }
        vmin = min(float(pcmg[sl].min()), float(psw[sl].min()))
        vmax = max(float(pcmg[sl].max()), float(psw[sl].max()))
        for ax, (title, arr) in zip(axes[row], fields_p.items()):
            if "diff" in title:
                lim = max(0.05, float(np.nanmax(np.abs(arr))))
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            else:
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("i (x)")
            ax.set_ylabel(ylab)
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.82)
    fig.suptitle(f"shailoil pressure  t={days[it]:.0f} d  |  xy k={kmid} / xz j={jmid}")
    fig.savefig(outdir / "02_pressure_maps.png")
    plt.close(fig)

    gcmg = _reshape(sg_cmg[it])
    gsw = _reshape(sg_sw[it])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), constrained_layout=True)
    for ax, arr, title, cmap, vmin, vmax in (
        (axes[0], gcmg[sl_xy], "CMG Sg", "viridis", 0.0, 1.0),
        (axes[1], gsw[sl_xy], "software Sg", "viridis", 0.0, 1.0),
        (axes[2], gcmg[sl_xy] - gsw[sl_xy], "diff Sg", "RdBu_r", -0.5, 0.5),
    ):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("i (x)")
        ax.set_ylabel("j (y)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"shailoil Sg  xy mid-z  t={days[it]:.0f} d")
    fig.savefig(outdir / "03_sg_maps.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)
    im = axes[0].imshow(_reshape(k_md)[sl_xy], origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title(f"inverted k / md  mean={float(np.mean(k_md)):.4g}  deck=0.03")
    axes[0].set_xlabel("i (x)")
    axes[0].set_ylabel("j (y)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="k / md")
    im = axes[1].imshow(_reshape(phi)[sl_xy], origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title(f"inverted phi  mean={float(np.mean(phi)):.4g}  deck=0.05")
    axes[1].set_xlabel("i (x)")
    axes[1].set_ylabel("j (y)")
    axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="phi")
    fig.suptitle("shailoil inverted rock  xy mid-z")
    fig.savefig(outdir / "04_rock_maps.png")
    plt.close(fig)

    print(json.dumps(metrics, indent=2), flush=True)
    print("wrote", sorted(p.name for p in outdir.glob("*.png")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
