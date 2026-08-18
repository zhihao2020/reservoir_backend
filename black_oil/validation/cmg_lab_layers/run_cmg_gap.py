"""Invert from CMG gauges, then measure field gap vs IMEX.

Two gaps:
  model  = F(K_CMG) vs CMG          (same rock, different physics)
  invert = F(k_post) vs CMG         (after ES-MDA on CMG gauges)
"""

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
sys.path[:0] = [str(ROOT), str(VAL), str(HERE)]

from cmg_io.grid_parse import parse_grid_series
from reservoir_backend.domain.types import Experiment
from reservoir_backend.inverse.parameterization import RegionParameterization
from reservoir_backend.physics.rock import Rock, log_permeability
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec
from run_invert_eval import (
    DAY_S,
    MD_TO_M2,
    OUT,
    TRUTH,
    _cmg_to_our,
    _grid,
    _nearest,
    _physics,
    _ports,
    _region,
    _same_cmg_controls,
)
from cmg_io.grid_parse import psi_to_pa

FIG = HERE / "figures"
REPORT = HERE / "cmg_gap_report.json"
PSI = 6894.757293168
DAYS = (0.25, 0.50, 1.00)


def _reshape(grid, flat):
    return np.asarray(flat, dtype=float).reshape(grid.nz, grid.ny, grid.nx)


def _rmse(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _simulate(grid, truth, k, days):
    times = np.asarray(days, dtype=float) * DAY_S
    region = _region(grid, int(truth["layers"]["n_top_high"]))
    param = RegionParameterization(region, phi=float(truth["controls"]["phi"]))
    inj, prod = _ports(grid, truth=truth)
    twin = DigitalTwin(
        grid,
        Experiment(size_m=grid.size_m(), sensors=[], controls=_same_cmg_controls(truth, times), observations=[]),
        [inj, prod],
        _physics(p_init=psi_to_pa(float(truth["controls"]["pres_psi"])), sw_init=float(truth["controls"]["swi"])),
        param,
        inverse=InverseSpec(n_ensemble=4, n_assimilations=1),
    )
    traj = twin.simulate(twin.rock_from_k(k), t_end=float(times[-1]), report_times=times)
    out = {}
    for t in times:
        st = traj.state_at(float(t))
        out[float(t) / DAY_S] = {"p": _reshape(grid, st.pressure) / PSI, "sw": _reshape(grid, st.sw)}
    return out


def _panel(ax, z, *, cmap, vmin, vmax, title):
    im = ax.imshow(z, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("i")
    ax.set_ylabel("k↑")
    return im


def main() -> int:
    FIG.mkdir(exist_ok=True)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    jmid = ny // 2

    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    cmg = {}
    for d in DAYS:
        tp = _nearest(np.array([t for t, _ in p_series]), d)
        ts = _nearest(np.array([t for t, _ in sw_series]), d)
        p = next(a for t, a in p_series if t == tp)
        s = next(a for t, a in sw_series if t == ts)
        cmg[d] = {"p": _cmg_to_our(p), "sw": _cmg_to_our(s)}

    k_true = np.load(HERE / "k_true.npy")
    k_post = np.load(HERE / "k_post_cmg_obs.npy")
    print("F(K_CMG) and F(k_post) ...", flush=True)
    f_true = _simulate(grid, truth, k_true, DAYS)
    f_post = _simulate(grid, truth, k_post, DAYS)

    gaps = []
    for d in DAYS:
        gaps.append(
            {
                "day": d,
                "p_rmse_model_psi": _rmse(f_true[d]["p"], cmg[d]["p"]),
                "p_rmse_invert_psi": _rmse(f_post[d]["p"], cmg[d]["p"]),
                "sw_rmse_model": _rmse(f_true[d]["sw"], cmg[d]["sw"]),
                "sw_rmse_invert": _rmse(f_post[d]["sw"], cmg[d]["sw"]),
                "p_mean_cmg": float(cmg[d]["p"].mean()),
                "p_mean_Ftrue": float(f_true[d]["p"].mean()),
                "p_mean_Fpost": float(f_post[d]["p"].mean()),
                "sw_mean_cmg": float(cmg[d]["sw"].mean()),
                "sw_mean_Ftrue": float(f_true[d]["sw"].mean()),
                "sw_mean_Fpost": float(f_post[d]["sw"].mean()),
            }
        )
    k_gap = {
        "logk_rmse_post_vs_cmg": float(np.sqrt(np.mean((log_permeability(k_post) - log_permeability(k_true)) ** 2))),
        "k_lo_md_cmg": 50.0,
        "k_hi_md_cmg": 500.0,
        "k_lo_md_post": float(np.mean(k_post[_region(grid, 3) == 0]) / MD_TO_M2),
        "k_hi_md_post": float(np.mean(k_post[_region(grid, 3) == 1]) / MD_TO_M2),
    }
    report = {
        "meaning": {
            "model": "F(K_CMG) vs IMEX — gap if rock is exact; this is the physics floor",
            "invert": "F(k_post) vs IMEX — after inverting CMG gauges with F_lab",
        },
        "k": k_gap,
        "fields": gaps,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    sl = np.s_[:, jmid, :]
    fig, axes = plt.subplots(len(DAYS), 6, figsize=(13.6, 2.2 * len(DAYS)), constrained_layout=True)
    p0, p1 = 2780.0, 3220.0
    s0, s1 = 0.18, 0.85
    for i, d in enumerate(DAYS):
        _panel(axes[i, 0], cmg[d]["p"][sl], cmap="coolwarm", vmin=p0, vmax=p1, title=f"t={d}d CMG p")
        _panel(axes[i, 1], f_true[d]["p"][sl], cmap="coolwarm", vmin=p0, vmax=p1, title="F(K_CMG) p")
        _panel(axes[i, 2], f_post[d]["p"][sl], cmap="coolwarm", vmin=p0, vmax=p1, title="F(k_post) p")
        _panel(axes[i, 3], cmg[d]["sw"][sl], cmap="YlGnBu", vmin=s0, vmax=s1, title="CMG Sw")
        _panel(axes[i, 4], f_true[d]["sw"][sl], cmap="YlGnBu", vmin=s0, vmax=s1, title="F(K_CMG) Sw")
        im = _panel(axes[i, 5], f_post[d]["sw"][sl], cmap="YlGnBu", vmin=s0, vmax=s1, title="F(k_post) Sw")
        for ax in axes[i]:
            ax.axhline(grid.nz / 2 - 0.5, color="k", ls="--", lw=0.4, alpha=0.35)
    fig.colorbar(im, ax=axes[:, 3:], shrink=0.5, label="Sw")
    fig.suptitle("CMG IMEX  vs  our F at CMG rock  vs  our F after invert", fontsize=12)
    fig.savefig(FIG / "cmg_gap_fields.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    ax.axis("off")
    lines = [
        ["t (d)", "p RMSE model", "p RMSE invert", "Sw RMSE model", "Sw RMSE invert"],
    ]
    for g in gaps:
        lines.append(
            [
                f"{g['day']:.2f}",
                f"{g['p_rmse_model_psi']:.0f} psi",
                f"{g['p_rmse_invert_psi']:.0f} psi",
                f"{g['sw_rmse_model']:.3f}",
                f"{g['sw_rmse_invert']:.3f}",
            ]
        )
    lines.append(["K", f"CMG 50/500 md", f"post {k_gap['k_lo_md_post']:.0f}/{k_gap['k_hi_md_post']:.0f}", f"logK RMSE {k_gap['logk_rmse_post_vs_cmg']:.2f}", ""])
    table = ax.table(cellText=lines, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.6)
    ax.set_title("Gap vs CMG: model = same K; invert = after ES-MDA on gauges")
    fig.savefig(FIG / "cmg_gap_table.png", dpi=150)
    plt.close(fig)
    print("wrote", FIG / "cmg_gap_fields.png", FIG / "cmg_gap_table.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
