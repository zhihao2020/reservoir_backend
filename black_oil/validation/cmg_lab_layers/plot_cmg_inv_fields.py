"""CMG vs inversion as 2-D field maps (xz mid-y, xy per layer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[3]
VAL = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(VAL), str(HERE)]

from cmg_io.grid_parse import parse_grid_series, psi_to_pa
from reservoir_backend.domain.types import Experiment
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

FIG = HERE / "figures"
PSI = 6894.757293168


def _pick(series, day: float):
    t = _nearest(np.array([x for x, _ in series], dtype=float), day)
    arr = next(a for tt, a in series if tt == t)
    return t, arr


def _reshape(grid, flat):
    return np.asarray(flat, dtype=float).reshape(grid.nz, grid.ny, grid.nx)


def _simulate(grid, truth, k_flat, days):
    times = np.asarray(days, dtype=float) * DAY_S
    region = _region(grid, int(truth["layers"]["n_top_high"]))
    from reservoir_backend.inverse.parameterization import RegionParameterization

    param = RegionParameterization(region, phi=float(truth["controls"]["phi"]))
    inj, prod = _ports(grid, truth=truth)
    twin = DigitalTwin(
        grid,
        Experiment(
            size_m=grid.size_m(),
            sensors=[],
            controls=_same_cmg_controls(truth, times),
            observations=[],
        ),
        [inj, prod],
        _physics(
            p_init=psi_to_pa(float(truth["controls"]["pres_psi"])),
            sw_init=float(truth["controls"]["swi"]),
        ),
        param,
        inverse=InverseSpec(n_ensemble=4, n_assimilations=1),
    )
    traj = twin.simulate(twin.rock_from_k(k_flat), t_end=float(times[-1]), report_times=times)
    out = {}
    for t in times:
        st = traj.state_at(float(t))
        out[float(t) / DAY_S] = {
            "p": _reshape(grid, st.pressure) / PSI,
            "sw": _reshape(grid, st.sw),
        }
    return out


def _panel(ax, field, *, cmap, vmin, vmax, title, xlabel=True):
    im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("k (z↑)")
    if xlabel:
        ax.set_xlabel("i (x)")
    else:
        ax.set_xticklabels([])
    return im


def main() -> None:
    FIG.mkdir(exist_ok=True)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    grid = _grid(truth)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    jmid = ny // 2
    days = [0.25, 0.50, 1.00]

    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    cmg = {}
    for d in days:
        _, p = _pick(p_series, d)
        _, s = _pick(sw_series, d)
        cmg[d] = {"p": _cmg_to_our(p), "sw": _cmg_to_our(s)}

    k_true = np.load(HERE / "k_true.npy")
    k_a = np.load(HERE / "k_post_self.npy")
    k_b = np.load(HERE / "k_post_cmg_obs.npy")
    print("simulate F(k_true), F(k_A), F(k_B) ...", flush=True)
    fa = _simulate(grid, truth, k_a, days)
    fb = _simulate(grid, truth, k_b, days)

    # --- Pressure and Sw xz field maps ---
    fig, axes = plt.subplots(len(days), 6, figsize=(14.2, 2.15 * len(days)), constrained_layout=True)
    pmin, pmax = 2780.0, 3220.0
    smin, smax = 0.18, 0.82
    for i, d in enumerate(days):
        last = i == len(days) - 1
        sl = np.s_[:, jmid, :]
        pairs = (
            (cmg[d]["p"][sl], fa[d]["p"][sl], "p", f"t={d} d  CMG p", f"F(k_A) p", f"CMG−A p"),
            (cmg[d]["sw"][sl], fb[d]["sw"][sl], "sw", f"t={d} d  CMG Sw", f"F(k_B) Sw", f"CMG−B Sw"),
        )
        # row: CMG p | A p | Δp | CMG Sw | B Sw | ΔSw
        cp, ap = cmg[d]["p"][sl], fa[d]["p"][sl]
        cs, bs = cmg[d]["sw"][sl], fb[d]["sw"][sl]
        im0 = _panel(axes[i, 0], cp, cmap="coolwarm", vmin=pmin, vmax=pmax, title=f"t={d}d  CMG  p", xlabel=last)
        im1 = _panel(axes[i, 1], ap, cmap="coolwarm", vmin=pmin, vmax=pmax, title="invert A  F(k) p", xlabel=last)
        dp = cp - ap
        lim = max(20.0, float(np.nanmax(np.abs(dp))))
        im2 = _panel(axes[i, 2], dp, cmap="RdBu_r", vmin=-lim, vmax=lim, title="CMG − A  Δp (psi)", xlabel=last)
        im3 = _panel(axes[i, 3], cs, cmap="YlGnBu", vmin=smin, vmax=smax, title="CMG  Sw", xlabel=last)
        im4 = _panel(axes[i, 4], bs, cmap="YlGnBu", vmin=smin, vmax=smax, title="invert B  F(k) Sw", xlabel=last)
        ds = cs - bs
        slim = max(0.05, float(np.nanmax(np.abs(ds))))
        im5 = _panel(axes[i, 5], ds, cmap="RdBu_r", vmin=-slim, vmax=slim, title="CMG − B  ΔSw", xlabel=last)
        for ax in axes[i, :]:
            ax.axhline(nz / 2 - 0.5, color="k", ls="--", lw=0.5, alpha=0.4)
    fig.colorbar(im0, ax=axes[:, 0:2], shrink=0.55, label="p (psi)")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.55, label="Δp")
    fig.colorbar(im3, ax=axes[:, 3:5], shrink=0.55, label="Sw")
    fig.colorbar(im5, ax=axes[:, 5], shrink=0.55, label="ΔSw")
    fig.suptitle("Field maps  xz @ j=mid    CMG simulation vs inversion forward", fontsize=13)
    fig.savefig(FIG / "cmg_vs_inv_fields_xz.png", dpi=150)
    plt.close(fig)

    # --- K field maps ---
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.1), constrained_layout=True)
    kt = _reshape(grid, k_true) / MD_TO_M2
    ka = _reshape(grid, k_a) / MD_TO_M2
    kb = _reshape(grid, k_b) / MD_TO_M2
    sl = np.s_[:, jmid, :]
    for ax, field, title in (
        (axes[0], kt[sl], "CMG / truth K"),
        (axes[1], ka[sl], "A posterior K"),
        (axes[2], kb[sl], "B posterior K"),
        (axes[3], np.log10(np.maximum(ka[sl], 1e-6) / np.maximum(kt[sl], 1e-6)), "A / truth  log10(k)"),
    ):
        if "log10" in title:
            im = ax.imshow(field, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        else:
            im = ax.imshow(field, origin="lower", aspect="auto", cmap="viridis", vmin=30, vmax=700)
        ax.set_title(title)
        ax.set_xlabel("i (x)")
        ax.set_ylabel("k (z↑)")
        ax.axhline(nz / 2 - 0.5, color="w", ls="--", lw=0.6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Permeability field  xz @ j=mid")
    fig.savefig(FIG / "cmg_vs_inv_k_field.png", dpi=150)
    plt.close(fig)

    # --- Sw areal maps: top layer vs bottom, t=1 d ---
    d = 1.0
    k_bot, k_top = 0, nz - 1
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), constrained_layout=True)
    for row, kk, lab in ((0, k_top, "top layer"), (1, k_bot, "bottom layer")):
        a = cmg[d]["sw"][kk]
        b = fb[d]["sw"][kk]
        _panel(axes[row, 0], a, cmap="YlGnBu", vmin=smin, vmax=smax, title=f"{lab}  t=1d  CMG Sw")
        _panel(axes[row, 1], b, cmap="YlGnBu", vmin=smin, vmax=smax, title=f"{lab}  invert B  Sw")
        _panel(axes[row, 2], a - b, cmap="RdBu_r", vmin=-0.4, vmax=0.4, title=f"{lab}  CMG − B")
        for ax in axes[row]:
            ax.set_xlabel("i (x)")
            ax.set_ylabel("j (y)")
    fig.suptitle("Areal Sw fields at t=1 day")
    fig.savefig(FIG / "cmg_vs_inv_sw_xy.png", dpi=150)
    plt.close(fig)
    print("wrote", sorted(p.name for p in FIG.glob("cmg_vs_inv_*.png")))


if __name__ == "__main__":
    main()
