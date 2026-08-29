"""真值 vs 反演（活油脱气）。中文表头。"""

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

from cmg_io.grid_parse import parse_grid_series, psi_to_pa
from reservoir_backend.domain.types import Experiment
from reservoir_backend.inverse.parameterization import ContrastParameterization
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec

import run_invert_eval as lib_eval

DAY_S = lib_eval.DAY_S
MD_TO_M2 = lib_eval.MD_TO_M2
_cmg_to_our = lib_eval._cmg_to_our
_grid = lib_eval._grid
_physics = lib_eval._physics
_ports = lib_eval._ports
_same_cmg_controls = lib_eval._same_cmg_controls

FIG = HERE / "figures"
PSI = 6894.757293168
OUT = HERE / "fault_channel_lib.out"
TRUTH = HERE / "truth_fault_channel_lib.json"
POST = HERE / "k_post.npy"

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 10,
        "figure.dpi": 160,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    }
)


def _zh_setup() -> None:
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei"):
        matches = [f.fname for f in font_manager.fontManager.ttflist if name.lower() in f.name.lower()]
        if matches:
            plt.rcParams["font.sans-serif"] = [name]
            return


def _reshape(grid, flat):
    return np.asarray(flat, dtype=float).reshape(grid.nz, grid.ny, grid.nx)


def _pick(series, day: float):
    ts = np.array([t for t, _ in series], dtype=float)
    t = float(ts[int(np.argmin(np.abs(ts - day)))])
    arr = next(a for tt, a in series if tt == t)
    return t, arr


def _simulate(grid, truth, k_flat, days):
    times = np.asarray(days, dtype=float) * DAY_S
    param = ContrastParameterization(lib_eval._rid(), phi=float(truth["controls"]["phi"]))
    inj, prod = _ports(grid, truth=truth)
    twin = DigitalTwin(
        grid,
        Experiment(size_m=grid.size_m(), sensors=[], controls=_same_cmg_controls(truth, times), observations=[]),
        [inj, prod],
        _physics(
            p_init=psi_to_pa(float(truth["controls"]["pres_psi"])),
            sw_init=float(truth["controls"]["swi"]),
            sg_init=float(truth["controls"].get("sgi", 0.0)),
            three_phase=True,
        ),
        param,
        face_mult_x=lib_eval._face_mult(),
        inverse=InverseSpec(max_iter=4),
    )
    traj = twin.simulate(twin.rock_from_k(k_flat), t_end=float(times[-1]), report_times=times)
    out = {}
    for t in times:
        st = traj.state_at(float(t))
        sg = _reshape(grid, st.sg) if st.sg is not None else np.zeros((grid.nz, grid.ny, grid.nx))
        out[float(t) / DAY_S] = {
            "p": _reshape(grid, st.pressure) / PSI,
            "sw": _reshape(grid, st.sw),
            "so": _reshape(grid, st.so()),
            "sg": sg,
        }
    return out


def _panel(ax, field, *, cmap, vmin, vmax, title, fault_i=5.5):
    im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x 方向格子")
    ax.set_ylabel("y 方向格子")
    ax.axvline(fault_i, color="k", ls="--", lw=0.7, alpha=0.55)
    return im


def main() -> None:
    _zh_setup()
    FIG.mkdir(exist_ok=True)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    inv = json.loads((HERE / "invert_eval_report.json").read_text(encoding="utf-8"))["invert"]
    grid = _grid(truth)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    k_chan = 3
    days = [0.25, 0.50, 1.00]
    p_series = parse_grid_series(OUT, field="pressure", nx=nx, ny=ny, nz=nz)
    sw_series = parse_grid_series(OUT, field="sw", nx=nx, ny=ny, nz=nz)
    so_series = parse_grid_series(OUT, field="so", nx=nx, ny=ny, nz=nz)
    sg_series = parse_grid_series(OUT, field="sg", nx=nx, ny=ny, nz=nz)
    truth_f = {}
    for d in days:
        _, p = _pick(p_series, d)
        _, s = _pick(sw_series, d)
        _, so = _pick(so_series, d)
        _, sg = _pick(sg_series, d)
        truth_f[d] = {
            "p": _cmg_to_our(p),
            "sw": _cmg_to_our(s),
            "so": _cmg_to_our(so),
            "sg": _cmg_to_our(sg),
        }
    k_true = lib_eval._k_true(truth)
    k_post = np.load(POST)
    print("正演 F(后验) ...", flush=True)
    fpost = _simulate(grid, truth, k_post, days)
    sl = np.s_[k_chan, :, :]

    pmin, pmax = 1780.0, 3220.0
    fig, axes = plt.subplots(len(days), 6, figsize=(14.6, 2.35 * len(days)), constrained_layout=True)
    for i, d in enumerate(days):
        tp, ip = truth_f[d]["p"][sl], fpost[d]["p"][sl]
        ts, ins = truth_f[d]["sw"][sl], fpost[d]["sw"][sl]
        im0 = _panel(axes[i, 0], tp, cmap="coolwarm", vmin=pmin, vmax=pmax, title=f"{d} 天  真值压力")
        _panel(axes[i, 1], ip, cmap="coolwarm", vmin=pmin, vmax=pmax, title="反演正演压力")
        dp = tp - ip
        lim = max(20.0, float(np.nanmax(np.abs(dp))))
        im2 = _panel(axes[i, 2], dp, cmap="RdBu_r", vmin=-lim, vmax=lim, title="真值 − 反演  压力差")
        im3 = _panel(axes[i, 3], ts, cmap="YlGnBu", vmin=0.15, vmax=0.85, title="真值含水")
        _panel(axes[i, 4], ins, cmap="YlGnBu", vmin=0.15, vmax=0.85, title="反演正演含水")
        ds = ts - ins
        slim = max(0.04, float(np.nanmax(np.abs(ds))))
        im5 = _panel(axes[i, 5], ds, cmap="RdBu_r", vmin=-slim, vmax=slim, title="真值 − 反演  含水差")
    fig.colorbar(im0, ax=axes[:, 0:2], shrink=0.55, label="压力 / psi")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.55, label="压力差 / psi")
    fig.colorbar(im3, ax=axes[:, 3:5], shrink=0.55, label="含水饱和度")
    fig.colorbar(im5, ax=axes[:, 5], shrink=0.55, label="含水差")
    fig.suptitle("脱气尺子：真值（CMG）对 反演正演（通道层 xy）", fontsize=13)
    fig.savefig(FIG / "zh_真值对反演_压力含水.png")
    plt.close(fig)

    gmin, gmax = 0.0, 0.20
    fig, axes = plt.subplots(len(days), 6, figsize=(14.6, 2.35 * len(days)), constrained_layout=True)
    for i, d in enumerate(days):
        to, io_ = truth_f[d]["so"][sl], fpost[d]["so"][sl]
        tg, ig = truth_f[d]["sg"][sl], fpost[d]["sg"][sl]
        im0 = _panel(axes[i, 0], to, cmap="YlOrBr", vmin=0.40, vmax=0.85, title=f"{d} 天  真值含油")
        _panel(axes[i, 1], io_, cmap="YlOrBr", vmin=0.40, vmax=0.85, title="反演正演含油")
        do = to - io_
        olim = max(0.04, float(np.nanmax(np.abs(do))))
        im2 = _panel(axes[i, 2], do, cmap="RdBu_r", vmin=-olim, vmax=olim, title="真值 − 反演  含油差")
        im3 = _panel(axes[i, 3], tg, cmap="Purples", vmin=gmin, vmax=gmax, title="真值含气")
        _panel(axes[i, 4], ig, cmap="Purples", vmin=gmin, vmax=gmax, title="反演正演含气")
        dg = tg - ig
        glim = max(0.03, float(np.nanmax(np.abs(dg))))
        im5 = _panel(axes[i, 5], dg, cmap="RdBu_r", vmin=-glim, vmax=glim, title="真值 − 反演  含气差")
    fig.colorbar(im0, ax=axes[:, 0:2], shrink=0.55, label="含油饱和度")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.55, label="含油差")
    fig.colorbar(im3, ax=axes[:, 3:5], shrink=0.55, label="含气饱和度")
    fig.colorbar(im5, ax=axes[:, 5], shrink=0.55, label="含气差")
    fig.suptitle("含油 / 含气：初值 Sg=0，井底 1800 < 泡点 2500，气来自放气", fontsize=12)
    fig.savefig(FIG / "zh_真值对反演_含油含气.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), constrained_layout=True)
    kt = _reshape(grid, k_true) / MD_TO_M2
    kp = _reshape(grid, k_post) / MD_TO_M2
    for ax, field, title, cmap, vmin, vmax in (
        (axes[0], kt[sl], "真值渗透率  50 / 2000 md", "viridis", 30, 2100),
        (axes[1], kp[sl], "反演后验渗透率", "viridis", 30, 2100),
        (axes[2], np.log10(np.maximum(kp[sl], 1e-6) / np.maximum(kt[sl], 1e-6)), "lg(反演 / 真值)", "RdBu_r", -0.4, 0.4),
    ):
        im = _panel(ax, field, cmap=cmap, vmin=vmin, vmax=vmax, title=title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("脱气尺子渗透率：真值对反演", fontsize=13)
    fig.savefig(FIG / "zh_真值对反演_渗透率.png")
    plt.close(fig)

    g_true = inv["gauges_truth"]
    t_true = np.asarray(g_true["times_day"], dtype=float)
    gp = inv["gauges_post"]
    t_post = np.asarray(gp["times_s"], dtype=float) / DAY_S
    hold = set(inv.get("holdout") or [])
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.4), sharex=True, constrained_layout=True)
    ax0, ax1, ax3 = axes
    for name, ys in g_true["series"].items():
        mark = "（留出）" if name in hold else ""
        if name.startswith("P"):
            ax0.plot(t_true, np.asarray(ys) / PSI, "o", ms=4, label=f"{name}  真值{mark}")
            if name in gp["series"]:
                ax0.plot(t_post, np.asarray(gp["series"][name]) / PSI, "-", lw=1.3, label=f"{name}  反演")
        elif name.startswith("S"):
            ax1.plot(t_true, ys, "o", ms=4, label=f"{name}  真值{mark}")
            if name in gp["series"]:
                ax1.plot(t_post, gp["series"][name], "-", lw=1.3, label=f"{name}  反演")
        elif name.startswith("G"):
            ax3.plot(t_true, ys, "o", ms=4, label=f"{name}  真值{mark}")
            if name in gp["series"]:
                ax3.plot(t_post, gp["series"][name], "-", lw=1.3, label=f"{name}  反演")
    ax0.axvspan(min(t_true), 1.0, color="0.92")
    ax0.axhline(2500.0, color="0.4", ls=":", lw=0.8)
    ax0.set_ylabel("压力 / psi")
    ax0.set_title("脱气尺子测点：真值（CMG）对 反演正演")
    ax0.legend(ncols=2, fontsize=6.5, frameon=False)
    ax1.set_ylabel("含水饱和度")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(ncols=2, fontsize=6.5, frameon=False)
    ax3.set_ylabel("含气饱和度")
    ax3.set_xlabel("时间 / 天")
    ax3.legend(ncols=2, fontsize=6.5, frameon=False)
    fig.savefig(FIG / "zh_真值对反演_测点时序.png")
    plt.close(fig)

    rows = []
    for d in days:
        rows.append(
            [
                f"{d:.2f}",
                f"{np.sqrt(np.mean((truth_f[d]['p'] - fpost[d]['p']) ** 2)):.1f}",
                f"{np.sqrt(np.mean((truth_f[d]['sw'] - fpost[d]['sw']) ** 2)):.3f}",
                f"{np.sqrt(np.mean((truth_f[d]['so'] - fpost[d]['so']) ** 2)):.3f}",
                f"{np.sqrt(np.mean((truth_f[d]['sg'] - fpost[d]['sg']) ** 2)):.3f}",
            ]
        )
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.2), constrained_layout=True)
    for ax in axes:
        ax.axis("off")
    head_k = ["", "基质 / md", "通道 / md", "对比度", "对数K均方根误差", "留出", "预报"]
    body_k = [
        ["真值", "50", "2000", "40", "—", "—", "—"],
        [
            "反演",
            f"{inv['k_lo_md_post']:.0f}",
            f"{inv['k_hi_md_post']:.0f}",
            f"{inv['k_contrast_post']:.2f}",
            f"{inv['logk_rmse_post']:.3f}",
            f"{inv['holdout_nrmse']:.2f}",
            f"{inv['forecast_nrmse']:.2f}",
        ],
    ]
    t0 = axes[0].table(cellText=body_k, colLabels=head_k, loc="center", cellLoc="center")
    t0.auto_set_font_size(False)
    t0.set_fontsize(10)
    t0.scale(1.05, 1.9)
    for (r, c), cell in t0.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#F7F7F5")
        else:
            cell.set_facecolor("#E8F2EE")
    axes[0].set_title("脱气尺子渗透率：真值对反演", fontsize=12, pad=8)
    t1 = axes[1].table(
        cellText=rows,
        colLabels=["时间 / 天", "压力 RMSE / psi", "含水 RMSE", "含油 RMSE", "含气 RMSE"],
        loc="center",
        cellLoc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1.05, 1.8)
    for (r, c), cell in t1.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7F7F5")
    axes[1].set_title("全场：真值（CMG）对 反演正演", fontsize=12, pad=8)
    fig.savefig(FIG / "zh_真值对反演_对照表.png")
    plt.close(fig)

    payload = {
        "真值md": [50.0, 2000.0],
        "反演md": [inv["k_lo_md_post"], inv["k_hi_md_post"]],
        "对比度_反演": inv["k_contrast_post"],
        "logk_rmse": inv["logk_rmse_post"],
        "留出": inv["holdout_nrmse"],
        "预报": inv["forecast_nrmse"],
        "全场": [
            {
                "天": d,
                "压力": float(np.sqrt(np.mean((truth_f[d]["p"] - fpost[d]["p"]) ** 2))),
                "含水": float(np.sqrt(np.mean((truth_f[d]["sw"] - fpost[d]["sw"]) ** 2))),
                "含油": float(np.sqrt(np.mean((truth_f[d]["so"] - fpost[d]["so"]) ** 2))),
                "含气": float(np.sqrt(np.mean((truth_f[d]["sg"] - fpost[d]["sg"]) ** 2))),
            }
            for d in days
        ],
    }
    (HERE / "zh_compare_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("写出", [p.name for p in sorted(FIG.glob("zh_真值对反演_*.png"))])


if __name__ == "__main__":
    main()
