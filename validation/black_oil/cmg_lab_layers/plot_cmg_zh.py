"""CMG 对照图：中文表头。不重跑 IMEX。"""

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
from plot_cmg_inv_fields import _pick, _reshape, _simulate
from run_invert_eval import DAY_S, MD_TO_M2, OUT, TRUTH, _cmg_to_our, _grid

FIG = HERE / "figures"
PSI = 6894.757293168

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
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


def _panel(ax, field, *, cmap, vmin, vmax, title, xlabel="x 方向格子", ylabel="z 方向（向上）"):
    im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def main() -> None:
    _zh_setup()
    FIG.mkdir(exist_ok=True)
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    report = json.loads((HERE / "invert_eval_report.json").read_text(encoding="utf-8"))
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
    print("正演 F(后验A)、F(后验B) ...", flush=True)
    fa = _simulate(grid, truth, k_a, days)
    fb = _simulate(grid, truth, k_b, days)
    ftrue = _simulate(grid, truth, k_true, days)

    pmin, pmax = 2780.0, 3220.0
    smin, smax = 0.18, 0.82

    fig, axes = plt.subplots(len(days), 6, figsize=(14.6, 2.25 * len(days)), constrained_layout=True)
    for i, d in enumerate(days):
        sl = np.s_[:, jmid, :]
        cp, ap, bp = cmg[d]["p"][sl], fa[d]["p"][sl], fb[d]["p"][sl]
        cs, asw, bsw = cmg[d]["sw"][sl], fa[d]["sw"][sl], fb[d]["sw"][sl]
        im0 = _panel(axes[i, 0], cp, cmap="coolwarm", vmin=pmin, vmax=pmax, title=f"{d} 天  CMG 压力")
        _panel(axes[i, 1], ap, cmap="coolwarm", vmin=pmin, vmax=pmax, title="协议A  反演正演压力")
        dp = cp - ap
        lim = max(20.0, float(np.nanmax(np.abs(dp))))
        im2 = _panel(axes[i, 2], dp, cmap="RdBu_r", vmin=-lim, vmax=lim, title="CMG − 协议A  压力差")
        im3 = _panel(axes[i, 3], cs, cmap="YlGnBu", vmin=smin, vmax=smax, title="CMG 含水饱和度")
        _panel(axes[i, 4], bsw, cmap="YlGnBu", vmin=smin, vmax=smax, title="协议B  反演正演含水")
        ds = cs - bsw
        slim = max(0.05, float(np.nanmax(np.abs(ds))))
        im5 = _panel(axes[i, 5], ds, cmap="RdBu_r", vmin=-slim, vmax=slim, title="CMG − 协议B  含水差")
        for ax in axes[i, :]:
            ax.axhline(nz / 2 - 0.5, color="k", ls="--", lw=0.5, alpha=0.4)
    fig.colorbar(im0, ax=axes[:, 0:2], shrink=0.55, label="压力 / psi")
    fig.colorbar(im2, ax=axes[:, 2], shrink=0.55, label="压力差 / psi")
    fig.colorbar(im3, ax=axes[:, 3:5], shrink=0.55, label="含水饱和度")
    fig.colorbar(im5, ax=axes[:, 5], shrink=0.55, label="含水差")
    fig.suptitle("全场对照（中间 y 切片）：CMG 仿真 对 实验室反演正演", fontsize=13)
    fig.savefig(FIG / "zh_场图_压力含水.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2), constrained_layout=True)
    kt = _reshape(grid, k_true) / MD_TO_M2
    ka = _reshape(grid, k_a) / MD_TO_M2
    kb = _reshape(grid, k_b) / MD_TO_M2
    sl = np.s_[:, jmid, :]
    for ax, field, title, cmap, vmin, vmax in (
        (axes[0], kt[sl], "CMG / 真值渗透率", "viridis", 30, 700),
        (axes[1], ka[sl], "协议A 后验渗透率", "viridis", 30, 700),
        (axes[2], kb[sl], "协议B 等效实验室渗透率", "viridis", 30, 700),
        (axes[3], np.log10(np.maximum(ka[sl], 1e-6) / np.maximum(kt[sl], 1e-6)), "协议A / 真值  lg(K比)", "RdBu_r", -0.3, 0.3),
    ):
        im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x 方向格子")
        ax.set_ylabel("z 方向（向上）")
        ax.axhline(nz / 2 - 0.5, color="w", ls="--", lw=0.6)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("渗透率场对照（中间 y 切片）", fontsize=13)
    fig.savefig(FIG / "zh_场图_渗透率.png")
    plt.close(fig)

    a = report["A_self_consistent_diverse"]
    b = report["B_cross_cmg_observations"]
    g_cmg = b["gauges_cmg"]
    g_post = b["gauges_post"]
    t_cmg = np.asarray(g_cmg["times_day"], dtype=float)
    t_post = np.asarray(g_post["times_s"], dtype=float) / DAY_S
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    p_names = [n for n in g_cmg["series"] if n.startswith("P")]
    s_names = [n for n in g_cmg["series"] if n.startswith("S")]
    hold = set(b.get("holdout") or [])
    for name in p_names:
        mark = "（留出）" if name in hold else ""
        ax0.plot(t_cmg, np.asarray(g_cmg["series"][name]) / PSI, "o", ms=5, label=f"{name}  CMG{mark}")
        if name in g_post["series"]:
            ax0.plot(t_post, np.asarray(g_post["series"][name]) / PSI, "-", lw=1.4, label=f"{name}  反演正演")
    for name in s_names:
        mark = "（留出）" if name in hold else ""
        ax1.plot(t_cmg, g_cmg["series"][name], "o", ms=5, label=f"{name}  CMG{mark}")
        if name in g_post["series"]:
            ax1.plot(t_post, g_post["series"][name], "-", lw=1.4, label=f"{name}  反演正演")
    ax0.axvspan(0.25, 1.0, color="0.90", label="同化窗口")
    ax1.axvspan(0.25, 1.0, color="0.90")
    ax0.set_ylabel("压力 / psi")
    ax0.set_title("协议B：CMG 测点 对 实验室 F(后验K)")
    ax0.legend(ncols=2, fontsize=7, frameon=False)
    ax1.set_xlabel("时间 / 天")
    ax1.set_ylabel("含水饱和度")
    ax1.legend(ncols=2, fontsize=7, frameon=False)
    fig.savefig(FIG / "zh_测点时序_协议B.png")
    plt.close(fig)

    if "gauges_truth" in a and "gauges_post" in a:
        gt, gp = a["gauges_truth"], a["gauges_post"]
        tt = np.asarray(gt["times_s"], dtype=float) / DAY_S
        tp = np.asarray(gp["times_s"], dtype=float) / DAY_S
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
        for name, ys in gt["series"].items():
            if name.startswith("P"):
                ax0.plot(tt, np.asarray(ys) / PSI, "o", ms=4, label=f"{name}  真值正演")
                if name in gp["series"]:
                    ax0.plot(tp, np.asarray(gp["series"][name]) / PSI, "-", lw=1.4, label=f"{name}  后验正演")
            elif name.startswith("S"):
                ax1.plot(tt, ys, "o", ms=4, label=f"{name}  真值正演")
                if name in gp["series"]:
                    ax1.plot(tp, gp["series"][name], "-", lw=1.4, label=f"{name}  后验正演")
        ax0.set_ylabel("压力 / psi")
        ax0.set_title("协议A（自洽）：本正演真值 对 反演后正演")
        ax0.legend(ncols=2, fontsize=7, frameon=False)
        ax1.set_xlabel("时间 / 天")
        ax1.set_ylabel("含水饱和度")
        ax1.legend(ncols=2, fontsize=7, frameon=False)
        fig.savefig(FIG / "zh_测点时序_协议A.png")
        plt.close(fig)

    rows_field = []
    for d in days:
        pr_a = float(np.sqrt(np.mean((cmg[d]["p"] - fa[d]["p"]) ** 2)))
        pr_b = float(np.sqrt(np.mean((cmg[d]["p"] - fb[d]["p"]) ** 2)))
        pr_m = float(np.sqrt(np.mean((cmg[d]["p"] - ftrue[d]["p"]) ** 2)))
        sr_a = float(np.sqrt(np.mean((cmg[d]["sw"] - fa[d]["sw"]) ** 2)))
        sr_b = float(np.sqrt(np.mean((cmg[d]["sw"] - fb[d]["sw"]) ** 2)))
        sr_m = float(np.sqrt(np.mean((cmg[d]["sw"] - ftrue[d]["sw"]) ** 2)))
        rows_field.append(
            [
                f"{d:.2f}",
                f"{pr_m:.0f}",
                f"{pr_a:.0f}",
                f"{pr_b:.0f}",
                f"{sr_m:.3f}",
                f"{sr_a:.3f}",
                f"{sr_b:.3f}",
            ]
        )

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 5.6), constrained_layout=True)
    for ax in axes:
        ax.axis("off")
    head_inv = ["方案", "层对比度（真值10）", "后验渗透率 / md", "对数K均方根误差", "同化残差", "留出残差", "预报残差"]
    body_inv = [
        [
            "协议A  自洽（本正演）",
            f"{a['k_contrast_post']:.2f}",
            f"{a['k_lo_md_post']:.0f} / {a['k_hi_md_post']:.0f}",
            f"{a['logk_rmse_post']:.3f}",
            f"{a['assimilate_nrmse']:.2f}",
            f"{a['holdout_nrmse']:.2f}",
            f"{a['forecast_nrmse']:.2f}",
        ],
        [
            "协议B  CMG测点（等效K）",
            f"{b['k_contrast_post']:.2f}（不收CMG的K）",
            f"{b['k_lo_md_post']:.0f} / {b['k_hi_md_post']:.0f}",
            "不对齐格子K",
            f"{b['assimilate_nrmse']:.2f}",
            f"{b['holdout_nrmse']:.2f}",
            f"{b['forecast_nrmse']:.2f}",
        ],
    ]
    t0 = axes[0].table(cellText=body_inv, colLabels=head_inv, loc="center", cellLoc="center")
    t0.auto_set_font_size(False)
    t0.set_fontsize(9)
    t0.scale(1.05, 1.85)
    for (r, c), cell in t0.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            cell.set_facecolor("#E8F2EE")
        else:
            cell.set_facecolor("#F7F7F5")
    axes[0].set_title("反演指标（产品尺子：测点拟合，不是格子K对齐CMG）", fontsize=12, pad=8)

    head_f = ["时间 / 天", "压力RMSE 同岩石", "压力RMSE 协议A", "压力RMSE 协议B", "含水RMSE 同岩石", "含水RMSE 协议A", "含水RMSE 协议B"]
    t1 = axes[1].table(cellText=rows_field, colLabels=head_f, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1.05, 1.85)
    for (r, c), cell in t1.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F7F7F5")
    axes[1].set_title("全场对CMG的均方根误差（压力单位 psi；同岩石 = 用50/500 md 正演）", fontsize=12, pad=8)
    fig.savefig(FIG / "zh_对照表.png")
    plt.close(fig)

    payload = {
        "A": {
            "对比度": a["k_contrast_post"],
            "后验md": [a["k_lo_md_post"], a["k_hi_md_post"]],
            "logk_rmse": a["logk_rmse_post"],
            "同化": a["assimilate_nrmse"],
            "留出": a["holdout_nrmse"],
            "预报": a["forecast_nrmse"],
        },
        "B": {
            "对比度": b["k_contrast_post"],
            "后验md": [b["k_lo_md_post"], b["k_hi_md_post"]],
            "同化": b["assimilate_nrmse"],
            "留出": b["holdout_nrmse"],
            "预报": b["forecast_nrmse"],
        },
        "全场对CMG": [
            {
                "天": d,
                "压力_同岩石": float(np.sqrt(np.mean((cmg[d]["p"] - ftrue[d]["p"]) ** 2))),
                "压力_A": float(np.sqrt(np.mean((cmg[d]["p"] - fa[d]["p"]) ** 2))),
                "压力_B": float(np.sqrt(np.mean((cmg[d]["p"] - fb[d]["p"]) ** 2))),
                "含水_同岩石": float(np.sqrt(np.mean((cmg[d]["sw"] - ftrue[d]["sw"]) ** 2))),
                "含水_A": float(np.sqrt(np.mean((cmg[d]["sw"] - fa[d]["sw"]) ** 2))),
                "含水_B": float(np.sqrt(np.mean((cmg[d]["sw"] - fb[d]["sw"]) ** 2))),
            }
            for d in days
        ],
        "图": [
            "zh_场图_压力含水.png",
            "zh_场图_渗透率.png",
            "zh_测点时序_协议A.png",
            "zh_测点时序_协议B.png",
            "zh_对照表.png",
        ],
    }
    (HERE / "zh_compare_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("写出", [p.name for p in sorted(FIG.glob("zh_*.png"))])


if __name__ == "__main__":
    main()
