"""CMG forward vs sensor inversion visual comparison.

Reuses the virtual-probe study path (wells + exclusive probes only; no CMG at runtime).

Usage (repo root):
  python validation/cmg_probe_study/plot_cmg_vs_inv.py
  python validation/cmg_probe_study/plot_cmg_vs_inv.py --cases channel --n-list 8,12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.pipeline import (  # noqa: E402
    WellPoint,
    place_uniform_probes,
    recommend_probes,
    run_time_series,
    split_n_probes,
)
from reservoir_backend.pipeline.probe_design import field_variance_over_time  # noqa: E402
from validation.cmg_io.grid_parse import parse_grid_series  # noqa: E402
from validation.cmg_probe_study.run_probe_study import (  # noqa: E402
    build_samples,
    dice_masks,
    mesh_with_probes,
    rel_l2,
    truth_mask,
)

HERE = Path(__file__).resolve().parent
CHANNEL = ROOT / "validation" / "cmg_channel_3d"
FAULT = ROOT / "validation" / "cmg_fault_3d"
CHANNEL_FINE = ROOT / "validation" / "cmg_channel_fine"
FIVESPOT = ROOT / "validation" / "cmg_fivespot"
OUT_DIR = HERE / "figures"


def _setup_cn_font() -> None:
    """Prefer a CJK-capable UI font so titles/tables render in Chinese."""
    from matplotlib import font_manager

    candidates = (
        "Microsoft YaHei",
        "SimHei",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "DengXian",
        "SimSun",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


CASE_CN = {
    "cmg_undulating_channel": "起伏通道",
    "cmg_faulted_dogleg": "断层狗腿",
}


def _layer_mean(arr: np.ndarray) -> np.ndarray:
    """Average over k (vertical) → (ny, nx) for map view. arr is (nz, ny, nx)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return a
    return np.nanmean(a, axis=0)


def _mid_layer(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return a
    return a[a.shape[0] // 2]


def _run_inversion(
    case_name: str,
    case_dir: Path,
    *,
    n_total: int,
    layout: str,
) -> dict | None:
    truth_files = list(case_dir.glob("truth_*.json"))
    if not truth_files:
        return None
    meta = json.loads(truth_files[0].read_text(encoding="utf-8"))
    out_path = next(case_dir.glob("*.out"), None)
    if out_path is None or not out_path.is_file():
        return None

    g = meta["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    sw_series = parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)
    p_series_psi = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    if len(sw_series) < 2 or len(p_series_psi) < 1:
        return None
    p_series_pa = [
        (t, np.asarray(arr, dtype=float) * 6894.757293168) for t, arr in p_series_psi
    ]

    base_mesh, ik, pk = mesh_with_probes(meta, [])
    n_p, n_s = split_n_probes(n_total)
    max_n = max(0, min(48, base_mesh.n_cells // 8))
    if n_p + n_s > max_n:
        scale = max_n / max(n_p + n_s, 1)
        n_p = int(n_p * scale)
        n_s = int(n_s * scale)
        n_p, n_s = split_n_probes(n_p + n_s)

    if n_p + n_s == 0:
        probe_pts: list[WellPoint] = []
        layout_used = "wells_only"
    elif layout == "uniform":
        specs = place_uniform_probes(base_mesh, n_p, n_s)
        probe_pts = [WellPoint(**s.as_well_point_kwargs()) for s in specs]
        layout_used = "uniform"
    else:
        var_p = field_variance_over_time(p_series_pa)
        var_s = field_variance_over_time(sw_series)
        specs = recommend_probes(
            base_mesh,
            n_p=n_p,
            n_s=n_s,
            mode="hybrid",
            prior_var_p=var_p,
            prior_var_s=var_s,
            seed=7,
        )
        probe_pts = [WellPoint(**s.as_well_point_kwargs()) for s in specs]
        layout_used = "adaptive"

    mesh, ik, pk = mesh_with_probes(meta, probe_pts)
    samples = build_samples(meta, mesh, out_path, ik, pk, sw_series, p_series_pa)
    if len(samples) < 2:
        return None

    k_bg = float(meta.get("background_perm_md", {}).get("kx", 50.0))
    k_prior = k_bg * 9.869233e-16
    n_p_obs = sum(
        1
        for n, r in mesh.well_role.items()
        if r in ("injector", "producer", "observer_p")
        and n in (samples[0].well_pressure or {})
    )
    # Product-default point-first path (no case-specific channel prior).
    _ = n_p_obs
    history = run_time_series(
        mesh,
        samples,
        permeability_prior_m2=k_prior,
        porosity_prior=0.3,
        viscosity_pa_s=1.0e-3,
        n_k_iterations=2 if (n_p + n_s) < 4 else 3,
        assimilate_k=False,
    )

    t0, sw0_c = sw_series[0]
    t1, sw1_c = sw_series[-1]
    # nearest CMG pressure to last Sw time
    p_last_c = min(p_series_pa, key=lambda x: abs(x[0] - t1))[1]

    last = history[-1]
    first = history[0]
    mask = truth_mask(meta)
    dsw_c = np.abs(np.asarray(sw1_c, dtype=float) - np.asarray(sw0_c, dtype=float))
    dsw_s = np.abs(np.asarray(last.sw, dtype=float) - np.asarray(first.sw, dtype=float))
    thr_c = max(float(np.quantile(dsw_c, 0.75)), 1e-4)
    thr_s = max(float(np.quantile(dsw_s, 0.75)), 1e-4)
    dice = dice_masks(dsw_s >= thr_s, dsw_c >= thr_c)
    sw_l2 = rel_l2(last.sw, sw1_c)
    k = last.permeability
    k_ch = float(np.mean(k[mask])) if np.any(mask) else float("nan")
    k_out = float(np.mean(k[~mask])) if np.any(~mask) else float("nan")
    k_ratio = k_ch / k_out if k_out and k_out > 0 else float("nan")

    # probe / well markers in cell i,j
    markers = []
    for name, cid in mesh.well_cell_id.items():
        i, j, kk = mesh.grid.ijk(cid)
        role = mesh.well_role.get(name, "")
        markers.append((name, role, i, j))

    return {
        "case": case_name,
        "layout": layout_used,
        "n_total": n_total,
        "n_p": n_p,
        "n_s": n_s,
        "sw_cmg": np.asarray(sw1_c, dtype=float),
        "sw_inv": np.asarray(last.sw, dtype=float),
        "dsw_cmg": dsw_c,
        "dsw_inv": dsw_s,
        "p_cmg": np.asarray(p_last_c, dtype=float),
        "p_inv": np.asarray(last.pressure, dtype=float),
        "k_inv": np.asarray(k, dtype=float),
        "mask": mask,
        "markers": markers,
        "metrics": {
            "sw_rel_l2": sw_l2,
            "delta_sw_dice": dice,
            "k_channel_over_out": k_ratio,
            "t_last": float(t1),
        },
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


def _add_markers(ax, markers, color_map=None):
    color_map = color_map or {
        "injector": "cyan",
        "producer": "lime",
        "observer_p": "white",
        "observer_s": "yellow",
        "observer": "orange",
    }
    for name, role, i, j in markers:
        # i,j are 0-based cell indices; map coords = cell centers in index space
        c = color_map.get(role, "w")
        m = "^" if role == "injector" else "v" if role == "producer" else "o"
        ax.plot(i, j, m, color=c, markersize=7, markeredgecolor="k", markeredgewidth=0.6)
        if role in ("injector", "producer"):
            ax.text(i + 0.3, j + 0.3, name, color="w", fontsize=7, fontweight="bold")


def _imshow(ax, data, title, cmap, vmin=None, vmax=None, cbar_label=""):
    im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("网格 i")
    ax.set_ylabel("网格 j")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return im


def plot_bundle(data: dict, out_path: Path) -> Path:
    """Write multi-panel CMG vs inversion comparison figure."""
    _setup_cn_font()
    m = data["metrics"]
    # vertical mean for robust map view
    sw_c = _layer_mean(data["sw_cmg"])
    sw_i = _layer_mean(data["sw_inv"])
    dsw_c = _layer_mean(data["dsw_cmg"])
    dsw_i = _layer_mean(data["dsw_inv"])
    p_c = _layer_mean(data["p_cmg"]) / 1.0e6  # MPa
    p_i = _layer_mean(data["p_inv"]) / 1.0e6
    k_i = _layer_mean(data["k_inv"]) / 9.869233e-16  # mD
    mask2 = _layer_mean(data["mask"].astype(float)) >= 0.5

    sw_err = np.abs(sw_i - sw_c)
    p_err = np.abs(p_i - p_c)
    thr_c = max(float(np.quantile(dsw_c, 0.75)), 1e-4)
    thr_i = max(float(np.quantile(dsw_i, 0.75)), 1e-4)
    m_c = dsw_c >= thr_c
    m_i = dsw_i >= thr_i
    # RGB overlay: R=CMG only, G=both, B=inv only
    overlay = np.zeros((*m_c.shape, 3), dtype=float)
    overlay[m_c & ~m_i] = (0.90, 0.25, 0.20)  # CMG only
    overlay[m_i & ~m_c] = (0.20, 0.45, 0.95)  # inv only
    overlay[m_c & m_i] = (0.20, 0.75, 0.35)  # both

    sw_vmin = float(min(sw_c.min(), sw_i.min()))
    sw_vmax = float(max(sw_c.max(), sw_i.max()))
    dsw_vmax = float(max(dsw_c.max(), dsw_i.max(), 1e-6))
    p_vmin = float(min(p_c.min(), p_i.min()))
    p_vmax = float(max(p_c.max(), p_i.max()))
    k_vmax = float(np.percentile(k_i, 99))

    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11.5), constrained_layout=True)
    case_cn = CASE_CN.get(str(data["case"]), str(data["case"]))
    layout_cn = {"uniform": "均匀布置", "adaptive": "自适应布置", "wells_only": "仅井点"}.get(
        str(data["layout"]), str(data["layout"])
    )
    fig.suptitle(
        f"{case_cn}  |  测点数 N={data['n_total']}（压力 {data['n_p']} / 饱和度 {data['n_s']}）"
        f"{layout_cn}\n"
        f"含水饱和度相对 L2={m['sw_rel_l2']:.3f}   ΔSw 重合 Dice={m['delta_sw_dice']:.3f}   "
        f"通道内外渗透率比={m['k_channel_over_out']:.2f}   时刻 t={m['t_last']:.0f} 天",
        fontsize=12,
        fontweight="bold",
    )

    # Row 0: Sw
    _imshow(axes[0, 0], sw_c, "CMG 含水饱和度（层平均）", "YlGnBu", sw_vmin, sw_vmax, "含水饱和度")
    _imshow(axes[0, 1], sw_i, "反演含水饱和度", "YlGnBu", sw_vmin, sw_vmax, "含水饱和度")
    _imshow(axes[0, 2], sw_err, "|反演 − CMG| 含水饱和度", "magma", 0.0, None, "|Δ含水饱和度|")
    for ax in axes[0]:
        _add_markers(ax, data["markers"])

    # Row 1: ΔSw footprint
    _imshow(axes[1, 0], dsw_c, "CMG |Δ含水饱和度|（初→末）", "OrRd", 0.0, dsw_vmax, "|Δ含水饱和度|")
    _imshow(axes[1, 1], dsw_i, "反演 |Δ含水饱和度|（初→末）", "OrRd", 0.0, dsw_vmax, "|Δ含水饱和度|")
    axes[1, 2].imshow(overlay, origin="lower", aspect="equal", interpolation="nearest")
    axes[1, 2].set_title("Δ含水饱和度足迹叠合\n（绿=一致，红=仅CMG，蓝=仅反演）")
    axes[1, 2].set_xlabel("网格 i")
    axes[1, 2].set_ylabel("网格 j")
    # truth channel contour on ΔSw panels
    for ax in (axes[1, 0], axes[1, 1], axes[1, 2]):
        if np.any(mask2):
            ax.contour(
                mask2.astype(float),
                levels=[0.5],
                colors=["k"],
                linewidths=0.9,
                origin="lower",
            )
        _add_markers(ax, data["markers"])

    # Row 2: pressure + k
    _imshow(axes[2, 0], p_c, "CMG 压力", "coolwarm", p_vmin, p_vmax, "MPa")
    _imshow(axes[2, 1], p_i, "反演压力", "coolwarm", p_vmin, p_vmax, "MPa")
    _imshow(axes[2, 2], k_i, "反演渗透率 (mD)\n白线为真值通道轮廓", "viridis", None, k_vmax, "mD")
    if np.any(mask2):
        axes[2, 2].contour(
            mask2.astype(float),
            levels=[0.5],
            colors=["w"],
            linewidths=1.0,
            origin="lower",
        )
    for ax in axes[2]:
        _add_markers(ax, data["markers"])

    # legend strip
    fig.text(
        0.5,
        0.01,
        "标记：▲注入井  ▼生产井  ○压力测点（白）/ 饱和度测点（黄）  |  "
        "黑/白线 = CMG 算例中的真值通道",
        ha="center",
        fontsize=8,
        color="0.25",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot CMG forward vs inversion maps")
    ap.add_argument("--cases", default="channel", help="channel,fault")
    ap.add_argument("--n-list", default="8,12", help="total exclusive probes")
    ap.add_argument("--layout", default="uniform", choices=["uniform", "adaptive"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    _setup_cn_font()

    case_map = {
        "channel": ("cmg_undulating_channel", CHANNEL),
        "fault": ("cmg_faulted_dogleg", FAULT),
        "channel_fine": ("cmg_undulating_channel_fine", CHANNEL_FINE),
        "fivespot": ("cmg_fivespot", FIVESPOT),
    }
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key in cases:
        if key not in case_map:
            print(f"skip unknown case {key}")
            continue
        cname, cdir = case_map[key]
        for n in n_list:
            print(f"running {cname} N={n} ...")
            data = _run_inversion(cname, cdir, n_total=n, layout=args.layout)
            if data is None:
                print(f"  failed {cname} N={n}")
                continue
            fname = f"{key}_N{n}_{args.layout}_cmg_vs_inv.png"
            path = plot_bundle(data, out_dir / fname)
            # also dump metrics json next to figure
            meta_path = path.with_suffix(".json")
            meta_path.write_text(
                json.dumps(
                    {
                        "case": data["case"],
                        "n_total": data["n_total"],
                        "n_p": data["n_p"],
                        "n_s": data["n_s"],
                        "layout": data["layout"],
                        **data["metrics"],
                        "figure": str(path.name),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            written.append(path)
            print(
                f"  wrote {path.name}  "
                f"SwL2={data['metrics']['sw_rel_l2']:.3f} "
                f"Dice={data['metrics']['delta_sw_dice']:.3f} "
                f"k={data['metrics']['k_channel_over_out']:.2f}"
            )

    # short index markdown
    md = out_dir / "README.md"
    lines = [
        "# CMG 正演 vs 反演对比图",
        "",
        "由 `plot_cmg_vs_inv.py` 生成。测点仅 wells + virtual exclusive probes。",
        "",
        "| 图件 | 算例 | 测点数 | 含水饱和度相对L2 | ΔSw重合Dice | 通道内外渗透率比 |",
        "|------|------|--------|------------------|-------------|------------------|",
    ]
    for p in written:
        j = p.with_suffix(".json")
        if j.is_file():
            meta = json.loads(j.read_text(encoding="utf-8"))
            cname = CASE_CN.get(str(meta["case"]), str(meta["case"]))
            lines.append(
                f"| `{p.name}` | {cname} | {meta['n_total']} | "
                f"{meta['sw_rel_l2']:.3f} | {meta['delta_sw_dice']:.3f} | "
                f"{meta['k_channel_over_out']:.2f} |"
            )
        else:
            lines.append(f"| `{p.name}` |  |  |  |  |  |")
    lines.append("")
    lines.append("## 读图")
    lines.append("")
    lines.append("- **第1行**：末时刻含水饱和度（CMG / 反演 / 绝对误差）")
    lines.append("- **第2行**：多时刻 |Δ含水饱和度| 足迹与重叠（绿=一致，红=仅CMG，蓝=仅反演）")
    lines.append("- **第3行**：压力场 + 反演渗透率（白线为 CMG 算例真通道轮廓）")
    lines.append("- 标记：▲注入井 ▼生产井 ○ 互斥测点")
    lines.append("- 反演路径为产品默认的**点优先**（不为单个工况更换先验）")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"index {md}")
    print(f"done: {len(written)} figures → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
