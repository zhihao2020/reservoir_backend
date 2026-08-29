"""CMG IMEX forward vs product F(k_post) after LM inversion (shale S1–S5).

Usage (repo root):
  python validation/shale_oil/cmg_shale_suite/plot_cmg_vs_inv.py --case S1
  python validation/shale_oil/cmg_shale_suite/plot_cmg_vs_inv.py --case all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
VAL = HERE.parent
FIG = HERE / "figures"

from build_shale_suite import CASE_DIR  # noqa: E402
from reservoir_backend.io.cmg_out import parse_grid_series, psi_to_pa  # noqa: E402
from reservoir_backend.io.shale_case import (  # noqa: E402
    frac_mask_from_truth,
    truth_half_length_m,
    twin_from_shale_truth,
)
from reservoir_backend.twin.offline import predict_from_trajectory  # noqa: E402

PSI = 6894.757293168
DAY_S = 86400.0


def _setup_cn() -> None:
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        if name in {f.name for f in font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _mid_k_layer(arr: np.ndarray, nz: int, ny: int, nx: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float).reshape(nz, ny, nx)
    return a[nz // 2]


def _run_case(
    case: str,
    *,
    n_times: int = 5,
    max_iter: int = 8,
) -> dict | None:
    case = case.upper()
    case_dir = VAL / CASE_DIR[case]
    truth_path = case_dir / f"truth_{case.lower()}.json"
    out_path = case_dir / f"mxshale_{case.lower()}.out"
    if not out_path.is_file():
        return None

    twin = twin_from_shale_truth(
        truth_path, out_path=out_path, n_times=n_times, max_iter=max_iter
    )
    from reservoir_backend.io.shale_case import _inflate_shale_sigmas
    from reservoir_backend.inverse.frac import decode_frac_theta

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    _inflate_shale_sigmas(twin, truth)
    twin.inverse.post_ensemble_enabled = False
    post = twin.calibrate(max_iter=max_iter, time_limit_s=900.0)
    g = truth["grid"]
    nx, ny, nz = int(g["nx"]), int(g["ny"]), int(g["nz"])
    eng = decode_frac_theta(twin.parameterization, post.theta)
    frac = frac_mask_from_truth(truth, twin.grid)
    mat = ~frac
    p_series = parse_grid_series(out_path, field="pressure", nx=nx, ny=ny, nz=nz)
    idx = np.unique(np.linspace(0, len(p_series) - 1, int(n_times)).astype(int))
    _, p_last_psi = p_series[int(idx[-1])]
    p_true = psi_to_pa(np.asarray(p_last_psi, dtype=float)).reshape(-1)
    dp_true = float(np.nanmean(p_true[mat]) - np.nanmean(p_true[frac]))
    p_inv_flat = post.history.states[-1].pressure
    dp_inv = float(np.mean(p_inv_flat[mat]) - np.mean(p_inv_flat[frac]))
    inv = {
        "ok": True,
        "k_frac_over_matrix": float(np.exp(post.theta[1]) / max(np.exp(post.theta[0]), 1e-30)),
        "inv_n_frac": eng.get("n_frac"),
        "truth_n_frac_planes": len(truth.get("frac_i_planes") or []),
        "dp_ratio": float(dp_inv / dp_true) if abs(dp_true) > 1.0 else None,
        "inv_x_f_m": eng.get("x_f_m"),
        "truth_x_f_m": truth_half_length_m(truth),
    }
    sw_series = parse_grid_series(out_path, field="sw", nx=nx, ny=ny, nz=nz)

    obs = twin.experiment.observations
    assim = twin.experiment.assimilate_observations()
    times_s = np.unique(np.concatenate([o.times_s for o in obs]))
    t_end = float(times_s.max())
    hist = twin.simulate(twin.rock_from_k(post.k), t_end=t_end, report_times=times_s)
    pred = predict_from_trajectory(twin.operator, twin.experiment, hist, assim)

    # pick last snapshot for field maps
    t_last_day, p_last_psi = p_series[-1]
    p_cmg_pa = psi_to_pa(np.asarray(p_last_psi, dtype=float)).reshape(nz, ny, nx)
    sw_cmg = np.asarray(sw_series[-1][1], dtype=float).reshape(nz, ny, nx)
    st_last = hist.states[-1]
    p_inv = st_last.pressure.reshape(nz, ny, nx)
    sw_inv = st_last.sw.reshape(nz, ny, nx)

    # representative perf sensors (first 4)
    names = sorted({o.sensor_name for o in assim})[:4]
    series_cmg: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    series_inv: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in names:
        o_c = next(o for o in obs if o.sensor_name == name)
        ts = np.asarray(o_c.times_s, dtype=float)
        series_cmg[name] = (ts / DAY_S, np.asarray(o_c.values, dtype=float) / PSI)
        inv_y = []
        for t in ts:
            st = hist.state_at(float(t))
            sensor = next(s for s in twin.experiment.sensors if s.name == name)
            inv_y.append(twin.operator.sample(sensor, st) / PSI)
        series_inv[name] = (ts / DAY_S, np.asarray(inv_y, dtype=float))

    return {
        "case": case,
        "inv": inv,
        "post": post,
        "t_last_day": float(t_last_day),
        "p_cmg_mid": _mid_k_layer(p_cmg_pa, nz, ny, nx) / PSI,
        "p_inv_mid": _mid_k_layer(p_inv, nz, ny, nx) / PSI,
        "sw_cmg_mid": _mid_k_layer(sw_cmg, nz, ny, nx),
        "sw_inv_mid": _mid_k_layer(sw_inv, nz, ny, nx),
        "series_cmg": series_cmg,
        "series_inv": series_inv,
        "pred_vals": pred,
        "assimilate_nrmse": float(post.assimilate_rmse),
        "nx": nx,
        "ny": ny,
    }


def plot_case(data: dict, dest: Path) -> Path:
    _setup_cn()
    case = data["case"]
    inv = data["inv"]
    fig = plt.figure(figsize=(13.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 1.0])

    ax_ts = fig.add_subplot(gs[0, :])
    for name in data["series_cmg"]:
        td, yc = data["series_cmg"][name]
        _, yi = data["series_inv"][name]
        ax_ts.plot(td, yc, "o-", ms=4, lw=1.2, label=f"{name} CMG")
        ax_ts.plot(td, yi, "--", lw=1.4, label=f"{name} F(k_post)")
    ax_ts.set_xlabel("时间 (day)")
    ax_ts.set_ylabel("完井压力 (psi)")
    ax_ts.set_title(
        f"{case}  CMG 正演 vs 反演后产品正演 F(k_post)  |  "
        f"assimilate nRMSE={data['assimilate_nrmse']:.2f}  "
        f"dp_ratio={inv.get('dp_ratio', float('nan')):.3f}"
    )
    ax_ts.legend(ncols=2, fontsize=7, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    p_vmin = float(min(data["p_cmg_mid"].min(), data["p_inv_mid"].min()))
    p_vmax = float(max(data["p_cmg_mid"].max(), data["p_inv_mid"].max()))
    for col, (fld, title) in enumerate(
        (
            (data["p_cmg_mid"], f"CMG 压力 (mid-k, t={data['t_last_day']:.0f} d)"),
            (data["p_inv_mid"], f"F(k_post) 压力 (mid-k)"),
        )
    ):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(fld, origin="lower", aspect="auto", cmap="coolwarm", vmin=p_vmin, vmax=p_vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        plt.colorbar(im, ax=ax, fraction=0.046, label="psi")

    sw_vmax = max(float(data["sw_cmg_mid"].max()), float(data["sw_inv_mid"].max()), 0.25)
    for col, (fld, title) in enumerate(
        (
            (data["sw_cmg_mid"], "CMG Sw (mid-k)"),
            (data["sw_inv_mid"], "F(k_post) Sw (mid-k)"),
        )
    ):
        ax = fig.add_subplot(gs[2, col])
        im = ax.imshow(fld, origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=sw_vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Sw")

    fig.suptitle(
        f"页岩 {case}：IMEX 尺子 vs LM frac-θ 反演后正演\n"
        f"k_frac/k_mat={inv.get('k_frac_over_matrix', float('nan')):.1f}  "
        f"n_frac inv/truth={inv.get('inv_n_frac')}/{inv.get('truth_n_frac_planes')}",
        fontsize=12,
        fontweight="bold",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="S1", help="S1..S5 or all")
    ap.add_argument("--n-times", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--out-dir", default=str(FIG))
    args = ap.parse_args(argv)
    cases = list(CASE_DIR) if str(args.case).lower() == "all" else [str(args.case).upper()]
    out_dir = Path(args.out_dir)
    written: list[Path] = []
    for case in cases:
        print(f"=== {case} ===", flush=True)
        data = _run_case(case, n_times=int(args.n_times), max_iter=int(args.max_iter))
        if data is None:
            print(f"  skip {case} (missing .out or invert failed)")
            continue
        path = plot_case(data, out_dir / f"{case.lower()}_cmg_vs_fpost.png")
        meta = {
            "case": case,
            "assimilate_nrmse": data["assimilate_nrmse"],
            "dp_ratio": data["inv"].get("dp_ratio"),
            "figure": path.name,
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        written.append(path)
        print(f"  wrote {path}", flush=True)

    md = out_dir / "README.md"
    lines = [
        "# 页岩 CMG vs 反演后正演",
        "",
        "| 图 | 算例 | assimilate nRMSE | dp_ratio |",
        "|----|------|------------------|----------|",
    ]
    for p in written:
        j = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        lines.append(
            f"| `{p.name}` | {j['case']} | {j['assimilate_nrmse']:.3f} | {j.get('dp_ratio')} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done: {len(written)} figures → {out_dir}")
    return 0 if written else 2


if __name__ == "__main__":
    raise SystemExit(main())
