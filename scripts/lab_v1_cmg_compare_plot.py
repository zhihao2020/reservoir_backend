"""GEM hidden vs F_ours(theta_true) maps.

Default: M2 4×4×2 DPDP (lab_v1_dev).
Pass --case for physical_3d (15³ single-porosity compositional).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reservoir_backend.twin.cmg_benchmark import (
    PRESSURE_SPAN_FLOOR_PA,
    _k_to_our,
    forward_at_theta,
    gem_kdir_down,
    load_hidden_truth,
    load_twin_case,
    nrmse_range,
    rmse,
    theta_true_from_spec,
    theta_true_from_twin,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1


def _setup_chinese_font() -> None:
    """Use a Windows CJK face so titles and axis labels render."""
    import matplotlib as mpl
    from matplotlib import font_manager

    candidates = ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "STSong")
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return


def _layer(field: np.ndarray, nx: int, ny: int, nz: int, k: int) -> np.ndarray:
    cube = np.asarray(field, dtype=float).reshape(nz, ny, nx)
    return cube[k]


def _xz_gem_k(field: np.ndarray, nx: int, ny: int, nz: int, j: int) -> np.ndarray:
    """xz slice with GEM *KDIR DOWN: row 0 (plot top) is k=1 at the top."""
    cube = np.asarray(field, dtype=float).reshape(nz, ny, nx)
    return cube[::-1, int(j), :]


def _maps(
    ax,
    arr: np.ndarray,
    title: str,
    *,
    vmin,
    vmax,
    cmap: str,
    cbar: str | None,
    xlabel="I（网格列）",
    ylabel="J（网格行）",
    origin: str = "lower",
    yticks=None,
    yticklabels=None,
) -> None:
    im = ax.imshow(arr, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if yticks is not None:
        ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar, fontsize=8)
        try:
            cb.formatter.set_useOffset(False)
            cb.update_ticks()
        except AttributeError:
            pass


def _monitor_k_our(case: Path | None, nz: int) -> list[tuple[str, int]]:
    """Schematic monitor layers as our 0-based k (z up)."""
    down = gem_kdir_down(case)
    gem_ks = [5, 11]
    spec = None
    if case is not None:
        spec_path = Path(case).parent / "spec.yaml"
        if spec_path.is_file():
            import yaml

            spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
            mon = spec.get("monitor") or {}
            found = []
            for name, row in mon.items():
                if isinstance(row, dict) and row.get("k") is not None:
                    found.append(int(row["k"]))
            if found:
                gem_ks = found
    out = []
    for gem_k in gem_ks:
        k0 = _k_to_our(int(gem_k), nz, kdir_down=down)
        k0 = int(np.clip(k0, 0, nz - 1))
        out.append((f"监测层 k={gem_k}", k0))
    return out


def _injector_j(twin) -> int:
    for port in twin.ports:
        if str(port.name).upper() == "INJ" and port.cell_ids.size:
            _i, j, _k = twin.grid.ijk(int(port.cell_ids[0]))
            return int(j)
    return twin.grid.ny // 2


def _cap_times(times: np.ndarray, t_end: float | None, n_cells: int) -> np.ndarray:
    t = np.asarray(times, dtype=float).ravel()
    t = t[np.isfinite(t)]
    cap = t_end
    if cap is None and n_cells >= 1000:
        # Default to the first actual GEM report.
        cap = float(t[t > 0.0][0]) if np.any(t > 0.0) else 864.0
    if cap is None:
        return t
    cap = float(cap)
    kept = t[t <= cap + 1.0e-9]
    if kept.size == 0 or float(np.max(kept)) <= 1.0e-12:
        extra = [0.0] if t.size and abs(float(t[0])) <= 1.0e-12 else []
        extra.append(cap)
        kept = np.asarray(extra, dtype=float)
    return np.unique(kept)


def _forward_capped(twin, theta, times: np.ndarray) -> tuple[dict, np.ndarray, bool]:
    times = np.asarray(times, dtype=float).ravel()
    if times.size == 0:
        times = np.array([8.64])
    ours = forward_at_theta(twin, theta, times)
    return ours, times, False


def _field_at(times: np.ndarray, arr: np.ndarray | None, t: float) -> np.ndarray | None:
    """Linear sample of a (n_times, n_cells) GEM field at t. None stays None."""
    if arr is None:
        return None
    times = np.asarray(times, dtype=float).ravel()
    field = np.asarray(arr, dtype=float)
    if field.ndim == 1:
        field = field.reshape(1, -1)
    if times.size == 0:
        return field[-1]
    if t <= float(times[0]) or times.size == 1:
        return np.asarray(field[0], dtype=float)
    if t >= float(times[-1]):
        return np.asarray(field[-1], dtype=float)
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = max(0, min(i, times.size - 2))
    dt = float(times[i + 1] - times[i])
    w = 0.0 if abs(dt) < 1.0e-18 else (float(t) - float(times[i])) / dt
    return (1.0 - w) * field[i] + w * field[i + 1]


def _plot_m2_layers(twin, truth, ours, dest: Path, metrics: dict) -> None:
    import matplotlib.pyplot as plt

    nx, ny, nz = twin.grid.nx, twin.grid.ny, twin.grid.nz
    it = -1
    gem_p = np.asarray(truth.pressure[it]) / 1.0e6
    ours_p = np.asarray(ours["pressure"][it]) / 1.0e6
    d_p = ours_p - gem_p
    gem_sg = None if truth.sg is None else np.asarray(truth.sg[it])
    ours_sg = np.asarray(ours["sg"][it])
    gem_pm = None if truth.pressure_matrix is None else np.asarray(truth.pressure_matrix[it]) / 1.0e6
    ours_pm = None if "pressure_matrix" not in ours else np.asarray(ours["pressure_matrix"][it]) / 1.0e6
    p_lo = min(float(gem_p.min()), float(ours_p.min()))
    p_hi = max(float(gem_p.max()), float(ours_p.max()))
    dabs = max(abs(float(d_p.min())), abs(float(d_p.max())), 1.0e-6)

    fig, axes = plt.subplots(nz, 3, figsize=(10.2, 3.2 * nz), constrained_layout=True)
    if nz == 1:
        axes = np.array([axes])
    for k in range(nz):
        _maps(axes[k, 0], _layer(gem_p, nx, ny, nz, k), f"GEM 裂缝压力  层 k={k}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar="MPa")
        _maps(axes[k, 1], _layer(ours_p, nx, ny, nz, k), f"本模型 裂缝压力  层 k={k}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar="MPa")
        _maps(axes[k, 2], _layer(d_p, nx, ny, nz, k), f"本模型 − GEM  层 k={k}", vmin=-dabs, vmax=dabs, cmap="RdBu_r", cbar="MPa")
    fig.suptitle(
        f"压力场对比  t={metrics['t_s']:.1f} s  均方根误差={metrics['rmse_p_pa']:.0f} Pa  "
        f"归一化均方根误差={metrics['nrmse_p_sigma']:.3f}",
        fontsize=11,
    )
    fig.savefig(dest / "pressure_layers.png", dpi=140)
    plt.close(fig)

    if gem_pm is not None:
        fig, axes = plt.subplots(1, 3 if ours_pm is None else 4, figsize=(12.0, 3.1), constrained_layout=True)
        _maps(axes[0], _layer(gem_p, nx, ny, nz, 0), "GEM 裂缝压力 层0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        _maps(axes[1], _layer(gem_pm, nx, ny, nz, 0), "GEM 基质压力 层0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        _maps(axes[2], _layer(ours_p, nx, ny, nz, 0), "本模型 裂缝压力 层0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        if ours_pm is not None:
            _maps(axes[3], _layer(ours_pm, nx, ny, nz, 0), "本模型 基质压力 层0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        fig.suptitle("裂缝与基质压力（层 k=0）")
        fig.savefig(dest / "continuum_k0.png", dpi=140)
        plt.close(fig)

    if gem_sg is not None:
        d_s = ours_sg - gem_sg
        fig, axes = plt.subplots(nz, 3, figsize=(10.2, 3.2 * nz), constrained_layout=True)
        if nz == 1:
            axes = np.array([axes])
        for k in range(nz):
            _maps(axes[k, 0], _layer(gem_sg, nx, ny, nz, k), f"GEM 气相饱和度  层 k={k}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            _maps(axes[k, 1], _layer(ours_sg, nx, ny, nz, k), f"本模型 气相饱和度  层 k={k}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            sabs = max(abs(float(d_s.min())), abs(float(d_s.max())), 0.05)
            _maps(axes[k, 2], _layer(d_s, nx, ny, nz, k), f"本模型 − GEM  层 k={k}", vmin=-sabs, vmax=sabs, cmap="RdBu_r", cbar="-")
        fig.suptitle(f"气相饱和度对比  均方根误差={metrics['rmse_sg']:.3f}", fontsize=11)
        fig.savefig(dest / "sg_layers.png", dpi=140)
        plt.close(fig)

    i_idx = np.arange(nx)
    gem_p_i = gem_p.reshape(nz, ny, nx).mean(axis=(0, 1))
    ours_p_i = ours_p.reshape(nz, ny, nx).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(i_idx, gem_p_i, "o-", label="GEM 裂缝压力（层平均）")
    ax.plot(i_idx, ours_p_i, "s-", label="本模型 裂缝压力（层平均）")
    if gem_pm is not None:
        ax.plot(i_idx, gem_pm.reshape(nz, ny, nx).mean(axis=(0, 1)), "o--", label="GEM 基质压力")
    if ours_pm is not None:
        ax.plot(i_idx, ours_pm.reshape(nz, ny, nx).mean(axis=(0, 1)), "s--", label="本模型 基质压力")
    ax.set_xlabel("I（入口 → 出口）")
    ax.set_ylabel("压力 / MPa")
    ax.set_xticks(i_idx)
    ax.legend(fontsize=8)
    ax.set_title("注采方向压力剖面")
    fig.savefig(dest / "pressure_profile.png", dpi=140)
    plt.close(fig)


def _plot_physical_3d(
    twin,
    truth,
    ours,
    dest: Path,
    metrics: dict,
    case: Path,
    obs_csv: Path | None,
    ours_series: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    nx, ny, nz = twin.grid.nx, twin.grid.ny, twin.grid.nz
    it = -1
    gem_p = np.asarray(truth.pressure[min(it, truth.pressure.shape[0] - 1)]) / 1.0e6
    ours_p = np.asarray(ours["pressure"][it]) / 1.0e6
    gem_sg = None if truth.sg is None else np.asarray(truth.sg[min(it, truth.sg.shape[0] - 1)])
    ours_sg = np.asarray(ours["sg"][it])
    p_lo = min(float(gem_p.min()), float(ours_p.min()))
    p_hi = max(float(gem_p.max()), float(ours_p.max()))
    p_cbar = "MPa"
    if (p_hi - p_lo) < 0.05:
        p_ref = 50.0
        gem_p = (gem_p - p_ref) * 1.0e6
        ours_p = (ours_p - p_ref) * 1.0e6
        p_lo = min(float(gem_p.min()), float(ours_p.min()))
        p_hi = max(float(gem_p.max()), float(ours_p.max()))
        p_cbar = f"相对 {p_ref:g} MPa / Pa"
    d_p = ours_p - gem_p
    dabs = max(abs(float(d_p.min())), abs(float(d_p.max())), 1.0)
    monitors = _monitor_k_our(case, nz)
    nmon = max(len(monitors), 1)
    fig, axes = plt.subplots(nmon, 3, figsize=(11.0, 3.15 * nmon), constrained_layout=True)
    if nmon == 1:
        axes = np.array([axes])
    for row, (label, k0) in enumerate(monitors):
        _maps(axes[row, 0], _layer(gem_p, nx, ny, nz, k0), f"GEM 压力  {label}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar=p_cbar)
        _maps(axes[row, 1], _layer(ours_p, nx, ny, nz, k0), f"本模型压力  {label}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar=p_cbar)
        _maps(axes[row, 2], _layer(d_p, nx, ny, nz, k0), f"本模型 − GEM  {label}", vmin=-dabs, vmax=dabs, cmap="RdBu_r", cbar="Pa" if p_cbar.endswith("Pa") else "MPa")
    stop = "（提前终止）" if metrics.get("truncated") else ""
    fig.suptitle(
        f"压力场对比  GEM时刻={metrics['t_gem_s']:.2f} s  本模型时刻={metrics['t_s']:.2f} s  "
        f"均方根误差={metrics['rmse_p_pa']:.0f} Pa{stop}",
        fontsize=11,
    )
    fig.savefig(dest / "pressure_monitors.png", dpi=140)
    plt.close(fig)

    if gem_sg is not None:
        d_s = ours_sg - gem_sg
        sabs = max(abs(float(d_s.min())), abs(float(d_s.max())), 0.05)
        fig, axes = plt.subplots(nmon, 3, figsize=(11.0, 3.15 * nmon), constrained_layout=True)
        if nmon == 1:
            axes = np.array([axes])
        for row, (label, k0) in enumerate(monitors):
            _maps(axes[row, 0], _layer(gem_sg, nx, ny, nz, k0), f"GEM 气相饱和度  {label}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            _maps(axes[row, 1], _layer(ours_sg, nx, ny, nz, k0), f"本模型 气相饱和度  {label}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            _maps(axes[row, 2], _layer(d_s, nx, ny, nz, k0), f"本模型 − GEM  {label}", vmin=-sabs, vmax=sabs, cmap="RdBu_r", cbar="-")
        fig.suptitle(f"气相饱和度对比  均方根误差={metrics.get('rmse_sg')}", fontsize=11)
        fig.savefig(dest / "sg_monitors.png", dpi=140)
        plt.close(fig)

    j_inj = _injector_j(twin)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), constrained_layout=True)
    gem_k_ticks = list(range(nz))
    gem_k_labels = [str(k) for k in range(1, nz + 1)]
    xz_kw = dict(
        vmin=p_lo,
        vmax=p_hi,
        cmap="viridis",
        cbar=p_cbar,
        xlabel="I（网格列）",
        ylabel="k（GEM，k=1 为顶）",
        origin="upper",
        yticks=gem_k_ticks,
        yticklabels=gem_k_labels,
    )
    _maps(axes[0], _xz_gem_k(gem_p, nx, ny, nz, j_inj), f"GEM 压力  xz剖面 j={j_inj}", **xz_kw)
    _maps(axes[1], _xz_gem_k(ours_p, nx, ny, nz, j_inj), f"本模型压力  xz剖面 j={j_inj}", **xz_kw)
    xz_kw["cmap"] = "RdBu_r"
    xz_kw["vmin"] = -dabs
    xz_kw["vmax"] = dabs
    xz_kw["cbar"] = "Pa" if p_cbar.endswith("Pa") else "MPa"
    _maps(axes[2], _xz_gem_k(d_p, nx, ny, nz, j_inj), "本模型 − GEM", **xz_kw)
    fig.suptitle("过注入井柱的 xz 剖面（层号与 CMG 一致）")
    fig.savefig(dest / "pressure_xz_inj.png", dpi=140)
    plt.close(fig)

    if obs_csv is None or not obs_csv.is_file():
        return
    from reservoir_backend.io.case import _read_observation_csv

    rows = _read_observation_csv(obs_csv)
    gem_by = {str(r["sensor"]): r for r in rows}
    names = sorted(s.name for s in twin.experiment.sensors if str(s.kind).lower() in {"pressure", "p", "bhp"})
    n = len(names)
    if n == 0:
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 2.35 * nrows), sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    series = ours if ours_series is None else ours_series
    t_ours = np.asarray(series.get("times_s", [metrics["t_s"]]), dtype=float).reshape(-1)
    p_hist = np.asarray(series["pressure"], dtype=float)
    if p_hist.ndim == 1:
        p_hist = p_hist.reshape(1, -1)
    for ax, name in zip(axes, names):
        sensor = next(s for s in twin.experiment.sensors if s.name == name)
        cell = int(np.argmin(np.sum((twin.grid.cell_centers() - np.array([sensor.x, sensor.y, sensor.z])) ** 2, axis=1)))
        ours_p_t = p_hist[:, cell] / 1.0e6
        ax.plot(t_ours[: ours_p_t.size], ours_p_t, "s-", label="本模型", markersize=4)
        gem = gem_by.get(name)
        if gem is not None:
            gt = np.asarray(gem["times"], dtype=float)
            gv = np.asarray(gem["values"], dtype=float) / 1.0e6
            ax.plot(gt, gv, "o-", label="GEM", markersize=4)
        ax.set_title(f"测点 {name}", fontsize=8)
        ax.set_ylabel("压力 / MPa", fontsize=8)
        ax.set_xlabel("时间 / s", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("测点压力（GEM 观测 vs 本模型格子）")
    fig.savefig(dest / "gauges.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, default=None, help="YAML case; default M2 case_dev DPDP")
    p.add_argument(
        "--hidden",
        type=Path,
        default=None,
        help="GEM hidden/ folder",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--t-end", type=float, default=None, help="cap F_ours end time (s); 15³ defaults to first GEM report > 0")
    args = p.parse_args(argv)

    case = Path(args.case) if args.case is not None else None
    if hidden_default := args.hidden:
        hidden = Path(hidden_default)
    elif case is not None:
        hidden = case.parent / "export" / "hidden"
    else:
        run = ROOT / "results" / "lab_v1" / "cmg_gem_run" / "hidden"
        export = ROOT / "examples" / "lab_v1" / "cmg_gem" / "export" / "hidden"
        hidden = run if (run / "pressure.npy").is_file() else export
    dest = Path(args.out) if args.out is not None else (
        ROOT / "results" / "lab_v1" / "cmg_gem_physical_3d_compare" if case is not None
        else ROOT / "results" / "lab_v1" / "cmg_compare"
    )
    dest.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    _setup_chinese_font()

    twin = load_twin_case(case) if case is not None else load_lab_v1(dev=True)
    truth = load_hidden_truth(hidden)
    times = _cap_times(truth.times_s, args.t_end, twin.grid.n_cells)
    theta = (
        theta_true_from_twin(twin)
        if case is not None
        else theta_true_from_spec(twin)
    )
    cache = dest / "ours.npz"
    truncated = False
    cache_valid = False
    if cache.is_file():
        with np.load(cache) as saved:
            cache_valid = (
                "times_s" in saved and "truncated" in saved
                and np.array_equal(saved["times_s"], times)
                and not bool(saved["truncated"])
            )
    if cache_valid:
        blob = np.load(cache)
        truncated = bool(np.asarray(blob["truncated"]).reshape(-1)[0]) if "truncated" in blob.files else False
        ours = {k: blob[k] for k in blob.files if k != "truncated"}
        if "times_s" in ours:
            times = np.asarray(ours["times_s"], dtype=float)
    else:
        ours, times, truncated = _forward_capped(twin, theta, times)
        ours = {k: np.asarray(v) for k, v in ours.items()}
        ours["times_s"] = times
        np.savez(cache, truncated=np.array(truncated), **ours)

    # GEM ASCII reports are 0 / 0.01 / 0.14 / 1 / 3 d. Sample the field at t_ours.
    t_ours = float(times[-1])
    gem_times = np.asarray(truth.times_s, dtype=float)
    p_gem = _field_at(gem_times, truth.pressure, t_ours)
    sg_gem = _field_at(gem_times, truth.sg, t_ours)
    so_gem = _field_at(gem_times, truth.so, t_ours)
    sw_gem = _field_at(gem_times, truth.sw, t_ours)
    pf_gem = _field_at(gem_times, truth.pressure_fracture, t_ours)
    pm_gem = _field_at(gem_times, truth.pressure_matrix, t_ours)
    gem_slice = type(truth)(
        times_s=np.array([t_ours], dtype=float),
        pressure=np.asarray(p_gem).reshape(1, -1),
        sg=None if sg_gem is None else np.asarray(sg_gem).reshape(1, -1),
        so=None if so_gem is None else np.asarray(so_gem).reshape(1, -1),
        sw=None if sw_gem is None else np.asarray(sw_gem).reshape(1, -1),
        pressure_fracture=None if pf_gem is None else np.asarray(pf_gem).reshape(1, -1),
        pressure_matrix=None if pm_gem is None else np.asarray(pm_gem).reshape(1, -1),
        meta=dict(truth.meta),
    )
    ours_last = {k: (v[-1:] if isinstance(v, np.ndarray) and v.ndim >= 1 and k != "times_s" else v) for k, v in ours.items()}
    p_ours = np.asarray(ours["pressure"][-1])
    lo = float(gem_times[gem_times <= t_ours + 1.0e-12].max()) if np.any(gem_times <= t_ours + 1.0e-12) else float(gem_times[0])
    hi = float(gem_times[gem_times >= t_ours - 1.0e-12].min()) if np.any(gem_times >= t_ours - 1.0e-12) else float(gem_times[-1])
    metrics = {
        "hidden": str(hidden),
        "case": None if case is None else str(case),
        "nrmse_p": nrmse_range(p_ours, p_gem),
        "nrmse_p_sigma": nrmse_range(p_ours, p_gem, span_floor=PRESSURE_SPAN_FLOOR_PA),
        "rmse_p_pa": rmse(p_ours, p_gem),
        "gem_pressure_min_max_pa": [float(p_gem.min()), float(p_gem.max())],
        "ours_pressure_min_max_pa": [float(p_ours.min()), float(p_ours.max())],
        "rmse_sg": None if sg_gem is None else rmse(np.asarray(ours["sg"][-1]), sg_gem),
        "t_s": t_ours,
        "t_gem_s": t_ours,
        "gem_bracket_s": [lo, hi],
        "truncated": bool(truncated),
        "n_cells": int(twin.grid.n_cells),
        "shape": [int(twin.grid.nx), int(twin.grid.ny), int(twin.grid.nz)],
        "note": "F_ours(k_GEM) vs GEM hidden (linear in t if no exact report). Not an M2a PASS claim.",
    }
    (dest / "compare.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if twin.grid.nz <= 4 and case is None:
        _plot_m2_layers(twin, gem_slice, ours_last, dest, metrics)
    else:
        obs_csv = hidden.parent / "observations.csv"
        _plot_physical_3d(
            twin,
            gem_slice,
            ours_last,
            dest,
            metrics,
            case or hidden.parent.parent / "case.yaml",
            obs_csv,
            ours_series=ours,
        )

    print(json.dumps({"out": str(dest), **metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
