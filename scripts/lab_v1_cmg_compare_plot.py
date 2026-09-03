"""M2a maps: GEM hidden vs F_ours(theta_true). Caches fields to ours.npz."""

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
    forward_at_theta,
    load_hidden_truth,
    nrmse_range,
    rmse,
    theta_true_from_spec,
)
from reservoir_backend.twin.lab_v1 import load_lab_v1


def _layer(field: np.ndarray, nx: int, ny: int, nz: int, k: int) -> np.ndarray:
    cube = np.asarray(field, dtype=float).reshape(nz, ny, nx)
    return cube[k]


def _maps(ax, arr: np.ndarray, title: str, *, vmin, vmax, cmap: str, cbar: str | None) -> None:
    im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("I")
    ax.set_ylabel("J")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar, fontsize=8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hidden",
        type=Path,
        default=None,
        help="GEM hidden/ folder. Default: latest results/lab_v1/cmg_gem_run/hidden",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "lab_v1" / "cmg_compare")
    args = p.parse_args(argv)
    hidden = Path(args.hidden) if args.hidden is not None else None
    if hidden is None:
        run = ROOT / "results" / "lab_v1" / "cmg_gem_run" / "hidden"
        export = ROOT / "examples" / "lab_v1" / "cmg_gem" / "export" / "hidden"
        hidden = run if (run / "pressure.npy").is_file() else export
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    twin = load_lab_v1(dev=True)
    nx, ny, nz = twin.grid.nx, twin.grid.ny, twin.grid.nz
    truth = load_hidden_truth(hidden)
    cache = dest / "ours.npz"
    if cache.is_file():
        blob = np.load(cache)
        ours = {k: blob[k] for k in blob.files}
    else:
        theta = theta_true_from_spec(twin)
        ours = forward_at_theta(twin, theta, truth.times_s)
        np.savez(cache, **{k: np.asarray(v) for k, v in ours.items()})

    it = -1
    gem_p = np.asarray(truth.pressure[it]) / 1.0e6
    ours_p = np.asarray(ours["pressure"][it]) / 1.0e6
    d_p = ours_p - gem_p
    gem_sg = None if truth.sg is None else np.asarray(truth.sg[it])
    ours_sg = np.asarray(ours["sg"][it])
    gem_pm = None if truth.pressure_matrix is None else np.asarray(truth.pressure_matrix[it]) / 1.0e6
    ours_pm = None if "pressure_matrix" not in ours else np.asarray(ours["pressure_matrix"][it]) / 1.0e6

    metrics = {
        "hidden": str(hidden),
        "nrmse_p": nrmse_range(ours["pressure"], truth.pressure),
        "nrmse_p_sigma": nrmse_range(
            ours["pressure"], truth.pressure, span_floor=PRESSURE_SPAN_FLOOR_PA
        ),
        "rmse_p_pa": rmse(ours["pressure"], truth.pressure),
        "rmse_sg": None if gem_sg is None else rmse(ours_sg, gem_sg),
        "gem_pf_span_pa": float(gem_p.max() - gem_p.min()) * 1.0e6,
        "span_floor_pa": PRESSURE_SPAN_FLOOR_PA,
        "gem_pf_mpa": [float(gem_p.min()), float(gem_p.max())],
        "ours_pf_mpa": [float(ours_p.min()), float(ours_p.max())],
        "diff_p_mpa": [float(d_p.min()), float(d_p.max())],
        "t_s": float(truth.times_s[it]),
        "note": "nrmse_p is the plan formula; nrmse_p_sigma floors the span at 2 kPa",
    }
    (dest / "compare.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    p_lo = min(float(gem_p.min()), float(ours_p.min()))
    p_hi = max(float(gem_p.max()), float(ours_p.max()))
    dabs = max(abs(float(d_p.min())), abs(float(d_p.max())), 1.0e-6)

    fig, axes = plt.subplots(nz, 3, figsize=(10.2, 3.2 * nz), constrained_layout=True)
    if nz == 1:
        axes = np.array([axes])
    for k in range(nz):
        _maps(axes[k, 0], _layer(gem_p, nx, ny, nz, k), f"GEM fracture P  k={k}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar="MPa")
        _maps(axes[k, 1], _layer(ours_p, nx, ny, nz, k), f"F_ours fracture P  k={k}", vmin=p_lo, vmax=p_hi, cmap="viridis", cbar="MPa")
        _maps(axes[k, 2], _layer(d_p, nx, ny, nz, k), f"ours − GEM  k={k}", vmin=-dabs, vmax=dabs, cmap="RdBu_r", cbar="MPa")
    fig.suptitle(
        f"M2a pressure  t={metrics['t_s']:.1f}s  RMSE={metrics['rmse_p_pa']:.0f} Pa  "
        f"NRMSE_σ={metrics['nrmse_p_sigma']:.3f}",
        fontsize=11,
    )
    fig.savefig(dest / "pressure_layers.png", dpi=140)
    plt.close(fig)

    if gem_pm is not None:
        fig, axes = plt.subplots(1, 3 if ours_pm is None else 4, figsize=(12.0, 3.1), constrained_layout=True)
        _maps(axes[0], _layer(gem_p, nx, ny, nz, 0), "GEM Pf k=0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        _maps(axes[1], _layer(gem_pm, nx, ny, nz, 0), "GEM Pm k=0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        _maps(axes[2], _layer(ours_p, nx, ny, nz, 0), "ours Pf k=0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        if ours_pm is not None:
            _maps(axes[3], _layer(ours_pm, nx, ny, nz, 0), "ours Pm k=0", vmin=11.78, vmax=11.90, cmap="viridis", cbar="MPa")
        fig.suptitle("Fracture vs matrix (k=0)")
        fig.savefig(dest / "continuum_k0.png", dpi=140)
        plt.close(fig)

    if gem_sg is not None:
        d_s = ours_sg - gem_sg
        fig, axes = plt.subplots(nz, 3, figsize=(10.2, 3.2 * nz), constrained_layout=True)
        if nz == 1:
            axes = np.array([axes])
        for k in range(nz):
            _maps(axes[k, 0], _layer(gem_sg, nx, ny, nz, k), f"GEM Sg  k={k}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            _maps(axes[k, 1], _layer(ours_sg, nx, ny, nz, k), f"F_ours Sg  k={k}", vmin=0.0, vmax=1.0, cmap="cividis", cbar="-")
            sabs = max(abs(float(d_s.min())), abs(float(d_s.max())), 0.05)
            _maps(axes[k, 2], _layer(d_s, nx, ny, nz, k), f"ours − GEM  k={k}", vmin=-sabs, vmax=sabs, cmap="RdBu_r", cbar="-")
        fig.suptitle(f"M2a gas saturation  RMSE={metrics['rmse_sg']:.3f}", fontsize=11)
        fig.savefig(dest / "sg_layers.png", dpi=140)
        plt.close(fig)

    i_idx = np.arange(nx)
    gem_p_i = gem_p.reshape(nz, ny, nx).mean(axis=(0, 1))
    ours_p_i = ours_p.reshape(nz, ny, nx).mean(axis=(0, 1))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(i_idx, gem_p_i, "o-", label="GEM Pf (layer-mean)")
    ax.plot(i_idx, ours_p_i, "s-", label="F_ours Pf (layer-mean)")
    if gem_pm is not None:
        ax.plot(i_idx, gem_pm.reshape(nz, ny, nx).mean(axis=(0, 1)), "o--", label="GEM Pm")
    if ours_pm is not None:
        ax.plot(i_idx, ours_pm.reshape(nz, ny, nx).mean(axis=(0, 1)), "s--", label="F_ours Pm")
    ax.set_xlabel("I (inlet → outlet)")
    ax.set_ylabel("P (MPa)")
    ax.set_xticks(i_idx)
    ax.legend(fontsize=8)
    ax.set_title("Inlet–outlet pressure")
    fig.savefig(dest / "pressure_profile.png", dpi=140)
    plt.close(fig)

    print(json.dumps({"out": str(dest), **metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
