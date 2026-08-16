"""Protocol A: observations from the same F. This is the generic invert, not CMG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reservoir_backend.apply import attach_two_layer_demo
from reservoir_backend.inverse.parameterization import CoarseFieldParameterization
from reservoir_backend.io.case import load_case
from reservoir_backend.physics.rock import log_permeability
from reservoir_backend.validation.synthetic import evaluate_synthetic, make_two_layer_waterflood

HERE = Path(__file__).resolve().parent
MD = 9.869232667160128e-16


def _injected_pv(twin, t_hist: float) -> dict:
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    pv = float(np.sum(twin.grid.cell_volumes()) * phi)
    q = 0.0
    for c in twin.experiment.controls:
        if c.port_name == "INJ" and c.kind == "rate" and c.values.size:
            q = float(np.max(np.abs(c.values)))
    return {
        "pore_volume_m3": pv,
        "q_inj_m3_s": q,
        "history_s": float(t_hist),
        "injected_pv": float(q * t_hist / max(pv, 1.0e-30)),
    }


def _layer_k(grid, k):
    z = grid.cell_centers()[:, 2]
    mid = float(np.median(z))
    lo = float(np.mean(k[z < mid]))
    hi = float(np.mean(k[z >= mid]))
    return lo, hi


def two_layer_matched() -> dict:
    case = make_two_layer_waterflood(n_times=6, t_end=700.0, seed=2, history_frac=0.85)
    post = case.twin.calibrate(n_ensemble=16, n_assimilations=4, seed=8)
    m = evaluate_synthetic(case, post)
    m["setup"] = "two_layer + matching 2-region θ"
    m["n_theta"] = int(case.twin.parameterization.n_params)
    m["k_lo_md_true"] = float(np.mean(case.k_true[case.twin.parameterization.region_id == 0]) / MD)
    m["k_hi_md_true"] = float(np.mean(case.k_true[case.twin.parameterization.region_id == 1]) / MD)
    m["k_lo_md_post"] = float(np.mean(post.esmda.k_mean[case.twin.parameterization.region_id == 0]) / MD)
    m["k_hi_md_post"] = float(np.mean(post.esmda.k_mean[case.twin.parameterization.region_id == 1]) / MD)
    m.update(_injected_pv(case.twin, 0.85 * 700.0))
    m["_k_true"] = case.k_true
    m["_k_post"] = post.esmda.k_mean
    m["_shape"] = (case.grid.nz, case.grid.ny, case.grid.nx)
    return m


def two_layer_blind_coarse() -> dict:
    case = make_two_layer_waterflood(n_times=6, t_end=700.0, seed=2, history_frac=0.85)
    grid = case.grid
    case.twin.parameterization = CoarseFieldParameterization(grid, 4, 3, 2, phi=0.20)
    case.twin.inverse.prior_mean = float(np.log(5.0e-13))
    post = case.twin.calibrate(n_ensemble=16, n_assimilations=4, seed=8)
    k_post = post.esmda.k_mean
    k_true = case.k_true
    logk = float(np.sqrt(np.mean((log_permeability(k_post) - log_permeability(k_true)) ** 2)))
    clo, chi = _layer_k(grid, k_post)
    tlo, thi = _layer_k(grid, k_true)
    m = {
        "setup": "two_layer truth, invert on 4x3x2 coarse K (no layer map)",
        "n_theta": 24,
        "posterior_logk_rmse": logk,
        "holdout_nrmse": float(post.holdout_rmse),
        "assimilate_nrmse": float(post.assimilate_rmse),
        "contrast_true": float(thi / max(tlo, 1e-30)),
        "contrast_post": float(chi / max(clo, 1e-30)),
        "k_lo_md_true": tlo / MD,
        "k_hi_md_true": thi / MD,
        "k_lo_md_post": clo / MD,
        "k_hi_md_post": chi / MD,
        "_k_true": k_true,
        "_k_post": k_post,
        "_shape": (grid.nz, grid.ny, grid.nx),
    }
    m.update(_injected_pv(case.twin, 0.85 * 700.0))
    return m


def apply_demo() -> dict:
    twin = load_case("config/lab_apply.yaml")
    k_true = attach_two_layer_demo(twin)
    post = twin.calibrate()
    k_post = post.esmda.k_mean
    logk = float(np.sqrt(np.mean((log_permeability(k_post) - log_permeability(k_true)) ** 2)))
    clo, chi = _layer_k(twin.grid, k_post)
    tlo, thi = _layer_k(twin.grid, k_true)
    hist = float(twin.experiment.history_end_s or 500.0)
    m = {
        "setup": "lab_apply.yaml --demo (z-quantile 2-region, same F)",
        "n_theta": int(twin.parameterization.n_params),
        "posterior_logk_rmse": logk,
        "holdout_nrmse": float(post.holdout_rmse),
        "assimilate_nrmse": float(post.assimilate_rmse),
        "contrast_true": float(thi / max(tlo, 1e-30)),
        "contrast_post": float(chi / max(clo, 1e-30)),
        "k_lo_md_true": tlo / MD,
        "k_hi_md_true": thi / MD,
        "k_lo_md_post": clo / MD,
        "k_hi_md_post": chi / MD,
        "_k_true": k_true,
        "_k_post": k_post,
        "_shape": (twin.grid.nz, twin.grid.ny, twin.grid.nx),
    }
    m.update(_injected_pv(twin, hist))
    return m


def _public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(10.2, 3.1 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = np.array([axes])
    for i, row in enumerate(rows):
        nz, ny, nx = row["_shape"]
        kt = np.asarray(row["_k_true"]).reshape(nz, ny, nx)[:, ny // 2, :] / MD
        kp = np.asarray(row["_k_post"]).reshape(nz, ny, nx)[:, ny // 2, :] / MD
        ratio = np.log10(np.maximum(kp, 1e-6) / np.maximum(kt, 1e-6))
        for ax, field, title, cmap, vmin, vmax in (
            (axes[i, 0], kt, f"{row['setup']}\nk_true (md)", "viridis", None, None),
            (axes[i, 1], kp, f"k_post (md)  contrast {row['contrast_post']:.2f}/{row['contrast_true']:.1f}", "viridis", None, None),
            (axes[i, 2], ratio, "log10(k_post/k_true)", "coolwarm", -1.5, 1.5),
        ):
            im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("i")
            ax.set_ylabel("k (z↑)")
            fig.colorbar(im, ax=ax, shrink=0.82)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rows = [two_layer_matched(), two_layer_blind_coarse(), apply_demo()]
    public = [_public(r) for r in rows]
    out = HERE / "protocol_a_generic.json"
    out.write_text(json.dumps(public, indent=2), encoding="utf-8")
    fig = HERE / "figures" / "protocol_a_generic_k.png"
    _plot(rows, fig)
    print(json.dumps(public, indent=2))
    print(f"wrote {out}")
    print(f"wrote {fig}")
