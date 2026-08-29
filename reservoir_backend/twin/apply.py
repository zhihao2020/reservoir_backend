"""Lab application: observations → invert → posterior fields.

This is the product path. It is not a CMG field matcher.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reservoir_backend.domain.types import ObservationSeries
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.physics.rock import Rock
from reservoir_backend.twin.offline import DigitalTwin
from reservoir_backend.synthetic import layered_permeability


def demo_sample_times(twin: DigitalTwin, *, n_hist: int = 5, n_fc: int = 2) -> NDArray[np.float64]:
    """History samples plus at least one time after history_end_s for forecast."""
    ctrl_end = max((float(c.times_s[-1]) for c in twin.experiment.controls), default=700.0)
    hist_end = float(twin.experiment.history_end_s) if twin.experiment.history_end_s is not None else ctrl_end
    hist_end = min(max(hist_end, 0.0), ctrl_end)
    if hist_end <= 0.0:
        hist_end = ctrl_end
    n_hist = max(int(n_hist), 1)
    hist = np.linspace(hist_end / n_hist, hist_end, n_hist)
    if ctrl_end > hist_end + 1.0e-12:
        n_fc = max(int(n_fc), 1)
        step = (ctrl_end - hist_end) / n_fc
        fc = np.linspace(hist_end + step, ctrl_end, n_fc)
        return np.unique(np.concatenate([hist, fc]))
    return np.unique(hist)


def attach_two_layer_demo(
    twin: DigitalTwin,
    *,
    k_lo: float = 2.0e-13,
    k_hi: float = 2.0e-12,
    seed: int = 3,
    holdout: list[str] | tuple[str, ...] | None = None,
) -> NDArray[np.float64]:
    """Fill empty observations from H(F(known-structure truth)) on this case's grid.

    If the case already has a 0/1 region map (layers or a channel), that map is
    the truth. Otherwise fall back to a mid-plane z split.
    """
    grid = twin.grid
    rid = getattr(twin.parameterization, "region_id", None)
    if rid is not None and int(np.max(rid)) >= 1:
        k_true = np.full(grid.n_cells, float(k_lo), dtype=float)
        k_true[np.asarray(rid, dtype=np.int64).ravel() == 1] = float(k_hi)
    else:
        z_cut = grid.origin[2] + 0.5 * grid.size_m()[2]
        k_true = layered_permeability(grid, k_lo, k_hi, z_cut)
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    times = demo_sample_times(twin)
    t_end = float(times[-1])
    traj = twin.simulate(Rock(k_true, np.full(grid.n_cells, phi)), t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    names = set(holdout or ())
    obs = []
    for s in twin.experiment.sensors:
        vals = []
        for i, t in enumerate(traj.times_s):
            rates, bhp = traj.rates_and_bhp_at(float(t))
            vals.append(twin.operator.sample(s, traj.state_at(float(t)), port_rates=rates, port_bhp=bhp))
        va = np.asarray(vals, dtype=float) + rng.normal(0.0, max(s.sigma, 1.0e-12), size=len(vals))
        obs.append(
            ObservationSeries(
                s.name,
                s.kind,
                np.asarray(traj.times_s, dtype=float),
                va,
                np.full(len(vals), max(s.sigma, 1.0e-12)),
                s.name in names,
            )
        )
    twin.experiment.observations = obs
    return k_true


def _layer_means(k: NDArray[np.float64], region_id: NDArray[np.int64]) -> tuple[float, float]:
    rid = np.asarray(region_id, dtype=np.int64).ravel()
    kk = np.asarray(k, dtype=float).ravel()
    lo = float(np.mean(kk[rid == 0]))
    hi = float(np.mean(kk[rid == 1]))
    return lo, hi


# Comparison-not-CMG: last-time Sw/p field nRMSE of F(m_post) vs F(m_true).
# Conservative product gate; not a CMG cell-map match.
SW_FIELD_NRMSE_MAX = 0.50
P_FIELD_NRMSE_MAX = 0.25


def demo_field_gate(sw_field_nrmse: float, p_field_nrmse: float) -> bool:
    """Product pass: displacement field nRMSE only. Contrast / logK / CMG are not required."""
    sw = float(sw_field_nrmse)
    p = float(p_field_nrmse)
    return bool(
        np.isfinite(sw)
        and np.isfinite(p)
        and sw < SW_FIELD_NRMSE_MAX
        and p < P_FIELD_NRMSE_MAX
    )


def accept_demo(twin: DigitalTwin, posterior, k_true: NDArray[np.float64]) -> dict:
    """P0 gate: waterflood similarity + F(m_post) vs F(m_true) Sw/p field nRMSE.

    Contrast / logK / hold-out stay in the report as extras. They are not the pass check.
    """
    from reservoir_backend.twin.offline import predict_from_trajectory, stack_observations

    rid = np.asarray(twin.parameterization.region_id, dtype=np.int64).ravel()
    k_post = np.asarray(posterior.k, dtype=float).ravel()
    k_lo_t, k_hi_t = _layer_means(k_true, rid)
    k_lo_p, k_hi_p = _layer_means(k_post, rid)
    contrast_true = k_hi_t / max(k_lo_t, 1.0e-30)
    contrast_post = k_hi_p / max(k_lo_p, 1.0e-30)
    logk_rmse = float(np.sqrt(np.mean((np.log(k_post) - np.log(np.asarray(k_true, dtype=float).ravel())) ** 2)))
    expand_err = float(np.max(np.abs(k_post - twin.parameterization.expand(posterior.theta))))

    assim = twin.experiment.assimilate_observations()
    stacked = stack_observations(assim)
    times = np.unique(np.concatenate([o.times_s for o in twin.experiment.observations]))
    t_end = float(times[-1])
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    true_hist = twin.simulate(Rock(k_true, np.full(twin.grid.n_cells, phi)), t_end=t_end, report_times=times)
    post_hist = twin.simulate(twin.rock_from_theta(posterior.theta), t_end=t_end, report_times=times)
    d_true = predict_from_trajectory(twin.operator, twin.experiment, true_hist, assim)
    d_post = predict_from_trajectory(twin.operator, twin.experiment, post_hist, assim)
    forward_match = float(np.sqrt(np.mean(((d_post - d_true) / stacked.sigma) ** 2)))
    from reservoir_backend.twin.similarity import report_from_trajectories

    sim = report_from_trajectories(twin, post_hist, true_hist, k=k_post)
    sw_field = float(sim["displacement"]["sw_field_nrmse"])
    p_field = float(sim["displacement"]["p_field_nrmse"])
    sg_post = getattr(post_hist.states[-1], "sg", None)
    sg_true = getattr(true_hist.states[-1], "sg", None)
    if sg_post is not None and sg_true is not None:
        from reservoir_backend.twin.similarity import field_nrmse

        sg_field = field_nrmse(sg_post, sg_true)
        sim.setdefault("displacement", {})["sg_field_nrmse"] = sg_field
    else:
        sg_field = float("nan")

    hold = float(posterior.holdout_rmse)
    forecast = float(posterior.forecast_rmse) if posterior.forecast_rmse is not None else float("nan")
    compositional = str(getattr(twin.physics, "model", "")).lower() in {"compositional", "comp", "eos"}
    if compositional:
        passed = bool(np.isfinite(p_field) and p_field < P_FIELD_NRMSE_MAX)
        if np.isfinite(sg_field):
            passed = passed and sg_field < SW_FIELD_NRMSE_MAX
    else:
        passed = demo_field_gate(sw_field, p_field)
    return {
        "pass": passed,
        "contrast_true": contrast_true,
        "contrast_post": contrast_post,
        "k_lo_md_post": k_lo_p / 9.869233e-16,
        "k_hi_md_post": k_hi_p / 9.869233e-16,
        "posterior_logk_rmse": logk_rmse,
        "forward_match_nrmse": forward_match,
        "sw_field_nrmse": sw_field,
        "sg_field_nrmse": sg_field,
        "p_field_nrmse": p_field,
        "k_vs_expand_max": expand_err,
        "holdout_nrmse": hold,
        "forecast_nrmse": forecast,
        "comparison": "F(m_post) vs F(m_true) plus waterflood similarity; not CMG",
        "gate": (
            f"sw_field_nrmse < {SW_FIELD_NRMSE_MAX} and p_field_nrmse < {P_FIELD_NRMSE_MAX} "
            "(F(m_post) vs F(m_true); comparison-not-CMG)"
        ),
        "similarity": sim,
    }


def mark_holdout(twin: DigitalTwin, names: list[str]) -> None:
    hold = set(names)
    twin.experiment.observations = [
        ObservationSeries(o.sensor_name, o.kind, o.times_s, o.values, o.sigma, o.sensor_name in hold)
        for o in twin.experiment.observations
    ]


def write_observation_csv(path: Path, twin: DigitalTwin) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "sensor", "kind", "value", "sigma", "holdout"])
        for o in twin.experiment.observations:
            for i, t in enumerate(o.times_s):
                w.writerow([f"{float(t):.6g}", o.sensor_name, o.kind, f"{float(o.values[i]):.8g}", f"{float(o.sigma[i]):.6g}", int(o.holdout)])


def plot_posterior_fields(
    grid: CartesianGrid,
    fields: dict[str, NDArray[np.float64]],
    out_dir: Path,
    *,
    k_true: NDArray[np.float64] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    jmid = grid.ny // 2
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    def sl(arr):
        return np.asarray(arr, dtype=float).reshape(nz, ny, nx)[:, jmid, :]

    written: list[Path] = []
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), constrained_layout=True)
    k_hat = fields["k"]
    pressure = fields["pressure"]
    sw = fields["sw"]
    so = fields["so"]
    panels = [
        (axes[0, 0], sl(k_true) if k_true is not None else sl(k_hat), "k_true" if k_true is not None else "k", "viridis"),
        (axes[0, 1], sl(k_hat), "k (posterior)", "viridis"),
        (axes[0, 2], sl(pressure), "p", "coolwarm"),
        (axes[1, 0], sl(sw), "Sw", "YlGnBu"),
        (axes[1, 1], sl(so), "So", "YlOrBr"),
        (axes[1, 2], sl(fields.get("sg", np.zeros_like(sw))), "Sg", "Purples"),
    ]
    for ax, field, title, cmap in panels:
        im = ax.imshow(field, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("i (x)")
        ax.set_ylabel("k (z↑)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    path = out_dir / "posterior_fields_xz.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written
