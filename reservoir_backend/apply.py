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
from reservoir_backend.validation.synthetic import layered_permeability


def attach_two_layer_demo(
    twin: DigitalTwin,
    *,
    k_lo: float = 2.0e-13,
    k_hi: float = 2.0e-12,
    seed: int = 3,
    holdout: list[str] | tuple[str, ...] | None = None,
) -> NDArray[np.float64]:
    """Fill empty observations from H(F(two-layer truth)) on this case's grid."""
    grid = twin.grid
    z_cut = grid.origin[2] + 0.5 * grid.size_m()[2]
    k_true = layered_permeability(grid, k_lo, k_hi, z_cut)
    phi = float(getattr(twin.parameterization, "phi", 0.20))
    times = twin.experiment.all_times_s()
    if times.size == 0:
        times = np.array([0.0, 300.0, 600.0], dtype=float)
    t_end = float(times[-1])
    traj = twin.simulate(Rock(k_true, np.full(grid.n_cells, phi)), t_end=t_end, report_times=times)
    rng = np.random.default_rng(seed)
    obs = []
    hold = set()
    # keep existing holdout names if the case listed them with no series
    for s in twin.experiment.sensors:
        vals = []
        for i, t in enumerate(traj.times_s):
            rates = traj.port_rates[i] if i < len(traj.port_rates) else {}
            vals.append(twin.operator.sample(s, traj.state_at(float(t)), port_rates=rates))
        va = np.asarray(vals, dtype=float) + rng.normal(0.0, max(s.sigma, 1.0e-12), size=len(vals))
        hold = s.name in set(holdout or ())
        obs.append(
            ObservationSeries(s.name, s.kind, np.asarray(traj.times_s, dtype=float), va, np.full(len(vals), max(s.sigma, 1.0e-12)), hold)
        )
    twin.experiment.observations = obs
    return k_true


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
    k_mean = fields["k_mean"]
    panels = [
        (axes[0, 0], sl(k_true) if k_true is not None else sl(k_mean), "k_true" if k_true is not None else "k_mean", "viridis"),
        (axes[0, 1], sl(k_mean), "k_mean (posterior)", "viridis"),
        (axes[0, 2], sl(fields.get("k_std", np.zeros_like(k_mean))), "k_std", "magma"),
        (axes[1, 0], sl(fields["pressure_mean"]), "p_mean", "coolwarm"),
        (axes[1, 1], sl(fields["sw_mean"]), "Sw_mean", "YlGnBu"),
        (axes[1, 2], sl(fields["so_mean"]), "So_mean", "YlOrBr"),
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
