"""Same two-layer model on several Cartesian meshes. Invert + figures.

Physical box and well/sensor coordinates stay fixed. Only Δx,Δy,Δz change.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VAL = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(VAL), str(HERE)]

from cmg_io.grid_parse import psi_to_pa
from reservoir_backend.domain.types import ControlSeries, Experiment
from reservoir_backend.grid.cartesian import CartesianGrid
from reservoir_backend.inverse.parameterization import RegionParameterization
from reservoir_backend.physics.rock import Rock, log_permeability
from reservoir_backend.ports.flow import FlowPort
from reservoir_backend.twin.offline import DigitalTwin, InverseSpec
from run_invert_eval import DAY_S, MD_TO_M2, _make_obs_from_traj, _physics, _score_k, diverse_sensors

FIG = HERE / "figures"
REPORT = HERE / "grid_compare_report.json"
# Same box as the IMEX ruler: 12×2 ft, 8×2 ft, 6×1.5 ft
LX_FT, LY_FT, LZ_FT = 24.0, 16.0, 9.0
FT_TO_M = 0.3048
K_LO_MD, K_HI_MD = 50.0, 500.0
GRIDS = (
    {"name": "N8", "nx": 8, "ny": 6, "nz": 4},
    {"name": "N12", "nx": 12, "ny": 8, "nz": 6},
    {"name": "N16", "nx": 16, "ny": 10, "nz": 8},
)


def _box_grid(nx: int, ny: int, nz: int) -> CartesianGrid:
    return CartesianGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=np.full(nx, LX_FT / nx * FT_TO_M),
        dy=np.full(ny, LY_FT / ny * FT_TO_M),
        dz=np.full(nz, LZ_FT / nz * FT_TO_M),
    )


def _region(grid: CartesianGrid) -> np.ndarray:
    n_top = grid.nz // 2
    rid = np.zeros(grid.n_cells, dtype=np.int64)
    for c in range(grid.n_cells):
        _i, _j, k = grid.ijk(c)
        if k >= grid.nz - n_top:
            rid[c] = 1
    return rid


def _ports(grid: CartesianGrid) -> tuple[FlowPort, FlowPort]:
    lx, ly, lz = grid.size_m()
    inj = FlowPort.at_point(grid, "INJ", "injector", "pressure", (0.04 * lx, 0.50 * ly, 0.45 * lz), sw_inj=1.0)
    prod = FlowPort.at_point(grid, "PROD", "producer", "pressure", (0.96 * lx, 0.50 * ly, 0.45 * lz))
    return inj, prod


def invert_one(spec: dict) -> dict:
    grid = _box_grid(spec["nx"], spec["ny"], spec["nz"])
    region = _region(grid)
    k_lo = K_LO_MD * MD_TO_M2
    k_hi = K_HI_MD * MD_TO_M2
    k_true = np.where(region == 1, k_hi, k_lo)
    param = RegionParameterization(region, phi=0.30)
    sensors, hold = diverse_sensors(grid, p_sigma=2.5e4, s_sigma=0.04, with_rate=True)
    times = np.array([0.125, 0.25, 0.375, 0.50]) * DAY_S
    p_inj, p_prod = psi_to_pa(3200.0), psi_to_pa(2800.0)
    inj, prod = _ports(grid)
    experiment = Experiment(
        size_m=grid.size_m(),
        sensors=sensors,
        controls=[
            ControlSeries("INJ", "pressure", times, np.full(times.size, p_inj)),
            ControlSeries("INJ", "composition", times, np.full(times.size, 1.0)),
            ControlSeries("PROD", "pressure", times, np.full(times.size, p_prod)),
        ],
        observations=[],
        history_end_s=float(0.375 * DAY_S),
    )
    twin = DigitalTwin(
        grid,
        experiment,
        [inj, prod],
        _physics(p_init=psi_to_pa(3000.0), sw_init=0.20),
        param,
        inverse=InverseSpec(
            n_ensemble=10,
            n_assimilations=3,
            seed=3,
            prior_mean=np.log(100.0 * MD_TO_M2),
            prior_std=0.8,
            n_workers=4,
        ),
    )
    print(f"{spec['name']} {grid.nx}x{grid.ny}x{grid.nz} n={grid.n_cells} ...", flush=True)
    t0 = time.perf_counter()
    traj = twin.simulate(twin.rock_from_k(k_true), t_end=float(times[-1]), report_times=times)
    twin.experiment.observations = _make_obs_from_traj(twin, sensors, hold, times, traj, seed=4)
    post = twin.calibrate()
    elapsed = time.perf_counter() - t0
    out = _score_k(k_true, region, post, k_lo, k_hi)
    out["elapsed_s"] = elapsed
    out["nx"], out["ny"], out["nz"] = grid.nx, grid.ny, grid.nz
    out["n_cells"] = grid.n_cells
    out["name"] = spec["name"]
    out["dx_m"] = float(grid.dx[0])
    out["k_true"] = (k_true / MD_TO_M2).reshape(grid.nz, grid.ny, grid.nx)
    out["k_post"] = (post.esmda.k_mean / MD_TO_M2).reshape(grid.nz, grid.ny, grid.nx)
    out["sw_end"] = traj.states[-1].sw.reshape(grid.nz, grid.ny, grid.nx)
    print(
        f"   contrast {out['k_contrast_post']:.2f}  logk_rmse {out['logk_rmse_post']:.3f}  "
        f"hold {out['holdout_nrmse']:.3f}  t={elapsed:.1f}s",
        flush=True,
    )
    return out


def plot(rows: list[dict]) -> None:
    FIG.mkdir(exist_ok=True)
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 8.2), constrained_layout=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    vmin, vmax = 30.0, 700.0
    for j, row in enumerate(rows):
        jmid = row["ny"] // 2
        for i, (field, title) in enumerate(
            (
                (row["k_true"], f"{row['name']} truth K"),
                (row["k_post"], f"{row['name']} posterior K"),
                (row["sw_end"], f"{row['name']} Sw at 0.5 d"),
            )
        ):
            ax = axes[i, j]
            if i < 2:
                imk = ax.imshow(field[:, jmid, :], origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            else:
                ims = ax.imshow(field[:, jmid, :], origin="lower", aspect="auto", cmap="YlGnBu", vmin=0.2, vmax=0.85)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("i (x)")
            if j == 0:
                ax.set_ylabel("k (z, 0=bottom)")
            ax.axhline(row["nz"] / 2 - 0.5, color="w", ls="--", lw=0.7, alpha=0.85)
    fig.colorbar(imk, ax=axes[:2, :].ravel().tolist(), shrink=0.7, label="k (md)")
    fig.colorbar(ims, ax=axes[2, :].ravel().tolist(), shrink=0.7, label="Sw")
    fig.suptitle("Same two-layer model, three meshes (mid-y slice)", fontsize=12)
    fig.savefig(FIG / "grid_compare_fields.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), constrained_layout=True)
    names = [r["name"] for r in rows]
    cells = [r["n_cells"] for r in rows]
    axes[0].bar(names, [r["k_contrast_post"] for r in rows], color="C0")
    axes[0].axhline(10.0, color="k", ls="--", lw=1)
    axes[0].set_ylabel("K contrast (truth 10)")
    axes[1].bar(names, [r["logk_rmse_post"] for r in rows], color="C1")
    axes[1].set_ylabel("log K RMSE")
    axes[2].bar(names, [r["holdout_nrmse"] for r in rows], color="C2")
    axes[2].set_ylabel("hold-out nRMSE")
    for ax, extra in zip(axes, [f"{c} cells" for c in cells]):
        ax.set_title(extra, fontsize=9)
    fig.suptitle("Invert metrics vs mesh")
    fig.savefig(FIG / "grid_compare_metrics.png", dpi=140)
    plt.close(fig)


def main() -> int:
    rows = [invert_one(g) for g in GRIDS]
    plot(rows)
    slim = []
    for r in rows:
        slim.append({k: v for k, v in r.items() if k not in {"k_true", "k_post", "sw_end"}})
    REPORT.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print("wrote", FIG / "grid_compare_fields.png", FIG / "grid_compare_metrics.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
