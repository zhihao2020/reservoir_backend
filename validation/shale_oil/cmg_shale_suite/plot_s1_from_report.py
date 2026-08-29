"""Fast S1 plot: IMEX BHP vs F(k_post) after LM (uses s1_inversion_report.json)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

from reservoir_backend.io.shale_case import (  # noqa: E402
    _align_rates_to_imex_bhp,
    _inflate_shale_sigmas,
    twin_from_shale_truth,
)
from reservoir_backend.twin.offline import predict_from_trajectory  # noqa: E402

PSI = 6894.757293168
DAY = 86400.0


def main() -> int:
    report_path = HERE / "s1_inversion_report.json"
    if not report_path.is_file():
        print("missing s1_inversion_report.json — run invert first")
        return 2
    rec = json.loads(report_path.read_text(encoding="utf-8"))
    theta = np.asarray(rec["run_report"]["posterior"]["theta"], dtype=float)
    truth = ROOT / "validation" / "shale_oil" / "cmg_s1_hw5frac" / "truth_s1.json"
    twin = twin_from_shale_truth(truth, n_times=4, fully_implicit=False)
    truth_d = json.loads(truth.read_text(encoding="utf-8"))
    _align_rates_to_imex_bhp(twin, truth_d, n_pass=3)
    _inflate_shale_sigmas(twin, truth_d, mode="cheap")
    post = twin.calibrate  # type: ignore
    rock = twin.rock_from_theta(theta)
    obs = twin.experiment.assimilate_observations()
    times = np.unique(np.concatenate([o.times_s for o in obs]))
    traj = twin.simulate(rock, t_end=float(times.max()), report_times=times)

    names = sorted({o.sensor_name for o in obs})[:4]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    for name in names:
        o = next(x for x in obs if x.sensor_name == name)
        td = o.times_s / DAY
        ax.plot(td, o.values / PSI, "o", ms=5, label=f"{name} IMEX")
        pred = []
        sensor = next(s for s in twin.experiment.sensors if s.name == name)
        for t in o.times_s:
            pred.append(twin.operator.sample(sensor, traj.state_at(float(t))) / PSI)
        ax.plot(td, pred, "-", lw=1.4, label=f"{name} F(k_post)")
    ax.set_xlabel("time (day)")
    ax.set_ylabel("completion pressure (psi)")
    ax.set_title(
        f"S1 IMEX vs F(k_post)  |  nRMSE={rec.get('assimilate_nrmse'):.2f}  "
        f"k_f/k_m={rec.get('k_frac_over_matrix'):.0f}  dp_ratio={rec.get('dp_ratio'):.3f}  "
        f"rate_scale={rec.get('rate_scale'):.2f}"
    )
    ax.legend(ncols=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    FIG.mkdir(parents=True, exist_ok=True)
    dest = FIG / "s1_cmg_vs_fpost.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
